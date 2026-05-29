import numpy as np
import pandas as pd
import pickle as pkl
from datetime import datetime
import random
import os
import pdb
from tqdm import tqdm, trange
from threading import Thread, Lock
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, average_precision_score, \
    jaccard_score, balanced_accuracy_score, matthews_corrcoef, confusion_matrix
from rdkit import Chem
from rdkit.Chem import AllChem
import esm
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from unicore.modules import init_bert_params
from unicore.data import (
    Dictionary, NestedDictionaryDataset, TokenizeDataset, PrependTokenDataset,
    AppendTokenDataset, FromNumpyDataset, RightPadDataset, RightPadDataset2D,
    RawArrayDataset, RawLabelDataset,
)
from unimol.data import (
    KeyDataset, ConformerSampleDataset, AtomTypeDataset,
    RemoveHydrogenDataset, CroppingDataset, NormalizeDataset,
    DistanceDataset, EdgeTypeDataset, RightPadDatasetCoord,
)
from unimol.models.transformer_encoder_with_pair import TransformerEncoderWithPair
from unimol.models.unimol import NonLinearHead, GaussianLayer
from utils.pytorchtools import EarlyStopping 

class UniMolModel(nn.Module):
    def __init__(self):
        super().__init__()
        dictionary = Dictionary.load('./datasets/total/raw/token_list.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), 512, self.padding_idx)
        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=15,
            embed_dim=512,
            ffn_embed_dim=2048,
            attention_heads=64,
            emb_dropout=0.1,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            max_seq_len=512,
            activation_fn='gelu',
            no_final_head_layer_norm=True,
        )

        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, 64, 'gelu'
        )
        self.gbf = GaussianLayer(K, n_edge_type)

        self.apply(init_bert_params)

    def forward(self, sample,):
        net_input = sample['input']
        src_tokens, src_distance, src_coord, src_edge_type = net_input['src_tokens'], net_input['src_distance'], \
                                                             net_input['src_coord'], net_input['src_edge_type']
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        output = {
            "molecule_embedding": encoder_rep,
            "molecule_representation": encoder_rep[:, 0, :],  # get cls token
            "smiles": sample['input']["smiles"],
        }
        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_normalization_cross_attention_1 = nn.LayerNorm(1280) 
        self.cross_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, kdim=1280, vdim=1280, batch_first=True)
        self.layer_normalization_cross_attention_2 = nn.LayerNorm(512)
        self.feed_forward_cross_attention = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
        )

    def forward(self, x, y, padding_mask):
        y = self.layer_normalization_cross_attention_1(y)
        y, _ = self.cross_attention(x, y, y, key_padding_mask=padding_mask)
        y_old = y
        y = self.layer_normalization_cross_attention_2(y)
        y = self.feed_forward_cross_attention(y)
        y = y + y_old
        return y

class TransformerEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_normalization_self_attention_1 = nn.LayerNorm(512) 
        self.self_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, kdim=512, vdim=512, batch_first=True)
        self.layer_normalization_self_attention_2 = nn.LayerNorm(512)
        self.feed_forward_self_attention = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
        )

    def forward(self, x):
        x_old = x
        x = self.layer_normalization_self_attention_1(x)
        x, _ = self.self_attention(x, x, x)
        x = x + x_old
        x_old = x
        x = self.layer_normalization_self_attention_2(x)
        x = self.feed_forward_self_attention(x)
        x = x + x_old
        return x

class EsmUnimolClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.molecule_encoder = UniMolModel()
        self.molecule_encoder.load_state_dict(th.load('/data1/shentao/Downloads/DeepP450/mol_pre_no_h_220816.pt')['model'], strict=False) 
        self.protein_encoder, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D() 
        self.protein_encoder.load_state_dict(th.load("/data1/shentao/Downloads/DeepP450/p450_eukaryota_esm_2.pth")) 
        self.batch_converter = self.alphabet.get_batch_converter(truncation_seq_length=2048)

        self.transformer_layer_cross_attention = TransformerDecoderLayer()
        self.transformer_layer_self_attention = TransformerEncoderLayer()
        self.mlp = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def move_data_batch_to_cuda(self, data_batch):
        data_batch['input'] = {k: v.cuda() if isinstance(v, th.Tensor) else v for k, v in data_batch['input'].items()}
        data_batch['target'] = {k: v.cuda() if isinstance(v, th.Tensor) else v for k, v in data_batch['target'].items()}
        return data_batch

    def forward(self, data_batch):
        data_batch = self.move_data_batch_to_cuda(data_batch)

       
        molecule_encoder_output = self.molecule_encoder(data_batch)
        molecule_embedding = molecule_encoder_output['molecule_embedding']

        
        sequence_batch = data_batch['input']['sequence']
        sequence_batch = [('', sequence) for sequence in sequence_batch]
        _, sequence_batch, token_batch = self.batch_converter(sequence_batch)
        token_batch = token_batch.cuda()

        protein_encoder_output = self.protein_encoder(token_batch, repr_layers=[33], return_contacts=False)
        protein_embedding = protein_encoder_output["representations"][33]

       
        x = self.transformer_layer_cross_attention(molecule_embedding, protein_embedding, None)
        x1 = self.transformer_layer_self_attention(x)
        x2 = x1[:, 0, :] 
        x3 = self.mlp(x2)
        return x3

def print_now():
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(formatted_time) 
                   
def set_random_seed(random_seed=1024):
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    th.manual_seed(random_seed)
    th.cuda.manual_seed(random_seed)
    th.cuda.manual_seed_all(random_seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.enabled = False

def get_smiles_list_(isoform: str):
    data_df = pd.read_csv(f"./datasets/{isoform}/raw/train_in-house.csv")
    smiles_list = data_df["smiles"].tolist()
    smiles_list = list(set(smiles_list))
    print(len(smiles_list))
    return smiles_list

def calculate_molecule_3D_structure_(smiles_list: list, isoform: str):
    n = len(smiles_list)
    global p
    index = 0
    while True:
        mutex.acquire()
        if p >= n:
            mutex.release()
            break
        index = p
        p += 1
        mutex.release()

        smiles = smiles_list[index]
        print(index, ':', round(index / n * 100, 2), '%', smiles)

        molecule = Chem.MolFromSmiles(smiles)
        molecule = AllChem.AddHs(molecule)
        atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        coordinate_list = []
        result = AllChem.EmbedMolecule(molecule, randomSeed=42, useRandomCoords=True, maxAttempts=1000)
        if result != 0:
            print('EmbedMolecule failed', result, smiles)
            mutex.acquire()
            with open(f"./datasets/{isoform}/result/invalid_smiles.txt", 'a') as f:
                f.write('EmbedMolecule failed' + ' ' + str(result) + ' ' + str(smiles) + '\n')
            mutex.release()
            continue
        try:
            AllChem.MMFFOptimizeMolecule(molecule)
        except:
            print('MMFFOptimizeMolecule error', smiles)
            mutex.acquire()
            with open(f"./datasets/{isoform}/result/invalid_smiles.txt", 'a') as f:
                f.write('MMFFOptimizeMolecule error' + ' ' + str(smiles) + '\n')
            mutex.release()
            continue
        coordinates = molecule.GetConformer().GetPositions()

        assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smiles)
        coordinate_list.append(coordinates.astype(np.float32))

        global smiles_to_conformation_dict
        mutex.acquire()
        smiles_to_conformation_dict[smiles] = {'smiles': smiles, 'atoms': atoms, 'coordinates': coordinate_list}
        mutex.release()

def calculate_molecule_3D_structure(isoforms: list):
    '''
    处理每个isoform，读取csv中的SMILES，返回分子三维结构字典pkl\n
    字典key为SMILES，value为字典{'smiles': smiles, 'atoms': atoms, 'coordinates': coordinate_list}
    '''
    for isoform in isoforms:
        print(isoform)
        smiles_list = get_smiles_list_(isoform)
        global smiles_to_conformation_dict
        smiles_to_conformation_dict = {}
        global p
        p = 0
        thread_count = 16
        threads = []
        for i in range(thread_count):
            threads.append(Thread(target=calculate_molecule_3D_structure_, args=(smiles_list, isoform, )))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        pkl.dump(smiles_to_conformation_dict,
                    open(f"./datasets/{isoform}/intermediate/smiles_to_conformation_dict.pkl", 'wb'))
        print('Valid smiles count:', len(smiles_to_conformation_dict))    

def construct_data_list(isoforms: list):
    '''
    处理每个isoform，读取csv及构象字典pkl\n
    返回列表pkl，列表元素为分子字典\n
    分子字典包含原子、坐标、smiles、一级序列、标签、数据集类型
    '''
    for isoform in isoforms:
        data_df = pd.read_csv(f"./datasets/{isoform}/raw/train_in-house.csv")
        smiles_to_conformation_dict = pkl.load(
            open(f"./datasets/{isoform}/intermediate/smiles_to_conformation_dict.pkl", 'rb'))
        data_list = []
        for _, row in data_df.iterrows():
            smiles = row["smiles"]
            if smiles in smiles_to_conformation_dict:
                data_item = {
                    "atoms": smiles_to_conformation_dict[smiles]["atoms"],
                    "coordinates": smiles_to_conformation_dict[smiles]["coordinates"],
                    "smiles": smiles,
                    "sequence": row["sequence"],
                    "label": row["label"],
                    "dataset_type": row["dataset_type"],
                }
                data_list.append(data_item)
        pkl.dump(data_list, open(f"./datasets/{isoform}/intermediate/data_list.pkl", 'wb'))
        
def convert_data_list_to_dataset_(data_list: list, isoform: str):
    dictionary = Dictionary.load(f"./datasets/{isoform}/raw/token_list.txt")
    dictionary.add_symbol("[MASK]", is_special=True)
    smiles_dataset = KeyDataset(data_list, "smiles")
    sequence_dataset = KeyDataset(data_list, "sequence")
    label_dataset = KeyDataset(data_list, "label")
    dataset = ConformerSampleDataset(data_list, 1024, "atoms", "coordinates")
    dataset = AtomTypeDataset(data_list, dataset)
    dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, False)
    dataset = CroppingDataset(dataset, 1, "atoms", "coordinates", 256)
    dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
    token_dataset = KeyDataset(dataset, "atoms")
    token_dataset = TokenizeDataset(token_dataset, dictionary, max_seq_len=512)
    coord_dataset = KeyDataset(dataset, "coordinates")
    src_dataset = AppendTokenDataset(PrependTokenDataset(token_dataset, dictionary.bos()), dictionary.eos())
    edge_type = EdgeTypeDataset(src_dataset, len(dictionary))
    coord_dataset = FromNumpyDataset(coord_dataset)
    coord_dataset = AppendTokenDataset(PrependTokenDataset(coord_dataset, 0.0), 0.0)
    distance_dataset = DistanceDataset(coord_dataset)
    return NestedDictionaryDataset({
        "input": {
            "src_tokens": RightPadDataset(src_dataset, pad_idx=dictionary.pad(), ),
            "src_coord": RightPadDatasetCoord(coord_dataset, pad_idx=0, ),
            "src_distance": RightPadDataset2D(distance_dataset, pad_idx=0, ),
            "src_edge_type": RightPadDataset2D(edge_type, pad_idx=0, ),
            "smiles": RawArrayDataset(smiles_dataset),
            "sequence": RawArrayDataset(sequence_dataset),
        },
        "target": {
            "label": RawLabelDataset(label_dataset),
        }
    })

def convert_data_list_to_data_loader(isoform: str, batch_size: int):
    data_list = pkl.load(open(f"./datasets/{isoform}/intermediate/data_list.pkl", 'rb'))
    # data_list_train = [data_item for data_item in data_list if data_item["dataset_type"] == "train"]
    # data_list_validate = [data_item for data_item in data_list if data_item["dataset_type"] == "validate"]
    data_list_test = [data_item for data_item in data_list if data_item["dataset_type"] == "test"]
    # dataset_train = convert_data_list_to_dataset_(data_list_train, isoform)
    # dataset_validate = convert_data_list_to_dataset_(data_list_validate, isoform)
    dataset_test = convert_data_list_to_dataset_(data_list_test, isoform)
    # data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
    #                                collate_fn=dataset_train.collater, num_workers=8)
    # data_loader_valid = DataLoader(dataset_validate, batch_size=batch_size, shuffle=True,
    #                                collate_fn=dataset_validate.collater, num_workers=8)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                  collate_fn=dataset_test.collater, num_workers=8)
    # return data_loader_train, data_loader_valid, data_loader_test
    return data_loader_test

def evaluate(model, data_loader, isoform, csv_save):
    model.eval()
    label_predict = th.tensor([], dtype=th.float32).cuda()
    label_true = th.tensor([], dtype=th.long).cuda()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    epoch_loss = 0.0
    num_samples = 0 
    with th.no_grad():
        for data_batch in data_loader:
            # for data_batch in tqdm(data_loader):
            label_predict_batch = model(data_batch)

            label_true_batch = data_batch['target']['label'].to(th.long)
            loss = criterion(label_predict_batch, label_true_batch)
            epoch_loss += loss.item()
            num_samples += len(label_predict_batch)
            label_predict = th.cat((label_predict, label_predict_batch.detach()), dim=0)
            label_true = th.cat((label_true, label_true_batch.detach()), dim=0)
    epoch_loss_mean = epoch_loss / num_samples
    label_predict = th.softmax(label_predict, dim=1)
    label_predict = label_predict.cpu().numpy()
    predict_label_bin = []
    for prob in label_predict[:, 1]:
        if prob>=0.5:
            predict_label_bin.append(1)
        else:
            predict_label_bin.append(0)
    predict_label_bin = np.array(predict_label_bin)    
    label_true = label_true.cpu().numpy()

    if csv_save == True:
        df = pd.DataFrame({'label_true': label_true, 'predict_label': predict_label_bin, 'label_predict': label_predict[:, 1]})
        df.to_csv(f"datasets/{isoform}/result/label_predict_test_{isoform}.csv", index=False)

    tn, fp, fn, tp = confusion_matrix(label_true, predict_label_bin).ravel()
    auc_roc = round(roc_auc_score(label_true, label_predict[:, 1]), 3)
    auc_prc = round(average_precision_score(label_true, label_predict[:, 1]), 3)
    accuracy = round(accuracy_score(label_true, predict_label_bin), 3)
    precision = round(precision_score(label_true, predict_label_bin), 3)
    recall = round(recall_score(label_true, predict_label_bin), 3)
    f1_score = round(2 * precision * recall / (precision + recall), 3)
    mcc = round(matthews_corrcoef(label_true, predict_label_bin), 3)
    jaccard = round(jaccard_score(label_true, predict_label_bin), 3)
    balanced_accuracy = round(balanced_accuracy_score(label_true, predict_label_bin), 3)
    metric = {'auc_roc': auc_roc, 'auc_prc': auc_prc, 'accuracy': accuracy, "balanced_accuracy": balanced_accuracy,
              'precision': precision, 'recall': recall, 'f1_score': f1_score, 'mcc':mcc, "jaccard": jaccard}
    print('Confusion matrix ', f"TN:{tn}, FP:{fp}, FN:{fn}, TP:{tp}")
    return metric, epoch_loss_mean

def train(data_loader_train, data_loader_validate, data_loader_test, isoform: str, max_bearable_epoch: int):
    model = EsmUnimolClassifier()
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 15)

    early_stopping = EarlyStopping(patience=5, verbose=True)
    for epoch in range(max_bearable_epoch):
        model.train()
        for step, data_batch in enumerate(data_loader_train):
            label_predict_batch = model(data_batch)
            label_true_batch = data_batch['target']['label'].to(th.long)

            loss = criterion(label_predict_batch, label_true_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 19 == 0:
                print('Epoch: {}, Step: {}, Loss: {}'.format(epoch+1, step, round(loss.item(), 3)))

        scheduler.step()

        metric_train, _ = evaluate(model, data_loader_train, isoform, csv_save=False)
        metric_validate, val_loss = evaluate(model, data_loader_validate, isoform, csv_save=False)
        metric_test, _ = evaluate(model, data_loader_test, isoform, csv_save=False)
        print("==================================================================================")
        print('Epoch', epoch+1)
        print('Train', metric_train)
        print('Validate', metric_validate)
        print('Test', metric_test)
        print("==================================================================================")

        medi_model_path = f"/data1/shentao/models/CYPi_pretrained/weight/{isoform}_e{epoch+1}_{val_loss:.4f}.pt"
        early_stopping.path = medi_model_path
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping!")
            break
    best_model_path = early_stopping.saved_paths[-1]
    best_epoch = int(os.path.basename(best_model_path).split("_")[1].replace('e', ''))
    best_val_loss = float(os.path.basename(best_model_path).split("_")[2].rstrip(".pt"))
    best_model_path_new = f"/data1/shentao/models/CYPi_pretrained/weight/{isoform}_best_e{best_epoch}_{best_val_loss:.4f}.pt"
    best_model = EsmUnimolClassifier()
    best_model.cuda()
    best_model.load_state_dict(th.load(best_model_path))
    th.save(best_model.state_dict(), best_model_path_new)
        
# def test(data_loader_train, data_loader_validate, data_loader_test, isoform, weight_path):

#     model = EsmUnimolClassifier()
#     model.cuda()
#     model.load_state_dict(th.load(weight_path))

#     metric_train, _ = evaluate(model, data_loader_train, isoform, csv_save=False)
#     metric_validate, _ = evaluate(model, data_loader_validate, isoform, csv_save=False)
#     metric_test, _ = evaluate(model, data_loader_test, isoform, csv_save=True)
#     print("Train", metric_train)
#     print("Validate", metric_validate)
#     print("Test", metric_test)
    
def test(data_loader_test, isoform, weight_path):

    model = EsmUnimolClassifier()
    model.cuda()
    model.load_state_dict(th.load(weight_path))

    metric_test, _ = evaluate(model, data_loader_test, isoform, csv_save=True)
    print("Test", metric_test)
                        
if __name__ == "__main__":
    set_random_seed(42)
    cuda_index = 0
    th.cuda.set_device(cuda_index)
    isoform_list = ['cyp1a2', 'cyp2c9', 'cyp2c19', 'cyp2d6', 'cyp3a4', 'total']
    mutex = Lock()
    calculate_molecule_3D_structure(isoform_list)

    construct_data_list(isoform_list)
    
    # isoform_single = "cyp2c9"
    # batch_size = 2
    # train_loader, valid_loader, test_loader = convert_data_list_to_data_loader(isoform_single, batch_size)
    # test_loader = convert_data_list_to_data_loader(isoform_single, batch_size)
    
    # print("train start!")
    # print_now()
    # n_epoch = 20
    # train(train_loader, valid_loader, test_loader, isoform_single, n_epoch)
    # print("train done!")
    # print_now()

    # batch_size = 2
    # n_epoch = 20
    # for isoform_single in isoform_list[2:]:
    #     train_loader, valid_loader, test_loader = convert_data_list_to_data_loader(isoform_single, batch_size)
    #     print(isoform_single)
    #     print("train start!")
    #     print_now()
    #     train(train_loader, valid_loader, test_loader, isoform_single, n_epoch)
    #     print("train done!")
    #     print_now()      
          
    # print("test start!")
    # print_now()
    # test(train_loader, valid_loader, test_loader, isoform_single,
    #      weight_path=f"/data1/shentao/models/CYPi_pretrained/weight/{isoform_single}/{isoform_single}_e4_0.3493.pt")
    # test(test_loader, isoform_single,
    #      weight_path=f"/data1/shentao/models/CYPi_pretrained/weight/{isoform_single}/{isoform_single}_best_e4_0.4103.pt")
    # print("test done!")
    # print_now()