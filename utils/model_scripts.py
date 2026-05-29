import esm
import torch
import torch.nn as nn
from unicore.modules import init_bert_params
from unicore.data import Dictionary
from unimol.models.transformer_encoder_with_pair import TransformerEncoderWithPair
from unimol.models.unimol import NonLinearHead, GaussianLayer


class UniMolModel(nn.Module):
    def __init__(self):
        super().__init__()
        dictionary = Dictionary.load('./datasets/cyp1a2/raw/token_list.txt')
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
        self.molecule_encoder.load_state_dict(torch.load('/data1/shentao/internal/ADME/deploy/HEC-CYPs/inhibition/model_pts/mol_pre_no_h_220816.pt')['model'], strict=False) 
        self.protein_encoder, self.alphabet = esm.pretrained.load_model_and_alphabet_local(
            "/data1/shentao/internal/ADME/deploy/HEC-CYPs/inhibition/model_pts/esm2_t33_650M_UR50D.pt"
        )
        self.protein_encoder.load_state_dict(torch.load("/data1/shentao/internal/ADME/deploy/HEC-CYPs/inhibition/model_pts/p450_eukaryota_esm_2.pth"))
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
        data_batch['input'] = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data_batch['input'].items()}
        if 'target' in data_batch:
            data_batch['target'] = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data_batch['target'].items()}
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