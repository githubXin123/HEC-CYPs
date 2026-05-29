import os
import json
import argparse
import pickle as pkl
import shutil
import torch
import torch.nn as nn
from torch import optim
from utils.model_scripts import EsmUnimolClassifier
from utils.configurator import Configurator
from utils.pytorchtools import EarlyStopping, set_random_seed, print_now
from utils.featurizer import convert_data_list_to_dataset_
from utils.benchmark import evaluate
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


def main(template_path):
    print(f'Configure the model based on {os.path.basename(template_path)} ...')
    config = Configurator(template_path)
    config.configurate()
    print('Configuration done.')
    set_random_seed(config.seed)
    torch.cuda.set_device(config.cuda_index)
    shutil.copytree(config.train_data, config.data_dir, dirs_exist_ok=True)
    
    isoforms = config.isoforms
    config.best_model_pts = {}
    for isoform in isoforms:
        model_weight_dir = f"/data3/xxx/models/CYPi_pretrained/{config.prj_name}_weight/{isoform}"
        os.makedirs(model_weight_dir)
        print("Current isoform: ", isoform)
        print('Prepare datasets for training and validation ...')
        data_list = pkl.load(open(f"{config.train_data}/{isoform}/intermediate/data_list.pkl", 'rb'))
        data_list_train = [data_item for data_item in data_list if data_item["dataset_type"] == "train"]
        data_list_validate = [data_item for data_item in data_list if data_item["dataset_type"] == "validate"]
        dataset_train = convert_data_list_to_dataset_(data_list_train)
        dataset_validate = convert_data_list_to_dataset_(data_list_validate)
        data_loader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True,
                                       collate_fn=dataset_train.collater, num_workers=config.num_workers)
        data_loader_valid = DataLoader(dataset_validate, batch_size=config.batch_size, shuffle=True,
                                       collate_fn=dataset_validate.collater, num_workers=config.num_workers)
        print('Preparation done.')

        print('Start training ...')
        sumwriter = SummaryWriter(log_dir=config.log_dir)
        model = EsmUnimolClassifier()
        model.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.000005, weight_decay=1e-4)
        early_stopping = EarlyStopping(patience=config.patience, verbose=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 15)
        for epoch in range(config.n_epochs):
            model.train()
            for step, data_batch in enumerate(data_loader_train):
                label_predict_batch = model(data_batch)
                label_true_batch = data_batch['target']['label'].to(torch.long)

                loss = criterion(label_predict_batch, label_true_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step + 1) % 19 == 0:
                    niter = epoch * len(data_loader_train) + step + 1
                    sumwriter.add_scalar("train loss", round(loss.item(), 3), niter)
                    if config.verbose:
                        print('Epoch: {}, Step: {}, Loss: {}'.format(epoch+1, step, round(loss.item(), 3)))
            scheduler.step()

            metric_train, _ = evaluate(model, data_loader_train, isoform, config.prj_dir, config.decision_threshold)
            metric_validate, val_loss = evaluate(model, data_loader_valid, isoform, config.prj_dir, config.decision_threshold)
            print("==================================================================================")
            print('Epoch', epoch+1)
            print('Train', metric_train)
            print('Validate', metric_validate)
            print("==================================================================================")

            medi_model_path = f"{model_weight_dir}/e{epoch+1}_{val_loss:.4f}.pt"
            early_stopping.path = medi_model_path
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping!")
                break
        best_model_path = early_stopping.saved_paths[-1]
        best_epoch = int(os.path.basename(best_model_path).split("_")[0].replace('e', ''))
        best_val_loss = float(os.path.basename(best_model_path).split("_")[1].rstrip(".pt"))
        best_model_path_new = f"{model_weight_dir}/best_e{best_epoch}_{best_val_loss:.4f}.pt"
        best_model = EsmUnimolClassifier()
        best_model.cuda()
        best_model.load_state_dict(torch.load(best_model_path))
        torch.save(best_model.state_dict(), best_model_path_new)            
        config.best_model_pts[isoform] = best_model_path_new
        print('Training done.')
        
    print('Saving config ...')
    config.config_path = os.path.join(config.prj_dir, 'config.json')
    with open(config.config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    print('Config saved.')    
    
if __name__ == '__main__':
    print('############ MODEL-TRAINING ############')
    print_now()
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("-t", "--template", type=str, default="template/model_training.json", help="input train template")
    args = parser.parse_args()
    main(args.template)
    print_now()
    print('############ MODEL-TRAINING ############')