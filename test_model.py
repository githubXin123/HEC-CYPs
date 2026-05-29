import os
import json
import argparse
import pickle as pkl
import shutil
import pandas as pd
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
    shutil.copytree(config.test_data, config.data_dir, dirs_exist_ok=True)

    metrics = []    
    isoforms = config.isoforms
    for isoform in isoforms:
        print("Current isoform: ", isoform)
        print('Prepare datasets for testing ...')
        data_list = pkl.load(open(f"{config.test_data}/{isoform}/intermediate/data_list.pkl", 'rb'))
        data_list_test = [data_item for data_item in data_list if data_item["dataset_type"] == "test"]
        dataset_test = convert_data_list_to_dataset_(data_list_test)
        data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False,
                                       collate_fn=dataset_test.collater, num_workers=config.num_workers)

        print('Preparation done.')
        print('Start testing ...')
        model = EsmUnimolClassifier()
        model.cuda()
        weight_path = config.loaded_model_pts[isoform]
        model.load_state_dict(torch.load(weight_path))

        metric_test, _ = evaluate(model, data_loader_test, isoform, config.prj_dir, config.decision_threshold, csv_save=True)
        if config.verbose:
            print("Test", metric_test)
        df_metric_single = pd.DataFrame(metric_test, index=[0])
        metrics.append(df_metric_single)
        print('Testing done.')
        
    df_metrics = pd.concat(metrics)
    df_metrics.index = isoforms
    df_metrics.to_csv(f"{config.prj_dir}/metrics.csv")
    
    print('Saving config ...')
    config.config_path = os.path.join(config.prj_dir, 'config.json')
    with open(config.config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    print('Config saved.')
            
if __name__ == '__main__':
    print('############ MODEL-TESTING ############')
    print_now()
    parser = argparse.ArgumentParser(description="Model Testing")
    parser.add_argument("-t", "--template", type=str, default="template/model_testing.json", help="input test template")
    args = parser.parse_args()
    main(args.template)
    print_now()
    print('############ MODEL-TESTING ############')