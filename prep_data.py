import os
import json
import argparse
import shutil
import pandas as pd
import pickle as pkl
from utils.globals import mutex
from utils.configurator import Configurator
from utils.pytorchtools import set_random_seed, print_now
from utils.featurizer import calculate_molecule_3D_structure


def main(template_path):
    print(f'Configure the model based on {os.path.basename(template_path)} ...')
    config = Configurator(template_path)
    config.configurate()
    print('Configuration done.')
    
    set_random_seed(config.seed)
    shutil.copytree("./datasets", config.data_dir, dirs_exist_ok=True)
    
    print("Calculating molecule conformation ...")
    calculate_molecule_3D_structure(config)
    print("Calculation done.")

    print("Saving data points ...")
    isoforms = config.isoforms
    for isoform in isoforms:
        data_df = pd.read_csv(f"{config.prj_dir}/datasets/{isoform}/raw/{config.task}.csv")
        smiles_to_conformation_dict = pkl.load(
            open(f"{config.prj_dir}/datasets/{isoform}/intermediate/smiles_to_conformation_dict.pkl", 'rb'))
        data_list = []
        for _, row in data_df.iterrows():
            smiles = row["smiles"]
            if smiles in smiles_to_conformation_dict:
                if config.task == "pred":
                    data_item = {
                        "atoms": smiles_to_conformation_dict[smiles]["atoms"],
                        "coordinates": smiles_to_conformation_dict[smiles]["coordinates"],
                        "smiles": smiles,
                        "sequence": row["sequence"],
                        "dataset_type": row["dataset_type"],
                    }   
                else:                 
                    data_item = {
                        "atoms": smiles_to_conformation_dict[smiles]["atoms"],
                        "coordinates": smiles_to_conformation_dict[smiles]["coordinates"],
                        "smiles": smiles,
                        "sequence": row["sequence"],
                        "label": row["label"],
                        "dataset_type": row["dataset_type"],
                    }
                data_list.append(data_item)
        pkl.dump(data_list, open(f"{config.prj_dir}/datasets/{isoform}/intermediate/data_list.pkl", 'wb'))
    print("All data points saved.")

    print('Saving config ...')
    config.config_path = os.path.join(config.prj_dir, 'config.json')
    with open(config.config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    print('Config saved.')
    
      
if __name__ == '__main__':
    print('############ DATA-PREPARATION ############')
    print_now()
    parser = argparse.ArgumentParser(description="Model Pretraining")
    parser.add_argument("-t", "--template", type=str, default="template/data_preparation.json", help="input data preparation template")
    args = parser.parse_args()
    main(args.template)
    print_now()
    print('############ DATA-PREPARATION ############')