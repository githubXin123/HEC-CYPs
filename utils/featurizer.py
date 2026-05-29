from .globals import mutex
from .configurator import Configurator
from rdkit import Chem
from rdkit.Chem import AllChem
from threading import Thread
import pandas as pd
import numpy as np
import pickle as pkl
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

def get_smiles_list_(isoform: str, prj_dir: str, task: str):
    data_df = pd.read_csv(f"{prj_dir}/datasets/{isoform}/raw/{task}.csv")
    smiles_list = data_df["smiles"].tolist()
    smiles_list = list(set(smiles_list))
    return smiles_list

def calculate_molecule_3D_structure_(smiles_list: list, isoform: str, prj_dir: str, verbose: bool):
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
        if verbose:
            print(index, ':', round(index / n * 100, 2), '%', smiles)
        molecule = Chem.MolFromSmiles(smiles)
        molecule = AllChem.AddHs(molecule)
        atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        coordinate_list = []
        result = AllChem.EmbedMolecule(molecule, randomSeed=42, useRandomCoords=True, maxAttempts=1000)
        if result != 0:
            print('EmbedMolecule failed', result, smiles)
            mutex.acquire()
            with open(f"{prj_dir}/datasets/{isoform}/result/invalid_smiles.txt", 'a') as f:
                f.write('EmbedMolecule failed' + ' ' + str(result) + ' ' + str(smiles) + '\n')
            mutex.release()
            continue
        try:
            AllChem.MMFFOptimizeMolecule(molecule)
        except:
            print('MMFFOptimizeMolecule error', smiles)
            mutex.acquire()
            with open(f"{prj_dir}/datasets/{isoform}/result/invalid_smiles.txt", 'a') as f:
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

def calculate_molecule_3D_structure(config: Configurator):
    isoforms = config.isoforms
    for isoform in isoforms:
        smiles_list = get_smiles_list_(isoform, config.prj_dir, config.task)
        global smiles_to_conformation_dict
        smiles_to_conformation_dict = {}
        global p
        p = 0
        thread_count = config.num_workers
        threads = []
        for i in range(thread_count):
            threads.append(Thread(target=calculate_molecule_3D_structure_, args=(smiles_list, isoform, config.prj_dir, config.verbose)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        pkl.dump(smiles_to_conformation_dict,
                    open(f"{config.prj_dir}/datasets/{isoform}/intermediate/smiles_to_conformation_dict.pkl", 'wb'))
        print(f"Valid smiles count in {isoform}:", len(smiles_to_conformation_dict))

def convert_data_list_to_dataset_(data_list: list):
    dictionary = Dictionary.load(f"./datasets/cyp1a2/raw/token_list.txt")
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

def convert_data_list_to_dataset_pred(data_list: list):
    dictionary = Dictionary.load(f"./datasets/cyp1a2/raw/token_list.txt")
    dictionary.add_symbol("[MASK]", is_special=True)
    smiles_dataset = KeyDataset(data_list, "smiles")
    sequence_dataset = KeyDataset(data_list, "sequence")
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
        }
    })