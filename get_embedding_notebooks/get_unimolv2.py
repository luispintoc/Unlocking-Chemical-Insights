#!/usr/bin/env python
"""
Script to generate UniMol intermediate-layer embeddings for ADMET tasks.
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.append('../external_repos/')
from UniMol.unimol_tools.unimol_tools.predictor import UniMolRepr

from tdc.benchmark_group import admet_group
from tdc.single_pred import ADME, Tox


def main():
    # pooling method: 'mean' or 'cls'
    method = 'cls'

    # initialize UniMol repr
    clf = UniMolRepr(data_type='molecule',
                     model_name='unimolv2',
                     model_size='84m',
                     remove_hs=True)

    # iterate over ADMET group tasks
    base_dir = '../input_data/tdcommons/admet_group'
    for task in os.listdir(base_dir):
        if task.startswith('.'):
            continue

        try:
            data = ADME(name = task)
        except:
            data = Tox(name = task)

        print(f"Processing task: {task}")
        path = os.path.join(base_dir, task)
        split = data.get_split(method = 'scaffold')

        # load dataframes
        train_df = split['train'].rename({'Drug': 'smiles', 'Y': 'target'}, axis=1).drop('Drug_ID', axis=1)
        test_df = split['test'].rename({'Drug': 'smiles', 'Y': 'target'}, axis=1).drop('Drug_ID', axis=1)

        # for each split
        for df, split in [(train_df, 'train'), (test_df, 'test')]:
            # determine layer count and hidden dim
            num_layers = len(clf.model.encoder.layers)
            hidden_dim = clf.model.args.encoder_embed_dim

            # initialize storage for each layer
            layer_storage = [ [] for _ in range(num_layers) ]

            # iterate over smiles
            for smi in df['smiles'].tolist():
                try:
                    # get representations
                    out = clf.get_repr([smi], return_atomic_reprs=False)
                    inter = out['intermediate_layer_reprs']

                    # pool each layer
                    for j, layer in enumerate(inter):
                        if method == 'cls':
                            emb = layer[0, 0, :]
                        else:
                            emb = layer.mean(axis=1)[0]
                        layer_storage[j].append(emb)

                except Exception as e:
                    # on failure, append NaNs
                    nan_emb = np.full((hidden_dim,), np.nan)
                    for j in range(num_layers):
                        layer_storage[j].append(nan_emb)

            # stack arrays and save per layer
            for i, lst in enumerate(layer_storage):
                arr = np.stack(lst, axis=0)
                cols = [f"univ2_layer_{i}_{method}_{k}" for k in range(arr.shape[1])]
                layer_df = pd.DataFrame(arr, columns=cols)

                out_df = pd.concat([df.reset_index(drop=True), layer_df], axis=1)
                out_path = os.path.join(path, f"{split}_univ2_layer{i}_emb.csv")
                out_df.to_csv(out_path, index=False)

            print(f"Saved {split} embeddings for task {task}")


if __name__ == '__main__':
    main()