import time
import torch
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
from eeg2vec.dataset import EpochDataset, CollateEpochs
from eeg2vec.transforms import UidToEpoch
from eeg2vec.model import FoundationalModel, Encoder, Contextualizer

from utils import *

def _split_data(df, train_prop=0.9, eval_prop=0.05, test_prop=0.05):
    """ 
    This function splits the data into train, test, and eval sets
    such that each set contains unique UIDs
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the UIDs to split
    
    train : float
        Proportion of data to use for training
    
    eval : float
        Proportion of data to use for evaluation
    
    test : float
        Proportion of data to use for testing
    
    """

    # Ensure the proportions sum to 1
    assert train_prop + test_prop + eval_prop == 1, "Proportions do not sum to 1."

    uids = np.random.permutation(df['UID'].unique())

    # Determine the size of each split
    train_size = int(train_prop * len(uids))
    test_size = int(test_prop * len(uids))

    # Split the UIDs accorsing to the proportions
    train_uids = uids[:train_size]
    test_uids = uids[train_size:train_size+test_size]
    eval_uids = uids[train_size+test_size:]
    
    # Split the data
    train_df = df[df['UID'].isin(train_uids)].sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = df[df['UID'].isin(test_uids)].sample(frac=1, random_state=42).reset_index(drop=True)
    eval_df = df[df['UID'].isin(eval_uids)].sample(frac=1, random_state=42).reset_index(drop=True)

    return train_df, eval_df, test_df

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_dir", type=str, default="", help="dir to save logs")
    parser.add_argument("--jobs", type=int, default=8, help="number of jobs for dataloading")
    args = parser.parse_args()
    
    if args.job_dir:
        # Configure logging
        log_file_path = initialize_logging(name="mayo_eegs_make_batches")
    
    # Load the meta data
    df = pd.read_csv("gs://ml-8880-phi-shared-aif-us-p/eeg_bendr/pretraining/"
                     "datasets/v20230819/mayo_eeg_pretraining_10405_epochs.csv", index_col=0)
    
    # Split the data
    train_df, eval_df, test_df = _split_data(df, train_prop=0.8, eval_prop=0.1, test_prop=0.1)
    
    
    
    
    
    
    