import mne
import torch
import warnings
import traceback
import numpy as np
import pandas as pd
from silver.io import EEGInterface
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from eeg2vec.pre_process import normalize_scale_interpolate, annotate_nans

warnings.filterwarnings("ignore", category=RuntimeWarning)


def process_eegs(uid: str, 
                 gcs_root: str, 
                 output_path: str,
                 tmin: int = 0,
                 tlen: int = 60,
                 new_sfreq: int =256,
                 reference = "average"):
    
    """ 
     This function does the following:
        1. Loads EEG data from GCS using UID
        2. Creates epochs from annotations
        3. Normalizes, scales, and interpolates the data
        4. Returns a dictionary of data, label, and UID
        
    Parameters
    ----------
    uid : str
        Unique ID of the EEG data
    gcs_root : str
        GCS root path of the EEG data
    tmin : int
        Start time of the EEG Epoch
    tlen : int
        Length of the EEG Epoch
    new_sfreq : int
        New sampling frequency
        
    Returns
    -------
    result : dict
        Dictionary of data, label, and UID
    """
    try:
        print(uid)
        
        final_result = defaultdict(list)
        
        EEG_20_div = ['fp1', 'fp2', 
                    'f7', 'f3', 'fz', 'f4', 'f8',
                    't7', 'c3', 'cz', 'c4', 't8', 
                    'p7', 'p3', 'pz', 'p4', 'p8', 
                    'o1', 'o2']
        
        # Initialize EEGInterface
        eeg_api = EEGInterface(gcs_root, 
                            file_format="parquet",
                                cache_size=None)
        
        # load eeg from gcs
        eeg = eeg_api.load_from_gcs(uid=uid, 
                                    channels = EEG_20_div,
                                    populate_missing_channels=False,
                                    return_metadata =False)
        
        # average referencing
        if reference == "average":
            eeg = eeg.set_eeg_reference(ref_channels='average', projection=False, verbose=False)
        
        # annotations
        eeg = annotate_nans(eeg)
        
        # epochs
        epochs = mne.make_fixed_length_epochs(eeg, 
                                              duration=60, 
                                              overlap=0,
                                              preload=True, 
                                              reject_by_annotation=True)
        
        # ignore first 5 minutes and last 1 minute
        good_epochs = epochs[5:-1]
        
        for epoch_index, epoch in enumerate(good_epochs):
            x = normalize_scale_interpolate(epoch,
                                            sequence_len=tlen,
                                            new_sfreq=new_sfreq,
                                            dataset_max = 0.007,
                                            dataset_min = -0.007)
            
            if x is not None:
                result = { 'UID': [uid],
                           'data': [x.numpy()],
                           'epoch': [epoch_index]
                }
                pd.DataFrame(result).to_pickle(f"{output_path}/{uid}_{epoch_index}.pkl")
                
                final_result['UID'] += [uid]
                final_result['epoch'] += [epoch_index]
                final_result['file_path'] += [f"{output_path}/{uid}_{epoch_index}.pkl"]
        
        return final_result
        
    except Exception as e:
        print(e)
        print(uid)
        traceback.print_exc()
        
        return {"UID" : [],
                "epoch": [],
                "file_path": []
               }


def pre_process(df: pd.DataFrame, 
                gcs_root,
                output_path: str,
                output_dataset: str):
    
    
    # Parallelize the process
    uids = df['UID'].to_list()
    gcs_root = [gcs_root] * len(uids)
    output_path = [output_path] * len(uids)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_eegs, uids, gcs_root, output_path))
    
    # Create a dataframe from the results
    result_df = pd.concat([pd.DataFrame(d) for d in results], ignore_index=True)

    # Save the dataframe in pkl format
    result_df.to_csv(output_dataset)

if __name__ == "__main__":

    df = pd.read_csv("gs://ml-8880-phi-shared-aif-us-p/eeg_bendr/pretraining/datasets/v20230819/eeg_pretraining_15984.csv").head(8)
    gcs_root = "gs://ml-8880-phi-shared-aif-us-p/eeg_prod/processed_parquet/eeg"
    
    print(" Processing Data")
    
    pre_process(df = df, 
                gcs_root = gcs_root, 
                output_path = "gs://ml-8880-phi-shared-aif-us-p/eeg_bendr/pretraining/pre_processed_data/v20230819/mayo_eeg_pretraining_15984_epochs",
                output_dataset = "gs://ml-8880-phi-shared-aif-us-p/eeg_bendr/pretraining/datasets/v20230819/mayo_eeg_pretraining_15984_epochs.csv")

    

    
