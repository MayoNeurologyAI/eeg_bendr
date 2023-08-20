import mne
import torch
import logging
import argparse
import warnings
import traceback
import subprocess
import numpy as np
import pandas as pd
from silver.io import EEGInterface
from collections import defaultdict
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from eeg2vec.pre_process import normalize_scale_interpolate, annotate_nans

warnings.filterwarnings("ignore", category=RuntimeWarning)

def execute_shell_command(command_string : str, verbose=1) -> bool:
    """
    Execute shell commands
    
    Parameters
    ----------
    command_string : str
        Command to execute
    verbose: int, default=1
        log verbose statements
        
    Returns
    -------
    bool: True/False
    
    """
    try:
        _mv_process = subprocess.Popen(command_string,
                                        shell=True,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
        stdout, stderr = _mv_process.communicate()
    except (OSError, KeyboardInterrupt) as e:
        _mv_process.kill()

    if _mv_process.returncode == 0:
        # if there is no command output then return True
        command_output = stdout.decode('utf-8')
        if command_output:
            return command_output
        else:
            return True

    if verbose >= 1:
        logging.error(stderr.decode('utf-8'))
        time_record = datetime.now(timezone.utc)
        logging.error(f"[{time_record}] Moving study failed")

    return False

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
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=str, default="", help="dir to save logs")
    args = parser.parse_args()
    
    if args.job_dir:
        # Configure logging
        log_file_path = "./mayo_eegs_pretrained.log"
        logging.basicConfig(level=logging.INFO, 
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            handlers=[
                                logging.FileHandler(log_file_path),  # Log to a file
                                logging.StreamHandler()  # Log to the console
                            ])
        
    

    df = pd.read_csv("gs://ml-8880-phi-shared-aif-us-p/eeg_bendr/pretraining/datasets/v20230819/eeg_pretraining_15984.csv")
    gcs_root = "gs://ml-8880-phi-shared-aif-us-p/eeg_prod/processed_parquet/eeg"
    
    print(" Processing Data")
    
    pre_process(df = df, 
                gcs_root = gcs_root, 
                output_path = "gs://ml-8880-phi-shared-aif-us-p/eeg_bendr/pretraining/pre_processed_data/v20230819/mayo_eeg_pretraining_15984_epochs",
                output_dataset = "gs://ml-8880-phi-shared-aif-us-p/eeg_bendr/pretraining/datasets/v20230819/mayo_eeg_pretraining_15984_epochs.csv")

    
    if args.job_dir:
        # write log file to job-dir
        command = f"gsutil -m cp -r {log_file_path} {args.job_dir}/"
        status = execute_shell_command(command)
    
