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

def _plot_curve(train_losses, 
                 valid_losses, 
                 output_path,
                 use_log_scale=False, 
                 metric="loss"
                 ):
    """
    Plots training and testing losses and saves the plot to a specified path.

    Parameters:
    ------------
    train_losses (list of float)
        List of training loss values
    test_losses (list of float)
        List of testing loss values
    use_log_scale (bool)
        Whether to plot on a logarithmic scale
    save_path (str): 
        Path where the plot will be saved

    Returns
    -------
        None
    """

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))

    if use_log_scale:
        plt.yscale('log')
        plt.ylabel(f'Log({metric})')
    else:
        plt.ylabel(f'{metric}')

    # Plotting train loss
    plt.plot(epochs, train_losses, label=f'Training {metric}', marker='o', linestyle='-')
    
    # Plotting test loss
    plt.plot(epochs, valid_losses, label=f'Validation {metric}', marker='o', linestyle='-')

    plt.title(f'Training and Validation {metric}')
    plt.xlabel('Epochs')
    plt.legend()

    # Saving the figure
    Path(f"{output_path}/images").mkdir(parents=True, exist_ok=True)
    if use_log_scale:
        plt.savefig(f'{output_path}/images/{metric}_log_curve.png')
    else:
        plt.savefig(f'{output_path}/images/{metric}_curve.png')
        
    plt.close()

def get_foundational_model() -> FoundationalModel:
    """ 
    This function returns a Pytorch FoundationalModel object to pre-train mayo EEGs
    
    Returns
    -------
    model : FoundationalModel
    """
    
    encoder = Encoder(in_features=20, encoder_h=512, dropout=0.0)
    contextualizer = Contextualizer(in_features=512, layer_drop=0.01)
    model = FoundationalModel(encoder, 
                              contextualizer, 
                              mask_rate=0.065, 
                              mask_span=10, 
                              temp=0.1, 
                              encoder_grad_frac=0.1, 
                              num_negatives=20, 
                              enc_feat_l2=1.0,
                              multi_gpu=True)
    
    return model

def train_model(model, 
                train_loader, 
                valid_loader, 
                optimizer,
                scheduler=None, 
                epochs=100, 
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
                output_path="./models/"):
    """ 
    This function trains a Pytorch FoundationalModel object to pre-train mayo EEGs
    
    Parameters
    ----------
    model : FoundationalModel
        
    train_loader : torch.utils.data.DataLoader
        Training data loader
    
    valid_loader : torch.utils.data.DataLoader
        Validation data loader
    
    optimizer : torch.optim.Optimizer
        Optimizer to use for training
        
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler
    
    epochs : int
        Number of epochs to train for
    
    device : torch.device
        Device to use for training
        
    output_path : str
        Path to save model checkpoints to
    
    """
    torch.autograd.set_detect_anomaly(True)
    train_loss = []
    train_scores = []
    valid_scores = []
    valid_loss = []
    for epoch in range(epochs):
        # Ensure the network is in train mode
        model.train()  
        total_loss = 0
        total_score = 0
        for i, inputs in enumerate(train_loader):
            inputs= inputs.float().to(device)
            
            if torch.isnan(inputs).any():
                print("Nans in the input")
            
            logits, z, mask, embedding = model(inputs)
            outputs = [logits, z, mask]
            
            # calculate the loss
            loss = model.calculate_loss(outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # calculate the accuracy
            total_score += FoundationalModel.contrastive_accuracy(outputs)
            
        # Calculate the average loss and accuracy for this epoch
        avg_loss = total_loss / len(train_loader)
        avg_score = total_score / len(train_loader) 

        # Store the metrics for this epoch
        train_loss.append(avg_loss)
        train_scores.append(avg_score)
        
        valid_score, val_loss = test_model(model, valid_loader, device)
        valid_scores.append(valid_score)
        valid_loss.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Training: {avg_score:.4f}, Validation: {valid_score:.4f}')
        
        # Save the model every 50 epochs
        Path(f"{output_path}/checkpoints").mkdir(parents=True, exist_ok=True)
        if (epoch+1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_score': train_scores,
                'valid_loss': valid_loss,
                'valid_score': valid_scores,
            }, f"{output_path}/checkpoints/checkpoint_{epoch+1}.pth")
        
        if scheduler is not None:
            scheduler.step()
        
        torch.cuda.empty_cache()
    
    return train_loss, train_scores, valid_loss, valid_scores

def test_model(model, test_loader, device):
    # Ensure the network is in eval mode
    model.eval()  
    with torch.no_grad():
        total_score = 0
        total_loss = 0
        for inputs in test_loader:
            inputs = inputs.float().to(device)
            
            logits, z, mask, embedding = model(inputs)
            outputs = [logits, z, mask]
            
            # calculate the loss
            total_loss += model.calculate_loss(outputs)
            
            # calculate the accuracy
            total_score += FoundationalModel.contrastive_accuracy(outputs)

    score = total_score / len(test_loader) 
    loss = total_loss / len(test_loader)
    
    return score, loss

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_dir", type=str, default="", help="dir to save logs")
    parser.add_argument("--jobs", type=int, default=8, help="number of jobs for dataloading")
    args = parser.parse_args()
    
    if args.job_dir:
        # Configure logging
        log_file_path = initialize_logging(name="mayo_eeg_pretraining_model")
    
    # Load the meta data
    df = pd.read_csv("gs://ml-8880-phi-shared-aif-us-p/eeg_bendr/pretraining/"
                     "datasets/v20230819/mayo_eeg_pretraining_10405_epochs.csv", index_col=0).head(10000)
    
    print(f"Number of unique UIDs: {len(df['UID'].unique())}")
    
    # Set the random seed
    np.random.seed(42)
    
    # Split the data into train, test, and eval
    train_df, valid_df, test_df = _split_data(df)
    
    print(f"Number of train UIDs: {len(train_df['UID'].unique())}, Epochs: {len(train_df)}")
    print(f"Number of val UIDs: {len(valid_df['UID'].unique())}, Epochs: {len(valid_df)}")
    print(f"Number of test UIDs: {len(test_df['UID'].unique())}, Epochs: {len(test_df)}")
    
    # Create the train, valid and test datasets
    epoch_transforms = transforms.Compose([
                        UidToEpoch()
                        ])
    
    dataset_train = EpochDataset(train_df, transform=epoch_transforms)
    dataset_test = EpochDataset(test_df, transform=epoch_transforms)
    dataset_valid = EpochDataset(valid_df, transform=epoch_transforms)
    
    # create the collate function
    collate_fn = CollateEpochs(transform=epoch_transforms)
    
    # Create the train, valid and test data loaders
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=64, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    
    # check if dataloader is working
    start_time = time.time()
    batch_train = next(iter(train_loader))
    time_elapsed = time.time() - start_time
    print (f"Train batch shape: {batch_train.shape}, Time elapsed: {time_elapsed} seconds")
    
    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print (f"Using device: {device}")
    
    # Get the foundational model
    model = get_foundational_model().to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    # Set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=0.00002, 
                                 weight_decay=0.01, 
                                 betas=[0.9, 0.98])
    
    print(f"Model Summary: {model}")
    
    # train the model
    output_path = "./pretraining_results/"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    train_loss, train_scores, valid_loss, valid_scores = train_model(model = model, 
                                                                    train_loader=train_loader,
                                                                    valid_loader=valid_loader,
                                                                    optimizer=optimizer,
                                                                    epochs=100,
                                                                    device=device,
                                                                    output_path=output_path)
    
    # Plot the training and validation loss and accuracy
    _plot_curve(train_losses=train_loss, 
                valid_losses=valid_loss, 
                metric= "loss", 
                use_log_scale=False, 
                output_path=output_path)
    
    # Plot the log training and validation loss and accuracy
    _plot_curve(train_losses=train_loss, 
                valid_losses=valid_loss, 
                metric= "loss", 
                use_log_scale=True, 
                output_path=output_path)
    
    # plot the training and validation accuracy
    _plot_curve(train_losses=train_scores, 
                valid_losses=valid_scores, 
                metric= "Accuracy", 
                use_log_scale=False, 
                output_path=output_path)
    
    # test the model
    test_score, test_loss = test_model(model, test_loader, device)
    print(f"Test Accuracy: {test_score:.4f}, Test Loss: {test_loss:.4f}")
    
    if args.job_dir:
        # write log file to job-dir
        save_log_to_gcs(log_file_path, args.job_dir)
        save_log_to_gcs(output_path, args.job_dir)