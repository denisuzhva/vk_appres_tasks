# trainer.py

import numpy as np
import pandas as pd
import time
import torch
from torch import nn
from torch.optim import RMSprop, Adam, SGD
from torch.utils.data import sampler, DataLoader 
from dataset import *
import matplotlib.pyplot as plt



def init_weights(m):
    """Xavier weight initializer."""
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight)

def train_model(df_path, model, n_classes, 
                sample_length, sample_channels, 
                batch_size, n_epochs, learning_rate, 
                device, log_df_path, model_dump_path, 
                validate_each_n_epoch,
                shuffle_dataset=True, valid_split=0.2):
    """
    The trainer function for network training and validation.

    Args:
        df_path:                Dataframe for the dataset maker
        model:                  A model for training
        n_classes:              Number of classes for classification
        sample_length:          Length of one sample in a single channel
        sample_channel:         Number of channels in a sample
        batch_size:             Size of a batch
        n_epochs:               Number of epochs
        learning_rate:          Learning rate
        device:                 Current device (cuda or cpu)
        log_df_path:            Path for training logs
        model_dump_path:        Path for the model to be stored
        validate_each_n_epoch:  An interval between epochs with validation performed
        shuffle_dataset:        True if wish to shuffle dataset
        valid_split:            A proportion of the dataset used for validation  
    """

    # Load dataset
    dataset = TIMIT_dataset(df_path, sample_length, sample_channels)
    dataset_size = len(dataset)
    indices = dataset.get_indices()

    # Split for train and validation
    split = int(np.floor(valid_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, valid_indices = indices[split:], indices[:split]

    train_sampler = sampler.SubsetRandomSampler(train_indices)
    valid_sampler = sampler.SubsetRandomSampler(valid_indices)

    train_loader = DataLoader(dataset, 
                              batch_size=batch_size, 
                              sampler=train_sampler,
                              pin_memory=True)
    valid_loader = DataLoader(dataset, 
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              pin_memory=True)

    n_train_batches = len(train_loader) 
    n_valid_batches = len(valid_loader) 

    # Loss and optimizer                 
    crit = nn.NLLLoss()
    optimizer = RMSprop(model.parameters(), lr=learning_rate, alpha=0.95, eps=1e-8)
    model.apply(init_weights)

    # Training iterations
    log_header = True
    start_t = time.time()
    min_valid_loss = 0
    for epoch in range(n_epochs):

        if epoch % validate_each_n_epoch == 0:
            do_eval = True
        else:
            do_eval = False

        if do_eval:
            tr_loss = 0.
            tr_acc = 0.
            vl_loss = 0.
            vl_acc = 0.

        # Training
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.float().to(device)
            labels = labels.to(device)

            pred = model(data)
            loss = crit(pred, labels)

            if do_eval:
                tr_loss += loss.cpu().item()
                pred_class = torch.argmax(pred, dim=1)
                tr_acc += (pred_class == labels).float().mean().cpu().item()
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if do_eval:
            tr_loss = tr_loss / n_train_batches
            tr_acc = tr_acc / n_train_batches

            # Validation
            model.eval()
            for batch_idx, (data, labels) in enumerate(valid_loader):
                data = data.float().to(device)
                labels = labels.to(device)

                pred = model(data)
                loss = crit(pred, labels)
            
                vl_loss += loss.cpu().item()
                pred_classes = torch.argmax(pred, dim=1)
                vl_acc += (pred_classes == labels).float().mean().cpu().item()

            vl_loss = vl_loss / n_valid_batches
            vl_acc = vl_acc / n_valid_batches

            d = {"epoch": [epoch], "train_loss": [tr_loss], "train_accuracy": [tr_acc], "validation_loss": [vl_loss], "validation_accuracy": [vl_acc]}
            print(d)
            df = pd.DataFrame(data=d)
            df.to_csv(log_df_path, mode='a', header=log_header, index=False)
            log_header = False

            torch.save(model, model_dump_path)
    
    print("t: ", time.time() - start_t)




