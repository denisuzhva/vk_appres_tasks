# trainer.py

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import RMSprop
from torch.utils.data import sampler, DataLoader 
from dataloader import *



def train_model(df_path, model, n_classes, 
                sample_length, sample_channels, 
                batch_size, n_epochs, learning_rate, 
                device, shuffle_dataset=True, valid_split=0.2):

    dataset = TIMIT_dataset(df_path, sample_length, sample_channels)
    dataset_size = len(dataset)
    indices = dataset.get_indices()

    # https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
    split = int(np.floor(valid_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(0)
        np.random.shuffle(indices)
    train_indices, valid_indices = indices[split:], indices[:split]

    train_sampler = sampler.SubsetRandomSampler(train_indices)
    valid_sampler = sampler.SubsetRandomSampler(valid_indices)

    train_loader = DataLoader(dataset, 
                              batch_size=batch_size, 
                              sampler=train_sampler)
    valid_loader = DataLoader(dataset, 
                              batch_size=batch_size,
                              sampler=valid_sampler)
                              
    crit = CrossEntropyLoss().to(device)
    optimizer = RMSprop(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):

        epoch_loss = 0.
        #tr_accuracy = 0

        # Train
        for batch_index, (data, labels) in enumerate(train_loader):
            data = data.float().to(device)
            labels = labels.to(device)
            pred = model.forward(data)
            #labels_ohe = torch.nn.functional.one_hot(labels, num_classes=n_classes).to(device)
            loss = crit(pred, labels)
            loss.backward()
            optimizer.step()
            batch_loss = torch.sum(loss)
            epoch_loss += batch_loss.cpu().item()

        print(epoch_loss)

        # Validation
        #for batch_index, (data, labels) in enumerate(valid_loader):
        #    #print(labels)
        #    pass
