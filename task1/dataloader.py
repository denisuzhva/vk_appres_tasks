# dataloader.py

import numpy as np
import pandas as pd 
import soundfile as sf
import torch
from torch.utils.data import Dataset



class TIMIT_dataset(Dataset):
    """TIMIT dataset loader."""
    
    def __init__(self, df_path, sample_len, n_channels):
        """
        Initialize the dataset.

        Args:
            df_path: Path to a TIMIT dataframe prepared using timit_make_labels.py
            sample_len: Length of a data sample
            n_channels: Number of channels in a sample
        """
        super(TIMIT_dataset, self).__init__()
        self.__df = pd.read_csv(df_path)
        self.__sample_len = sample_len
        self.__n_channels = n_channels

    def __len__(self):
        return self.__df.shape[0]

    def __getitem__(self, idx):
        """Get n_channels random samples of the specified sentence."""
        row = self.__df.iloc[idx]
        path = row["path"]
        label = row["label"]
        sentence, _ = sf.read(path)
        sentence = sentence.astype(np.float32)
        dsample = np.zeros((self.__n_channels, self.__sample_len))
        for c in range(self.__n_channels):
            sample_start = torch.randint(sentence.size - self.__sample_len, (1,))
            sample_end = sample_start + self.__sample_len
            dsample[c] = sentence[sample_start:sample_end]
        dsample = torch.from_numpy(dsample)
        return dsample, label

    def get_indices(self):
        indices = self.__df.index.values
        return indices