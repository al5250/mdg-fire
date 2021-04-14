from typing import Iterable, Tuple, Optional
import os

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

import pdb


class MadagascarFiresDataset(Dataset):

    # Class constants
    num_bins = 32
    bin_cols = [f'bin_{s}' for s in range(32)]
    num_bands = 7
    history = 368
    period_length = 16

    def __init__(
        self,
        data_dir: str, 
        train_start: str = '2015-02-01', 
        test_start: str = '2020-02-01', 
        batch_size: int = 32, 
        max_regions: int = 2053
    ):
        self.data_dir = data_dir
        self.train_start = train_start
        self.test_start = test_start
        self.batch_size = batch_size
        self.max_regions = max_regions 

        # Load data
        filenames = sorted(os.listdir(data_dir))
        dfs = []
        for f in filenames:
            if '.csv' in f:
                fpath = f'{data_dir}/{f}'
                print(f'Loading {fpath}...')
                df = pd.read_csv(fpath)
                dfs.append(df)
        self.df = pd.concat(dfs, ignore_index=True)
        self.df['date'] = pd.to_datetime(self.df['date'])

        # Filter based on place and time, split into train and test
        place_time_pairs = self.df[['rect_id', 'date']].drop_duplicates()
        train_time_mask = (place_time_pairs['date'] > self.train_start) & (place_time_pairs['date'] <= self.test_start)
        test_time_mask = place_time_pairs['date'] > self.test_start
        place_mask = place_time_pairs['rect_id'] < self.max_regions
        self.train_place_times = place_time_pairs[train_time_mask & place_mask].values
        self.test_place_times = place_time_pairs[test_time_mask & place_mask].values 
        print(
            f'Prepared training set with {self.train_size} pairs from {self.train_start} to '
            f'{test_start} for {self.max_regions} regions.'
        )
        print(f'Prepared test set with {self.test_size} pairs starting {self.test_start} for {self.max_regions} regions.')
    
    @property
    def train_size(self) -> int:
        return len(self.train_place_times)
    
    @property 
    def test_size(self) -> int:
        return len(self.test_place_times)

    def __len__(self) -> int:
        return self.train_size + self.test_size
    
    def __getitem__(self, idx: int) -> Tensor:
        # Get place and time id from index
        if idx < self.train_size:
            place, time = self.train_place_times[idx]
        else:
            place, time = self.test_place_times[idx - self.train_size]
        
        # Extract history
        rect_df = self.df[(self.df['rect_id'] == place)]
        hist_start = time - pd.Timedelta(days=self.history)
        year_mask = (rect_df['date'] > hist_start) & (rect_df['date'] <= time)
        year_rect_df = rect_df[year_mask]
        year_rect_df = year_rect_df.set_index(['date'])
        
        # Divide into history // period_length periods
        agg_year_rect_df = year_rect_df.groupby('band').resample(
            f'{self.period_length}D', origin=hist_start, closed='right', label='right'
        ).sum()
        if time not in agg_year_rect_df.index.get_level_values(1):
            pdb.set_trace()
        data = agg_year_rect_df[self.bin_cols].values
        count = agg_year_rect_df['count'].values
        with np.errstate(divide='ignore', invalid='ignore'):
            data = data / count.reshape(-1, 1)
        data = data.reshape((self.num_bands, -1, self.num_bins)).transpose([1, 0, 2])
        data = np.nan_to_num(data, 0) # Fill in no data with zeros
        
        # Pad if needed
        expected_seq_len = self.history // self.period_length
        if data.shape[0] < expected_seq_len:
            data = np.concatenate([np.zeros((expected_seq_len - data.shape[0], self.num_bands, self.num_bins)), data], axis=0)
        
        label = torch.tensor(data[0, 0, 0] < 0.5, dtype=torch.long)
        return (torch.tensor(data, dtype=torch.float), label)