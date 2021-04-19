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
    
    def __init__(self, place_time_pairs: np.ndarray, labels: np.ndarray, data_df: pd.DataFrame, cache: bool):
        self.place_time_pairs = place_time_pairs
        self.labels = labels
        self.df = data_df
        self.cache = cache
        if self.cache:
            self.cached_data = {}

    def __len__(self) -> int:
        return len(self.place_time_pairs)
    
    def __getitem__(self, idx: int) -> Tensor:
        has_fire = self.labels[idx]

        if self.cache and idx in self.cached_data:
            data = self.cached_data[idx]
       
        else:
            place, time = self.place_time_pairs[idx]
        
            # Extract history
            rect_df = self.df.loc[place]
            hist_start = time - pd.Timedelta(days=self.history)
            year_mask = (rect_df.index > hist_start) & (rect_df.index <= time)
            year_rect_df = rect_df[year_mask]

            agg_year_rect_df = year_rect_df.resample(
                f'{self.period_length}D', origin=hist_start, closed='right', label='right'
            ).sum()
            data = agg_year_rect_df.values.reshape((-1, self.num_bins, self.num_bands)).transpose([0, 2, 1])

            with np.errstate(divide='ignore', invalid='ignore'):
                data = data / data.sum(axis=-1, keepdims=True)
            data = np.nan_to_num(data, 0).astype('float32')

            # Pad if needed
            expected_seq_len = self.history // self.period_length
            if data.shape[0] < expected_seq_len:
                data = np.concatenate([np.zeros((expected_seq_len - data.shape[0], self.num_bands, self.num_bins)), data], axis=0)

            if self.cache:
                self.cached_data[idx] = data
        
        features = torch.tensor(data, dtype=torch.float)
        label = torch.tensor(has_fire, dtype=torch.long)
        return (features, label)
    
    @classmethod
    def create_datasets_from_dir(
        cls,
        data_dir: str, 
        train_start: str = '2015-01-01', 
        test_start: str = '2019-01-01', 
        test_stop: str = '2019-12-31',
        max_regions: int = 2053,
        cache: bool = False
    ):

        # Load data
        filenames = sorted(os.listdir(data_dir))
        dfs = []
        for f in filenames:
            if '.csv' in f:
                fpath = f'{data_dir}/{f}'
                print(f'Loading {fpath}...')
                df = pd.read_csv(fpath)
                dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        df['date'] = pd.to_datetime(df['date'])
        df['has_fire'] = (df['fire_counts'] > 0)

        # Filter based on place and time, split into train and test
        place_time_fire = df[['rect_id', 'date', 'has_fire']].drop_duplicates()
        train_time_mask = (place_time_fire['date'] > train_start) & (place_time_fire['date'] <= test_start)
        test_time_mask = (place_time_fire['date'] > test_start) & (place_time_fire['date'] <= test_stop)
        place_mask = place_time_fire['rect_id'] < max_regions
        train_place_time_fire = place_time_fire[train_time_mask & place_mask].values
        test_place_time_fire = place_time_fire[test_time_mask & place_mask].values 

        # Set index and columns for dataframe
        df = df.pivot_table(index=['rect_id', 'date'], columns='band', values=cls.bin_cols)

        # Create datasets
        train_dataset = cls(train_place_time_fire[:, :2], train_place_time_fire[:, -1].astype('bool'), df, cache)
        test_dataset = cls(test_place_time_fire[:, :2], test_place_time_fire[:, -1].astype('bool'), df, cache)

        print(
            f'Prepared training set with {len(train_dataset)} pairs from {train_start} to '
            f'{test_start} for {max_regions} regions with fire rate {train_place_time_fire[:, -1].mean():.2f}.'
        )
        print(
            f'Prepared test set with {len(test_dataset)} pairs starting {test_start} for '
            f'{max_regions} regions with fire rate {test_place_time_fire[:, -1].mean():.2f}.'
        )

        return train_dataset, test_dataset