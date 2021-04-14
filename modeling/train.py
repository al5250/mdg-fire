import pdb

from omegaconf import DictConfig
import hydra
import torch
from torch.utils.data import SubsetRandomSampler
from torch import nn
import torch.nn.functional as F
import numpy as np
from fastai.learner import Learner, DataLoader, DataLoaders
from fastai.metrics import accuracy, Recall, Precision

from dataset import MadagascarFiresDataset


@hydra.main(config_path='configs/', config_name='train')
def train(config: DictConfig) -> None:

    # Create dataset and dataloaders
    dataset = MadagascarFiresDataset(**config.dataset)
    train_sampler = SubsetRandomSampler(np.arange(dataset.train_size))
    test_sampler = SubsetRandomSampler(np.arange(dataset.train_size, len(dataset)))
    train_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=test_sampler)
    data = DataLoaders(train_loader, test_loader)

    # Create model
    if config.model_type == 'logistic_regression':
        num_features = dataset.history // dataset.period_length * dataset.num_bands * dataset.num_bins
        model = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=num_features, out_features=2)
        )
    elif config.model_type == 'neural_network':
        raise NotImplementedError()
    else:
        raise ValueError()

    # Create learner and run
    learner = Learner(data, model, loss_func=F.cross_entropy, metrics=[accuracy, Recall(), Precision()])
    learner.fit(n_epoch=config.n_epoch, lr=config.lr)

if __name__ == "__main__":
    train()
