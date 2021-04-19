import pdb

from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from fastai.learner import Learner, DataLoaders, DataLoader
from fastai.callback.data import WeightedDL
from fastai.metrics import accuracy, Recall, Precision, BalancedAccuracy
from fastai.callback.tensorboard import TensorBoardCallback

from dataset import MadagascarFiresDataset
from models import LogisticClassifier


@hydra.main(config_path='configs/', config_name='train_logistic')
def train(config: DictConfig) -> None:

    # Create datasets and dataloaders
    train_dataset, test_dataset = MadagascarFiresDataset.create_datasets_from_dir(**config.dataset)
    train_weights = np.ones(len(train_dataset))
    train_weights[train_dataset.labels] = config.fire_weight
    train_loader = WeightedDL(
        train_dataset, bs=config.batch_size, wgts=train_weights, shuffle=True, drop_last=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(test_dataset, bs=config.batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
    data = DataLoaders(train_loader, test_loader, device=config.device)

    # Create model
    model = instantiate(config.model).to(config.device)

    # Create learner and run
    learner = Learner(data, model, loss_func=F.cross_entropy, metrics=[accuracy, Recall(), Precision(), BalancedAccuracy()])
    learner.fit(n_epoch=config.n_epoch, lr=config.lr, cbs=[TensorBoardCallback(log_preds=False)])

if __name__ == "__main__":
    train()
