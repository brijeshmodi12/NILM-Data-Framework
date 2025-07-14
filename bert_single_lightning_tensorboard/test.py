# Basic BERT4NILM with Lightning and Tensorboard
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os, sys
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from utils import *
from models import BERT4NILM
from dataloader import NILMDataModule, NILMDataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Hyper-Parameters
n_epochs = 10
learning_rate = 1e-3
batch_size = 64
norm_method = 'minmax' # or zscore
temperature = 0.1 # use lower values to make rare events stand out

class LightningBert(pl.LightningModule):
    def __init__(self, seq_len, lr, temperature):
        super().__init__()
        self.save_hyperparameters()
        self.model = BERT4NILM(seq_len=seq_len)
        
    def forward(self, x):
        return self.model(x)

    def custom_loss(self, prediction, truth, temperature):
        # using MSE + KL Divergence + L1 
        return bert4nilm_loss_continuous(prediction, truth, temperature)

    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = self.custom_loss(prediction, y, temperature)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = self.custom_loss(prediction, y, temperature)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = self.custom_loss(prediction, y, temperature)
        self.log('test_loss', loss)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }


def run_experiment(
    train_x, train_y, val_x, val_y, test_x, test_y,
    temperature=0.1, learning_rate=1e-3, batch_size=64,
    n_epochs=10, norm_method='minmax', log_dir='lightning_logs'
):
    # Wrap datasets
    train_ds = NILMDataset(train_x, train_y)
    val_ds = NILMDataset(val_x, val_y)
    test_ds = NILMDataset(test_x, test_y)
    datamodule = NILMDataModule(train_ds, val_ds, test_ds, batch_size=batch_size)

    # Set up logger — experiment name reflects hyperparameters
    experiment_name = f"temp_{temperature}_lr_{learning_rate}_bs_{batch_size}"
    logger = TensorBoardLogger(save_dir=log_dir, name=experiment_name)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        filename='best-{epoch:02d}-{val_loss:.4f}'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True
    )

    # Initialize model
    model = LightningBert(
        seq_len=train_x.shape[1],
        lr=learning_rate,
        temperature=temperature,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        accelerator='auto',
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10
    )

    # Train + Test
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    # Load datasets
    dataset_path = r"C:\Users\brind\OneDrive - Universitetet i Oslo\Codes\Alva\datasets\refit_clean"
    house_list = range(3,4)
    target_appliance = 'washing_machine' 
    x, y = load_multiple_houses(house_list, dataset_path, target_appliance)

    print('\nData Loaded: \n',x.shape, y.shape)
    print('--------------------')

    # Split the data
    full_dataset = TensorDataset(x,y)
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

    # Normalize using train statistics  
    (train_x, train_y), (val_x, val_y), (test_x, test_y), norm_stats = normalize_using_train_stats(
                                    train_dataset, val_dataset, test_dataset, method=norm_method)

    # Run multiple experiments
    for temp in [0.1, 0.3]:
        for lr in [1e-3, 5e-4]:
            for bs in [32, 64]:
                run_experiment(
                    train_x, train_y, val_x, val_y, test_x, test_y,
                    temperature=temp,
                    learning_rate=lr,
                    batch_size=bs,
                    n_epochs=n_epochs,
                    norm_method=norm_method
                )

