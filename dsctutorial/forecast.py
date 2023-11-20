import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import CSVLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

class SlidingWindow(Dataset):
    def __init__(self, target, cov, window_size):
        self.target = target
        self.cov = cov
        self.window_size = window_size
        self.step_size = 1
        self.shift_size = 1

    def __len__(self):
        return (len(self.target) - self.window_size - self.shift_size + 1) // self.step_size

    def __getitem__(self, idx):
        input_target = self.target[range(idx*self.step_size, idx*self.step_size + self.window_size)]
        input_cov = self.cov[range(idx*self.step_size, idx*self.step_size + self.window_size)]
        output_target = self.target[range(
            idx*self.step_size + self.shift_size,
            idx*self.step_size + self.window_size + self.shift_size
        )]
        input_target = torch.FloatTensor(input_target)
        input_cov = torch.FloatTensor(input_cov)
        output_target = torch.FloatTensor(output_target)
        return torch.cat([input_target, input_cov], dim=1), output_target

class LSTMModule(LightningModule):
    def __init__(self, target_dim, cov_dim, hidden_dim, lr):
        super().__init__()
        self.rnn = nn.LSTM(target_dim + cov_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, target_dim)
        self.lr = lr
    def training_step(self, batch, batch_idx):
        input_seq, output_seq = batch
        h, _ = self.rnn(input_seq)
        pred = self.proj(h)
        loss = F.mse_loss(pred, output_seq)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
    
class LSTMForecaster:
    def __init__(self, hidden_dim, window_size, lr, num_epochs, batch_size):
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def fit(self, target, cov):
        self.target_scaler = StandardScaler()
        self.cov_scaler = StandardScaler()
        self.model = LSTMModule(target.shape[1], cov.shape[1], self.hidden_dim, self.lr)
        self.dataset = SlidingWindow(
            target=self.target_scaler.fit_transform(target),
            cov=self.cov_scaler.fit_transform(cov),
            window_size=self.window_size,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.trainer = Trainer(
            accelerator='auto',
            max_epochs=self.num_epochs,
            log_every_n_steps=np.ceil(len(self.dataloader) * 0.1),
            logger=CSVLogger('.'),
        )
        self.trainer.fit(
            model=self.model,
            train_dataloaders=self.dataloader,
        )

    def predict(self, horizon, _past_target, _past_cov, _future_cov):
        past_target = self.target_scaler.transform(_past_target)
        past_cov = self.cov_scaler.transform(_past_cov)
        future_cov = self.cov_scaler.transform(_future_cov)
        columns = _past_target.columns
        index = _past_target.index
        past_target = torch.FloatTensor(past_target).to(self.model.device)
        past_cov = torch.FloatTensor(past_cov).to(self.model.device)
        future_cov = torch.FloatTensor(future_cov).to(self.model.device)[None, ...]
        input_seq = torch.cat([past_target, past_cov], dim=1)[None, ...]
        with torch.no_grad():
            output, (hn, cn) = self.model.rnn(input_seq)
            predn = self.model.proj(output[:, [-1], :])
            pred = [predn]
            for i in range(horizon - 1):
                input = torch.cat([predn, future_cov[:, [i], :]], dim=2)
                output, (hn, cn) = self.model.rnn(input, (hn, cn))
                predn = self.model.proj(output)
                pred.append(predn)
        pred = torch.cat(pred, dim=1)[0].cpu()
        pred = self.target_scaler.inverse_transform(pred)
        pred = pd.DataFrame(
            pred,
            columns=columns,
            index=pd.date_range(start=index[-1], periods=horizon+1, freq=index.freq)[1:])
        return pred

    def load_from_checkpoint(self, path, target, cov):
        self.target_scaler = StandardScaler()
        self.cov_scaler = StandardScaler()
        self.model = LSTMModule.load_from_checkpoint(
            path,
            map_location='cuda' if torch.cuda.is_available() else 'cpu',
            target_dim=target.shape[1],
            cov_dim=cov.shape[1],
            hidden_dim=self.hidden_dim,
            lr=self.lr
        )
        self.target_scaler.fit_transform(target)
        self.cov_scaler.fit_transform(cov)
