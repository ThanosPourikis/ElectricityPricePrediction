import copy
import logging
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
from torch.nn import L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.lstm.utils import RequirementsSample, sliding_windows
from models.utils import get_metrics_df


class LstmMVInput:
    def __init__(
        self,
        loss_function,
        data,
        name,
        LSTM,
        learning_rate,
        validation_size,
        sequence_length,
        batch_size,
        hidden_size,
        num_layers,
        output_dim,
        num_epochs,
        model=None,
    ):
        # data = data.set_index('Date')
        self.export = pd.DataFrame(data["SMP"], index=data.index)
        if data.isnull().values.any():
            self.inference = data[-24:]
            self.test = data[-(8 * 24) : -24]
            data = data[: -(8 * 24)]
        else:
            self.test = data[-(7 * 24) :]
            data = data[: -(7 * 24)]

        self.LSTM = LSTM
        self.features = data.loc[:, data.columns != "SMP"]
        self.labels = data["SMP"]
        self.input_size = self.features.columns.shape[0]
        self.validation_size = validation_size
        self.loss_function = loss_function
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.name = name
        self.model = model

    def train(self):
        (
            self.x_train,
            self.x_validate,
            self.y_train,
            self.y_validate,
        ) = train_test_split(
            self.features,
            self.labels,
            random_state=96,
            test_size=self.validation_size,
            shuffle=True,
        )

        x_train, y_train = sliding_windows(
            self.x_train,
            self.y_train,
            sequence_len=self.sequence_length,
            window_step=1,
        )
        x_validate, y_validate = sliding_windows(
            self.x_validate,
            self.y_validate,
            sequence_len=self.sequence_length,
            window_step=1,
        )

        feature_t_s = MinMaxScaler(feature_range=(-1, 1))
        feature_v_s = MinMaxScaler(feature_range=(-1, 1))
        labels_t_s = MinMaxScaler(feature_range=(-1, 1))
        labels_v_s = MinMaxScaler(feature_range=(-1, 1))

        x_train = feature_t_s.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
        x_validate = feature_v_s.fit_transform(x_validate.reshape(-1, x_validate.shape[-1])).reshape(x_validate.shape)

        y_train = labels_t_s.fit_transform(y_train.squeeze())
        y_validate = labels_v_s.fit_transform(y_validate.squeeze())

        model = self.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_dim=self.output_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.criterion = L1Loss()
        optimiser = Adam(model.parameters(), self.learning_rate)

        self.error_train = np.empty(0)
        self.error_val = np.empty(0)
        start_time = time.time()

        train_data_loader = DataLoader(
            RequirementsSample(x_train, y_train),
            self.batch_size,
            shuffle=False,
            drop_last=True,
        )
        val_data_loader = DataLoader(
            RequirementsSample(x_validate, y_validate),
            self.batch_size,
            shuffle=False,
            drop_last=True,
        )
        while True:  # Early Stopping If error hasnt decreased in N epochs STOP
            # for i in range(self.num_epochs):
            model.train()
            err = []
            for j, k in train_data_loader:
                y_train_pred = model(j.float())
                loss = self.criterion(y_train_pred.squeeze(), k.squeeze().float())
                err.append(loss.detach().item())
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            self.error_train = np.append(self.error_train, (sum(err) / len(err)))

            model.eval()
            err = []
            for j, k in val_data_loader:
                y_val_pred = model(j.float())
                loss = self.criterion(y_val_pred.squeeze(), k.squeeze().float())
                err.append(loss.detach().item())
            self.error_val = np.append(self.error_val, (sum(err) / len(err)))

            logging.info(
                f"{self.name}\t Time {time.time() - start_time:.4f}\t Epoch {self.error_val.shape[0]} {self.loss_function} \t Training\t{self.error_train[-1]:.4f}\t Validation\t{self.error_val[-1]:.4f}"
            )

            if self.error_val[-1] <= self.error_val.min():
                self.model = copy.deepcopy(model)

            if (self.error_val.shape[0] - self.error_val.argmin()) > self.num_epochs:
                break

        self.best_epoch = self.error_val.argmin()
        logging.info(f"{self.name} Training Completed Best_epoch : {self.best_epoch} Training Time {time.time() - start_time:.4f}")

        self.hist = pd.DataFrame()
        self.hist["Traing Error"] = self.error_train.tolist()
        self.hist["Validation Error"] = self.error_val.tolist()

    def get_results(self, test=None):
        scaler_f = MinMaxScaler((-1, 1))
        scaler_l = MinMaxScaler((-1, 1))
        scaler_l.fit_transform(np.array(self.y_train).reshape(-1, 1))
        self.model.eval()

        x_train = scaler_f.fit_transform(np.array(self.x_train))
        x_train = Tensor(x_train.reshape(1, -1, self.x_train.shape[1]))
        train = self.model(x_train)
        train = train.detach().numpy().reshape(-1, 1)
        train = scaler_l.inverse_transform(train)
        self.export = self.export.join(pd.DataFrame(train, index=self.y_train.index, columns=["Training"]))

        x_validate = scaler_f.transform(np.array(self.x_validate))
        x_validate = Tensor(x_validate.reshape(1, -1, self.x_validate.shape[1]))
        val = self.model(x_validate)
        val = scaler_l.inverse_transform(val.detach().numpy().reshape(-1, 1))
        self.export = self.export.join(pd.DataFrame(val, index=self.y_validate.index, columns=["Validation"]))

        x_test, y_test = (
            self.test.loc[:, self.test.columns != "SMP"],
            self.test.loc[:, "SMP"],
        )
        x_test = scaler_f.transform(np.array(x_test))
        x_test = Tensor(x_test.reshape(1, -1, x_test.shape[1]))
        test = self.model(x_test)
        test = scaler_l.inverse_transform(test.detach().numpy().reshape(-1, 1))
        self.export = self.export.join(pd.DataFrame(test, index=y_test.index, columns=["Testing"]))

        metrics = get_metrics_df(self.y_train, train, self.y_validate, val, y_test, test)

        try:
            inference = self.inference.loc[:, self.inference.columns != "SMP"]
            inference = scaler_f.transform(np.array(inference))
            inference = Tensor(inference.reshape(1, -1, inference.shape[1]))
            inference = self.model(inference)
            inference = scaler_l.inverse_transform(inference.detach().numpy().reshape(-1, 1))
            self.export["Inference"] = pd.DataFrame(inference, index=self.inference.index)
        finally:
            return self.export, metrics, self.hist, self.best_epoch
