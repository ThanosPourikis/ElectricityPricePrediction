import numpy as np
import torch

# input shape = (batch_size, seq_len, features)


def sliding_windows(features, labels, sequence_len=24, window_step=1):
    xX = []
    yY = []
    for i in range(0, features.shape[0] - sequence_len + 1, window_step):
        xX.append(features[i : i + sequence_len])
        yY.append(labels[i : i + sequence_len])

    xX = np.array(xX)
    yY = np.array(yY)

    return xX, yY


def split_data(data, lookForward, validation_size=0.2):
    validate_size = int(len(data) * validation_size)
    train_size = len(data) - validate_size
    y_train = np.array(
        [
            data.iloc[i : i + lookForward, data.columns == "SMP"]
            for i in range(0, train_size, lookForward)
        ]
    ).squeeze()
    y_validate = np.array(
        [
            data.iloc[i : i + lookForward, data.columns == "SMP"]
            for i in range(train_size, (int(len(data) / 24) * 24), lookForward)
        ][:-1]
    ).squeeze()

    x_train = np.array(
        [
            data.iloc[i : i + lookForward, data.columns != "SMP"]
            for i in range(0, train_size, lookForward)
        ]
    )
    x_validate = np.array(
        [
            data.iloc[i : i + lookForward, data.columns != "SMP"]
            for i in range(train_size, (int(len(data) / 24) * 24), lookForward)
        ][:-1]
    )
    return x_train, y_train, x_validate, y_validate


class RequirementsSample(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return np.array(self.features[index]), np.array(self.labels[index])


def get_tensors(data, lookforward, x_scaler, y_scaler):
    # x_train, y_train, x_validation, y_validation = split_data(data, lookforward)
    x_train, y_train, x_validation, y_validation = sliding_windows(
        data, lookforward
    )

    x_train = x_scaler.fit_transform(
        x_train.reshape(-1, x_train.shape[-1])
    ).reshape(x_train.shape)
    x_validation = x_scaler.transform(
        x_validation.reshape(-1, x_validation.shape[-1])
    ).reshape(x_validation.shape)

    y_train = y_scaler.fit_transform(y_train.reshape(-1, y_train.shape[1]))
    y_validation = y_scaler.fit_transform(
        y_validation.reshape(-1, y_validation.shape[1])
    )

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_validation = torch.from_numpy(x_validation).type(torch.Tensor)

    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_validation = torch.from_numpy(y_validation).type(torch.Tensor)

    return x_train, x_validation, y_train, y_validation
