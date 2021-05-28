import numpy as np


# def split_data(data, lookback, scaler, test_size=0.2):
#
#     data = scaler.fit_transform(data)
#
#     windows = []
#     for i in range(lookback, len(data) - lookback, lookback):
#         # windows.append(data_raw[i: i + lookback])
#         windows.append(data[i: i + lookback])
#
#     windows = np.array(windows)
#
#     test_size = int(np.round(data.shape[0]*test_size))
#     train_size = data.shape[0] - test_size
#     x_train = windows[:train_size, :, :]
#     y_train = windows[:train_size, :, -1]
#
#     x_validate = windows[train_size:, :-1, :]
#     y_validate = windows[train_size:, :, -1]
#     return x_train, y_train, x_validate, y_validate


def split_data(data, lookback, scaler, validation_size=0.2):

    validate_size = int(len(data) * validation_size)
    train_size = len(data) - validate_size
    data = scaler.fit_transform(data)
    y_train = np.array([data[i: i + lookback, -1] for i in range(lookback, train_size, lookback)])
    y_validate = np.array([data[i: i + lookback, -1] for i in range(train_size, int(len(data)/24)*24, lookback)][:-1])

    x_train = np.array([data[lookback:train_size]])
    x_validate = np.array([data[train_size:-lookback]])

    return x_train, y_train, x_validate, y_validate
