import pandas as pd


def training_data():

    # data = pd.DataFrame(target_model_data())
    # data.to_csv('data.csv')
    smp_values = pd.read_csv('SMP_VALUES.csv')
    data = pd.read_csv('data.csv')

    dataframe = pd.concat([data, smp_values['SMP']], axis=1, join='inner')
    del dataframe['Date']
    del dataframe['Unnamed: 0']
    dataframe.to_csv('data_for_training.csv')

    return dataframe
