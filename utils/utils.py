import math

import pandas as pd
import sqlalchemy


import plotly
import plotly.express as px
import json


import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

MSE = 'MSE'
MAE = 'MAE'
HuberLoss = 'HuberLoss'

extended_features_list = ['Date', 'Res_Total', 'Load Total', 'Hydro Total', 'sum_imports', 'sum_exports',
				 'weekdays', 'weekdays0', 'bankdays', 'bankdays0', 'winter', 'spring', 'autumn',
				 'summer', 't1_weekdays', ' t1_weekdays0', 't1_bankdays', 't1_bankdays0',
				 't1_winter', 't1_spring', 't1_autumn', 't1_summer', 'SMP']

features_list = ['Res_Total','Load Total','Hydro Total','Date','sum_imports','sum_exports','SMP']

def error_calculation(function, y_train, y_train_prediction, y_test, y_test_prediction):
	if MAE == function:
		# calculate mean absolute error
		train_score = mean_absolute_error(y_train[:, 0], y_train_prediction[:, 0])
		print('Train Score: %.2f MAE' % train_score)
		test_score = mean_absolute_error(y_test[:, 0], y_test_prediction[:, 0])
		print('Test Score: %.2f MAE' % test_score)

	elif MSE == function:
		# calculate root mean squared error
		train_score = math.sqrt(mean_squared_error(y_train[:, 0, ], y_train_prediction[:, 0, ]))
		print('Train Score: %.2f RMSE' % train_score)
		test_score = math.sqrt(mean_squared_error(y_test[:, 0, ], y_test_prediction[:, 0, ]))
		print('Test Score: %.2f RMSE' % test_score)
	return [train_score, test_score]


def loss_function_selection(function):
	if MAE == function:
		return torch.nn.L1Loss()
	elif MSE == function:
		return torch.nn.MSELoss(reduction='mean')
	elif HuberLoss == function:
		return torch.nn.SmoothL1Loss(reduction='mean')


def get_conn():
	engine = sqlalchemy.create_engine('sqlite:///database.db')
	return engine.connect()


def save_df_to_db(dataframe, df_name):
	connection = get_conn()
	dataframe.to_sql(df_name, connection, if_exists='replace', index=0)


def get_data(table, columns):
	connection = get_conn()
	return pd.read_sql(f'SELECT {columns} FROM {table}', connection)

def get_json_for_line_fig(df,x,y):
	fig = px.line(df,x=x,y=y)
	fig = fig.update_xaxes(rangeslider_visible=True)
	fig.update_layout(width=1500, height=500)
	return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 

def get_json_for_fig_scatter(df,x,y):
	fig = px.scatter(df,x=x,y=y,trendline="ols")
	fig.update_layout(width=1500, height=500)
	fig = fig.update_xaxes(rangeslider_visible=True)
	return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 

