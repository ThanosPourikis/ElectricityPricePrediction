import flask
from flask import render_template

from utils.database_interface import DB
from utils.web_utils import get_json_for_line_fig,get_json_for_fig_scatter,get_metrics,get_json_for_line_scatter,get_candlesticks
from datetime import date
import pandas as pd
import json

app = flask.Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')
today = pd.to_datetime(date.today())
@app.route('/')
def index():
	db = DB()
	df = db.get_data('*', 'dataset')[-(7*24):].set_index('Date')
	return render_template('charts.jinja',title = 'Train Data For The Past 7 Days',df=df,get_json = get_json_for_line_fig,candlestick = get_candlesticks(df.SMP))

@app.route('/Correlation')
def corrolations():
	db = DB()
	df = db.get_data('*', 'dataset')[-(7*24):].set_index('SMP')
	df = df.iloc[:,df.columns!='Date'].dropna()
	return render_template('charts.jinja',title = 'Correlation For The Past 7 Days',df=df,get_json = get_json_for_fig_scatter)

@app.route('/Linear')
def Linear_page():
	db = DB()
	df = db.get_data('*','Linear').set_index('Date')
	if not 'Inference' in df.columns:
		df['Previous Prediction'] = db.get_data('*','infernce').set_index('Date')['Linear']


	metrics = get_metrics('Linear',db).iloc[0]

	return render_template('model.jinja', title = 'Linear Model Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_scatter(df,df.columns),
							train_error= metrics['train_error'],
							validate_error = metrics['validate_error'],
							test_error = metrics['test_error']
							)

@app.route('/KnnR')
def Knn():
	db = DB()
	df = db.get_data('*','Knn').set_index('Date')
	if not 'Inference' in df.columns:
		df['Previous Prediction'] = db.get_data('*','infernce').set_index('Date')['Knn']

	metrics = get_metrics('Knn',db).iloc[0]
	return render_template('model.jinja', title = 'KnnR Model Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_scatter(df,df.columns),
							train_error= metrics['train_error'],
							validate_error = metrics['validate_error'],
							test_error = metrics['test_error'])


@app.route('/XgB')
def XgB():
	db = DB()
	df = db.get_data('*','XgB').set_index('Date')
	if not 'Inference' in df.columns:
		df['Previous Prediction'] = db.get_data('*','infernce').set_index('Date')['XgB']

	metrics = get_metrics('XgB',db).iloc[0]
	return render_template('model.jinja', title = 'XgB Model Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_scatter(df,df.columns),
							train_error= metrics['train_error'],
							validate_error = metrics['validate_error'],
							test_error = metrics['test_error'])

@app.route('/Lstm')
def lstm():
	db = DB()
	df = db.get_data('*','Lstm').set_index('Date')
	if not 'Inference' in df.columns:
		df['Previous Prediction'] = db.get_data('*','infernce').set_index('Date')['Lstm']

	hist = db.get_data('*','hist_lstm')

	metrics = get_metrics('Lstm',db).iloc[0]
	return render_template('lstm.jinja', title = 'Lstm Model Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_scatter(df,df.columns),
							train_error= metrics['train_error'],
							validate_error = metrics['validate_error'],
							test_error = metrics['test_error'],
							hist_json = get_json_for_line_scatter(hist,['hist_train','hist_val'],metrics['best_epoch']))

@app.route('/Hybrid_Lstm')
def hybrid_lstm():
	db = DB()
	df = db.get_data('*','Hybrid_Lstm').set_index('Date')
	if not 'Inference' in df.columns:
		df['Previous Prediction'] = db.get_data('*','infernce').set_index('Date')['Hybrid_Lstm']
	hist = db.get_data('*','hist_Hybrid_Lstm')
	
	metrics = get_metrics('Hybrid_Lstm',db).iloc[0]
	return render_template('lstm.jinja', title = 'Hybrid Lstm Model Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_scatter(df,df.columns),
							train_error= metrics['train_error'],
							validate_error = metrics['validate_error'],
							test_error = metrics['test_error'],
							hist_json = get_json_for_line_scatter(hist,['hist_train','hist_val'],metrics['best_epoch']))

@app.route('/api')
def api():
	try:
		db = DB()
		df = {}
		df['Date'] = db.get_data('*','linear')[-24:]['Date'].astype(str).to_list()
		df['Linear'] = db.get_data('*','linear').set_index('Date')['Inference'].dropna().to_list()
		df['Knn'] = db.get_data('*','Knn').set_index('Date')['Inference'].dropna().to_list()
		df['XgB'] = db.get_data('*','XgB').set_index('Date')['Inference'].dropna().to_list()
		df['Lstm'] = db.get_data('*','Lstm').set_index('Date')['Inference'].dropna().to_list()
		df['Hybrid_Lstm'] = db.get_data('*','Hybrid_Lstm').set_index('Date')['Inference'].dropna().to_list()
		return json.dumps(df)
	except:
		return 'No Prediction Possible'



if __name__ == '__main__':
	
	app.run(host="localhost", port=8000, debug=True)
