import flask
from flask import render_template
from werkzeug.utils import redirect

from utils.database_interface import DB
from utils.web_utils import get_json_for_line_fig,get_json_for_fig_scatter,get_metrics,get_json_for_line_scatter,get_candlesticks
from datetime import date, timedelta
import pandas as pd
from pytz import timezone


localTz = timezone('CET')

app = flask.Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')
today = pd.to_datetime(date.today())
week = str(localTz.localize(today - timedelta(weeks=1)))
datasets = ['requirements','requirements_units','requirements_weather','requirements_units_weather']
models = ['Linear','KnnModel','XgbModel','Lstm','Hybrid_Lstm']
# dataset = 'requirements'
# database = 'requirements_units'

@app.route('/Api/<route>')
def api(route):


	if route =='datasets':
		return pd.DataFrame(datasets).to_json()
	elif route == 'models':
		return pd.DataFrame(models).to_json()
	elif route == 'docs':
		return render_template('api.jinja',datasets= datasets,models = models)

@app.route('/')
def home():
	return redirect('requirements/Linear')

@app.route('/Api')
def api_redict():
	return redirect('Api/docs')

@app.route('/<dataset>/Dataset')
def index(dataset):
	db = DB(dataset)
	df = db.get_data('*', dataset)
	return render_template('home.jinja',title = 'Train Data For The Past 7 Days',
	df=df,get_json = get_json_for_line_fig,candlestick = get_candlesticks(df.SMP),dataset = dataset)

@app.route('/<dataset>/Correlation')
def corrolations(dataset):
	db = DB(dataset)
	df = db.get_data('*', dataset,f'"index" > "{week}"').set_index('SMP').dropna()
	return render_template('correlation.jinja',title = 'Correlation For The Past 7 Days',
	df=df,get_json = get_json_for_fig_scatter,dataset = dataset)

@app.route('/<dataset>/<name>')
def page_for_ml_model(dataset,name):
	db = DB(dataset)
	df = db.get_data('*',name)
	df['Previous Prediction'] = db.get_data(f'"index","{name}"','infernce')

	metrics = get_metrics(name,db)

	return render_template('model.jinja', title = f'{name} Model Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_scatter(df,df.columns),
							metrics = metrics,dataset = dataset)

@app.route('/<dataset>/Lstm')
def lstm(dataset):
	db = DB(dataset)
	df = db.get_data('*','Lstm')
	df['Previous Prediction'] = db.get_data('"index","Lstm"','infernce')
	hist = db.get_data('*','hist_lstm')

	metrics = get_metrics('Lstm',db)
	return render_template('lstm.jinja', title = 'Lstm Model Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_scatter(df,df.columns),
							metrics = metrics,
							hist_json = get_json_for_line_scatter(hist,['hist_train','hist_val'],metrics.iloc[0]['best_epoch']),dataset = dataset)

@app.route('/<dataset>/Hybrid_Lstm')
def hybrid_lstm(dataset):
	db = DB(dataset)
	df = db.get_data('*','Hybrid_Lstm')
	df['Previous Prediction'] = db.get_data('"index","Hybrid_Lstm"','infernce')
	hist = db.get_data('*','hist_Hybrid_Lstm')
	
	metrics = get_metrics('Hybrid_Lstm',db)
	return render_template('lstm.jinja', title = 'Hybrid Lstm Model Last 7days Prediction vs Actual Price And Inference',
							chart_json = get_json_for_line_scatter(df,df.columns),
							metrics = metrics,
							hist_json = get_json_for_line_scatter(hist,['hist_train','hist_val'],metrics.iloc[0]['best_epoch']),dataset = dataset)

@app.route('/prices_api/<dataset>')
def prices_api(dataset):
	try:
		db = DB(dataset)
		df = pd.DataFrame()
		for model in models:
			df[model] = db.get_data('"index","Inference"',model).dropna()
		return df.reset_index(drop=True).to_json()
	except:
		return 'No Prediction Possible'

@app.route('/<dataset>/metrics_api/<model>')
def metrics_api(dataset,model):
	db = DB(dataset)
	try : 
		if model == 'all':
			df = {}
			for model in models:
				df[model] = get_metrics(model,db).loc[:,['Train','Validation','Test']].to_dict()
			return df
		else:
			return get_metrics(model,db).to_json()
	except:
		return "WRONG"





if __name__ == '__main__':
	
	app.run(host="localhost", port=8000, debug=True)
