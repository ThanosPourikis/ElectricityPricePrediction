from datetime import datetime,timedelta
import requests
import os
import pandas as pd

dt = datetime.now() + timedelta(days=1)

def get_excel_data(folder_path,filetype):
	df = {}

	yyyy = dt.year
	mm = dt.month
	dd = dt.day
	

	if os.path.exists(folder_path):
		print('Directory already exists')
	else:
		print('Creating Directory')
		os.mkdir(folder_path)

	url = f'https://www.admie.gr/getOperationMarketFilewRange?dateStart=2020-11-01&dateEnd={yyyy}-{mm}-{dd}&FileCategory={filetype}'

	name = ''
	data = requests.get(url)
	for file in data.json():
		if name[:8] != file['file_path'].split('/')[-1][:8] :
			name = file['file_path'].split('/')[-1][:8] + '.xlsx'
			if os.path.exists(folder_path+name):
				print(f'Reading File : {name}')
				df[name] = pd.read_excel(folder_path+name)
			else:
				print(f'Downloading File : {name}')
				with open(folder_path + name,'wb') as xlsx:
					xlsx.write(requests.get(file['file_path']).content)
				df[name] = pd.read_excel(folder_path+name)
	return df
