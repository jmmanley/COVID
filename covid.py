import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np


# load the time series data into a pandas data structure
def load_data(filename='COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'):
	df = pd.read_csv(filename)
	return df.drop(columns=['Lat','Long'])


# put the data into a nice numpy format by country
def get_country_data(df,country_list):
	df = df[df['Country/Region'].isin(country_list)]

	country_data = []
	for country in country_list:
		cdata = df[df['Country/Region'] == country]
		cdata = cdata.drop(columns=['Country/Region', 'Province/State'])
		by_date = np.sum(np.array(cdata.values),axis=0)
		country_data.append(by_date)

	return np.array(country_data).T

# put the data into a nice numpy format by state
def get_state_data(df,state_list):
	df = df[df['Province/State'].isin(state_list)]

	country_data = []
	for state in state_list:
		cdata = df[df['Province/State'] == state]
		cdata = cdata.drop(columns=['Country/Region', 'Province/State'])
		by_date = np.sum(np.array(cdata.values),axis=0)
		country_data.append(by_date)

	return np.array(country_data).T

# plot the number of confirmed cases
def plot_confirmed_cases(country_info,country_list):
	plt.semilogy(country_info)
	plt.legend(country_list)
	plt.show()


# plot the number of confirmed cases shifted by days since THRESH cases
# (OPTIONAL: date_offset lets you start plotting a few days before this convergence)
def plot_confirmed_cases_by_thresh(country_info,country_list,case_thresh,date_offset=0,per_capita=None, out=None, date=None):
	if per_capita is None:
		start_date = [np.where(ci > case_thresh)[0][0] + date_offset for ci in country_info.T]
	else:
		start_date = [np.where(ci/per_capita[ind] > case_thresh)[0][0] + date_offset for ind,ci in enumerate(country_info.T)]

	end_date = country_info.shape[0]

	shape_length = end_date - np.min(start_date) - date_offset
	shaped_data = np.empty((len(country_list),shape_length))
	shaped_data[:] = np.NaN

	for ind,info in enumerate(country_info.T):
		# print(end_date - start_date[ind])
		# print(info)
		if per_capita is None:
			shaped_data[ind,:end_date - start_date[ind]] = info[start_date[ind]:]
		else:
			shaped_data[ind,:end_date - start_date[ind]] = info[start_date[ind]:]/per_capita[ind]


	plt.semilogy(range(date_offset,end_date - np.min(start_date)),shaped_data.T, marker='o')
	# sns.lineplot(shaped_data.T)
	plt.legend(country_list)
	if date is not None:
		plt.title('growth in COVID-19 cases as of ' + date)
	if per_capita is None:
		plt.ylabel('number of confirmed COVID-19 cases')
		plt.xlabel('days since ' + str(case_thresh) + ' cases reported')
		if out is not None:
			plt.savefig(out)
		else:
			plt.savefig('cases.png')
	else:
		plt.ylabel('number of confirmed COVID-19 cases per capita (millions)')
		plt.xlabel('days since ' + str(case_thresh) + 'cases per capita (millions) reported')
		if out is not None:
			plt.savefig(out)
		else:
			plt.savefig('cases_capita.png')
	plt.show()

if __name__=='__main__':
	country_list = ['Italy','Iran','US','Germany','France','Spain','Korea, South', 'China', 'Japan']
	country_pop = [60.48,81.16,327.2,82.79,66.99,46.66,51.47,1386,126.8]
	df = load_data()
	date = df.columns[-1]

	country_info = get_country_data(df,country_list)

	# plot_confirmed_cases(country_info,country_list)
	# plot_confirmed_cases_by_thresh(country_info,country_list,case_thresh=200,date_offset=-3)
	plot_confirmed_cases_by_thresh(country_info,country_list,case_thresh=200,date_offset=0, date=date)
	plot_confirmed_cases_by_thresh(country_info,country_list,case_thresh=1,date_offset=-3,per_capita=country_pop, date=date)

	state_list = ['Washington', 'New York']

	state_info = get_state_data(df, state_list)

	plot_confirmed_cases_by_thresh(state_info,state_list,case_thresh=300,date_offset=-3, out='cases_states.png', date=date)
	nyc_current = state_info[-3:,1]
	p = np.polyfit(range(len(nyc_current)), np.log(nyc_current),1)
	print('Predicted # cases in NY tomorrow:', np.exp(p[1]) * np.exp(p[0] * (len(nyc_current)+1)))
