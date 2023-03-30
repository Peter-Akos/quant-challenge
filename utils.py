import numpy as np
import pandas as pd

county_station_df = pd.read_csv('county-station.csv', index_col=0)
columns = np.load('columns.npy')
metrics = ['avg', 'min', 'max', 'prec']
county_names = list(set(county_station_df['County Name'].to_list()))
dates = np.load('dates.npy')


def fill_missing(row):
    if np.sum(np.isnan([row['avg'], row['min'], row['max']])) == 1:
        if np.isnan(row['avg']):
            row['avg'] = (row['min'] + row['max']) / 2
        if np.isnan(row['min']):
            row['min'] = row['avg'] - (row['max'] + row['max'])
        if np.isnan(row['max']):
            row['max'] = row['avg'] + (row['avg'] - row['min'])
    return row


def get_stations(county):
    a = county_station_df[county_station_df['County Name'] == county]
    return a['Station'].to_list()


def get_data_from_station(station):
    station_df = pd.read_csv('data/weather/minnesota_daily/' + station + '.csv', names=['data', 'avg', 'min', 'max', 'prec'], header=None, index_col='data').interpolate()

    if station_df.isna().sum()[:3].sum() > 0:
        print("now")
        station_df = station_df.apply(fill_missing, axis=1)

    first_year = int(station_df.index[0][:4])
    last_year = int(station_df.index[-1][:4])

    #     print(first_year, last_year)
    #     print(station_df)

    data = {}
    for year in range(first_year, last_year + 1):
        # selected_df = station_df[str(year)+"-04-30":str(year)+"-08-25"]
        # print(len(selected_df))
        current_year = []
        for date in dates:
            for metric in metrics:
                if str(year) + '-' + date in station_df.index:
                    current_year.append(station_df.loc[str(year) + '-' + date][metric])
                else:
                    current_year.append(None)

        data[year] = np.asarray(current_year)

    # return pd.DataFrame(index=data.keys(), data=data.values(), columns=columns)
    return data


def unite_stations(station_data, county_name=''):
    years = []
    for data in station_data:
        years = years + list(data.keys())
    years = list(set(years))
    final_data = {}
    for year in years:
        curr = []
        for i in range(len(dates)*len(metrics)):
            correct_values = []
            for d in station_data:
                if year in d.keys() and not pd.isnull(d[year][i]):
                    correct_values.append(d[year][i])
            if len(correct_values) == 0:
                curr.append(None)
            else:
                curr.append(sum(correct_values) / len(correct_values))

        # if (len([x for x in curr if x is not None])) == 354:
        final_data[str(year) + '-' + county_name] = curr
    return final_data
