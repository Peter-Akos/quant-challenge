import numpy as np
import pandas as pd
import torch

county_station_df = pd.read_csv('../data/county-station.csv', index_col=0)
columns = np.load('../data/columns.npy')
metrics = ['avg', 'min', 'max', 'prec']
county_names = list(set(county_station_df['County Name'].to_list()))
dates = np.load('../data/dates.npy')


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


def get_data_pred(file_name, model=None, adjust=False):
    df = pd.read_csv('../data/weather/prediction_targets_daily/' + file_name + ".csv",
                     names=['avg', 'min', 'max', 'prec'], header=None)

    first_year = int(df.index[0][:4])
    last_year = int(df.index[-1][:4])

    data = {}
    for year in range(first_year, last_year + 1):
        # selected_df = station_df[str(year)+"-04-30":str(year)+"-08-25"]
        # print(len(selected_df))
        current_year = []
        for date in dates:
            for metric in metrics:
                if str(year) + '-' + date in df.index:
                    current_year.append(df.loc[str(year) + '-' + date][metric])
                else:
                    current_year.append(None)

        data[year] = np.asarray(current_year)

    if model is not None:
        with torch.no_grad():
            inputs = pd.DataFrame(index=data.keys(), data=data.values(), columns=columns).fillna(0)
            weather = torch.tensor(inputs.to_numpy(), dtype=torch.float32)
            weather = weather.reshape(-1, 4, 222)
            outputs = model(weather)
            i = 0
            for year in range(first_year, last_year + 1):
                outputs[i] -= (1950-year)*1.96966271
                i += 1
            return outputs


    return pd.DataFrame(index=data.keys(), data=data.values(), columns=columns)
    # return data