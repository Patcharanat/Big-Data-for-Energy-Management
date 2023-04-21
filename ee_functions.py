# Functions for EE Project

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def make_data_sample(floor=7):
    """
    Create a sample dataset for developing ML model Using Floor3 all zone year 2018, 2019 as the sample
    """
    date_format = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    try:
        df_floorX_2018 = pd.read_csv(f'dataset/2018Floor{floor}.csv',
                        parse_dates=['Date'],
                        date_parser=date_format)

        df_floorX_2019 = pd.read_csv(f'dataset/2019Floor{floor}.csv',
                        parse_dates=['Date'],
                        date_parser=date_format)
    except:
        df_floorX_2018 = pd.read_csv(f'dataset/2018Floor{floor}.csv')
        df_floorX_2018 = df_floorX_2018[df_floorX_2018.Date.str.len() == 19]
        df_floorX_2018['Date'] = pd.to_datetime(df_floorX_2018['Date'])
        df_floorX_2018['Date'] = pd.to_datetime(df_floorX_2018["Date"].dt.strftime("%Y-%m-%d %H:%M:%S"))

        df_floorX_2019 = pd.read_csv(f'dataset/2019Floor{floor}.csv')
        df_floorX_2019 = df_floorX_2019[df_floorX_2019.Date.str.len() == 19]
        df_floorX_2019['Date'] = pd.to_datetime(df_floorX_2019['Date'])
        df_floorX_2019['Date'] = pd.to_datetime(df_floorX_2019["Date"].dt.strftime("%Y-%m-%d %H:%M:%S"))

    df_sample = pd.concat([df_floorX_2018, df_floorX_2019]) \
                  .reset_index(drop=True)
    return df_sample

def sep_zone(df):
    """
    Create and return dict containing columns name for each zone.
    """
    zone = {}
    for column in df.columns:
        if 'Date' in column:
            continue
        if column[:2] not in zone:
            # each zone contains own Date column.
            zone[column[:2]] = ['Date', column]
        else:
            zone[column[:2]].append(column)
    return zone

def choose_zone(zone: dict, selected=[]):
    """
    Select zone(s) return dict of selected zone and columns name
    """
    if len(selected) == 0:
        cols_name = [col for key in zone.keys() for col in zone[key] if col != 'Date']
        cols_name.insert(0, 'Date')
        return zone, cols_name
    else:
        zone_new = {key: zone[key] for key in selected}
        cols_name = [col for key in zone_new.keys() for col in zone_new[key] if col != 'Date']
        cols_name.insert(0, 'Date')
        return zone_new, cols_name

def create_timeseries_features(df):
    """
    Create time series features based on time series index.
    """
    df_temp = df.copy()
    df_temp['hour'] = df_temp.index.hour
    df_temp['dayofweek'] = df_temp.index.dayofweek
    df_temp['quarter'] = df_temp.index.quarter
    df_temp['month'] = df_temp.index.month
    df_temp['year'] = df_temp.index.year
    df_temp['dayofyear'] = df_temp.index.dayofyear
    df_temp['dayofmonth'] = df_temp.index.day
    df_temp['weekofyear'] = df_temp.index.isocalendar().week
    df_temp['weekofyear'] = df_temp['weekofyear'].astype('int64')
    return df_temp

def create_features(df):
    """
    Feature Engineering. Can be applied after sum_features() and set_date_as_index().
    """
    df_temp = df.copy()
    df_temp['SMA30'] = df_temp['sum_power_consumption(kW)'].rolling(window='30D', min_periods=1).mean()
    df_temp['SMA15'] = df_temp['sum_power_consumption(kW)'].rolling(window='15D', min_periods=1).mean()
    df_temp['SMA7'] = df_temp['sum_power_consumption(kW)'].rolling(window='7D', min_periods=1).mean()
    return df_temp

def sum_features(df):
    """
    SUM sub-features into features and drop the old ones
    """
    AC = [col for col in df.columns if 'AC' in col]
    Light = [col for col in df.columns if 'Light' in col]
    Plug = [col for col in df.columns if 'Plug' in col]
    df_temp = df.copy()
    df_temp['sum_ac(kW)'] = df_temp[AC].sum(axis=1)
    df_temp['sum_light(kW)'] = df_temp[Light].sum(axis=1)
    df_temp['sum_plug(kW)'] = df_temp[Plug].sum(axis=1)
    df_temp['sum_power_consumption(kW)'] = df_temp[AC+Light+Plug].sum(axis=1)
    df_temp = df_temp.drop(columns=AC+Light+Plug)
    return df_temp

def form_dataframe(data_array, old_dataframe):
    """
    Re-form DataFrame from array using index and columns from the old one.
    """
    df_temp = pd.DataFrame(data = data_array,
                            index = old_dataframe.index,
                            columns = old_dataframe.drop(columns=['Date']).columns)
    df_temp.insert(loc=0, column='Date', value=old_dataframe['Date'])
    return df_temp

def set_date_as_index(dataframe):
    """
    Set column Date as index
    """
    dataframe['Date'] = pd.to_datetime(dataframe['Date'].dt.strftime("%Y-%m-%d %H:%M:%S"))
    df = dataframe.set_index('Date')
    # change index dtype to datetime
    # df.index = pd.to_datetime(df.index)
    return df

def set_index_as_index(df):
    """
    Reset index from Dataframe that have Date column as index
    """
    df_temp = df.copy()
    df_temp.insert(loc=0, column='Date', value=df.index)
    df_temp = df_temp.reset_index(drop=True)
    return df_temp

def sep_input_output(df):
    """
    Separate Input and Output columns. use before train_test_split_data()
    """
    df_input = df.drop(columns=['sum_power_consumption(kW)', 'sum_ac(kW)', 'sum_light(kW)', 'sum_plug(kW)'])
    df_output = df[['Date', 'sum_ac(kW)', 'sum_light(kW)', 'sum_plug(kW)', 'sum_power_consumption(kW)']]
    return df_input, df_output

def train_test_split_data(df_input, df_output, train_size=0.8):
    """
    Timeseries train test split. df should be regular index, not date index.
    """
    cutoff = int(df_input.shape[0]*train_size)
    
    X_train = df_input.loc[:cutoff, :]
    X_test = df_input.loc[cutoff+1:, :]
    y_train = df_output.loc[:cutoff, :]
    y_test = df_output.loc[cutoff+1:, :]

    X_train_copy = set_date_as_index(X_train)
    X_test_copy = set_date_as_index(X_test)
    y_train_copy = set_date_as_index(y_train)
    y_test_copy = set_date_as_index(y_test)
    
    return X_train_copy, X_test_copy, y_train_copy, y_test_copy

def reduce_resolution(df, interval=15):
    """
    Reduce data resolution by aggregating records for more interval.
    """
    i = 0
    reduced_resolution_df = pd.DataFrame()
    time_col = pd.DataFrame(df[(df.index % interval) == 0].Date)
    while i < df.shape[0]:
        # average value of all columns except Date column
        if i+interval < df.shape[0]:
            mean_interval = pd.DataFrame(df.iloc[i: i+interval].mean()).transpose()
        else :
            mean_interval = pd.DataFrame(df.iloc[i: ].mean()).transpose()

        reduced_resolution_df = pd.concat([reduced_resolution_df, mean_interval], axis=0)
        i += interval

    reduced_resolution_df = reduced_resolution_df.set_index(time_col.index, drop=True)
    df_temp = pd.concat([time_col, reduced_resolution_df], axis=1)
    completedly_reduced_resolution_df = df_temp.reset_index(drop=True)
    return completedly_reduced_resolution_df

def process_data(df):
    """
    Main function to process a raw data.
    Missing values should be imputed before apllying this function.
    """
    df_temp = df.copy()
    df_temp = sum_features(df_temp)
    df_temp = set_date_as_index(df_temp)
    df_temp = create_features(df_temp)
    df_temp = create_timeseries_features(df_temp)
    df_temp = set_index_as_index(df_temp)
    return df_temp

def make_full_sample():
    """
    Create full sample dataset, including all floors year 2018 and 2019.
    """
    # create a list storing all dataframes
    datasets_list = [make_data_sample(floor=i) for i in range(1, 8)]
    Date_col = datasets_list[0]['Date']

    # add suffix to identify dataframes
    for count, value in enumerate(datasets_list):
        datasets_list[count] = value.drop(columns=['Date']) \
                                    .add_suffix(f"_floor{count+1}")
    
    # concatenate all dataframe into a new dataframe
    df = pd.concat(datasets_list, axis=1)
    df.insert(loc=0, column='Date', value=Date_col)
    return df

def make_predict_score(y_test, prediction, print_result=False):
    """
    Evalute a predicted result by using several metrics in regression task.
    Print scores and also return scores.
    """
    r2 = r2_score(y_true=y_test, y_pred=prediction).round(4)
    mse = mean_squared_error(y_true=y_test, y_pred=prediction).round(4)
    # rmse = mean_squared_error(y_true=y_test, y_pred=prediction, squared=False).round(4)
    mae = mean_absolute_error(y_true=y_test, y_pred=prediction).round(4)
    
    if print_result == True:
        print(f'R2: {r2:.3f} \nMSE: {mse:.3f} \nMAE: {mae:.3f}')
    return [r2, mse, mae]

def init_score_df():
    """
    Create score DataFrame used for store evalution of different ML development
    """
    comparing_score = {'R-squared': [], 
                       'MSE': [], 
                       'MAE': [],
                       'causal': []}
    score_df = pd.DataFrame(comparing_score)
    return score_df

def insert_score(score_df, score_list_add: list, causal: str):
    """
    Insert scores into a score dataframe used for experiment
    """
    score_df_add = pd.DataFrame({'R-squared': score_list_add[0], 
                                 'MSE': score_list_add[1], 
                                 'MAE': score_list_add[2], 
                                 'causal': causal},
                                 index = [0])
    score_df_new = pd.concat([score_df, score_df_add], axis=0)
    return score_df_new.reset_index(drop=True)

### Function Transformer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.impute import KNNImputer

def imputer_iterative(df, zone):
    imputer = IterativeImputer(estimator=BayesianRidge(), # default estimator 
                        max_iter=10,
                        initial_strategy='median',
                        random_state=1)
    data_zone_list = []
    for e in zone:
        array_temp = imputer.fit_transform(df[zone[e]].drop(columns=['Date']))
        df_temp = form_dataframe(data_array=array_temp, old_dataframe=df[zone[e]])
        data_zone_list.append(df_temp.drop(columns=['Date']))
    df_sample = pd.concat(data_zone_list, axis=1)
    df_sample.insert(loc=0, column='Date', value=df['Date'])

    df_sample = process_data(df_sample)
    return df_sample

def imputer_knn(df, zone):
    imputer = KNNImputer(n_neighbors=5, weights='uniform') # default parameters

    data_zone_list = []
    for e in zone:
        df_temp1 = set_date_as_index(df[zone[e]])
        df_temp2 = create_timeseries_features(df_temp1)
        df_temp2 = set_index_as_index(df_temp2)
        
        array_temp = imputer.fit_transform(df_temp2.drop(columns=['Date']))
        df_temp = form_dataframe(data_array=array_temp, old_dataframe=df_temp2)
        df_temp = df_temp[df_temp1.columns]
        
        data_zone_list.append(df_temp)

    df_sample = pd.concat(data_zone_list, axis=1)
    df_sample.insert(loc=0, column='Date', value=df['Date'])

    df_sample = process_data(df_sample)
    return df_sample

def imputer_meangroupby(df):
    df_sample = set_date_as_index(df)
    df_sample = create_timeseries_features(df_sample)
    for col in df_sample.columns:
        df_sample[col] = df_sample[col].fillna(df_sample.groupby(['year', 'month', 'dayofweek', 'hour'])[col].transform('mean'))
        df_sample[col] = df_sample[col].fillna(df_sample.groupby(['month', 'dayofweek', 'hour'])[col].transform('mean'))
        df_sample[col] = df_sample[col].fillna(df_sample.groupby(['dayofweek', 'hour'])[col].transform('mean'))
    df_sample = sum_features(df_sample)
    df_sample = create_features(df_sample)
    df_sample = set_index_as_index(df_sample)
    return df_sample