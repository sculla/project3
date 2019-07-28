# Out[31]:
# 0    99453
# 1      547
# Name: 8, dtype: int64
# 547/99453
# Out[32]: 0.005500085467507265
# 1-0.005500085467507265
# Out[33]: 0.9944999145324928
# 0.9955- 0.9944999145324928
# Out[34]: 0.0010000854675072945


import numpy as np
from pandas import DataFrame
from sqlalchemy import create_engine
import pandas as pd
import seaborn as sns
from sklearn import model_selection
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline as pipe
import arrow
import dotenv
import pickle
import os
from os import path
import datetime

def log(message):
    f = open('why_is_this_taking_so_fucking_long.log', 'a')
    f.write(message+'\n')
    print(message)
    f.close()

def local_cur():
    tab = 'project3.phone_data_sample'
    connection_string = 'postgresql://localhost:5432/sculla'
    engine = create_engine(connection_string, isolation_level='AUTOCOMMIT')  # ,echo=True)
    conn = engine.connect()
    cursor = conn.connection.cursor()
    return cursor

def get_cursor():

    awsqlKey = dotenv.get_key('.env', 'awsqlKey')

    from psycopg2 import connect
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    params = {
        'host': 'project3.czq1askywdkq.us-west-2.rds.amazonaws.com',
        'user': 'sculla',
        'password': awsqlKey,
        'port': 5432
    }

    connection = connect(**params, dbname='sculla')
    connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    #connection_string = 'postgresql://localhost:5432/sculla'
    #engine = create_engine(connection_string, echo=True, isolation_level='AUTOCOMMIT')
    # conn = engine.connect()
    # cursor = conn.connection.cursor()
    cursor = connection.cursor()
    cursor.execute('SET search_path TO project3;') # for console afterwards
    return cursor


def run_csv():
    with open('names_list.pickle', 'rb') as f:
        name_list = pickle.load(f)
    os.chdir('/Volumes/Seibu Ryu/data_slice')

    for file in name_list:
        if path.exists(f'new_{file}'):
            log(f'File already exists: new_{file}')
            continue
        lap_time = arrow.now()
        log(f'{file} = start')
        log(f"Start Time: {(arrow.now() - lap_time).seconds//60}:{(arrow.now() - lap_time).seconds%60} minutes.")
        df = pd.read_csv(f'{file}', parse_dates=True,low_memory=False,
                         names=['ip', 'app', 'device', 'os',
                                'channel', 'click_time', 'attributed_time',
                                'is_attributed']
                         # dtype={'ip': ,
                         #        'app': int,
                         #        'device': int,
                         #        'os': int,
                         #        'channel': int,
                         #        'is_attributed': int}
                         )
        df.fillna(0, inplace=True)
        df = pd.get_dummies(df, columns=['app', 'device', 'os', 'channel'])
        df.to_csv(f'new_{file}')
        log(f"End Time: {(arrow.now() - lap_time).seconds//60}:{(arrow.now() - lap_time).seconds%60} minutes.")
        log(f'{file} = done')


def feature_eng(cursor):


    #cursor.execute(f'alter table {tab} add column time_since_click_sec int;\n')

    cursor.execute(f'SELECT * FROM {tab} limit 100000;')
    table = cursor.fetchall()
    df = pd.DataFrame(table, columns=['index', 'app', 'ip', 'device', 'os',
                                      'channel', 'click_time', 'attributed_time',
                                      'is_attributed'])
    df.fillna(0, inplace=True)
    df = pd.get_dummies(df, columns=['app', 'device', 'os', 'channel'])
    df.sort_values(by='index', inplace=True)
    new = pd.DataFrame(df.groupby(['ip']))

    # d = dict()
    # col = 'time_since_click_sec'
    # #cursor.execute(f'alter table {tab} add column {col} int;\n')
    #
    # for idx, val in enumerate(df[6].values):
    #     tab_index = df[0][idx]
    #     if df[2][idx] in d: #if ip addr in dictionary
    #
    #         new_val = (val - d[tab_index]).seconds
    #         log(new_val)
    #         conn.execute(f'UPDATE {tab} '
    #                        f'SET {col} = {new_val} '
    #                        f'WHERE index = {tab_index};\n')
    #         d[tab_index] = val  #update dictionary to new time
    #     else: #initialize the delta in table & first timestamp
    #         conn.execute(f'UPDATE {tab} '
    #                        f'SET {col} = 0 '
    #                        f'WHERE index = {tab_index};\n')
    #
    #         d[df[2][idx]]= val #update dictionary to new time
    #         log(tab_index in d)
    ['index', 'app', 'ip', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']

    # cursor.execute(f'SELECT * FROM {tab} limit 100000;')
    # table = cursor.fetchall()
    # df = pd.DataFrame(table)
    #0, index 6, click time 7, attr time, 8, is attr -- not yet features


    X_train, X_testval, y_train, y_testval = model_selection.train_test_split(
        X,
        y,
        test_size=.5,
        stratify=y,  # stratify ensures that each split has the same number in each class
    )
    X_test, X_val, y_test, y_val = model_selection.train_test_split(
        X_testval,
        y_testval,
        test_size=.5,
        stratify=y_testval,  # stratify ensures that each split has the same number in each class
    )
    logreg = linear_model.LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_val)

    log(metrics.classification_report(y_val, y_pred))

if __name__ == '__main__':
    run_csv()

# if __name__ == '__main__':
#
#
#     mm_row = 10
#     tab = 'project3.phone_data'
#     cursor = get_cursor()
#     lap_time = arrow.now()
#     log(f"Start Time: {(arrow.now() - lap_time).seconds//60}:{(arrow.now() - lap_time).seconds%60} minutes.")
#     log(f'Starting {mm_row}mm rows:')
#     cursor.execute(f'SELECT ip, app, device, os, channel, is_attributed FROM {tab} limit {mm_row}'
#                    f'000000;')
#     table = cursor.fetchall()
#     log('Received from PSQL server')
#     log(f"Start Time: {(arrow.now() - lap_time).seconds//60}:{(arrow.now() - lap_time).seconds%60} minutes.")
#     df = pd.DataFrame(table, columns=['ip', 'app', 'device', 'os',
#                                       'channel', 'is_attributed'])
#     log('In DataFrame')
#     log(f"Start Time: {(arrow.now() - lap_time).seconds//60}:{(arrow.now() - lap_time).seconds%60} minutes.")
#     df.fillna(0, inplace=True)
#     log('Filled NA')
#     log(f"Start Time: {(arrow.now() - lap_time).seconds//60}:{(arrow.now() - lap_time).seconds%60} minutes.")
#     df = pd.get_dummies(df, columns=['app', 'device', 'os', 'channel'])
#     log('Got Dummies')
#     # log(f"Start Time: {(arrow.now() - lap_time).seconds//60}:{(arrow.now() - lap_time).seconds%60} minutes.")
#     # df.sort_values(by='index', inplace=True)
#     # log('Sorted')
#     log(f"Start Time: {(arrow.now() - lap_time).seconds//60}:{(arrow.now() - lap_time).seconds%60} minutes.")
#     X, y = df.drop(columns=['is_attributed'], axis=1), df['is_attributed']
#     log('X, y split')
#     log(f"Start Time: {(arrow.now() - lap_time).seconds//60}:{(arrow.now() - lap_time).seconds%60} minutes.")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
#                                                         stratify=y)
#     log('Train/Test Split')
#     log(f"Start Time: {(arrow.now() - lap_time).seconds//60}:{(arrow.now() - lap_time).seconds%60} minutes.")
#     pipe_baseline = pipe([
#         ('base', DummyClassifier())
#     ])
#     pipe_gnb = pipe([
#         ('Pass', DummyClassifier())
#         #('Gauss', GaussianNB()) #pass... recall is next to zero
#     ])
#     pipe_bnb = pipe([
#         ('Bernn', BernoulliNB())
#     ])
#     pipe_mnb = pipe([
#         ('Multi', MultinomialNB())
#     ])
#
#     pipes = [pipe_baseline, pipe_gnb, pipe_bnb, pipe_mnb]
#     pipe_dict = {0:'Baseline Dummy Classifier', 1:'Pass is dummy: Gaussian Na\u00EFve Bayes',
#                  2: 'Bernoulli Na\u00EFve Bayes', 3: 'Multinomial Na\u00EFve Bayes'}
#     log('Performing Train & Tests...')
#     log(f"Start Time: {(arrow.now() - lap_time).seconds//60}:{(arrow.now() - lap_time).seconds%60} minutes.")
#     for idx, pip in enumerate(pipes):
#         log(f'\nWorking on {pipe_dict[idx]}')
#         log(f"Start Time: {(arrow.now() - lap_time).seconds//60}:{(arrow.now() - lap_time).seconds%60} minutes.")
#         log(f'Starting train {pipe_dict[idx]}')
#         pip.fit(X_train, y_train)
#         log(f"Start Time: {(arrow.now() - lap_time).seconds//60}:{(arrow.now() - lap_time).seconds%60} minutes.")
#         log(f'Starting Predict {pipe_dict[idx]}')
#         log(metrics.classification_report(pip.predict(X_test), y_test))
#         log(f"Start Time: {(arrow.now() - lap_time).seconds//60}:{(arrow.now() - lap_time).seconds%60} minutes.")
#         log('End')
#
