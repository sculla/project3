import dotenv
import pandas as pd
from psycopg2 import connect
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']


def get_cursor():
    awsqlKey = dotenv.get_key('.env', 'awsqlKey')

    params = {
        'host': 'project3.czq1askywdkq.us-west-2.rds.amazonaws.com',
        'user': 'sculla',
        'password': awsqlKey,
        'port': 5432
    }

    connection = connect(**params, dbname='sculla')
    connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    # connion_string = 'postgresql://localhost:5432/sculla'
    # engine = create_engine(connion_string, echo=True, isolation_level='AUTOCOMMIT')
    # cursor = engine.conn()
    # cursor = cursor.connion.cursor()
    cursor = connection.cursor()
    cursor.execute('SET search_path TO project3;')  # for console afterwards
    return cursor


def merge_c_s():
    name = 'clicks_sec'
    x = pd.read_csv('data/train.csv')
    y = x.groupby(by=['ip', 'click_time']).count()
    y = y.reset_index()
    y = y.rename(columns={'channel': f'{name}'})
    main2 = main.merge(y[['ip', 'click_time', name]], how='left', on=['ip', 'click_time'])

    # print('done3')
    # print(main.head())
    # main.to_csv('new_train3-3.csv')
    # print('done4')


def load_csv():
    x = pd.read_csv('data/train.csv', skiprows=1, names=columns, dtype=
    dict(zip(['ip', 'app', 'device', 'os', 'channel', 'attributed_time', 'is_attributed'],
             [int, int, int, int, int, str, int])))  # , parse_dates=['click_time'])
    # x['click_time'] = x['click_time'].astype('datetime64[s]')
    return x.drop(columns=['attributed_time'])


def tc(column='ip'):
    print(column)
    name = f'total_clicks_{column}'
    y = df.groupby(by=column).count()
    print(column, 'grouped')
    y = y.reset_index()
    y = y.rename(columns={'click_time': f'{name}'})
    print(column, 'renamed')
    y = y[['ip', f'{name}']]
    y.to_csv(f'data/{name}.csv', index=False)
    print(column, 'exported')


def c_s():
    # clicks / seconds
    x = load_csv()
    name = 'clicks_sec'
    y = x.groupby(by=['ip', 'click_time']).count()
    y = y.reset_index()
    y = y.rename(columns={'channel': f'{name}'})
    y = y[['ip', 'click_time', f'{name}']]
    y.to_csv(f'data/ip_{name}.csv', index=False)


def dl():
    # total downloads
    x = load_csv()
    name = 'total_downloads'
    y = x.groupby(by=['ip', 'is_attributed']).count()
    y = y.reset_index()
    y = y.rename(columns={'channel': f'{name}'})
    y = y[['ip', f'{name}']]
    y.to_csv(f'data/ip_{name}.csv', index=False)




def gbc_test():

    # imp_dtype = {'ip': int,
    #              'app': int,
    #              'device': int,
    #              'os': int,
    #              'channel': int,
    #              'is_attributed': int,
    #              'attr_clicks_app': int,
    #              'attr_clicks_device': int,
    #              'attr_clicks_os': float,
    #              'attr_clicks_channel': int}
    print('starting')
    main = pd.read_csv('0_new_clicks.csv', header=0, low_memory=False)
    main.drop(['attributed_time'], axis=1, inplace=True)
    main.fillna(0, axis=1, inplace=True)
    print('loaded')
    X_train, X_test, y_train, y_test = train_test_split(
        main.drop(['click_time', 'is_attributed'], axis=1),
        main['is_attributed'],
        random_state=42
    )
    print('split')
    fet = 'None sqrt log2'.split()
    for idx, fig in enumerate(['exponential','deviance']):
        gbc = GradientBoostingClassifier(max_depth=7,
                                         loss=fig,
                                         max_features=feat,
                                         verbose=3,
                                         tol=1e-6,
                                         random_state=42,
                                         subsample=1,
                                         n_estimators=10)

        gbc.fit(X_train, y_train)
        print('metrics',fig)
        print(metrics.classification_report(y_test, gbc.predict(X_test)))
        with open(f'gbc_model_50mm_{idx}-{fig}.pkl', 'wb') as f:
            pickle.dump(gbc, f)

def prep_layer():
    cursor = get_cursor()
    # x = load_csv()
    # name = 'clicks_sec'
    # y = x.groupby(by=['ip', 'click_time']).count()
    # y = y.reset_index()
    # y = y.rename(columns={'channel': f'{name}'})
    # y = y[['ip', 'click_time', f'{name}']]
    # y.to_csv(f'data/ip_{name}.csv', index=False)
    col = ['app', 'device', 'os', 'channel']
    main = pd.read_csv('data/train.csv', names=columns,
                       low_memory=False, skiprows=1, nrows=5e7, parse_dates=True)
    for idx, column in enumerate(col):
        if not os.path.exists(f'sql_{column}_click_time.pkl'):
            cursor.execute(
                f'select {column}, click_time, count(is_attributed) from project3.phone_data where is_attributed = 1 group by {column}, click_time;\n')
            table = cursor.fetchall()
            with open(f'sql_{column}_click_time.pkl', 'wb') as f:
                pickle.dump(table, f)
        else:
            with open(f'sql_{column}_click_time.pkl', 'rb') as f:
                table = pickle.load(f)
            continue
            print(column)
            name = f'{column}_click_time'
            main_grouped = pd.DataFrame(table, columns=[column, 'click_time',
                                                        name], )  # df.groupby(by=[column, 'is_attributed']).count()    #table, names=[col, 'is_attributed', 'is_attributed_count'])

            print(column, 'grouped')

            del table
            print('loaded, and merging')

            main = main.merge(main_grouped, how='left', on=[column,'click_time'])
            try:
                main = main.drop(['Unnamed: 0'], axis=1)
            except KeyError:
                pass

            print('output')
        print('weeee')
        main.to_csv(f'0_train_8_5.csv', index=False)
        return main


def first_layer(df=False):
    if df == False:
        raise 'ffs pass in a data frame'
    cursor = get_cursor()
    dt = {'ip': int,
          'app': int,
          'device': int,
          'os': int,
          'channel': int,
          'attributed_time': str,
          'is_attributed': int}
    # df = pd.read_csv('data/test/train.csv', names=columns, low_memory=False)

    columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']
    main = df
    for skip_num in range(1): #range(37):
        skip = 1 + skip_num * 5e6

        col = ['app', 'device', 'os', 'channel']
        if df == 1:
            main = pd.read_csv('0_train_8_5.csv', names=columns,
                               low_memory=False) #, skiprows=int(skip), nrows=5e6)
        else:
            pass
        for idx, column in enumerate(col):
            if not os.path.exists(f'sql_{column}.pkl'):
                cursor.execute(
                    f'select {column},is_attributed, count(is_attributed) from project3.phone_data where is_attributed = 1 group by {column}, is_attributed;\n')
                table = cursor.fetchall()
                with open(f'sql_{column}.pkl', 'wb') as f:
                    pickle.dump(table, f)
            else:
                with open(f'sql_{column}.pkl', 'rb') as f:
                    table = pickle.load(f)

            print(column)
            name = f'attr_clicks_{column}'
            main_grouped = pd.DataFrame(table, columns=[column, 'is_attributed',
                                                        name])  # df.groupby(by=[column, 'is_attributed']).count()    #table, names=[col, 'is_attributed', 'is_attributed_count'])
            del table
            print(column, 'grouped')
            main_grouped = main_grouped.reset_index()
            main_grouped = main_grouped.rename(columns={'click_time': f'{name}'})
            print(column, 'renamed')
            main_grouped = main_grouped[[f'{column}', f'{name}']]
            print('loaded, and merging')

            main = main.merge(main_grouped, how='left', on=column)
            try:
                main = main.drop(['Unnamed: 0'], axis=1)
            except:
                pass

            print('output')
        print('weeee')
        main.to_csv(f'1_train_8_5.csv.csv', index=False)
        return main


def second_layer(df=False):
    cursor = get_cursor()


    columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']


    if df == False:
        raise 'ffs pass in a data frame'
        # main = pd.read_csv('2_train_8_5.csv',
        #                    low_memory=False,
        #                    dtype=dict(zip(col, [int] * 4)))  # , nrows=5e6)
    else:
        pass
    col = ['app', 'device', 'os', 'channel']
    main = df
    for idx, column in enumerate(col):
        if not os.path.exists(f'sql_{column}_click.pkl'):
            cursor.execute(
                f'select {column}, count(*) from project3.phone_data group by {column};\n')
            table = cursor.fetchall()
            with open(f'sql_{column}_click.pkl', 'wb') as f:
                pickle.dump(table, f)
        else:
            with open(f'sql_{column}_click.pkl', 'rb') as f:
                table = pickle.load(f)

        print(column)
        name = f'clicks_per_{column}'
        main_grouped = pd.DataFrame.from_records(table, columns=[column,
                                                    name])  # df.groupby(by=[column, 'is_attributed']).count()    #table, names=[col, 'is_attributed', 'is_attributed_count'])
        del table
        print(column, 'grouped')
        main_grouped = main_grouped.reset_index()
        main_grouped = main_grouped.rename(columns={'click_time': f'{name}'})
        print(column, 'renamed')
        main_grouped = main_grouped[[f'{column}', f'{name}']]
        print('loaded, and merging')

        main = main.merge(main_grouped, how='left', on=column)
        try:
            main = main.drop(['Unnamed: 0'], axis=1)
        except:
            pass

        print('output')
    print('weeee')
    main.to_csv(f'2_new_clicks.csv', index=False)
    del main


def vc():
    # gbc_test(load_new=True)
    #gbc_test(datatable=second_layer(datatable=first_layer()))
    from sklearn.ensemble import VotingClassifier
    from sklearn.preprocessing import LabelEncoder
    print('starting')
    main = pd.read_csv('data/submission/test-worked.csv', header=0, low_memory=False)
    main.fillna(0, axis=1, inplace=True)

    with open('gbc_training_i36.pkl', 'rb') as f:
        gbc_36 = pickle.load(f)
    estimators = ('gbc_36', gbc_36)

    vc = VotingClassifier(estimators, voting='soft', n_jobs=-1)
    vc.estimators_ = [gbc_36]

    X, y = main.drop(['click_id', 'click_time'], axis=1), main['click_id']

    print('split')
    #vc.fit(X_train, y_train)

    y = pd.DataFrame(y)
    print('predicting')
    y['is_attributed'] = gbc_36.predict(X)

    y.to_csv('data/submission/submission-test.csv')
    return y
    # print(metrics.classification_report(y_test, vc.predict(X_test)))
    # with open(f'gbc_voting.pkl', 'wb') as f:
    #     pickle.dump(vc, f)

def gbc_test(datatable=None, load_new=False):

    # imp_dtype = {'ip': int,
    #              'app': int,
    #              'device': int,
    #              'os': int,
    #              'channel': int,
    #              'is_attributed': int,
    #              'attr_clicks_app': int,
    #              'attr_clicks_device': int,
    #              'attr_clicks_os': float,
    #              'attr_clicks_channel': int}
    if load_new:
        print('starting')
        datatable = pd.read_csv(f'{path}', header=0, low_memory=False)
    main=datatable
    main.drop(['attributed_time'], axis=1, inplace=True)
    main.fillna(0, axis=1, inplace=True)
    print('loaded')
    for idx in range(36,50):
        X_train, X_test, y_train, y_test = train_test_split(
            main.drop(['click_time', 'is_attributed'], axis=1),
            main['is_attributed'],
            #random_state=42
        )
        print('split')
        gbc = GradientBoostingClassifier(max_depth=4,
                                         loss='deviance',
                                         max_features='sqrt',
                                         verbose=3,
                                         tol=1e-6,
                                         #random_state=42,
                                         subsample=1,
                                         n_estimators=20)

        gbc.fit(X_train, y_train)
        print('metrics',idx)
        print(metrics.classification_report(y_test, gbc.predict(X_test)))
        with open(f'gbc_training_i{idx}.pkl', 'wb') as f:
            pickle.dump(gbc, f)
    # TODO add bits for the voting classifier
    # vc = VotingClassifier(vote='soft')
    # vc.estimators_ = pickled models


if __name__ == '__main__':
    # gbc_test(load_new=True)
    li = ['data/test/1-0_test_new.csv', 'data/test/1-1_test_new.csv',
          'data/test/1-2_test_new.csv']
    gbc = GradientBoostingClassifier(max_depth=7,
                                     loss='deviance',
                                     max_features='sqrt',
                                     verbose=3,
                                     tol=1e-6,
                                     # random_state=42,
                                     subsample=1,
                                     n_estimators=20,
                                     warm_start=True)
    for idx, csv in enumerate(li):
        main = pd.read_csv(f'{csv}', header=0, low_memory=False)
        main.drop(['attributed_time'], axis=1, inplace=True)
        main.fillna(0, axis=1, inplace=True)
        print(f'loaded {csv}')
        X_train, y_train = main.drop(['click_time', 'is_attributed'], axis=1),\
                           main['is_attributed']
        del main
        print(f'split {csv}')
        gbc.fit(X_train, y_train)
        print(f'trained {csv}')
        with open(f'data/test/gbc_testing_i{idx}.pkl', 'wb') as f:
            pickle.dump(gbc, f)
        del X_train, y_train

    main = pd.read_csv(f'data/test/1-3_test_new.csv', header=0, low_memory=False)
    main.drop(['attributed_time'], axis=1, inplace=True)
    main.fillna(0, axis=1, inplace=True)
    print('loaded')
    X_test, y_test = main.drop(['click_time', 'is_attributed'], axis=1), \
                       main['is_attributed']
    del main
    print(metrics.classification_report(y_test, gbc.predict(X_test)))

    from xgboost import XGBClassifier
    ####################
    model_params = {
        'verbosity':3,
        'n_jobs':-1,
    }
    model = XGBClassifier(**model_params)
    del X_test, y_test
    main = pd.read_csv(f'data/test/1-0_test_new.csv', header=0, low_memory=False)
    main.drop(['attributed_time'], axis=1, inplace=True)
    main.fillna(0, axis=1, inplace=True)
    X_train, y_train = main.drop(['click_time', 'is_attributed'], axis=1), \
                       main['is_attributed']
    del main
    model.fit(X_train, y_train)
    del X_train, y_train
    main = pd.read_csv(f'data/test/1-1_test_new.csv', header=0, low_memory=False)
    main.drop(['attributed_time'], axis=1, inplace=True)
    main.fillna(0, axis=1, inplace=True)
    X_test, y_test = main.drop(['click_time', 'is_attributed'], axis=1), \
                       main['is_attributed']
    del main
    print(metrics.classification_report(y_test, model.predict(X_test)))