
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
import datetime


columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']

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
                             [int, int, int, int, int, str, int]))) #, parse_dates=['click_time'])
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
    y = y[['ip',f'{name}']]
    y.to_csv(f'data/{name}.csv', index=False)
    print(column, 'exported')

def c_s():
    # clicks / seconds
    x = load_csv()
    name = 'clicks_sec'
    y = x.groupby(by=['ip', 'click_time']).count()
    y = y.reset_index()
    y = y.rename(columns={'channel': f'{name}'})
    y = y[['ip','click_time',f'{name}']]
    y.to_csv(f'data/ip_{name}.csv', index=False)

def dl():
    # total downloads
    x = load_csv()
    name = 'total_downloads'
    y = x.groupby(by=['ip', 'is_attributed']).count()
    y = y.reset_index()
    y = y.rename(columns={'channel': f'{name}'})
    y = y[['ip',f'{name}']]
    y.to_csv(f'data/ip_{name}.csv', index=False)

def test():
    main = load_csv()
    print('loaded csv')
    print(main.head())
    name = 'total_clicks'
    tc = pd.read_csv(f'data/ip_{name}.csv', index_col=0)
    print('loaded, and merging')
    main = main.merge(tc, how='left', on='ip')
    del tc
    main.to_csv('new_train1-1.csv')
    gbc = GradientBoostingClassifier(max_depth=50, verbose=1, tol=1e-7, random_state=42, subsample=1, n_estimators=1000)



if __name__ == '__main__':
    columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']
    df = pd.read_csv('data/train.csv', names=columns)
    main = pd.read_csv('data/train.csv', skiprows=1, names=columns, dtype=
                    dict(zip(['ip', 'app', 'device', 'os', 'channel', 'attributed_time', 'is_attributed'],
                             [int, int, int, int, int, str, int])))
    col = ['ip','app', 'device', 'os', 'channel']
    for idx, column in enumerate(col):
        print(column)
        name = f'attr_clicks_{column}'
        main_grouped = main.groupby(by=[column, 'is_attributed']).count()
        print(column, 'grouped')
        main_grouped = main_grouped.reset_index()
        main_grouped = main_grouped.rename(columns={'click_time': f'{name}'})
        print(column, 'renamed')
        main_grouped = main_grouped[[f'{column}',f'{name}']]
        print('loaded, and merging')

        df = df.merge(main_grouped,how='left', on=column)
        try:
            df = df.drop(['Unnamed: 0'], axis=1)
        except:
            pass

        print('output')
    df.to_csv(f'new_train.csv')







