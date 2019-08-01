
import pandas as pd



columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']


def load_csv():
    x = pd.read_csv('data/xaa.csv', skiprows=1, names=columns)
    return x.drop(columns=['attributed_time'])

def tc():
    x = load_csv()
    name = 'total_clicks'
    y = x.groupby(by='ip').count()
    y = y.reset_index()
    y = y.rename(columns={'channel': f'{name}'})
    y = y[['ip',f'{name}']]
    y.to_csv(f'data/ip_{name}.csv')

def c_s():
    # clicks / seconds
    x = load_csv()
    name = 'clicks_sec'
    y = x.groupby(by=['ip', 'click_time']).count()
    y = y.reset_index()
    y = y.rename(columns={'channel': f'{name}'})
    y = y[['ip','click_time',f'{name}']]
    y.to_csv(f'data/ip_{name}.csv')

def dl():
    # total downloads
    x = load_csv()
    name = 'total_downloads'
    y = x.groupby(by=['ip', 'is_attributed']).count()
    y = y.reset_index()
    y = y.rename(columns={'channel': f'{name}'})
    y = y[['ip',f'{name}']]
    y.to_csv(f'data/ip_{name}.csv')

if __name__ == '__main__':
    main = load_csv()
    name = 'total_clicks'
    tc = pd.read_csv(f'data/ip_{name}.csv')
    main = main.merge(tc, how='left', on='ip')
    del tc
    print('done1')

    name = 'total_downloads'
    dl = pd.read_csv(f'data/ip_{name}.csv')
    print('loaded, and merging'
    main = main.merge(dl, how='left', on='ip')
    del dl
    print('done2')
    main.to_csv('new_train.csv')
    c_s = pd.read_csv(f'data/main.csv', low_memory=False)
    print('loaded, and merging')
    main = main.merge(c_s, how='left', on=['ip','click_time'])
    del c_s
    print('done3')
    main.to_csv('new_train.csv')
    print('done4')


