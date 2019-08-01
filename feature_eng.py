
import pandas as pd


columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']


def load_csv():
    x = pd.read_csv('data/train.csv', skiprows=1, names=columns)
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
    print('start')

    print('mid')
    c_s()
    print('done')



