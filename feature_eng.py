
import pandas as pd


columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']

def c_s():
    # clicks / seconds
    y = x.groupby(by=['ip', 'click_time']).count()
    y = y.reset_index()
    x = x.merge(y[['ip', 'n_clicks']], on='ip', how='left')
    x = x.rename(columns={'n_clicks_y': 'clicks_sec', 'n_clicks_x': 'n_clicks'})

def dl():
    # total downloads
    y = x.groupby(by=['ip', 'is_attributed']).count()
    y = y.reset_index()
    x = x.merge(y[['ip', 'is_attributed']], on='ip', how='left')
    x = x.rename(columns={'is_attributed_y': 'total_downloads', 'is_attributed_x': 'is_attributed'})

if __name__ == '__main__':
    x = pd.read_csv('data/train.csv', skiprows=1, names=columns)
    x.drop(columns=['attributed_time'], inplace=True)

    # total clicks by ip
    y = x.groupby(by='ip').count()
    y = y.reset_index()
    # x = x.merge(y[['ip','click_time']], on='ip', how='left')
    # x = x.rename(columns={'click_time_y': 'n_clicks','click_time_x': 'click_time'})


