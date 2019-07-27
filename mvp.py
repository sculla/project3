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


import pandas as pd
from sqlalchemy import create_engine
import pandas as pd
import seaborn as sns
from sklearn import model_selection
from sklearn import linear_model, metrics
tab = 'project3.phone_data_sample2'

if __name__ == '__main__':
    connection_string = 'postgresql://localhost:5432/sculla'
    engine = create_engine(connection_string,  isolation_level='AUTOCOMMIT' )#,echo=True)
    conn = engine.connect()
    cursor = conn.connection.cursor()

    #cursor.execute(f'alter table {tab} add column time_since_click_sec int;\n')

    cursor.execute(f'SELECT * FROM {tab} limit 100000;')
    table = cursor.fetchall()
    df = pd.DataFrame(table)

    d = dict()
    col = 'time_since_click_sec'
    #cursor.execute(f'alter table {tab} add column {col} int;\n')

    for idx, val in enumerate(df[6].values):
        tab_index = df[0][idx]
        if df[2][idx] in d: #if ip addr in dictionary

            new_val = (val - d[tab_index]).seconds
            print(new_val)
            conn.execute(f'UPDATE {tab} '
                           f'SET {col} = {new_val} '
                           f'WHERE index = {tab_index};\n')
            d[tab_index] = val  #update dictionary to new time
        else: #initialize the delta in table & first timestamp
            conn.execute(f'UPDATE {tab} '
                           f'SET {col} = 0 '
                           f'WHERE index = {tab_index};\n')

            d[df[2][idx]]= val #update dictionary to new time
            print(tab_index in d)


    cursor.execute(f'SELECT * FROM {tab} limit 100000;')
    table = cursor.fetchall()
    df = pd.DataFrame(table)
    #0, index 6, click time 7, attr time, 8, is attr -- not yet features
    df.fillna(0, inplace=True)
    df.sort_values(by=8, inplace=True)
    X, y = df.drop(columns=[0,1,2,3,4,5,6,7,8], axis=1), df[8]
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

    print(metrics.classification_report(y_val, y_pred))


