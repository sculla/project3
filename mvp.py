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
from pandas import DataFrame
from sqlalchemy import create_engine
import pandas as pd
import seaborn as sns
from sklearn import model_selection
from sklearn import linear_model, metrics
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline as pipe
tab = 'project3.phone_data'

if __name__ == '__main__':
    connection_string = 'postgresql://localhost:5432/sculla'
    engine = create_engine(connection_string,  isolation_level='AUTOCOMMIT' )#,echo=True)
    conn = engine.connect()
    cursor = conn.connection.cursor()

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
    #         print(new_val)
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
    #         print(tab_index in d)
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

    print(metrics.classification_report(y_val, y_pred))


X, y = df.drop(columns=['index', 'ip', 'click_time', 'attributed_time', 'is_attributed', 'click2'], axis=1), df['is_attributed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                    stratify=y)
pipe_baseline = pipe([
    ('base', DummyClassifier())
])
pipe_gnb = pipe([
    ('Gauss', GaussianNB())
])
pipe_bnb = pipe([
    ('Bernn', BernoulliNB())
])
pipe_mnb = pipe([
    ('Multi', MultinomialNB())
])

pipes = [pipe_baseline, pipe_gnb, pipe_bnb, pipe_mnb]
pipe_dict = {0:'Baseline Dummy Classifier', 1:'Gaussian Na\u00EFve Bayes',
             2: 'Bernoulli Na\u00EFve Bayes', 3: 'Multinomial Na\u00EFve Bayes'}
print('Performing Train & Tests...')
for idx, pip in enumerate(pipes):
    print(f'\nWorking on {pipe_dict[idx]}')
    pip.fit(X_train, y_train)
    print(metrics.classification_report(pip.predict(X_test), y_test))

