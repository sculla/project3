import pandas as pd
from sklearn import svm
from sklearn import kernel_approximation, linear_model, pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import ensemble
#up sample the training set for tuning not for test
#gradiant boost, random forest, logistic
import arrow

def log(message):
    f = open('mvp.log', 'a')
    f.write(message + '\n')
    print(message)
    f.close()

if __name__ == '__main__':
    lap_time = arrow.now()
    data_tr = pd.read_csv(f'data/train.csv', nrows=1e6)
    log(f"Start Time: {(arrow.now() - lap_time).seconds // 60}:{(arrow.now() - lap_time).seconds % 60} minutes.")
    # hours = [x for x in range(24)]
    # for h in hours:
    #     data_tr[f'{h}'] = (data_tr['hour'] == h).astype('int')
    log(f"Start Time: {(arrow.now() - lap_time).seconds // 60}:{(arrow.now() - lap_time).seconds % 60} minutes.")

    X, y = data_tr.drop(columns=['is_attributed', 'click_time', 'attributed_time'], axis=1), data_tr['is_attributed']
    X_tr, X_te, y_tr, y_te = train_test_split(X,y, test_size=.5, random_state=42)
    log(f"Start Time: {(arrow.now() - lap_time).seconds // 60}:{(arrow.now() - lap_time).seconds % 60} minutes.")

    GBC = ensemble.GradientBoostingClassifier(max_depth=20)

    for _ in range(1):
        log(f"Start Time: {(arrow.now() - lap_time).seconds // 60}:{(arrow.now() - lap_time).seconds % 60} minutes.")

        GBC.fit(X_tr, y_tr)
        print('train done')
        log(f"Start Time: {(arrow.now() - lap_time).seconds // 60}:{(arrow.now() - lap_time).seconds % 60} minutes.")

        print(metrics.classification_report(GBC.predict(X_te),y_te))
        log(f"Start Time: {(arrow.now() - lap_time).seconds // 60}:{(arrow.now() - lap_time).seconds % 60} minutes.")
