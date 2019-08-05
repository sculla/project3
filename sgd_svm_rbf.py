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
    data_tr = pd.read_csv(f'data/test/1-0_test_new.csv')
    log(f"Start Time: {(arrow.now() - lap_time).seconds // 60}:{(arrow.now() - lap_time).seconds % 60} minutes.")
    # hours = [x for x in range(24)]
    # for h in hours:
    #     data_tr[f'{h}'] = (data_tr['hour'] == h).astype('int')
    log(f"Start Time: {(arrow.now() - lap_time).seconds // 60}:{(arrow.now() - lap_time).seconds % 60} minutes.")
    data_tr.fillna(0, inplace=True)

    X, y = data_tr.drop(columns=['is_attributed', 'click_time', 'attributed_time'], axis=1), data_tr['is_attributed']
    X_tr, X_te, y_tr, y_te = train_test_split(X,y, test_size=.5, random_state=42, stratify=y)
    log(f"Start Time: {(arrow.now() - lap_time).seconds // 60}:{(arrow.now() - lap_time).seconds % 60} minutes.")

    #sgd = svm.SVC()
    logreg = linear_model.LogisticRegression()
    logreg.fit(X_tr, y_tr)
    y_pred = logreg.predict(X_te)

    log(metrics.classification_report(y_te, y_pred))


    # for _ in range(1):
    #     log(f"Start Time: {(arrow.now() - lap_time).seconds // 60}:{(arrow.now() - lap_time).seconds % 60} minutes.")
    #
    #     GBC.fit(X_tr, y_tr)
    #     print('train done')
    #     log(f"Start Time: {(arrow.now() - lap_time).seconds // 60}:{(arrow.now() - lap_time).seconds % 60} minutes.")
    #
    #     print(metrics.classification_report(GBC.predict(X_te),y_te))
    #     log(f"Start Time: {(arrow.now() - lap_time).seconds // 60}:{(arrow.now() - lap_time).seconds % 60} minutes.")

    svm_sgd = linear_model.SGDClassifier(max_iter=500, tol=1e-3)
    svm_sgd = pipeline.make_pipeline(
        kernel_approximation.Nystroem(
            kernel='rbf'), svm_sgd)
    svm_sgd.fit(X_tr, y_tr)
    y_pred = svm_sgd.predict(X_te)

    log(metrics.classification_report(y_te, y_pred))