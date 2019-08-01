import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import kernel_approximation, linear_model, pipeline

columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']

if __name__ == '__main__':
    x = pd.read_csv('data/train.csv', skiprows=1, nrows=5e6, names=columns)
    x.drop(columns=['app', 'device', 'os', 'channel', 'attributed_time'], inplace=True)

    svm_sgd = linear_model.SGDClassifier(max_iter=500,verbose=1,
                                         penalty='l2',tol=1e-6, n_jobs=-1, learning_rate='adaptive', n_iter_no_change=5, eta0=.1, class_weight='balanced')

    svm_sgd = pipeline.make_pipeline(
        kernel_approximation.Nystroem(
            kernel='rbf'), svm_sgd)

    y = x.groupby(by='ip').count()
    y = y.reset_index()
    x = x.merge(y, on='ip')
    x=x.rename(columns={'click_time_y':'n_cli'})

    x = x.drop(['is_attributed_y'],axis=1)

    xtr, xte, ytr, yte = train_test_split(x[['ip','n_cli']],x['is_attributed_x'])
    from sklearn import metrics
    print(metrics.classification_report(yte,svm_sgd.predict(xte)))