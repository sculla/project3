import pandas as pd
from sklearn import svm
from sklearn import kernel_approximation, linear_model, pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import ensemble
#up sample the training set for tuning not for test
#gradiant boost, random forest, logistic
data_tr = pd.read_csv(f'data/new_xaa.csv')
hours = [x for x in range(24)]
for h in hours:
    data_tr[f'{h}'] = (data_tr['hour'] == h).astype('int')
print('train done')
X, y = data_tr.drop(columns=['is_attributed'], axis=1), data_tr['is_attributed']
X_tr, X_te, y_tr, y_te = train_test_split(X,y, test_size=.5, random_state=42)

GBC = ensemble.GradientBoostingClassifier(max_depth=5, criterion='mae')

for _ in range(1):
    GBC.fit(X_tr, y_tr)
    print(metrics.classification_report(GBC.predict(X_te),y_te))