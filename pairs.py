import os
import re

from sklearn import metrics
from sklearn import naive_bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    reviews = dict()
    for thing in ['neg', 'pos']:
        path = f'pairs/txt_sentoken/{thing}'
        os.chdir(path)
        for filename in os.listdir(os.getcwd()):
            with open(filename, 'r') as f:
                words = f.readlines()
                clean_file = ''
                for word in words:
                    s = re.sub(r'[^\w\s]', '', word)
                    clean_file += s
                if thing == 'neg':
                    rev_type = 0
                else:
                    rev_type = 1
                reviews[(filename, rev_type)] = clean_file.replace('\n', '').replace('  ', ' ')
        os.chdir('/Users/sculla/Documents/PyCharm_Projects/project3')

    features = CountVectorizer().fit_transform(reviews.values())

    keys = reviews.keys()
    r_vec = []
    for _, r_type in keys:
        r_vec.append(r_type)

    X_tr, X_te, y_tr, y_te = train_test_split(features, r_vec, random_state=42)
    nb = naive_bayes.MultinomialNB()
    nb.fit(X_tr, y_tr)
    metrics.classification_report(nb.predict(X_te), y_te)
