import os
import numpy as np

from scipy.sparse import coo_matrix
from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

from cocoML.cocoML import cocoModel


def main():

    files = ['Fix_EN.spa', 'Non_Fix_En.spa']

    for file in files:

        data_path = os.path.join(r'C:\Users\coco\Desktop', file)

        classifiers = [
            RandomForestClassifier(n_estimators=100),
            ExtraTreesClassifier(n_estimators=300),
            XGBClassifier(n_estimators=100)
        ]

        X, y = load_svmlight_file(data_path)
        X = np.array(coo_matrix(X, dtype=np.float).todense())

        kfold = StratifiedKFold(n_splits=5)

        for model in classifiers:
            for train_idx, test_idx in kfold.split(X, y):
                X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
                model = cocoModel(model)
                model.fit(X_train, y_train)

                model.get_score(X_train, y_train, metric='acc', mark='train')
                model.get_score(X_train, y_train, metric='err', mark='train')
                model.get_score(X_train, y_train, metric='prec', mark='train')
                model.get_score(X_train, y_train, metric='recall', mark='train')
                model.get_score(X_train, y_train, metric='f1-score', mark='train')

                model.get_score(X_train, y_train, metric='acc', mark='test')
                model.get_score(X_train, y_train, metric='err', mark='test')
                model.get_score(X_train, y_train, metric='prec', mark='test')
                model.get_score(X_train, y_train, metric='recall', mark='test')
                model.get_score(X_train, y_train, metric='f1-score', mark='test')

            model.report()


if __name__ == '__main__':

    main()