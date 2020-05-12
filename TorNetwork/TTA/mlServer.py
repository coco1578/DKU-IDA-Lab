import numpy as np
from sklearn.externals import joblib


class mlDiscriminator:

    def __init__(self):
        # Reshape your data either using array.reshpae(-1, 1) if your data has
        # a single feature or array.reshape(1, -1) if it contains a single sample.
        self.clf = None
        self.binary_cls_tor_or_non_tor = joblib.load('model/tor_non_tor_clf.pkl').set_params(n_jobs=1)
        self.binary_cls_tor_or_hidden = joblib.load('model/tor_hidden_clf.pkl').set_params(n_jobs=1)
        self.multi_cls_tor_sites = joblib.load('model/TBB/tor_sites_clf.pkl').set_params(n_jobs=1)
        self.multi_cls_hidden_sites = joblib.load('model/TBB/hidden_sites_clf.pkl').set_params(n_jobs=1)

    def _reshape_test_data(self, X_test):
        X_test = np.array(X_test)
        if X_test.ndim == 1:
            return X_test.reshape(1, -1)
        else:
            return X_test

    def is_tor_or_non_tor(self, X_test):
        X_test = self._reshape_test_data(X_test)
        y_pred = self.binary_cls_tor_or_non_tor.predict(X_test)

        return y_pred

    def is_tor_or_hidden_service(self, X_test):
        X_test = self._reshape_test_data(X_test)
        y_pred = self.binary_cls_tor_or_hidden.predict(X_test)

        return y_pred

    def close_world_tor_site(self, X_test):
        X_test = self._reshape_test_data(X_test)
        y_pred = self.multi_cls_tor_sites.predict(X_test)

        return int(y_pred[0])

    def close_world_hidden_site(self, X_test):
        X_test = self._reshape_test_data(X_test)
        y_pred = self.multi_cls_hidden_sites.predict(X_test)

        return int(y_pred[0])

