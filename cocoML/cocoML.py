# -*- coding: utf-8 -*-
"""


__author__ = "coco"
__email__ = "rlawnsgh0826@gmail.com"
__date__ = "2019-08-26"

__last modified by__ = "coco1578"
__last modified time__ = "2019-08-28"


"""
import sys
import time
import pickle
import numpy as np

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize


class Metric:

    def __init__(self):
        self.eta = 1e-5

    def get_metric(self, metric):

        if metric == 'acc':
            return self.accuracy

        elif metric == 'err':
            return self.error

        elif metric == 'roc_auc':
            return self.roc_auc

        elif metric == 'prc':
            return self.prc

        elif metric == 'tpr':
            return self.tpr

        elif metric == 'fpr':
            return self.fpr

        elif metric == 'prec':
            return self.prec

        elif metric == 'recall':
            return self.recall

        elif metric == 'f1-score':
            return self.f1_score

        elif metric == 'bacc':
            return self.bacc

        elif metric == 'berr':
            return self.berr

        else:
            print('[ERROR] Wrong metric')
            sys.exit(0)

    def accuracy(self, y_true, y_pred):

        acc = np.mean(y_true == y_pred)
        return acc

    def error(self, y_true, y_pred):

        acc = self.accuracy(y_true, y_pred)
        err = 1 - acc
        return err

    def roc_auc(self, y_true, y_score):

        fpr, tpr, threshold = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    def prc(self, y_true, y_score):

        prc = average_precision_score(y_true, y_score)
        return prc

    def tpr(self, y_true, y_pred):

        # TPR = TP / (TP + FN)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = (tp + self.eta) / (float(tp + fn) + self.eta)
        return tpr

    def fpr(self, y_true, y_pred):

        # FPR = FP / (FP + TN)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = (fp + self.eta) / (float(fp + tn) + self.eta)
        return fpr

    def prec(self, y_true, y_pred):

        # prec = tp / (tp + fp)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        prec = (tp + self.eta) / (float(tp + fp) + self.eta)
        return prec

    def recall(self, y_true, y_pred):

        # recall = tp / (tp + fn)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        recall = (tp + self.eta) / (float(tp + fn) + self.eta)
        return recall

    def f1_score(self, y_true, y_pred):

        prec = self.prec(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        fscore = (2 * prec * recall) / float(prec + recall)
        return fscore

    def bacc(self, y_true, y_pred):

        berr = self.berr(y_true, y_pred)
        bacc = 1 - berr
        return bacc

    def berr(self, y_true, y_pred):

        unique = np.unique(y_true)

        soo = [np.size(np.where(y_true == c)) for c in unique]
        soo = np.array(soo)
        soo = soo.astype(np.float)

        C = confusion_matrix(y_true, y_pred)
        right = np.diag(C)
        missed = soo - right

        berr = 1.0 / np.size(unique) * np.sum(np.divide(missed, soo))
        return berr


class cocoModel:

    def __init__(self, model):

        self.model = model

        self.X = None
        self.y = None

        self.metric_list = list()
        self.result = dict()
        self.result['train'] = dict()
        self.result['test'] = dict()

        self.score_cls_result = dict()

        self.common_metric = ['acc', 'err']
        self.label_based_metric = ['tpr', 'fpr', 'prec', 'recall', 'f1-score', 'bacc', 'berr']
        self.proba_based_metric = ['roc_auc', 'prc']

        self.metric = Metric()

    def fit(self, X, y):

        self.X = X
        self.y = y

        self.classes = set(self.y)

        t0 = time.time()
        self.model.fit(self.X, self.y)
        training_time = time.time() - t0

        if 'time' not in self.result:
            self.result['time'] = list()

        self.result['time'].append(training_time)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save_model(self, filename):

        fd = open(filename, 'w')
        pickle.dump(self.model, fd)
        fd.close()

    def load_model(self, filename):

        fd = open(filename)
        self.model = pickle.load(fd)
        fd.close()

    def get_model(self):
        return self.model

    def get_score(self, X_test, y_test, metric, mark=None):

        if metric in self.common_metric:
            score = self._get_common_score(X_test, y_test, metric)

        elif metric in self.label_based_metric:
            score = self._get_label_score(X_test, y_test, metric)

        elif metric in self.proba_based_metric:
            score = self._get_proba_score(X_test, y_test, metric)

        else:
            print('[ERROR] Wrong metric')
            sys.exit(0)

        if metric not in self.result[mark]:
            self.result[mark][metric] = list()

        self.result[mark][metric].append(score)

        return score

    def _get_common_score(self, X_test, y_test, metric):

        metric_func = self.metric.get_metric(metric)

        y_pred = self.predict(X_test)
        y_true = y_test

        score = metric_func(y_true, y_pred)
        return score

    def _get_proba_score(self, X_test, y_test, metric):

        metric_func = self.metric.get_metric(metric)

        y_prob = self.predict_proba(X_test)

        if len(self.classes) < 3:
            score = metric_func(y_test, y_prob[:, 1])
        else:
            self.score_cls_result[metric] = dict()

            total_score = 0
            y_test_encoded = label_binarize(y_test, classes=np.unique(y_test))

            for cls in range(len(self.classes)):
                score_cls = metric_func(y_test_encoded[:, cls], y_prob[:, cls])
                self.score_cls_result[metric][cls] = score_cls

                total_score += score_cls

            score = total_score / float(len(self.classes))

        return score

    def _get_label_score(self, X_test, y_test, metric):

        metric_func = self.metric.get_metric(metric)

        y_pred = self.predict(X_test)

        if len(self.classes) < 3:
            score = metric_func(y_test, y_pred)
        else:
            self.score_cls_result[metric] = dict()
            total_score = 0

            for cls in range(len(self.classes)):
                y_pred_temp = np.where(y_pred == cls, 1, 0)
                y_test_temp = np.where(y_test == cls, 1, 0)

                score_cls = metric_func(y_test_temp, y_pred_temp)
                self.score_cls_result[metric][cls] = score_cls
                total_score += score_cls

            score = total_score / float(len(self.classes))

        return score

    def report(self):

        marks = ['train', 'test']

        for mark in marks:
            print('\n[INFO] ======== Classification Performance : %s ========' % mark)
            metrics = self.result[mark].keys()
            for metric in metrics:
                score_list = self.result[mark][metric]
                score_str = list(map(lambda x: float('%.3f' % x), score_list))
                print('[INFO] {0} = {1}, mean = {2:.3f}, std = {3:.3f}'.format(metric, score_str, np.mean(score_list),
                                                                               np.std(score_list)))

        print('\n[INFO] ======== Classification Performance : time ========')
        score_list = self.result['time']
        score_str = list(map(lambda x: float('%.3f' % x), score_list))
        print('[INFO] {0} = {1}, mean = {2:.3f}, std = {3:.3f}'.format('time', score_str, np.mean(score_list),
                                                                       np.std(score_list)))

        metrics = self.score_cls_result.keys()

        if len(metrics) < 1:
            return

        print('\n[INFO] ====== Classification Performance Per Each Class ======')
        for metric in metrics:
            class_score = self.score_cls_result[metric]

            for cls in class_score.keys():
                class_score[cls] = float('%.3f' % class_score[cls])

            print()
            print('[INFO] {0} = {1}'.format(metric, class_score))
            score_list = [class_score[cls] for cls in class_score]
            print('[INFO] mu = {0:.3f}'.format(np.mean(score_list)))
            print('[INFO] std = {0:.3f}'.format(np.std(score_list)))

    def draw_roc_curve(self, X_test, y_test, filepath=None):

        metric_func = self.metric.get_metric('roc_auc')

        y_test_encoded = label_binarize(y_test, classes=np.unique(y_test))
        y_prob = self.predict_proba(X_test)

        tpr = dict()
        fpr = dict()
        roc_auc = dict()

        classes = list()

        if len(self.classes) > 2:
            for cls in range(len(self.classes)):
                classes.append(cls)
                fpr[cls], tpr[cls], _ = roc_curve(y_test_encoded[:, cls], y_prob[:, cls])
                roc_auc[cls] = auc(fpr[cls], tpr[cls])
        else:
            classes.append(0)
            fpr[0], tpr[0], _ = roc_curve(y_test, y_prob[:, 1])
            roc_auc[0] = auc(fpr[0], tpr[0])

        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt

        plt.style.use('classic')
        plt.figure()

        for cls in classes:
            plt.plot(fpr[cls], tpr[cls], lw=2, label='ROC curve - %d (area = %.2f)' % (cls, roc_auc[cls]))
        plt.plot([0, 1], [0, 1], color='0.5', lw=2, linestyle='--')

        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.grid(True)
        plt.legend(loc='best', prop={'size': 8})

        if filepath is None:
            filepath = './' + str(int(time.time())) + '_roc_curve.png'
        print('[INFO] roc_auc curve saved to [%s]' % filepath)
        plt.savefig(filepath)
