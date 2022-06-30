from math import sqrt
from sklearn.metrics import accuracy_score, precision_score, f1_score, matthews_corrcoef, recall_score, roc_auc_score
import numpy as np


class BaseClassifier:
    def __init__(self, metric_choose='*'):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.classifier = None
        self.metric_choose = '*'
        self.metric_choose = metric_choose

    def k_fold_cross_validation(self, n, best_parameters, classifierAlgo):
        print("parameters : ", best_parameters)
        for metric in best_parameters:
            if self.metric_choose == '*' or metric == self.metric_choose:
                metrics_y_true = []
                metrics_y_pred = []
                metrics_y_score = []

                self.train(best_parameters[metric]["parameters"], classifierAlgo)
                results_true, results_pred, results_score = self.predict()
                metrics_y_true = results_true
                metrics_y_pred = metrics_y_pred + list(results_pred)
                metrics_y_score = metrics_y_score + list(results_score)

                print(metric, ": ", self.evaluationMetrics(metrics_y_true, metrics_y_pred, metrics_y_score))

    def train(self, parameters, classifierAlgo):
        self.classifier = classifierAlgo()
        self.classifier.set_params(**parameters)

        X_train = np.loadtxt("{}/X_train.txt".format('src/dataset_flattened'))
        y_train = np.loadtxt("{}/Y_train.txt".format('src/dataset_flattened'))

        self.classifier.fit(X_train, y_train)

    def predict(self):
        X_test = np.loadtxt("{}/X_test.txt".format('src/dataset_flattened'))
        y_test = np.loadtxt("{}/Y_test.txt".format('src/dataset_flattened'))

        y_score = self.classifier.predict_proba(X_test)

        return y_test, self.classifier.predict(X_test), y_score[:, 1]

    def predict2(self, x_test):
        # x_test = [x_test]
        y_test = self.classifier.predict(x_test)
        y_score = self.classifier.predict_proba(x_test)
        return y_test, y_score[:, 1]

    def predict_proba(self, x):
        p = self.classifier.predict_proba(x)
        return p

    def evaluationMetrics(self, y_true, y_pred, y_score):
        return {"accuracy": accuracy_score(y_true, y_pred)}
                #"precision": precision_score(y_true, y_pred),
                #"recall": recall_score(y_true, y_pred),
                #"f_measure": f1_score(y_true, y_pred),
                #"MCC": matthews_corrcoef(y_true, y_pred),
                #"AUC": roc_auc_score(y_true, y_score)}

    def reset_counters(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
