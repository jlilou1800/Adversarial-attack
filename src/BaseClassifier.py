import io
from math import sqrt

from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, f1_score, matthews_corrcoef, recall_score, roc_auc_score, \
    confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import torch


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
            # print("Metric: ", metric)
            # if self.metric_choose == '*' or metric == self.metric_choose:
            # print("OK")
            metrics_y_true = []
            metrics_y_pred = []
            metrics_y_score = []
            self.train(best_parameters[metric]["parameters"], classifierAlgo)
            results_true, results_pred, results_score = self.predict()
            metrics_y_true = results_true
            metrics_y_pred = metrics_y_pred + list(results_pred)
            metrics_y_score = metrics_y_score + list(results_score)
            # end if
            # print("évaluation ...")
            # print(metric, ": ", self.evaluationMetrics([int(a) for a in metrics_y_true], [int(a) for a in metrics_y_pred]))

    def train(self, parameters, classifierAlgo):
        self.classifier = classifierAlgo()
        self.classifier.set_params(**parameters)

        X_train = np.loadtxt("{}/X_train.txt".format('src/dataset_flattened'))
        y_train = np.loadtxt("{}/Y_train.txt".format('src/dataset_flattened'))

        self.classifier.fit(X_train, y_train)

    def predict(self, X=None, Y=None):
        if X is None or Y is None:
            X_test = np.loadtxt("{}/X_test.txt".format('src/dataset_flattened'))
            y_test = np.loadtxt("{}/Y_test.txt".format('src/dataset_flattened'))
        else:
            X_test = X
            y_test = Y
        y_pred = self.classifier.predict(X_test)
        y_score = self.classifier.predict_proba(X_test)

        metrics_y_true = [int(a) for a in y_test]
        metrics_y_pred = [int(a) for a in y_pred]

        # self.reset_counters()
        # matrix = self.confusion_matrix(metrics_y_true, metrics_y_pred)
        # self.compute_counters(matrix)

        # print("PLOT LOADING")
        # self.plot_images(X_test[:18], metrics_y_true[:18], metrics_y_pred[:18], 3, 6)
        # print("PLOT OKKKKK")
        return y_test, y_pred, y_score[:, 1]

    def predict2(self, x_test):
        # x_test = [x_test]
        y_test = self.classifier.predict(x_test)
        y_score = self.classifier.predict_proba(x_test)
        return y_test, y_score[:, 1]

    def predict_proba(self, x):
        # x = [x]
        p = self.classifier.predict_proba(x)
        return p

    def score(self, x, y):
        p = self.classifier.score(x, y)
        return p

    def evaluationMetrics(self, y_true, y_pred):
        return {"accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='weighted'),
                "recall": recall_score(y_true, y_pred, average='weighted'),
                "f_measure": f1_score(y_true, y_pred, average='weighted')}


    def confusion_matrix(self, y_test, y_pred):
        nbr_of_char = 26
        n = len(y_test)
        # initialization of confusion matrix
        mat = list()
        for i in range(nbr_of_char):
            tmp = list()
            for j in range(nbr_of_char):
                tmp.append(0)
            mat.append(tmp)
        # computation of confusion matrix
        for i in range(n):
            mat[y_pred[i]-97][y_test[i]-97] += 1
        # return mat
        return None

    def plot_images(self, X, y, yp, M, N):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        f, ax = plt.subplots(M, N, sharex=True, sharey=True, figsize=(N, M))
        cpt = 0
        for i in range(M):
            for j in range(N):
                x_adv = X[cpt]

                x_adv = x_adv.squeeze(0)
                x_adv = x_adv.mul(torch.FloatTensor(std).view(3, 1, 1)).add(
                    torch.FloatTensor(mean).view(3, 1, 1)).numpy()  # reverse of normalization op
                x_adv = np.transpose(x_adv, (1, 2, 0))  # C X H X W  ==>   H X W X C
                x_adv = np.clip(x_adv, 0, 1)

                # print(x_adv)

                # x = X[cpt]
                # x_array_2d = x.reshape(100, 100)
                # img = Image.fromarray(x_array_2d)
                # img = img.convert("RGB")
                #
                # byte_io = io.BytesIO()
                # img.save(byte_io, format="JPEG")
                # jpg_buffer = byte_io.getvalue()
                # byte_io.close()
                # x_adv = Image.open(io.BytesIO(jpg_buffer))


                ax[i][j].imshow(x_adv)  # .cpu().numpy())

                title = ax[i][j].set_title("Pred: {}".format(chr(int(yp[i * N + j]))))
                plt.setp(title, color=('g' if yp[i * N + j] == y[i * N + j] else 'r'))
                ax[i][j].set_axis_off()
                cpt += 1
        plt.tight_layout()
        plt.show()

    def compute_counters(self, matrix):
        nbr_of_char = 26
        self.reset_counters()
        for i in range(nbr_of_char):
            self.TP += matrix[i][i]
        for i in range(nbr_of_char):
            self.FP += sum(matrix[i]) - matrix[i][i]
        tmp = list(map(list, zip(*matrix)))
        for i in range(nbr_of_char):
            self.FN += sum(tmp[i]) - tmp[i][i]

    def reset_counters(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
