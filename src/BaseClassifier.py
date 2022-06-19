from math import sqrt
from sklearn.metrics import accuracy_score, precision_score, f1_score, matthews_corrcoef, recall_score, roc_auc_score


class BaseClassifier:
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    datasets = None
    datasets_X_train = []
    datasets_y_train = []
    classifier = None
    metric_choose = '*'

    def __init__(self, datasets, metric_choose='*'):
        self.datasets = datasets
        self.metric_choose = metric_choose

    def k_fold_cross_validation(self, n, best_parameters, classifierAlgo):
        print("parameters : ", best_parameters)
        for metric in best_parameters:
            if self.metric_choose == '*' or metric == self.metric_choose:
                metrics_y_true = []
                metrics_y_pred = []
                metrics_y_score = []

                for i in range(n):
                    self.train(self.datasets[i], best_parameters[metric]["parameters"], classifierAlgo)
                    results_true, results_pred, results_score = self.predict(self.datasets[i])
                    metrics_y_true = metrics_y_true + results_true
                    metrics_y_pred = metrics_y_pred + list(results_pred)
                    metrics_y_score = metrics_y_score + list(results_score)

                print(metric, ": ", self.evaluationMetrics(metrics_y_true, metrics_y_pred, metrics_y_score))

    def train(self, dataset, parameters, classifierAlgo):
        self.classifier = classifierAlgo()
        self.classifier.set_params(**parameters)
        X_train = []
        y_train = []

        # resample the dataset to match X,y form
        for i in range(len(dataset["training"])):
            X_train.append(list(dataset["training"][i]["X"].values()))
            y_train.append(dataset["training"][i]["y"])
        # Train the classifier
        self.classifier.fit(X_train, y_train)

    def predict(self, dataset):
        X_test = []
        y_test = []

        for j in range(len(dataset["test"])):
            X_test.append(list(dataset["test"][j]["X"].values()))
            y_test.append(dataset["test"][j]["y"])

        y_score = self.classifier.predict_proba(X_test)

        return y_test, self.classifier.predict(X_test), y_score[:, 1]

    def evaluationMetrics(self, y_true, y_pred, y_score):
        return {"accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "f_measure": f1_score(y_true, y_pred),
                "MCC": matthews_corrcoef(y_true, y_pred),
                "AUC": roc_auc_score(y_true, y_score)}

    def reset_counters(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
