from sklearn.naive_bayes import GaussianNB
from src import BaseClassifier
import numpy as np


class BayesianNetwork(BaseClassifier.BaseClassifier):

    BNBestParameters = None

    def __init__(self, datasets, metric_choose='*'):
        BaseClassifier.BaseClassifier.__init__(self, metric_choose)
        print("Optimisation of the parameters ...")
        self.BNBestParameters = self.parameter_optimize()
        print("applying the best parameters to our dataset...")
        self.k_fold_cross_validation(10, self.BNBestParameters, GaussianNB)

    # def get_optimized_paramater(self):
    #     return

    def parameter_optimize(self):
        best_parameters = {}
        X_train = np.loadtxt("{}/X_train.txt".format('src/dataset_flattened'))
        y_train = np.loadtxt("{}/Y_train.txt".format('src/dataset_flattened'))
        X_test = np.loadtxt("{}/X_test.txt".format('src/dataset_flattened'))
        y_test = np.loadtxt("{}/Y_test.txt".format('src/dataset_flattened'))


        metrics_y_true = []
        metrics_y_test = []
        metrics_y_score = []

        parameters = {}
        classifier = GaussianNB(**parameters)
        classifier.fit(X_train, y_train)
        results_test = classifier.predict(X_test)
        results_score = classifier.predict_proba(X_test)
        metrics_y_true = y_test
        metrics_y_test = metrics_y_test + list(results_test)
        metrics_y_score = metrics_y_score + list(results_score[:, 1])

        evaluated_test_metrics = self.evaluationMetrics(metrics_y_true, metrics_y_test, metrics_y_score)

        for key in evaluated_test_metrics:
            if key not in best_parameters.keys():
                best_parameters[key] = {"value": 0, "parameters": None}
            if best_parameters[key]["value"] <= evaluated_test_metrics[key]:
                best_parameters[key]["value"] = evaluated_test_metrics[key]
                best_parameters[key]["parameters"] = classifier.get_params()
        return best_parameters

    def get_best_parameters(self):
        return self.BNBestParameters
