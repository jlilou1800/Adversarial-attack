from sklearn.neighbors import KNeighborsClassifier
from src import BaseClassifier
import numpy as np


class K_NearestNeighbors(BaseClassifier.BaseClassifier):

    KNNBestParameters = None

    def __init__(self, metric_choose='*'):
        BaseClassifier.BaseClassifier.__init__(self, metric_choose)
        print("Optimisation of the parameters ...")
        self.KNNBestParameters = self.parameter_optimize()
        # self.KNNBestParameters = self.get_optimized_paramater()
        print("applying the best parameters to our dataset...")
        self.k_fold_cross_validation(10, self.KNNBestParameters, KNeighborsClassifier)

    def get_optimized_paramater(self):
        return {'accuracy': {'value': 0.938, 'parameters': {'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 18, 'p': 1, 'weights': 'distance'}}}


    def parameter_optimize(self):
        best_parameters = {}
        X_train = np.loadtxt("{}/X_train.txt".format('src/dataset_flattened'))
        y_train = np.loadtxt("{}/Y_train.txt".format('src/dataset_flattened'))
        X_test = np.loadtxt("{}/X_test.txt".format('src/dataset_flattened'))
        y_test = np.loadtxt("{}/Y_test.txt".format('src/dataset_flattened'))

        for n_neighbors in range(1, 20):
            for weights in ['uniform', 'distance']:
                for p in range(1, 2):
                    metrics_y_true = []
                    metrics_y_test = []
                    metrics_y_score = []

                    parameters = {'n_neighbors': n_neighbors,
                                      'weights': weights,
                                      'p': p}
                    classifier = KNeighborsClassifier(**parameters)
                    classifier.fit(X_train, y_train)
                    results_test = classifier.predict(X_test)
                    results_score = classifier.predict_proba(X_test)
                    metrics_y_true = y_test
                    metrics_y_test = metrics_y_test + list(results_test)
                    metrics_y_score = metrics_y_score + list(results_score[:, 1])

                    evaluated_test_metrics = self.evaluationMetrics(metrics_y_true, metrics_y_test)

                    for key in evaluated_test_metrics:
                        if key not in best_parameters.keys():
                            best_parameters[key] = {"value": 0, "parameters": None}
                        if best_parameters[key]["value"] <= evaluated_test_metrics[key]:
                            best_parameters[key]["value"] = evaluated_test_metrics[key]
                            best_parameters[key]["parameters"] = classifier.get_params()
        return best_parameters

    def get_best_parameters(self):
        return self.KNNBestParameters