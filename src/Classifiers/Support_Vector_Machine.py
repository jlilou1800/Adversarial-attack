from sklearn import svm
from src import BaseClassifier
import numpy as np


class Support_Vector_Machine(BaseClassifier.BaseClassifier):

    SVCBestParameters = None

    def __init__(self, metric_choose='*'):
        BaseClassifier.BaseClassifier.__init__(self, metric_choose)
        print("Optimisation of the parameters ...")
        self.SVCBestParameters = self.parameter_optimize()
        # self.SVCBestParameters = self.get_optimized_paramater()
        print("applying the best parameters to our dataset...")
        self.k_fold_cross_validation(10, self.SVCBestParameters, svm.SVC)

    def get_optimized_paramater(self):
        return {'accuracy': {'value': 0.965, 'parameters': {'C': 100, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'probability': True, 'random_state': 0, 'shrinking': True, 'tol': 0.001, 'verbose': False}}}


    def parameter_optimize(self):
        best_parameters = {}
        X_train = np.loadtxt("{}/X_train.txt".format('src/dataset_flattened'))
        y_train = np.loadtxt("{}/Y_train.txt".format('src/dataset_flattened'))
        X_test = np.loadtxt("{}/X_test.txt".format('src/dataset_flattened'))
        y_test = np.loadtxt("{}/Y_test.txt".format('src/dataset_flattened'))

        for C in [0.1, 1, 10, 100]:
            for gamma in ['auto', 'scale']:
                metrics_y_true = []
                metrics_y_test = []
                metrics_y_score = []

                parameters = {'kernel': 'rbf',
                              'gamma': gamma,
                              'C': C,
                              'probability': True,
                              'random_state': 0}
                classifier = svm.SVC(**parameters)
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
        return self.SVCBestParameters