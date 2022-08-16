from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from src import BaseClassifier
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class Logistic_Regression(BaseClassifier.BaseClassifier):

    LRBestParameters = None

    def __init__(self, metric_choose='*'):
        BaseClassifier.BaseClassifier.__init__(self, metric_choose)
        print("Optimisation of the parameters ...")
        self.LRBestParameters = self.parameter_optimize()
        # self.LRBestParameters = self.get_optimized_paramater()
        print("applying the best parameters to our dataset...")
        self.k_fold_cross_validation(10, self.LRBestParameters, LogisticRegression)

    def get_optimized_paramater(self):
        return {'accuracy': {'value': 0.931, 'parameters': {'C': 0.1, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 500, 'multi_class': 'multinomial', 'n_jobs': None, 'penalty': 'l2', 'random_state': 0, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}}}


    def parameter_optimize(self):
        best_parameters = {}
        X_train = np.loadtxt("{}/X_train.txt".format('src/dataset_flattened'))
        y_train = np.loadtxt("{}/Y_train.txt".format('src/dataset_flattened'))
        X_test = np.loadtxt("{}/X_test.txt".format('src/dataset_flattened'))
        y_test = np.loadtxt("{}/Y_test.txt".format('src/dataset_flattened'))

        for solver in ['newton-cg', 'lbfgs']:
            for C in [0.0001, 0.001, 0.01, 0.1]:
                metrics_y_true = []
                metrics_y_test = []
                metrics_y_score = []

                parameters = {'multi_class': 'multinomial',
                                      'C': C,
                                      'solver': solver,
                                      'penalty': 'l2',
                                      'random_state': 0,
                                        'max_iter': 500}
                classifier = LogisticRegression(**parameters)
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
        return self.LRBestParameters