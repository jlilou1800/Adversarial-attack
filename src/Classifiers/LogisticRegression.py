from sklearn.linear_model import LogisticRegression
from src import BaseClassifier


class Logistic_Regression(BaseClassifier.BaseClassifier):

    LRBestParameters = None

    def __init__(self, datasets, metric_choose='*'):
        BaseClassifier.BaseClassifier.__init__(self, datasets, metric_choose)
        print("Optimisation of the parameters ...")
        self.LRBestParameters = self.parameter_optimize()
        print("applying the best parameters to our dataset...")
        self.k_fold_cross_validation(10, self.LRBestParameters, LogisticRegression)

    def parameter_optimize(self):
        best_parameters = {}
        X_train = [[] for i in range(10)]
        y_train = [[] for i in range(10)]
        X_test = [[] for i in range(10)]
        y_test = [[] for i in range(10)]

        for dataset_ID in range(0, 10):
            for i in range(len(self.datasets[dataset_ID]["training"])):
                X_train[dataset_ID].append(list(self.datasets[dataset_ID]["training"][i]["X"].values()))
                y_train[dataset_ID].append(self.datasets[dataset_ID]["training"][i]["y"])
            for j in range(len(self.datasets[dataset_ID]["test"])):
                X_test[dataset_ID].append(list(self.datasets[dataset_ID]["test"][j]["X"].values()))
                y_test[dataset_ID].append(self.datasets[dataset_ID]["test"][j]["y"])

        for solver in ['newton-cg', 'lbfgs']:
            for C in [0.0001, 0.001, 0.01, 0.1]:
                metrics_y_true = []
                metrics_y_test = []
                metrics_y_score = []
                for dataset_ID in range(0, 10):
                    parameters = {'multi_class': 'multinomial',
                                          'C': C,
                                          'solver': solver,
                                          'penalty': 'l2',
                                          'random_state': 0,
                                            'max_iter': 500}
                    classifier = LogisticRegression(**parameters)
                    classifier.fit(X_train[dataset_ID], y_train[dataset_ID])
                    results_test = classifier.predict(X_test[dataset_ID])
                    results_score = classifier.predict_proba(X_test[dataset_ID])
                    metrics_y_true = metrics_y_true + y_test[dataset_ID]
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
        return self.LRBestParameters