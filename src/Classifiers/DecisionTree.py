from sklearn import tree
from src import BaseClassifier
import numpy as np


class DecisionTree(BaseClassifier.BaseClassifier):
    DTBestParameters = None

    def __init__(self, datasets, metric_choose='*'):
        BaseClassifier.BaseClassifier.__init__(self, metric_choose)
        print("Optimisation of the parameters ...")
        self.DTBestParameters = self.parameter_optimize()
        # print("___________________________")
        # print(self.DTBestParameters)
        # self.DTBestParameters = self.get_optimized_paramater()
        # print("2___________________________")
        # print(self.DTBestParameters)
        print("applying the best parameters to our dataset...")
        self.k_fold_cross_validation(10, self.DTBestParameters, tree.DecisionTreeClassifier)

    def get_optimized_paramater(self):
        return {'accuracy': {'value': 0.936, 'parameters': {'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'entropy', 'max_depth': 20, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': 0, 'splitter': 'best'}}}


    def parameter_optimize(self):
        best_parameters = {}
        X_train = np.loadtxt("{}/X_train.txt".format('src/dataset_flattened'))
        y_train = np.loadtxt("{}/Y_train.txt".format('src/dataset_flattened'))
        X_test = np.loadtxt("{}/X_test.txt".format('src/dataset_flattened'))
        y_test = np.loadtxt("{}/Y_test.txt".format('src/dataset_flattened'))

        for criterionArg in ["gini", "entropy"]:
            for sample_split in range(2, 10):
                for maxDepthArg in range(10, 22, 2):
                    metrics_y_true = []
                    metrics_y_test = []
                    metrics_y_score = []

                    parameters = {'criterion': criterionArg,
                                  'max_depth': maxDepthArg,
                                  'min_samples_split': sample_split,
                                  'random_state': 0}
                    classifier = tree.DecisionTreeClassifier(**parameters)
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
        return self.DTBestParameters
