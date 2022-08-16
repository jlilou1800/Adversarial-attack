from sklearn.ensemble import AdaBoostClassifier
from src import BaseClassifier
from numpy import arange
import numpy as np


class AdaBoost(BaseClassifier.BaseClassifier):

	def __init__(self, metric_choose='*'):
		BaseClassifier.BaseClassifier.__init__(self, metric_choose)
		print("Optimisation of the parameters ...")
		ABBestParameters = self.parameter_optimize()
		print("applying the best parameters to our dataset...")
		self.k_fold_cross_validation(10, ABBestParameters, AdaBoostClassifier)

	def get_optimized_paramater(self):
		return {'n_estimators': 31, 'algorithm': 'SAMME.R', 'learning_rate': 0.5, 'random_state': 0}

	def parameter_optimize(self):

		best_parameters = {}
		X_train = np.loadtxt("{}/X_train.txt".format('src/dataset_flattened'))
		y_train = np.loadtxt("{}/Y_train.txt".format('src/dataset_flattened'))
		X_test = np.loadtxt("{}/X_test.txt".format('src/dataset_flattened'))
		y_test = np.loadtxt("{}/Y_test.txt".format('src/dataset_flattened'))

		for n_estimators_value in range(1, 52, 10):
			for algorithm_value in ["SAMME", "SAMME.R"]:
				for learning_rate_value in arange(0.5, 2.5, 0.5):
					metrics_y_true = []
					metrics_y_test = []
					metrics_y_score = []

					parameters = {'n_estimators': n_estimators_value,
								  'algorithm': algorithm_value,
								  'learning_rate': learning_rate_value,
								  'random_state': 0}
					classifier = AdaBoostClassifier(**parameters)
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
