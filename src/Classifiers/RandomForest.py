from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from src import BaseClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np


class RandomForest(BaseClassifier.BaseClassifier):

	def __init__(self,dataset, metric_choose='*'):
		BaseClassifier.BaseClassifier.__init__(self, metric_choose)
		print("Optimisation of the parameters ...")
		RFBestParameters = self.parameter_optimize()
		# RFBestParameters = self.get_optimized_paramater()
		print("applying the best parameters to our dataset...")
		self.k_fold_cross_validation(10, RFBestParameters, RandomForestClassifier)

	def get_optimized_paramater(self):
		return {'accuracy': {'value': 0.911, 'parameters': {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': 14, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 45, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 70, 'n_jobs': None, 'oob_score': False, 'random_state': 0, 'verbose': 0, 'warm_start': False}}}


	def parameter_optimize(self):

		best_parameters = {}
		X_train = np.loadtxt("{}/X_train.txt".format('src/dataset_flattened'))
		y_train = np.loadtxt("{}/Y_train.txt".format('src/dataset_flattened'))
		X_test = np.loadtxt("{}/X_test.txt".format('src/dataset_flattened'))
		y_test = np.loadtxt("{}/Y_test.txt".format('src/dataset_flattened'))

		for criterionArg in ["gini", "entropy"]:
			for n_estimators in range(50, 80, 10):
				for sample_split in range(45, 55, 5):
					for maxDepthArg in range(9, 15):
						metrics_y_true = []
						metrics_y_test = []
						metrics_y_score = []

						parameters = {'criterion': criterionArg,
									  'max_depth': maxDepthArg,
									  'n_estimators': n_estimators,
									  'min_samples_split': sample_split,
									  'random_state': 0}
						classifier = RandomForestClassifier(**parameters)
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
