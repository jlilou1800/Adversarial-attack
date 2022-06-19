"""
	Decorate Algorithm : Diverse En-semble Creation by Oppositional Relabeling of Artificial Training Examples
"""
from math import ceil, sqrt
from random import choices, gauss
from sklearn import tree
from src.BaseClassifier import BaseClassifier


class Decorate(BaseClassifier):

	features_probs_mu = {}
	features_probs_sigma = {}
	target_probs = {'bot': 0.0}
	classifier_list = {}
	best_parameters = None

	def __init__(self, dataset, best_parameters=dict(), class_a_set=True, metric_choose='*'):
		BaseClassifier.__init__(self, dataset, metric_choose)

		if best_parameters == {}:
			# Optimize the parameters TODO
			best_parameters = self.parameter_optimize(class_a_set)
			print("Best parameters established : ",best_parameters)
		else:
			self.best_parameters = best_parameters

		# Train the classifier
		for metrics in self.best_parameters:
			if self.metric_choose == '*' or self.metric_choose == metrics:
				for iter in range(len(self.datasets)):
					self.decorate_algorithm(iter, dataset[iter]["training"], parameters=self.best_parameters[metrics]["parameters"], class_a_set=class_a_set)

				# Predict the dataset
				y_test, y_pred, y_score = self.predict()
				print(metrics, ": ", self.evaluationMetrics(y_test, y_pred, y_score))

	def get_distribution(self, dataset):
		self.features_probs_mu = {}
		self.features_probs_sigma = {}
		self.target_probs["bot"] = 0
		for key in dataset[0]["X"]:
			self.features_probs_mu[key] = 0.0
			self.features_probs_sigma[key] = 0.0

		for i in range(len(dataset)):
			for key in dataset[i]["X"]:
				self.features_probs_mu[key] += dataset[i]["X"][key]
				self.features_probs_sigma[key] += (dataset[i]["X"][key] * dataset[i]["X"][key])
			self.target_probs["bot"] += dataset[i]["y"]

		for key in self.features_probs_mu:
			self.features_probs_mu[key] = self.features_probs_mu[key]/len(dataset)
			self.features_probs_sigma[key] = self.features_probs_sigma[key] / len(dataset)

		for key in self.features_probs_sigma:
			self.features_probs_sigma[key] -= (self.features_probs_mu[key] * self.features_probs_mu[key])
			self.features_probs_sigma[key] = sqrt(self.features_probs_sigma[key])

		self.target_probs["bot"] = self.target_probs["bot"]/len(dataset)
	
	def create_synthetic_data(self, n, class_a_set=True):
		synth_data = []
		class_a_numeric_key = ["nb_friends", "nb_tweets", "friends/(followers^2)", "age", "following_rate"]
		for i in range(n):
			tmpline = {"id": "tmp", "X": {},
					   'y': choices([0, 1], [1 - self.target_probs['bot'], self.target_probs['bot']])[0]}

			for key in self.features_probs_mu:
				if class_a_set:
					if key not in class_a_numeric_key:
						tmpline["X"][key] = choices([1, 0], [1 - self.features_probs_mu[key], self.features_probs_mu[key]])[0]
					else:
						tmpline["X"][key] = gauss(self.features_probs_mu[key], self.features_probs_sigma[key])
				else:
					tmpline["X"][key] = gauss(self.features_probs_mu[key], self.features_probs_sigma[key])
			synth_data.append(tmpline)

		return synth_data

	def decorate_algorithm(self, iter, t, c_size=15, i_max=20, r_size=0.5, parameters=dict(), class_a_set=True):
		self.get_distribution(self.datasets[iter]["training"])
		i = 1
		trials = 1
		c_i = self.base_learn(t, parameters)
		c_set = [c_i]  # Set of target values

		error = self.ensemble_error(c_set, t)

		while i < c_size and trials < i_max:
			max_n = ceil(r_size * len(t))
			r = self.create_synthetic_data(max_n, class_a_set)
			t_2 = t + r
			c_2 = self.base_learn(t_2, parameters)
			c_set.append(c_2)
			error_2 = self.ensemble_error(c_set, t)
			if error_2 <= error:
				i += 1
				error = error_2
			else:
				c_set.pop()
			trials += 1
		self.classifier_list[iter] = c_set

	def ensemble_error(self, c_set, t):
		incorrect = 0
		total = 0

		x_test = []
		y_test = []

		for i in range(len(t)):
			x_test.append(list(t[i]["X"].values()))
			y_test.append(t[i]["y"])

		for j in range(len(c_set)):
			y_predict = c_set[j].predict(x_test)
			for k in range(len(y_test)):
				if y_test[k] != y_predict[k]:
					incorrect += 1
				total += 1
		return incorrect/total

	def base_learn(self, training_set, parameters):
		classifier = tree.DecisionTreeClassifier()
		classifier.set_params(**parameters)
		x_train = []
		y_train = []

		# resample the dataset to match X,y form
		for i in range(len(training_set)):
			x_train.append(list(training_set[i]["X"].values()))
			y_train.append(training_set[i]["y"])

		# Train the classifier
		classifier.fit(x_train, y_train)
		return classifier

	def predict(self):
		total_y_test = []
		total_y_pred = []
		total_y_score = []
		for dataset_id in range(len(self.datasets)):
			X_test = []
			y_test = []

			for j in range(len(self.datasets[dataset_id]["test"])):
				X_test.append(list(self.datasets[dataset_id]["test"][j]["X"].values()))
				y_test.append(self.datasets[dataset_id]["test"][j]["y"])

			for classifier in self.classifier_list[dataset_id]:
				total_y_test += y_test
				total_y_pred += list(classifier.predict(X_test))
				total_y_score += list(classifier.predict_proba(X_test)[:, 1])

		return total_y_test, total_y_pred, total_y_score

	def parameter_optimize(self, class_a_set):
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

		for criterionArg in ["gini", "entropy"]:
			for sample_split in range(30, 70, 5):
				for maxDepthArg in range(5, 15):
					metrics_y_true = []
					metrics_y_test = []
					metrics_y_score = []
					for dataset_ID in range(0, 10):
						parameters = {'criterion': criterionArg,
									  'max_depth': maxDepthArg,
									  'min_samples_split': sample_split,
									  'random_state': 0}
						self.decorate_algorithm(dataset_ID, self.datasets[dataset_ID]["training"], parameters=parameters)
					y_test, y_pred, y_score = self.predict()
					metrics_y_true += y_test
					metrics_y_test += y_pred
					metrics_y_score += y_score[:,1]

					evaluated_test_metrics = self.evaluationMetrics(metrics_y_true, metrics_y_test, metrics_y_score)

					for key in evaluated_test_metrics:
						if key not in best_parameters.keys():
							best_parameters[key] = {"value": 0, "parameters": None}
						if best_parameters[key]["value"] <= evaluated_test_metrics[key]:
							best_parameters[key]["value"] = evaluated_test_metrics[key]
							best_parameters[key]["parameters"] = parameters
		return best_parameters