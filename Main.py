# coding: utf8
import os

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


import numpy as np
import matplotlib.pyplot as plt
from src import Database
from src.AdversarialAttack import AdversarialAttack
from src.Classifiers import AdaptiveBoosting, LogisticRegression, BayesianNetwork, DecisionTree, RandomForest, Support_Vector_Machine, K_NearestNeighbors
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


warnings.simplefilter(action='ignore', category=FutureWarning)

COLUMN = 20
LINE = 25


def adversarial_attack(X_test, eps, classifierAlgo):
    adversarial_img = []
    for img in X_test:
        # applying random noise does not fool the classifier
        img = img.reshape(LINE, COLUMN)
        # plt.show()
        quantized_noise = np.round(np.random.normal(loc=0.0, scale=eps, size=img.shape) * 255.) / 255.
        noisy_img = np.clip(img + quantized_noise, 0., 1.)
        plt.imshow(noisy_img.reshape(LINE, COLUMN), vmin=0., vmax=1.)
        adversarial_img.append(noisy_img.flatten())
        # plt.show()
    noisy_prediction = classifierAlgo.predict2(adversarial_img)
    print("-" * 15)
    print("Y prédiction = ", classifierAlgo.predict2(X_test)[0])
    print("-" * 15)
    print("X_adv prédiction = ", noisy_prediction[0])
    print("-" * 15)
    y_pred = noisy_prediction[0]
    print("proba adversarial")
    print(classifierAlgo.predict_proba(adversarial_img))
    return y_pred

def main():
    epsilons = [.05, .1, .15, .2, .25, .3]
    methods = ["fgsm", "ostcm", "bim"]
    eps = epsilons[-1]
    # KNN = None
    db = Database.Database(1000)
    # db.createDb()
    # db.kFoldCrossValidation(10, 1)
    db.define_labels()
    KNN = K_NearestNeighbors.K_NearestNeighbors('f_measure')

    X_test = np.loadtxt("{}/X_test.txt".format('src/dataset_flattened'))
    Y_test = np.loadtxt("{}/Y_test.txt".format('src/dataset_flattened'))
    Attack = AdversarialAttack()
    Attack.adversarial_attack(methods[0], X_test[0], Y_test[0], KNN, eps)

    # for eps in epsilons:


    # db = Database.Database(100)
    # db.define_labels()
    # db.createDb()
    # db.kFoldCrossValidation(10, 1)

    # parameter_testing()

    # X_test = np.loadtxt("{}/X_test.txt".format('src/dataset_flattened'))
    # plt.imshow(X.reshape(LINE, COLUMN), vmin=0., vmax=1.)
    # plt.show()

    # print("Decision Tree - Best Parameters establishing ...")                 # accuracy = 0.936
    # DT = DecisionTree.DecisionTree('f_measure')

    # print("K Nearest Neighbors - Best Parameters establishing ...")             # accuracy = 0.938
    # KNN = K_NearestNeighbors.K_NearestNeighbors('f_measure')
    #
    # adversarial_attack(Y[:5], KNN)

    # X_test = np.loadtxt("{}/X_test.txt".format('src/dataset_flattened'))
    # print("y_true = ", 78)
    # x_pred = DT.predict2(X_test[0])
    # print("X prédiction = ", x_pred)
    # x_p = DT.predict_proba(X_test[0])
    # print("X prédiction proba = ", x_p)
    # adversarial_attack(X_test[:5], DT)



# def model(X, theta):
#     return X.dot(theta)
#
# def cost_function(X, y, theta):
#     m = len(y)
#     return 1 / (2 * m) * np.sum((model(X, theta) - y) ** 2)
#
# def grad(X, y, theta):
#     m = len(y)
#     return 1 / m * X.T.dot(model(X, theta) - y)
#
# def gradient_descent(X, y, theta, learning_rate=0.01, n_iterations=1000):
#     # création d'un tableau de stockage pour enregistrer l'évolution du Cout du modele
#     cost_history = np.zeros(n_iterations)
#
#     for i in range(0, n_iterations):
#         theta = theta - learning_rate * grad(X, y, theta)  # mise a jour du parametre theta (formule du gradient descent)
#         cost_history[i] = cost_function(X, y, theta)  # on enregistre la valeur du Cout au tour i dans cost_history[i]
#
#     return theta, cost_history

def parameter_testing():
    print("Decision Tree - Best Parameters establishing ...")                 # accuracy = 0.936
    DT = DecisionTree.DecisionTree('f_measure')

    print("--------------------------------------------------------")
    print("Random Forest - Best Parameters establishing ...")                 # accuracy = 0.911
    RF = RandomForest.RandomForest('f_measure')

    # print("--------------------------------------------------------")
    # print("AdaBoost - Best Parameters establishing ...")                      # accuracy = 0.624
    # AB = AdaptiveBoosting.AdaBoost('f_measure')

    print("--------------------------------------------------------")
    print("K Nearest Neighbors - Best Parameters establishing ...")             # accuracy = 0.938
    KNN = K_NearestNeighbors.K_NearestNeighbors('f_measure')

    print("--------------------------------------------------------")
    print("Logistic Regression - Best Parameters establishing ...")             # accuracy = 0.931
    LR = LogisticRegression.Logistic_Regression('f_measure')

    print("--------------------------------------------------------")
    print("Support Vector Machine - Best Parameters establishing ...")          # accuracy = 0.965
    SVM = Support_Vector_Machine.Support_Vector_Machine('f_measure')

    # print("--------------------------------------------------------")
    # print("Bayesian Network - Best Parameters establishing ...")              # accuracy = 0.631
    # BN = BayesianNetwork.BayesianNetwork('f_measure')

def clearRepository(repo_name):
    try:
        for file in os.listdir(repo_name):
            file = os.path.join(repo_name, file)
            try:
                if os.path.isfile(file):
                    os.unlink(file)
            except Exception as e:
                print(e)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
