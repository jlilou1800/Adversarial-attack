# coding: utf8
import os

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


import numpy as np
import matplotlib.pyplot as plt
from src import Database
from src.Classifiers import AdaptiveBoosting, LogisticRegression, BayesianNetwork, DecisionTree, RandomForest, Support_Vector_Machine, K_NearestNeighbors
import warnings

import tensorflow as tf
from keras.utils import np_utils
# from cleverhans.tf2.attacks import fast_gradient_method, basic_iterative_method, momentum_iterative_method

import cleverhans.attacks.

warnings.simplefilter(action='ignore', category=FutureWarning)

COLUMN = 20
LINE = 25


def adversarial_attack(X_test, classifierAlgo):
    adversarial_img = []
    for img in X_test:
        # applying random noise does not fool the classifier
        img = img.reshape(LINE, COLUMN)
        # plt.show()
        quantized_noise = np.round(np.random.normal(loc=0.0, scale=0.3, size=img.shape) * 255.) / 255.
        noisy_img = np.clip(img + quantized_noise, 0., 1.)
        plt.imshow(noisy_img.reshape(LINE, COLUMN), vmin=0., vmax=1.)
        adversarial_img.append(noisy_img.flatten())
        # plt.show()
    noisy_prediction = classifierAlgo.predict2(adversarial_img)
    print("-" * 15)
    print("X_adv prédiction = ", noisy_prediction[0])
    y_pred = noisy_prediction[0]
    return y_pred

def create_adv_examples(model, input_t, x_to_adv, attack_dict):
    """
    This fn may seem bizarre and pointless, but the point of it is to
    enable the entire attack to be specified as a dict from the command line without
    editing this script, which is convenient for storing the settings used for an attack
    """
    if attack_dict['method'] == 'fgm':
        attack = FastGradientMethod(model, sess=K.get_session(), back='tf')
    elif attack_dict['method'] == 'bim':
        attack = attacks.BasicIterativeMethod(model, sess=K.get_session(), back='tf')
    elif attack_dict['method'] == 'mim':
        attack = attacks.MomentumIterativeMethod(model, sess=K.get_session(), back='tf')
    else:
        assert False, 'Current attack needs to be added to the create attack fn'
    adv_tensor = attack.generate(input_t, **{k: a for k, a in attack_dict.items() if
                                             k != 'method'})  # 'method' key for this fn use
    x_adv = batch_eval(adv_tensor, input_t, x_to_adv, batch_size=args.batch_size, verbose="Generating adv examples")
    return x_adv

def main():
    # db = Database.Database(10000)
    # db.createDb()
    # db.kFoldCrossValidation(10, 1)
    # parameter_testing()

    X = np.loadtxt("{}/X_test.txt".format('src/dataset_flattened'))[0]
    # plt.imshow(X.reshape(LINE, COLUMN), vmin=0., vmax=1.)
    # plt.show()
    # fgsm(X, ord("N"),0.1)





    # print("Decision Tree - Best Parameters establishing ...")                 # accuracy = 0.936
    # DT = DecisionTree.DecisionTree('f_measure')
    #
    # X_test = np.loadtxt("{}/X_test.txt".format('src/dataset_flattened'))
    # x_pred = DT.predict2(X_test[:5])[0]
    # print("X prédiction = ", x_pred)
    # adversarial_attack(X_test[:5], DT)


    # print(img)
    # prediction = DT.predict2(img)
    # print("-"*15)
    # print("X prédiction = ", prediction)


    # perturbations = create_adversarial_pattern(X_test[0], label)

    # X_adv = gen_tf2_fgsm_attack(1, X_test[0])
    # print("-------------------------------------")
    # print(X_test[0])
    # print("-------------------------------------")
    # print(X_adv)
    # # # print(np.reshape(X_test[0], (-1, COLUMN)))
    # for data in X_test:
    #     array = np.reshape(data, (-1, COLUMN))

#     np.random.seed(0)  # pour toujours reproduire le meme dataset
#
#     n_samples = 100  # nombre d'echantillons a générer
#     x = np.linspace(0, 10, n_samples).reshape((n_samples, 1))
#     y = x + np.random.randn(n_samples, 1)
#
#     plt.scatter(x, y)  # afficher les résultats. X en abscisse et y en ordonnée
#     plt.show()
#
#     # ajout de la colonne de biais a X
#     X = np.hstack((x, np.ones(x.shape)))
#     print(X.shape)
#
#     # création d'un vecteur parametre theta
#     theta = np.random.randn(2, 1)
#     print(theta)
#     # Example de test :
#     print(grad(X, y, theta))
#
#

# def create_tf_model(input_size, num_of_class):
#     """ This method creates the tensorflow classification model """
#     model_kddcup = tf.keras.Sequential([
#         tf.keras.layers.Dense(200, input_dim=input_size, activation=tf.nn.relu),
#         tf.keras.layers.Dense(500, activation=tf.nn.relu),
#         tf.keras.layers.Dense(200, activation=tf.nn.relu),
#         tf.keras.layers.Dense(num_of_class),
#         # We seperate the activation layer to be able to access
#         # the logits of the previous layer later
#         tf.keras.layers.Activation(tf.nn.softmax)
#         ])
#     model_kddcup.compile(loss='categorical_crossentropy',
#                          optimizer='adam',
#                          metrics=['accuracy'])
#     return model_kddcup
#
# def gen_tf2_fgsm_attack(org_model, x_test):
#     """ This method creates adversarial examples with fgsm """
#     logits_model = tf.keras.Model(500, 1)
#     epsilon = 0.1
#     adv_fgsm_x = fast_gradient_method(logits_model,
#                                       x_test,
#                                       epsilon,
#                                       np.inf,
#                                       targeted=False)
#     return adv_fgsm_x
#
# def gen_tf2_bim(org_model, x_test):
#     """ This method creates adversarial examples with bim """
#     logits_model = tf.keras.Model(org_model.input, model.layers[-1].output)
#
#     epsilon = 0.1
#     adv_bim_x = basic_iterative_method(logits_model,
#                                        x_test,
#                                        epsilon,
#                                        0.1,
#                                        nb_iter=10,
#                                        norm=np.inf,
#                                        targeted=True)
#     return adv_bim_x

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
#         theta = theta - learning_rate * grad(X, y,
#                                              theta)  # mise a jour du parametre theta (formule du gradient descent)
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
