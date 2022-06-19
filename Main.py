from src.Classifiers import AdaptiveBoosting, LogisticRegression, BayesianNetwork, DecisionTree, RandomForest, Decorate, Support_Vector_Machine #, K_NearestNeighbors


def main():
    pass


def parameter_testing(data):
    print("Decision Tree - Best Parameters establishing ...")
    DT = DecisionTree.DecisionTree('f_measure')

    # print("Decorate - Best Parameters establishing ...")
    # Deco = Decorate.Decorate(best_parameters=DT.get_best_parameters(), metric_choose='f_measure')
    #
    # print("Random Forest - Best Parameters establishing ...")
    # RF = RandomForest.RandomForest('f_measure')
    #
    # print("AdaBoost - Best Parameters establishing ...")
    # AB = AdaptiveBoosting.AdaBoost('f_measure')
    #
    # # print("K Nearest Neighbors - Best Parameters establishing ...")
    # # KNN = K_NearestNeighbors.K_NearestNeighbors('f_measure')
    #
    # print("Logistic Regression - Best Parameters establishing ...")
    # LR = LogisticRegression.Logistic_Regression('f_measure')
    #
    # print("Support Vector Machine - Best Parameters establishing ...")
    # SVM = Support_Vector_Machine.Support_Vector_Machine('f_measure')
    #
    # print("Bayesian Network - Best Parameters establishing ...")
    # BN = BayesianNetwork.BayesianNetwork('f_measure')


if __name__ == "__main__":
    main()
