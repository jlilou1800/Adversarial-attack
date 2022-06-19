from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import metrics
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline
from sklearn import tree
import matplotlib.pyplot as plt
import pydotplus
# from sklearn.externals.six import StringIO
from sklearn.metrics import classification_report, confusion_matrix


class KNearestNeighbors:
    def __init__(self, k=1):
        self.X_train, self.Y_train, self.X_test, self.Y_test = None, None, None, None
        self.loadData()
        self.Y_pred_test, self.Y_pred_train = None, None
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.knn.fit(self.X_train, self.Y_train)

    def fit(self):
        self.knn.fit(self.X_train, self.Y_train)

    def predict(self):
        self.Y_pred_test = self.knn.predict(self.X_test)
        self.Y_pred_train = self.knn.predict(self.X_train)

    def getAccuracy(self):
        print("Accuracy:", metrics.accuracy_score(self.Y_test, self.Y_pred_test))


    def loadData(self):
        self.X_train = np.loadtxt("{}/X_train.txt".format('dataset_flattened'))
        self.Y_train = np.loadtxt("{}/Y_train.txt".format('dataset_flattened'))
        self.X_test = np.loadtxt("{}/X_test.txt".format('dataset_flattened'))
        self.Y_test = np.loadtxt("{}/Y_test.txt".format('dataset_flattened'))

#
# # for k in range(1, 10):
# #     # create KNN Classifier
# #     knn = KNeighborsClassifier(n_neighbors=k)
# #     # train the model using the training sets
# #     knn.fit(X_train, Y_train)
# #     # predict the response for test dataset
# #     y_pred = knn.predict(X_test)
# #     #print(y_pred)
# #     print(k)
# #     print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))
#
#
# def getDecisionTreeMaxDepth(x_train, y_train, x_test, y_test, max_depth):
#     # Creation of decision tree
#     clf = tree.DecisionTreeClassifier(max_depth=max_depth, criterion="entropy")
#     clf.fit(x_train, y_train)
#     # Test prediction
#     y_pred_test = clf.predict(x_test)
#     y_pred_train = clf.predict(x_train)
#
#     print("Accuracy:", metrics.accuracy_score(Y_test, y_pred_test))
#
#     # # Accuracy of our algorithm (evaluation)
#     # getEvaluation(y_train, y_pred_train, y_test, y_pred_test, 'evaluation_max_depth_'+str(max_depth)+'.txt')
#     # # illustrations maker (graph and decision tree)
#     # treeToPng(clf, 'dt_max_depth'+str(max_depth))
#
#
# def treeToPng(model, filename):
#     dot_data = StringIO()
#     tree.export_graphviz(model, out_file=dot_data,
#                          filled=True, rounded=True,
#                          special_characters=True, )
#     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#     graph.write_png('%s.png' % filename)
#
# def getErrorPlotMaxDepth(x_train, y_train, x_test, y_test, filename):
#     x = list()
#     ytrain = list()
#     ytest = list()
#     for i in range(1, 101):
#         x.append(i)
#         clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=i)
#         clf = clf.fit(x_train, y_train)
#
#         pred_train = clf.predict(x_train)
#         ytrain.append(1-metrics.accuracy_score(y_train, pred_train))
#         pred_test = clf.predict(x_test)
#         ytest.append(1-metrics.accuracy_score(y_test, pred_test))
#
#     getPlot(x, ytrain, ytest, filename, 'max_depth')
#
#
# def getErrorPlotMaxLeafNodes(x_train, y_train, x_test, y_test, filename):
#     x = list()
#     _ytrain = list()
#     _ytest = list()
#     for i in range(2, 100):
#         x.append(i)
#         clf = tree.DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=i)
#         clf = clf.fit(x_train, y_train)
#
#         pred_train = clf.predict(x_train)
#         _ytrain.append(1-metrics.accuracy_score(y_train, pred_train))
#         pred_test = clf.predict(x_test)
#         _ytest.append(1-metrics.accuracy_score(y_test, pred_test))
#
#     getPlot(x, _ytrain, _ytest, filename, 'max_leaf_nodes')
#
# def getEvaluation(y_train, y_pred_train, y_test, y_pred_test, filename):
#     with open( filename, 'w') as f:
#         print("Evaluation for training set:", file=f)
#         print(confusion_matrix(y_train, y_pred_train), file=f)
#         print(classification_report(y_train, y_pred_train), file=f)
#         print("Error for training set: " + str(1 - metrics.accuracy_score(y_train, y_pred_train)), file=f)
#         print("\n", file=f)
#         print("Evaluation for test set:", file=f)
#         print(confusion_matrix(y_test, y_pred_test), file=f)
#         print(classification_report(y_test, y_pred_test), file=f)
#         print("Error for test set: " + str(1 - metrics.accuracy_score(y_test, y_pred_test)), file=f)
#
# def getPlot(x, y_train, y_test, filename, meta_parameter):
#     plt.plot(x, y_train, 'b', label="training set error")
#     plt.plot(x, y_test, 'r', label="testing set error")
#     plt.xlabel(meta_parameter)
#     plt.ylabel("errors")
#     plt.legend()
#     plt.savefig(filename + ".png")
#     plt.clf()
#
# # for i in range(1, 20):
# #     getDecisionTreeMaxDepth(X_train, Y_train, X_test, Y_test, i)
