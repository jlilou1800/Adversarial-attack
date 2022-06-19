# import cv2
# from src import DetectPlates, Preprocess, DecisionTree, KNearestNeighbors, Database
import os

#Color code
from src import Database

def main():
    db = Database.Database(10000)
    # db.createDb()
    # db.kFoldCrossValidation(10, 1)
    X_test_path = "dataset_flattened/X_test.txt"



    # DT = DecisionTree.DecisionTree()
    # DT.fit()
    # DT.predict()
    # DT.getAccuracy()


    # KNN = KNearestNeighbors.KNearestNeighbors()
    # KNN.fit()
    # KNN.predict()
    # KNN.getAccuracy()

    #
    # y_pred = KNN.predict(X_test_path)
    # for _ord in y_pred:
    #     print(chr(int(_ord)))
    # print("-------------------------")


    # for _ord in y_pred:
    #     print(chr(int(_ord)))
    # print("-------------------------")
    # #############################################################################################
    #
    # print("------------------------------------------------------------------------------")


# def createflattenedTest(img_path):
#     img = cv2.imread(img_path)
#     x_db = open('charsPlateFlattened/X_test.txt', 'a')
#     img_str = ""
#     for i in range(len(img)):
#         for j in range(len(img[i])):
#             _pix = sum(img[i][j][0:3])
#             if _pix > 0:                    # if pixel is activated return 1, else return 0
#                 img_str += ' ' + str(1)     # only two color in flattened image file (0 and 1)
#             else:
#                 img_str += ' ' + str(0)
#     x_db.write(img_str + '\n')
#     x_db.close()


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
