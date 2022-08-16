import cv2
import numpy as np
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from random import choice
import os
from shutil import copy
import json
# from src import Preprocess, PossibleChar
from src import PossibleChar, Preprocess

run = True

FACTOR_SCALE = 1

class Database:
    def __init__(self, nbr_of_file, repository_name="src/dataset", font_pack_path="src/Font_Pack", img_width=100, img_height=100, font_size=110):
        self.nbrOfFile = nbr_of_file
        self.alphaNum = "abcdefghijklmnopqrstuvw"   #0123456789"
        self.repository = repository_name
        self.imgWidth = img_width
        self.imgHeight = img_height
        self.fontPackPath = font_pack_path
        self.fontSize = font_size
        self.labels = {}

    def define_labels(self):
        for char in self.alphaNum:
            self.labels[ord(char)] = char
        with open('src/dataset_flattened/labels.json', 'w') as f:
            json.dump(self.labels, f)

    def createDb(self):
        for repo in [self.repository, '{}_flattened'.format(self.repository)]:      # create dataset folders
            if not os.path.exists(repo):     # if data set repository doesn't exist
                os.makedirs(repo)
            clearRepository(repo)          # clean repository of data set
        n = 0
        while n < self.nbrOfFile:
            if n % 200 == 0:
                print(n)
            rand_char = choice(self.alphaNum)
            font_files = os.listdir(self.fontPackPath)
            # rand_font = self.fontPackPath + "/" + 'arialbd.ttf'
            rand_font = self.fontPackPath + "/" + choice(font_files)
            success, img = self.createImage(n+1, rand_char, rand_font)
            if success:
                n += 1
                self.updateFlattenedDataset(img, rand_char)
            else:
                print('error')

    def updateFlattenedDataset(self, img, char):
        x_db = open('{0}_flattened/X_dataset.txt'.format(self.repository), 'a')
        y_db = open('{0}_flattened/Y_dataset.txt'.format(self.repository), 'a')
        img_str = ""
        for i in range(len(img)):
            for j in range(len(img[i])):
                _pix = sum(img[i][j][0:3])
                # if _pix > 0:                    # if pixel is activated return 1, else return 0
                #     img_str += ' ' + str(1)     # only two color in flattened image file (0 and 1)
                # else:
                #     img_str += ' ' + str(0)
                img_str += ' ' + str(_pix)
        x_db.write(img_str + '\n')
        y_db.write(str(ord(char)) + '\n')
        x_db.close(), y_db.close()

    def center_text(self, img, font, text, color=(0, 0, 0)):
        draw = ImageDraw.Draw(img)
        text_width, text_height = draw.textsize(text, font)
        position = ((self.imgWidth - text_width)/2, (self.imgHeight - text_height)/2)
        draw.text(position, text, color, font=font)
        return img

    def createImage(self, n, char, font, updt=False, repo=False):
        path = "{}/data_{}_{}.jpeg".format(self.repository, n, char)
        if updt:
            path = "charsPlate/data_{}_{}.jpeg".format(repo, n, char)
        font = ImageFont.truetype(font, self.fontSize)
        img = Image.new("RGBA", (self.imgWidth, self.imgHeight), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        # w, h = draw.textsize(char)
        self.center_text(img, font, char)
        # draw.text((self.imgWidth, 0), char, (255, 255, 255), font=font, align="center")
        # img.show()
        # success, imgCropped = self.resizeImg(img, path)
        # return success, imgCropped
        # imgOriginal = np.asarray(img).copy()
        imgOriginal = np.array(img)
        cv2.imwrite(path + '_test.jpeg', imgOriginal)
        return True, imgOriginal

    def resizeImg(self, img, imgPath):
        imgOriginal = np.asarray(img).copy()
        imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(imgOriginal)  # preprocess to get grayscale and threshold images
        imgThreshScene = imgThreshScene.copy()
        contours, npaHierarchy = cv2.findContours(imgThreshScene, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # find all contours
        ext = [self.imgHeight, 0, self.imgWidth, 0]
        for i in range(len(contours)):  # for each contour
            possibleChar = PossibleChar.PossibleChar(contours[i])
            tmp = possibleChar.getExtrema()
            ext = [min(ext[0], tmp[0]), max(ext[1], tmp[1]), min(ext[2], tmp[2]), max(ext[3], tmp[3])]
        succes = False
        crop_img = None
        if len(contours) > 0:
            succes = True
            crop_img = imgOriginal[ext[0]:ext[1], ext[2]:ext[3]]
            crop_img = cv2.resize(crop_img, (self.imgHeight, self.imgWidth))
            cv2.imwrite(imgPath + '_test.jpeg', crop_img)
        return succes, crop_img
        # return True, img

    def updateTestOrTrainingFlattened(self, i, j):
        x_db = open('{0}_flattened/X_dataset.txt'.format(self.repository), 'r')
        y_db = open('{0}_flattened/Y_dataset.txt'.format(self.repository), 'r')
        x_lines = x_db.readlines()
        y_lines = y_db.readlines()
        x_train = open('{0}_flattened/X_train.txt'.format(self.repository), 'w')
        y_train = open('{0}_flattened/Y_train.txt'.format(self.repository), 'w')
        x_test = open('{0}_flattened/X_test.txt'.format(self.repository), 'w')
        y_test = open('{0}_flattened/Y_test.txt'.format(self.repository), 'w')
        x_train.write(''.join(x_lines[:i] + x_lines[j:]))
        y_train.write(''.join(y_lines[:i] + y_lines[j:]))
        x_test.write(''.join(x_lines[i:j]))
        y_test.write(''.join(y_lines[i:j]))
        x_train.close(), y_train.close(), x_test.close(), y_test.close()

    def kFoldCrossValidation(self, k, iteration):
        repository_names = ("training", "test")
        for repo in repository_names:
            if not os.path.exists(repo):    # if data set repository doesn't exist
                os.makedirs(repo)
            clearRepository(repo)           # clean repository of data set
        step = self.nbrOfFile // k
        index1 = step*(iteration-1)
        index2 = step*iteration
        self.updateTestOrTrainingFlattened(index1, index2)
        for file in os.listdir(self.repository):
            if index1 < int(file.split('_')[1]) <= index2:
                copy("{}/{}".format(self.repository, file), "test")
            else:
                copy("{}/{}".format(self.repository, file), "training")

    def update(self, repo, i):
        f = open('README.txt', 'r')
        tmp = f.readlines()[i]
        clearRepository(repo)
        rand_font = self.fontPackPath + "/" + 'arialbd.ttf'
        n = 0
        for char in tmp:
            self.createImage(n, char, rand_font, True, repo)
            n += 1


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





def main():
    #clearRepository('dataset_flattened')
    if run:
        test = Database(10000)
        test.createDb()
        test.kFoldCrossValidation(10, 1)

        print("Process finished")


if __name__ == "__main__":
    main()
