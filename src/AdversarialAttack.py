#import required libs
import torch
import torch.nn
# from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import numpy as np
import requests, io
import matplotlib.pyplot as plt
from matplotlib import cm
import json
from torch.autograd import Variable
# %matplotlib inline

from PIL import TiffImagePlugin, JpegImagePlugin

COLUMN = 250
LINE = 250

class AdversarialAttack:
    def __init__(self):
        self.inceptionv3 = models.inception_v3(pretrained=True)  # download and load pretrained inceptionv3 model
        self.inceptionv3.eval()
        self.output = None
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.labels = self.get_labels()

    def adversarial_attack(self, method, x_test, y_true, classifier, eps):
        img = self.array_to_jpeg(x_test)
        image_tensor, img_variable = self.image_to_tensor(img)
        self.output = self.inceptionv3.forward(img_variable)
        y_test = classifier.predict2([x_test])
        y_test = int(y_test[0][0])
        y_pred_label = self.labels[str(y_test)]
        pred_index = ord(y_pred_label) - 65
        x_pred_prob = classifier.predict_proba([x_test])[0][pred_index]

        if method == "fgsm":
            x_adversarial, x_grad, y_adv_pred, y_adv_pred_label, adv_pred_prob = self.fgsm(y_true, img_variable, eps, classifier)


        visualize(image_tensor, x_adversarial, x_grad, eps, y_pred_label, y_adv_pred_label, x_pred_prob, adv_pred_prob)

    def array_to_jpeg(self, x_array):
        x_array_2d = x_array.reshape(LINE, COLUMN)
        img = Image.fromarray(x_array_2d)
        img = img.convert("RGB")

        byte_io = io.BytesIO()
        img.save(byte_io, format="JPEG")
        jpg_buffer = byte_io.getvalue()
        byte_io.close()
        img = Image.open(io.BytesIO(jpg_buffer))

        return img

    def image_to_tensor(self, img):
        preprocess = transforms.Compose([
            transforms.Resize((LINE, COLUMN)),
            # transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        image_tensor = preprocess(img)  # preprocess an i
        image_tensor = image_tensor.unsqueeze(0)  # add batch dimension.  C X H X W ==> B X C X H X W
        img_variable = Variable(image_tensor, requires_grad=True)  # convert tensor into a variable

        return image_tensor, img_variable

    def get_labels(self):
        labels_link = "src/dataset_flattened/labels.json"
        with open(labels_link, 'r') as f:
            labels_json = json.load(f)
        # labels = {int(idx): label for idx, label in labels_json.items()}
        return labels_json

    def fgsm(self, y_true, img_variable, eps, classifier):
        target = Variable(torch.LongTensor([y_true]), requires_grad=False)
        # perform a backward pass in order to get gradients
        loss = torch.nn.CrossEntropyLoss()
        loss_cal = loss(self.output, target)
        loss_cal.backward(retain_graph=True)  # this will calculate gradient of each variable (with requires_grad=True) and can be accessed by "var.grad.data"

        x_grad = torch.sign(img_variable.grad.data)  # calculate the sign of gradient of the loss func (with respect to input X) (adv)
        x_adversarial = img_variable.data + eps * x_grad  # find adv example using formula shown above
        x_adv = x_adversarial.data.numpy()[0][0]
        x_adv = x_adv.flatten()
        y_adv_pred = classifier.predict2([x_adv])
        y_adv_pred = int(y_adv_pred[0][0])
        y_adv_pred_label = self.labels[str(y_adv_pred)]

        # output_adv = self.inceptionv3.forward(Variable(x_adversarial))  # perform a forward pass on adv example

        pred_index = ord(y_adv_pred_label) - 65
        adv_pred_prob = classifier.predict_proba([x_adv])[0][pred_index]

        return x_adversarial, x_grad, y_adv_pred, y_adv_pred_label, adv_pred_prob

    # def pdgm(self):
    #     pass
    #
    # def ostcm(self):
    #     # targeted class can be a random class or the least likely class predicted by the network
    #     y_target = 288  # leopard
    #     y_target = Variable(torch.LongTensor([y_target]), requires_grad=False)
    #     print(y_target)
    #
    #     zero_gradients(img_variable)  # flush gradients
    #     loss_cal2 = loss(output, y_target)
    #     loss_cal2.backward()
    #     epsilons = [0.002, 0.01, 0.15, 0.5]
    #     x_grad = torch.sign(img_variable.grad.data)
    #     for i in epsilons:
    #         x_adversarial = img_variable.data - i * x_grad
    #         output_adv = self.inceptionv3.forward(Variable(x_adversarial))
    #         x_adv_pred = labels[torch.max(output_adv.data, 1)[1][0]]
    #         op_adv_probs = F.softmax(output_adv, dim=1)
    #         adv_pred_prob = round((torch.max(op_adv_probs.data, 1)[0][0]) * 100, 4)
    #         visualize(image_tensor, x_adversarial, x_grad, i, x_pred, x_adv_pred, x_pred_prob, adv_pred_prob)
    #
    # #
    # def bim(self, epsilon=0.25, num_steps=5, alpha=0.025):
    #     y_true = Variable(torch.LongTensor([282]), requires_grad=False)  # tiger cat
    #     # epsilon = 0.25
    #     # num_steps = 5
    #     # alpha = 0.025
    #     # above three are hyperparameters
    #     for i in range(num_steps):
    #         # zero_gradients(img_variable)  # flush gradients
    #         output = inceptionv3.forward(img_variable)  # perform forward pass
    #         loss = torch.nn.CrossEntropyLoss()
    #         loss_cal = loss(output, y_true)
    #         loss_cal.backward()
    #         x_grad = alpha * torch.sign(img_variable.grad.data)  # as per the formula
    #         adv_temp = img_variable.data + x_grad  # add perturbation to img_variable which also contains perturbation from previous iterations
    #         total_grad = adv_temp - image_tensor  # total perturbation
    #         total_grad = torch.clamp(total_grad, -epsilon, epsilon)
    #         x_adv = image_tensor + total_grad  # add total perturbation to the original image
    #         img_variable.data = x_adv
    #
    #     # final adversarial example can be accessed at- img_variable.data
    #     output_adv = inceptionv3.forward(img_variable)
    #     x_adv_pred = labels[torch.max(output_adv.data, 1)[1][0]]  # classify adversarial example
    #     output_adv_probs = F.softmax(output_adv, dim=1)
    #     x_adv_pred_prob = round((torch.max(output_adv_probs.data, 1)[0][0]) * 100, 4)
    #     visualize(image_tensor, img_variable.data, total_grad, epsilon, x_pred, x_adv_pred, x_pred_prob,
    #               x_adv_pred_prob)  # class and prob of original ex will remain same
    # #
    # def illcm(self):
    #     pass
    #
    # def cw(self):
    #     pass



def visualize(x, x_adv, x_grad, epsilon, clean_pred, adv_pred, clean_prob, adv_prob):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = x.squeeze(0)  # remove batch dimension # B X C H X W ==> C X H X W
    x = x.mul(torch.FloatTensor(std).view(3, 1, 1)).add(
        torch.FloatTensor(mean).view(3, 1, 1)).numpy()  # reverse of normalization op- "unnormalize"
    x = np.transpose(x, (1, 2, 0))  # C X H X W  ==>   H X W X C
    # x = np.clip(x, 0, 1)

    x_adv = x_adv.squeeze(0)
    x_adv = x_adv.mul(torch.FloatTensor(std).view(3, 1, 1)).add(
        torch.FloatTensor(mean).view(3, 1, 1)).numpy()  # reverse of normalization op
    x_adv = np.transpose(x_adv, (1, 2, 0))  # C X H X W  ==>   H X W X C
    x_adv = np.clip(x_adv, 0, 1)

    x_grad = x_grad.squeeze(0).numpy()
    x_grad = np.transpose(x_grad, (1, 2, 0))
    x_grad = np.clip(x_grad, 0, 1)

    figure, ax = plt.subplots(1, 3, figsize=(18, 8))
    ax[0].imshow(x)
    ax[0].set_title('Clean Example', fontsize=20)

    ax[1].imshow(x_grad)
    ax[1].set_title('Perturbation', fontsize=20)
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    ax[2].imshow(x_adv)
    ax[2].set_title('Adversarial Example', fontsize=20)

    ax[0].axis('off')
    ax[2].axis('off')

    ax[0].text(1.1, 0.5, "+{}*".format(round(epsilon, 3)), size=15, ha="center",
               transform=ax[0].transAxes)

    ax[0].text(0.5, -0.13, "Prediction: {}\n Probability: {}".format(clean_pred, clean_prob), size=15, ha="center",
               transform=ax[0].transAxes)

    ax[1].text(1.1, 0.5, " = ", size=15, ha="center", transform=ax[1].transAxes)

    ax[2].text(0.5, -0.13, "Prediction: {}\n Probability: {}".format(adv_pred, adv_prob), size=15, ha="center",
               transform=ax[2].transAxes)

    plt.show()
