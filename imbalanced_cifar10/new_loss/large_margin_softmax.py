import sys
sys.path.append("../")

import tqdm

import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data import random_split
from torch.utils.data import DataLoader

from typing import Callable

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from imbalance_cifar import IMBALANCECIFAR10

SPACING = 30
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class LargeMarginSoftmaxLoss:

    def __init__(self, m: int):
        self._m: int = int(m)

        # slope_direction_changes -> values where cos(m*theta) will
        #                            change the direction of its slope
        # cos(1 * theta)          -> pi
        # cos(2 * theta)          -> pi/2, pi
        # cos(3 * theta)          -> pi/3, 2pi/3, pi
        # ...                     -> ...

        slope_direction_changes = np.linspace(0, torch.pi, self._m + 1)[1:]
        self._slope_direction_changes = slope_direction_changes

        self._psy: Callable = self._select_psy(self._m)

    def _select_psy(self, m: int) -> Callable:
        # for m=1,2,3,4 a hardcoded version of psy has been implemented
        # in order to speed up the loss-calculation

        if m == 1:
            # hardcoded
            return self._psy_m1
        elif m == 2:
            # hardcoded
            return self._psy_m2
        elif m == 3:
            # hardcoded
            return self._psy_m3
        elif m == 4:
            # hardcoded
            return self._psy_m4
        else:
            # dynamic calculation of psy
            # for arbitrary values of m
            return self._psy_mx

    def chebyshev_polynomial(self, cos_theta: torch.Tensor, degree: int) -> torch.Tensor:
        # chebyshev_polynomial for arbitrary m:
        # more information at: https://en.wikibooks.org/wiki/Trigonometry/For_Enthusiasts/Chebyshev_Polynomials

        iterations = int(degree // 2) + 1
        collect_partial_sums = torch.zeros_like(cos_theta)

        for k in range(iterations + 1):
            binomial_coefficient = math.comb(degree, 2 * k)
            term = torch.pow(torch.square(cos_theta) - 1, k) * torch.pow(cos_theta, degree - 2 * k)
            collect_partial_sums += binomial_coefficient * term
        return collect_partial_sums

    def _psy_mx(self, cos_theta: torch.Tensor) -> torch.Tensor:
        # dynamic calculation of psy for arbitrary values of m
        chebyshev_polynomial = self.chebyshev_polynomial(cos_theta, self._m)
        r = torch.zeros_like(cos_theta)
        for value in self._slope_direction_changes:
            r += (cos_theta < value).int()
        return torch.pow(-1, r) * chebyshev_polynomial - (2 * r)

    def _psy_m1(self, cos_theta: torch.Tensor) -> torch.Tensor:
        # psy(theta)                        =  (-1)^r * cos(m*theta) - 2r
        # slope direction changes
        #       theta in [0.0, pi]          => r = 0
        # chebyshev_polynomial for m=1      => cos(1*theta) = cos(theta)
        # psy(theta) with m=1               =  cos(theta)

        return cos_theta

    def _psy_m2(self, cos_theta: torch.Tensor) -> torch.Tensor:
        # psy(theta)                        =  (-1)^r * cos(m*theta) - 2r
        # slope direction changes
        #       theta in [0.0, pi/2]        => r = 0
        #       theta in (pi/2, pi]         => r = 1
        # chebyshev_polynomial for m=2      => 2*cos(theta)^2 -1

        limit = 0.0  # aka cos(pi/2)
        r = (cos_theta < limit).int()
        chebyshev_polynomial = 2 * torch.square(cos_theta) - 1
        return torch.pow(-1, r) * chebyshev_polynomial - (2 * r)

    def _psy_m3(self, cos_theta: torch.Tensor) -> torch.Tensor:
        # psy(theta)                        =  (-1)^r * cos(m*theta) - 2r
        # slope direction changes
        #       theta in [0.0, pi/3]        => r = 0
        #       theta in (pi/2, pi]         => r = 1
        #       theta in [0.0, pi / 2]      => r = 2
        # chebyshev_polynomial for m=3      => 3*cos(theta)^3 - 3*cos(theta)

        limit_1 = 0.5  # aka cos(pi/3)
        limit_2 = -0.5  # aka cos(2pi/3)
        r = (cos_theta < limit_1).int() + (cos_theta < limit_2).int()
        chebyshev_polynomial = 3 * torch.pow(cos_theta, 3) - 3 * cos_theta
        return torch.pow(-1, r) * chebyshev_polynomial - (2 * r)

    def _psy_m4(self, cos_theta: torch.Tensor) -> torch.Tensor:
        # psy(theta)                        =  (-1)^r * cos(m*theta) - 2r
        # slope direction changes
        #       theta in [0.0, pi/3]        => r = 0
        #       theta in (pi/2, pi]         => r = 1
        #       theta in [0.0, pi / 2]      => r = 2
        # chebyshev_polynomial for m=4      => 8*cos(theta)^4 - 8*cos(theta)^2 + 1

        limit_1 = math.cos(math.pi/4)
        limit_2 = math.cos(math.pi/2)
        limit_3 = math.cos(3*math.pi/4)
        r = (cos_theta < limit_1).int() + (cos_theta < limit_2).int() + (cos_theta < limit_3).int()
        chebyshev_polynomial = 8 * torch.pow(cos_theta, 4) - 8 * torch.pow(cos_theta, 2) + 1
        return torch.pow(-1, r) * chebyshev_polynomial - (2 * r)

    def calculate_loss(self, output: torch.Tensor, features: torch.Tensor,
                       weights: torch.Tensor, labels: torch.Tensor):

        # calculate ||w|| * ||f||
        feature_norm = torch.norm(features, dim=1)
        weight_norm = torch.norm(weights[labels], dim=1)
        norm = (feature_norm * weight_norm)
        is_not_zero = norm != 0

        # model output where class == y_true (aka. correct class label)
        y_true_output = output[range(len(output)), labels]

        # set cos_theta to zero by default, because
        # if ||w|| * ||f|| = 0
        # then ||w|| * ||f|| * psy = 0
        cos_theta = torch.zeros_like(y_true_output)

        # avoid div by zero
        cos_theta[is_not_zero] = y_true_output[is_not_zero] / norm[is_not_zero]

        # re-weight model output
        output[range(len(output)), labels] = norm * self._psy(cos_theta)

        # softmax loss aka. -log(softmax(output))
        return F.cross_entropy(output, labels)

    def __call__(self, *args, **kwargs):
        return self.calculate_loss(*args, **kwargs)


class AdaptedVGG16(nn.Module):

    def __init__(self):
        super().__init__()

        vgg16 = models.vgg16(weights='VGG16_Weights.DEFAULT')

        vgg16.avgpool = nn.Identity()

        # add new classifier
        vgg16.classifier = nn.Sequential(nn.Linear(512, 128, bias=True),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(p=0.25, inplace=False),
                                         nn.Linear(128, 32, bias=True),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(p=0.25, inplace=False),
                                         )

        # train the entire model
        for param in vgg16.parameters():
            param.requires_grad = True

        self.vgg16 = vgg16

        # the last layer of the new classifier is
        # outside the Sequential module
        self.last_linear_layer = nn.Linear(32, 10, bias=False)

    def forward(self, x):
        features = self.vgg16(x)
        outputs = self.last_linear_layer(features)
        weights = self.last_linear_layer.weight
        return outputs, features, weights


class AdaptedResNet18(nn.Module):

    def __init__(self):
        super().__init__()

        resnet18 = models.resnet18(weights='ResNet18_Weights.DEFAULT')

        # reduce the kernel size for fst conv-layer
        # source --> https://ai-pool.com/a/s/end-to-end-pytorch-example-of-image-classification-with-convolutional-neural-networks
        resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1), bias=False)

        # add new classifier
        resnet18.fc = nn.Sequential(nn.Linear(512, 64, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0.25, inplace=False),
                                    )

        # train the entire model
        for param in resnet18.parameters():
            param.requires_grad = True

        self.resnet18 = resnet18

        # the last layer of the new classifier is
        # outside the Sequential module
        self.last_linear_layer = nn.Linear(64, 10, bias=False)

    def forward(self, x):
        features = self.resnet18(x)
        outputs = self.last_linear_layer(features)
        weights = self.last_linear_layer.weight
        return outputs, features, weights


def nice_print(segments: [str]):
    text = ""
    for line in segments:
        white_space = " " * max(0, SPACING - len(line))
        text += line + white_space
    print(text)


def load_imbalanced_CIFAR10(batch_size: int,
                            train_validation_split: float,
                            imbalance_factor: float,
                            imbalance_type: str) -> (DataLoader, DataLoader, DataLoader):

    image_transformations = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(
                                                    (0.4914, 0.4822, 0.4465),
                                                    (0.2023, 0.1994, 0.2010))
                                                ])

    training_dataset = IMBALANCECIFAR10(root='../../dataset',
                                        train=True,
                                        download=True,
                                        imb_type=imbalance_type,
                                        imb_factor=imbalance_factor,
                                        transform=image_transformations)

    test_dataset = torchvision.datasets.CIFAR10(root='../../dataset',
                                                train=False,
                                                download=True,
                                                transform=image_transformations)

    sample_size = len(training_dataset)
    training_dataset_size = int(sample_size * train_validation_split)
    validation_dataset_size = sample_size - training_dataset_size

    split = torch.utils.data.random_split(training_dataset,
                                          lengths=(training_dataset_size, validation_dataset_size))

    training_dataset, validation_dataset = split

    training_data_loader: DataLoader = torch.utils.data.DataLoader(training_dataset,
                                                                   batch_size=batch_size,
                                                                   pin_memory=True,
                                                                   shuffle=True)

    validation_data_loader: DataLoader = torch.utils.data.DataLoader(validation_dataset,
                                                                     batch_size=batch_size,
                                                                     pin_memory=True,
                                                                     shuffle=True)

    test_data_loader: DataLoader = torch.utils.data.DataLoader(test_dataset,
                                                               batch_size=batch_size,
                                                               pin_memory=True,
                                                               shuffle=False)

    return training_data_loader, validation_data_loader, test_data_loader


def fit(model: nn.Module, criterion: Callable, optimizer: optim.Optimizer,
        training_data: DataLoader, epoch: int or str):

    model.train()
    training_loss = []
    training_accuracy = []
    for samples, labels in tqdm.tqdm(training_data, leave=False):
        samples = samples.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs, features, weights = model(samples)
        loss = criterion(outputs, features, weights, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        _, prediction = torch.max(outputs, dim=1)
        accuracy = torch.tensor(torch.sum(prediction == labels).item() / len(prediction))

        training_loss.append(loss.cpu().item())
        training_accuracy.append(accuracy.cpu().item())

    segments = ["Model has been fitted",
                f"epoch {epoch}",
                f"accuracy {np.mean(training_accuracy):.3}",
                f"loss {np.mean(training_loss):.3}",
                ]

    nice_print(segments)


def evaluate(model: nn.Module,  evaluation_data: DataLoader, epoch: int or str):

    model.eval()
    with torch.no_grad():
        collect_labels = []
        collect_predictions = []
        for samples, labels in tqdm.tqdm(evaluation_data, leave=False):
            samples = samples.to(DEVICE)
            labels = labels.to(DEVICE)
            output, _, _ = model(samples)
            _, prediction = torch.max(output, dim=1)
            collect_labels.append(labels.cpu().numpy())
            collect_predictions.append(prediction.cpu().numpy())

        all_labels = np.concatenate(collect_labels)
        all_predictions = np.concatenate(collect_predictions)
        f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
        pre = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
        rec = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
        acc = accuracy_score(all_labels, all_predictions)

        segments = ["Model has been evaluated",
                    f"epoch {epoch}",
                    f"accuracy {acc:.3}",
                    f"f1 score {f1:.3}",
                    f"precision {pre:.3}",
                    f"recall {rec:.3}",
                    ]

        nice_print(segments)


def example():

    # select any adapted model
    # model = AdaptedResNet18()
    model = AdaptedVGG16()
    model.to(DEVICE)

    # select any value for m
    m = 2

    # build the loss-function
    criterion = LargeMarginSoftmaxLoss(m=m)

    # load the data
    train_dl, valid_dl, test_dl = \
        load_imbalanced_CIFAR10(batch_size=32, train_validation_split=0.9,
                                imbalance_type="exp", imbalance_factor=0.33)

    # use SGD
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # simple training loop
    num_epochs = 25
    for epoch in range(num_epochs):
        fit(model, criterion, optimizer, train_dl, epoch)
        evaluate(model, valid_dl, epoch)

    # test on never seen data
    print("\n\nFinal evaluation on the Test-Data")
    evaluate(model, test_dl, "test-phase")


if __name__ == '__main__':
    example()
