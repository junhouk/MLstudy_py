import os

import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
import numpy as np
import math


MNIST_classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
CIFAR10_classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

learning_rate = 1e-2
total_epoch =300
print_steps = 500


class Dataset(object):
    def __init__(self, dataset_name, data_root="./", batch_size=1):
        assert dataset_name in ("mnist", "cifar10")
        if dataset_name == "mnist":
            self.dataset = MNIST
        else:
            self.dataset = CIFAR10

        transformer = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        trainset = self.dataset(root=data_root, train=True, download=True, transform=transformer)
        testset = self.dataset(root=data_root, train=False, download=True, transform=transformer)

        self.train = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.test = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True)


def main(dataset_name, data_root, batch_size):
    dataset_name = dataset_name.lower()
    is_mnist = dataset_name == "mnist"
    classes = MNIST_classes if is_mnist else CIFAR10_classes
    class_num = len(classes)

    print("Download dataset...")
    print(dataset_name, data_root, batch_size)
    data_loader = Dataset(dataset_name, data_root=data_root, batch_size=batch_size)
    train_set, test_set = data_loader.train, data_loader.test
    num_trainset, num_testset = len(train_set), len(test_set)

    in_planes = 28 * 28 if is_mnist else 32 * 32
    hidden_num1 = math.floor(512)
	hidden_num2 = math.floor(256)
	hidden_num3 = math.floor(128)
	hidden_num4 = math.floor((128+class_num) / 2)
	
    print("hidden layer node num : ", hidden_num)
	#stack layer
    mlp = nn.Sequential(
        nn.Linear(in_planes, hidden_num1),
        nn.ReLU(),
        nn.Linear(hidden_num1, hidden_num2),
        nn.ReLU(),
        nn.Linear(hidden_num2, hidden_num3),
        nn.ReLU(),
        nn.Linear(hidden_num3, hidden_num4),
        nn.ReLU(),
        nn.Linear(hidden_num4, class_num),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=learning_rate)

    print("Start training...")
    for epoch in range(total_epoch):
        total_loss = 0.0
        num_data = 0
        for i, data in enumerate(train_set):
            inputs, labels = data
            inputs = inputs.reshape(-1, in_planes)
            outputs = mlp(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()x

            total_loss += loss.item()
            num_data += batch_size
            if i % print_steps == 0 and (i != 0 or i != num_trainset - 1):
                print(f"Epoch: {epoch}, {i}/{num_trainset}. Average loss: {total_loss / num_data: .3f}")
        with torch.no_grad():
            accuracytmp = np.zeros(num_testset, )

            for j, data in enumerate(test_set):
                testin, testlabel = data
                testin = testin.reshape(-1, in_planes)
                prediction = mlp(testin)
                correct_prediction = torch.argmax(prediction, 1) == testlabel
                accuracytmp[j, ] = correct_prediction.float().mean()
            accuracyfin = accuracytmp.mean()
            print("Accuracy: ",epoch, " : ", accuracyfin.item())
    print("end training")




#main("mnist", "./dataset", batch_size=32)
main("cifar10", "./dataset", batch_size=32)