import pickle

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from mnist_classifier import DigitClassifier
from utils import ImagesDataset


def model_accuracy(test_model, data):
    correct = 0
    for i, (ims, lbls) in enumerate(data):
        y = test_model(ims.type(torch.FloatTensor))
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum(predictions == lbls)
    return correct / len(data.dataset.labels)


def encoded_classification_accuracy():
    mnist_classifier = torch.load("trained_models/MNIST_classifier.pt")
    data_path = "data/encoded/train_encoded_data.pickle"
    data = ImagesDataset(data_path)
    data_loader = DataLoader(data, batch_size=1000)
    return model_accuracy(mnist_classifier, data_loader)


if __name__ == "__main__":
    accuracy = encoded_classification_accuracy()
    print(f"Encoded images digit accuracy = {100*accuracy:.2f}%")

