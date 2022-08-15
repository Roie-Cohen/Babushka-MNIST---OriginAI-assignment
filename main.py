import pickle

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from mnist_classifier import DigitClassifier


class ImagesDataset(Dataset):

    def __init__(self, filepath, transform=None, target_transform=None):
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.images = [d[0] for d in data]
        self.labels = [d[1] for d in data]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].type(torch.FloatTensor)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def model_accuracy(test_model, data):
    correct = 0
    for i, (ims, lbls) in enumerate(data):
        y = test_model(ims.type(torch.FloatTensor))
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum(predictions == lbls)
    return correct / len(data.dataset.labels)


def encoded_classification_accuracy():
    model = torch.load("trained_models/MNIST_classifier.pt")
    data_path = "data/encoded/encoded_data.pickle"
    data = ImagesDataset(data_path)
    data_loader = DataLoader(data, batch_size=1000)
    return model_accuracy(model, data_loader)


if __name__ == "__main__":
    accuracy = encoded_classification_accuracy()
    print(f"Encoded images digit accuracy = {100*accuracy:.2f}%")

