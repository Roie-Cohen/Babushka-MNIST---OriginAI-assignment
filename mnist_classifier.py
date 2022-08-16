import copy

import torch
import torch.nn as nn
from numpy import ceil
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from utils import plot_loss_graph


class DigitClassifier(nn.Module):

    def __init__(self):
        super(DigitClassifier, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1),  # 28x28 (x1) --> 24x24 (x16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # --> 12x12 (x16)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1),  # --> 10x10 (x16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # --> 5x5 (x16)
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),   # 5x5 (x16) --> 400
            nn.Linear(400, 10)  # --> 10
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        return x


def model_accuracy(test_model, data):
    correct = 0
    for i, (ims, lbls) in enumerate(data):
        y = test_model(ims)
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum(predictions == lbls)
    return correct/data.dataset.data.size(0)


def train():
    torch.manual_seed(100)

    learning_rate = 1E-3
    batch_size = 128
    epochs = 5

    train_dataset = MNIST(root='data', train=True, download=True, transform=ToTensor())
    test_dataset = MNIST(root='data', train=False, download=True, transform=ToTensor())
    batches = ceil(train_dataset.data.size(0) / batch_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    loss_func = nn.CrossEntropyLoss()

    model = DigitClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    min_loss = torch.inf
    best_model = copy.deepcopy(model)
    t = 0
    losses = []
    # train the classifier
    for epoch in range(epochs):
        model.train()
        for b_i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(images)
            loss = loss_func(out, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if b_i % 100 == 0:
                accuracy = 100*model_accuracy(model, test_loader)
                t = (b_i+1)/(batches*epochs) + epoch/epochs
                print(f'process={100*t:.2f}  loss={loss.item():.3f}  accuracy={accuracy:.2f}%')

            if t > 0.5 and loss < min_loss:
                min_loss = loss
                best_model = copy.deepcopy(model)

    return best_model, losses


if __name__ == "__main__":
    classifier, losses = train()
    plot_loss_graph(losses, save_path="figures/MNIST_classifier_loss")
    torch.save(classifier, "trained_models/MNIST_classifier.pt")

