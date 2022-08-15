import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


class DigitClassifier(nn.Module):

    def __init__(self):
        super(DigitClassifier, self).__init__()
        # convolutional later 1: 28x28 --> 12x12 (x16)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # convolutional later 1: 12x12 (x16) --> 5x5 (x16)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # output linear layer: 400 --> 10
        self.out_layer = nn.Linear(400, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.out_layer(x)
        return x


def model_accuracy(test_model, data):
    correct = 0
    for i, (ims, lbls) in enumerate(data):
        y = test_model(ims)
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum(predictions == lbls)
    return correct/data.dataset.data.size(0)


def train():

    learning_rate = 1E-3
    batch_size = 128
    epochs = 10
    torch.manual_seed(100)

    # load data
    train_data = MNIST(root='data', train=True, download=True, transform=ToTensor())
    test_data = MNIST(root='data', train=False, download=True, transform=ToTensor())

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    loss_func = nn.CrossEntropyLoss()

    model = DigitClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train the classifier
    for epoch in range(epochs):
        model.train()
        for b_i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(images)
            loss = loss_func(out, labels)
            loss.backward()
            optimizer.step()

            if b_i % 100 == 0:
                print(f'epoch={epoch+1}/{epochs}   loss={loss.item()}    accuracy={100*model_accuracy(model, test_loader)}%')

    return model


if __name__ == "__main__":
    model = train()
    torch.save(model, "trained_models/MNIST_classifier.pt")

