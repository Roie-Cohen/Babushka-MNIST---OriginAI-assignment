import pickle

import torch
import torch.nn as nn
import torchvision.transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset


class WatermarkDecoder(nn.Module):

    def __init__(self):
        super(WatermarkDecoder, self).__init__()
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
        # output linear layer: 400 --> 11
        self.out_layer = nn.Linear(400, 11)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.out_layer(x)
        return x


class EncodedImagesDataset(Dataset):

    FILE_PATH = "data/encoded/encoded_data.pickle"

    def __init__(self, transform=None, target_transform=None):
        with open(self.FILE_PATH, "rb") as f:
            encoded = pickle.load(f)

        self.images = [enc[0] for enc in encoded]
        self.img_labels = [(enc[1], enc[2]) for enc in encoded]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.images[idx].type(torch.FloatTensor)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def model_accuracy(test_model, data):
    correct = 0
    for i, (ims, lbls) in enumerate(data):
        y = test_model(ims)
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum(predictions == lbls)
    return correct/len(data.dataset.img_labels)


def train():

    learning_rate = 1E-3
    batch_size = 128
    epochs = 10
    torch.manual_seed(100)
    test_size = 0.1

    # load data
    data = EncodedImagesDataset(target_transform=torchvision.transforms.Lambda(lambda x: x[1]))
    train_data = data
    test_data = data
    # train_samples = int(len(data.images) * (1-test_size))
    # train_data = [im.type(torch.FloatTensor) for im in data.images[:train_samples]]
    # test_data = [im.type(torch.FloatTensor) for im in data.images[train_samples:]]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    loss_func = nn.CrossEntropyLoss()

    model = WatermarkDecoder()
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
                print(f'epoch={epoch+1}/{epochs}   loss={loss.item()}    accuracy={100*model_accuracy(model, train_loader)}%')

    return model


if __name__ == "__main__":
    model = train()
    torch.save(model, "trained_models/watermark_decoder.pt")

