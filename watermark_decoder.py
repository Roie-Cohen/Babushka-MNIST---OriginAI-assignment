import copy
import pickle

import torch
import torch.nn as nn
import torchvision.transforms
from numpy import ceil
from torch.utils.data import DataLoader, Dataset
from utils import plot_loss_graph


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

    def __init__(self, data_path, transform=None, target_transform=None):
        with open(data_path, "rb") as f:
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


def train_watermark_decoder():
    torch.manual_seed(100)

    learning_rate = 1E-3
    batch_size = 128
    epochs = 15

    # load data
    train_dataset = EncodedImagesDataset("data/encoded/train_encoded_data.pickle", target_transform=torchvision.transforms.Lambda(lambda x: x[1]))
    test_dataset = EncodedImagesDataset("data/encoded/test_encoded_data.pickle", target_transform=torchvision.transforms.Lambda(lambda x: x[1]))
    batches = ceil(len(train_dataset.images) / batch_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    loss_func = nn.CrossEntropyLoss()

    model = WatermarkDecoder()
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
                accuracy = 100 * model_accuracy(model, test_loader)
                t = (b_i + 1) / (batches * epochs) + epoch / epochs
                print(f'process={100 * t:.2f}  loss={loss.item():.3f}  accuracy={accuracy:.2f}%')

            if t > 0.5 and loss < min_loss:
                min_loss = loss
                best_model = copy.deepcopy(model)

    return best_model, losses


def main(is_train_decoder=False):
    if is_train_decoder:
        decoder, losses = train_watermark_decoder()
        plot_loss_graph(losses, "figures/watermark_decoder_loss")
        torch.save(decoder, "trained_models/watermark_decoder.pt")


if __name__ == "__main__":
    main(is_train_decoder=True)
