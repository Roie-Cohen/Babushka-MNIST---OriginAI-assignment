import copy
import os
import pickle

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from numpy import ceil
from utils import ImagesDataset, classification_accuracy, plot_loss_graph

from utils import load_mnist_watermarks

ENCODED_FOLDER = "data/encoded"


class WatermarkClassifier(nn.Module):
    def __init__(self):
        super(WatermarkClassifier, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),    # 7x7 --> 5x5
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),  # 5x5 --> 3x3
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(144, 64),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(32, 10)

    def forward(self, w):
        w = self.conv1(w)
        w = self.conv2(w)
        w = self.fc1(w)
        w = self.fc2(w)
        w = self.fc3(w)
        return w


class WatermarkEncoder(nn.Module):

    def __init__(self):
        super(WatermarkEncoder, self).__init__()

        self.fc1 = nn.Linear(10, 28*28)

        self.blend_layers = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, padding=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, w, image):
        x = self.fc1(w)
        x = x.view(1, 1, 28, 28)
        x = torch.cat([x, image], dim=1)
        watermark = self.blend_layers(x)
        return watermark


def create_encoded_watermarks(save_folder=ENCODED_FOLDER, train=True):
    mnist_data = MNIST(root='data', train=train)
    watermarks = load_mnist_watermarks(train=train)
    watermark_encoder = torch.load("trained_models/watermark_encoder.pt")
    watermark_encoder.eval()

    watermark_classifier = torch.load("trained_models/watermark_classifier.pt")
    watermark_classifier.eval()

    n = len(watermarks)
    image_inds = np.random.permutation(n)
    watermark_inds = np.random.permutation(n)

    encoded = []
    with torch.no_grad():
        for ii, wi in tqdm(zip(image_inds, watermark_inds)):
            canvas = mnist_data.data[ii].type(torch.FloatTensor)
            canvas = canvas.view(1, 1, *canvas.size())
            stamp = watermarks[wi][0].type(torch.FloatTensor)
            stamp_embedding = watermark_classifier(stamp.view(1, *stamp.size()))
            watermark = watermark_encoder(stamp_embedding, canvas)
            enc = canvas + watermark

            enc = torch.maximum(enc, torch.tensor(0))
            enc = torch.minimum(enc, torch.tensor(255))
            enc = enc.type(torch.ByteTensor)
            image_label = mnist_data.targets[ii].item()
            watermark_label = watermarks[wi][1].item()
            encoded.append((enc[0, :, :, :], image_label, watermark_label))

        # add blank canvases for "None" label (label = 10)
        image_inds = np.random.permutation(n)[:int(n/10)]
        for ii in tqdm(image_inds):
            canvas = mnist_data.data[ii].type(torch.ByteTensor)
            canvas = canvas.view(1, *canvas.size())
            canvas = copy.deepcopy(torch.tensor(canvas.numpy()))    # necessary unexplainable black magic line...
            image_label = mnist_data.targets[ii].item()
            encoded.append((canvas, image_label, 10))

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    file_name = "train_encoded_data" if train else "test_encoded_data"
    save_path = os.path.join(save_folder, f"{file_name}.pickle")
    with open(save_path, "wb") as f:
        pickle.dump(encoded, f)
    return encoded


def train_watermark_classifier():

    learning_rate = 1E-4
    batch_size = 128
    epochs = 20
    torch.manual_seed(100)

    # load data
    train_dataset = ImagesDataset("data/watermarks/train_watermarks.pickle")
    test_dataset = ImagesDataset("data/watermarks/test_watermarks.pickle")
    batches = ceil(len(train_dataset.images) / batch_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    loss_func = nn.CrossEntropyLoss()

    model = WatermarkClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    min_loss = torch.inf
    best_model = copy.deepcopy(model)
    t = 0
    losses = []
    # train the classifier
    model.train()
    for epoch in range(epochs):
        for b_i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(images)
            loss = loss_func(out, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if b_i % 100 == 0:
                accuracy = 100 * classification_accuracy(model, test_loader)
                t = (b_i + 1) / (batches * epochs) + epoch / epochs
                print(f'process={100 * t:.2f}  loss={loss.item():.3f}  accuracy={accuracy:.2f}%')

            if t > 0.5 and loss < min_loss:
                min_loss = loss
                best_model = copy.deepcopy(model)

    return best_model, losses


def invisible_watermark_loss(watermark, min_pix_val=1, max_pix_val=3):
    loss1 = torch.mean(torch.maximum(min_pix_val - torch.abs(watermark), torch.tensor(0)))
    loss2 = torch.mean(torch.maximum(torch.abs(watermark) - max_pix_val, torch.tensor(0)))
    loss = loss1 + loss2
    return loss


def train_watermark_encoder():
    torch.manual_seed(100)

    watermarks = load_mnist_watermarks()

    learning_rate = 1E-4
    steps = 1000

    # load data
    mnist_data = MNIST(root='data', train=True, download=True, transform=ToTensor())
    n_images = mnist_data.data.size(0)

    watermark_classifier = torch.load("trained_models/watermark_classifier.pt")
    watermark_classifier.eval()

    loss_func = invisible_watermark_loss

    model = WatermarkEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    min_loss = torch.inf
    best_model = copy.deepcopy(model)
    losses = []
    # train the classifier
    model.train()
    for step in range(steps):
        optimizer.zero_grad()

        watermark = watermarks[np.random.randint(len(watermarks))][0].type(torch.FloatTensor)
        label_vector = watermark_classifier(watermark.view(1, *watermark.size()))

        canvas = mnist_data.data[np.random.randint(n_images)]
        canvas = canvas.view(1, 1, *canvas.size())

        w = model(label_vector, canvas)

        loss = loss_func(w)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        t = step / steps
        if step % 10 == 0:
            print(f'process={100 * t:.2f}  loss={loss.item():.3f}  w_pixel={w.abs().median().item():.2f}')

        if t > 0.5 and loss < min_loss:
            min_loss = loss
            best_model = copy.deepcopy(model)

    return best_model, losses


def main(is_train_encoder=False, is_train_classifier=False, is_create_encoded=False):
    if is_train_classifier:
        classifier, losses = train_watermark_classifier()
        torch.save(classifier, "trained_models/watermark_classifier.pt")
        plot_loss_graph(losses, save_path="figures/watermark_classifier_loss")
    if is_train_encoder:
        encoder, losses = train_watermark_encoder()
        torch.save(encoder, "trained_models/watermark_encoder.pt")
        plot_loss_graph(losses, save_path="figures/watermark_encoder_loss")
    if is_create_encoded:
        create_encoded_watermarks(train=True)
        create_encoded_watermarks(train=False)


if __name__ == "__main__":
    main(is_train_classifier=False, is_train_encoder=False, is_create_encoded=True)
    print('done')
