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
from utils import ImagesDataset, classification_accuracy

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

        self.fc2 = nn.Linear(64, 10)

    def forward(self, w):
        w = self.conv1(w)
        w = self.conv2(w)
        embedding = self.fc1(w)
        w = self.fc2(embedding)
        return w, embedding


class WatermarkEncoder(nn.Module):

    def __init__(self):
        super(WatermarkEncoder, self).__init__()

        self.fc1 = nn.Linear(10, 28*28)

        # self.upsample_layers = nn.Sequential(
        #     nn.ConvTranspose2d(1, 16, kernel_size=4),   # 8x8 --> 11x11
        #     nn.Upsample(scale_factor=2),    # 11x11 --> 22x22
        #     nn.ConvTranspose2d(16, 1, kernel_size=7),  # 22x22 --> 28x28
        # )
        #
        self.blend_layers = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, padding=2),
            # nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

        # self.embedding_layers = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=5),    # 28x28 --> 24x24
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),        # --> 12x12
        #     nn.Conv2d(16, 32, kernel_size=3),   # --> 10x10
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),        # --> 5x5
        #     nn.Conv2d(32, 1, kernel_size=1),
        #     nn.Flatten(),
        # )

    def forward(self, w, image):
        # upsample watermark to 28x28
        x = self.fc1(w)
        x = x.view(1, 1, 28, 28)
        # x = self.upsample_layers(w)
        # combine with the image to stamp
        x = torch.cat([x, image], dim=1)
        # create watermark and blend to image
        watermark = self.blend_layers(x)
        # create embedding of the watermark
        # embedding = self.embedding_layers(watermark)
        return watermark


def load_watermarks():
    ws = load_mnist_watermarks()
    watermarks = {i: [] for i in range(10)}
    for w, label in ws:
        watermarks[label.item()].append(w.type(torch.FloatTensor))
    return watermarks


def create_encoded_watermarks(save_folder=ENCODED_FOLDER):
    mnist_data = MNIST(root='data', train=True)
    watermarks = load_mnist_watermarks()
    model = torch.load("trained_models/watermark_encoder.pt")
    model.eval()

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
            stamp = stamp.view(1, *stamp.size())
            stamp_embedding, _ = watermark_classifier(stamp)
            watermark = model(stamp_embedding, canvas)
            enc = canvas + watermark

            enc = torch.maximum(enc, torch.tensor(0))
            enc = torch.minimum(enc, torch.tensor(255))
            enc = enc.type(torch.ByteTensor)
            image_label = mnist_data.targets[ii].item()
            watermark_label = watermarks[wi][1].item()
            encoded.append((enc[0, :, :, :], image_label, watermark_label))

        # # add blank canvases for "None" label
        # image_inds = np.random.permutation(n)[:6000]
        # for ii in tqdm(image_inds):
        #     canvas = mnist_data.data[ii].type(torch.ByteTensor)
        #     canvas = canvas.view(1, *canvas.size())
        #     image_label = mnist_data.targets[ii].item()
        #     encoded.append((canvas, image_label, 11))

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    save_path = os.path.join(save_folder, "encoded_data.pickle")
    with open(save_path, "wb") as f:
        pickle.dump(encoded, f)
    return encoded


# return a tuple of watermarks representing an anchor, positive and negative
def get_triplet(watermarks_dict):
    l1, l2 = np.random.permutation(10)[:2]
    anchor_ind = np.random.randint(len(watermarks_dict[l1]))
    positive_ind = -1
    while positive_ind < 0 or positive_ind == anchor_ind:
        positive_ind = np.random.randint(len(watermarks_dict[l1]))
    negative_ind = np.random.randint(len(watermarks_dict[l2]))

    anchor = watermarks_dict[l1][anchor_ind]
    positive = watermarks_dict[l1][positive_ind]
    negative = watermarks_dict[l2][negative_ind]
    return anchor, positive, negative


def invisible_watermark_loss(watermark, image, margin=0.1):
    # loss = torch.divide(torch.abs(watermark), image)
    # loss = torch.add(loss, 1)
    # loss = torch.log(loss)
    # loss = torch.subtract(loss, margin)
    # loss = torch.mean(torch.maximum(loss, torch.tensor(0)))
    # loss = torch.maximum(watermark, -image)     # encoded image values are non negative
    # loss = torch.minimum(loss, 255 - image)    # encoded image values up to 255
    loss1 = torch.mean(torch.maximum(1 - torch.abs(watermark), torch.tensor(0)))
    loss2 = torch.mean(torch.maximum(torch.abs(watermark) - 3, torch.tensor(0)))  # TODO: fix
    loss = loss1 + loss2
    return loss


def plot_watermarks(watermarks):
    n = len(watermarks)
    fig, ax = plt.subplots(1, n)
    for i, watermark in enumerate(watermarks):
        ax[i].imshow(watermark.detach())


def train_watermark_classifier():

    learning_rate = 1E-4
    batch_size = 128
    epochs = 20
    torch.manual_seed(100)

    # load data
    train_data = ImagesDataset("data/watermarks/train_watermarks.pickle")
    test_data = ImagesDataset("data/watermarks/test_watermarks.pickle")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    loss_func = nn.CrossEntropyLoss()

    model = WatermarkClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train the classifier
    for epoch in range(epochs):
        model.train()
        for b_i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            out, _ = model(images)
            loss = loss_func(out, labels)
            loss.backward()
            optimizer.step()

            if b_i % 100 == 0:
                print(f'epoch={epoch+1}/{epochs}   loss={loss.item()}    accuracy={100*classification_accuracy(model, test_loader)}%')

    return model


def train():
    watermarks = load_watermarks()

    learning_rate = 1E-4
    steps = 1000
    distance_margin = 1

    # load data
    mnist_data = MNIST(root='data', train=True, download=True, transform=ToTensor())
    n_images = mnist_data.data.size(0)

    loss_distance = nn.TripletMarginLoss(margin=distance_margin)

    watermark_classifier = torch.load("trained_models/watermark_classifier.pt")
    watermark_classifier.eval()

    model = WatermarkEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train the classifier
    for step in range(steps):
        model.train()
        optimizer.zero_grad()

        watermark = watermarks[np.random.randint(10)][np.random.randint(1000)]
        label_vector, _ = watermark_classifier(watermark.view(1, *watermark.size()))

        canvas = mnist_data.data[np.random.randint(n_images)]
        canvas = canvas.view(1, 1, *canvas.size())

        w = model(label_vector, canvas)

        # loss = loss_distance(em_a, em_p, em_n)
        loss = invisible_watermark_loss(w, canvas)
        # loss = 0
        # for w in (w_a, w_p, w_n):
        #     loss = loss + invisible_watermark_loss(w, canvas)

        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f'step={step+1}/{steps}   loss={loss.item()}   S={w.abs().mean().item()}')

    return model


if __name__ == "__main__":
    model = train()
    torch.save(model, "trained_models/watermark_encoder.pt")
    create_encoded_watermarks()
    print('done')
