import os.path
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToPILImage, PILToTensor
import pickle
from PIL.Image import Image
import matplotlib.pyplot as plt


WATERMARK_FOLDER = "data/watermarks"
ENCODED_FOLDER = "data/encoded"
MODEL_FOLDER = "trained_models"


class ImagesDataset(Dataset):

    def __init__(self, data_path, transform=None, target_transform=None, label_ind=1):
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.images = [d[0] for d in data]
        if type(self.images[0]) is Image:
            self.images = [PILToTensor()(image) for image in self.images]

        self.labels = [d[label_ind] for d in data]
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


# saves a list of tuples of type (watermark 7x7, corresponding label)
def create_mnist_watermarks(save_folder=WATERMARK_FOLDER, train=True):
    transform = ToPILImage()
    data = MNIST(root='data', train=train)
    images = [transform(image).resize((7, 7)) for image in data.data]
    labels = [label.item() for label in data.targets]
    watermarks = [(image, label) for image, label in zip(images, labels)]
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    filename = "train_watermarks" if train else "test_watermarks"
    save_path = os.path.join(save_folder, f"{filename}.pickle")
    with open(save_path, "wb") as f:
        pickle.dump(watermarks, f)


# loads the watermarks and convert to tensor format
def load_mnist_watermarks(load_folder=WATERMARK_FOLDER, train=True):
    file_name = "train_watermarks" if train else "test_watermarks"
    file_path = os.path.join(load_folder, f"{file_name}.pickle")
    with open(file_path, "rb") as f:
        watermarks = pickle.load(f)
    watermarks = [(PILToTensor()(image).view(1, *image.size), torch.tensor(label)) for image, label in watermarks]
    return watermarks


def classification_accuracy(test_model, data):
    correct = 0
    for i, (ims, lbls) in enumerate(data):
        y = test_model(ims)
        if type(y) is tuple:
            y = y[0]
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum(predictions == lbls)
    return correct/len(data.dataset.labels)


def plot_loss_graph(losses, save_path=None):
    t = np.linspace(0, 1, len(losses))
    plt.plot(t, losses, linewidth=2)
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    create_mnist_watermarks(train=False)
