import os.path
import torch
import numpy as np
from torchvision.datasets import MNIST
from torchvision.transforms import ToPILImage, PILToTensor
import pickle
import matplotlib.pyplot as plt


WATERMARK_FOLDER = "data/watermarks"
ENCODED_FOLDER = "data/encoded"
MODEL_FOLDER = "trained_models"


# saves a list of tuples of type (watermark 7x7, corresponding label)
def create_mnist_watermarks(save_folder=WATERMARK_FOLDER):
    transform = ToPILImage()
    data = MNIST(root='data', train=True)
    images = [transform(image).resize((7, 7)) for image in data.data]
    labels = [label.item() for label in data.targets]
    watermarks = [(image, label) for image, label in zip(images, labels)]
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    save_path = os.path.join(save_folder, "train_watermarks.pickle")
    with open(save_path, "wb") as f:
        pickle.dump(watermarks, f)


# loads the watermarks and convert to tensor format
def load_mnist_watermarks(load_folder=WATERMARK_FOLDER):
    file_path = os.path.join(load_folder, "train_watermarks.pickle")
    with open(file_path, "rb") as f:
        watermarks = pickle.load(f)
    watermarks = [(PILToTensor()(image), torch.tensor(label)) for image, label in watermarks]
    return watermarks


def create_encoded_watermarks(save_folder=ENCODED_FOLDER):
    mnist_data = MNIST(root='data', train=True)
    watermarks = load_mnist_watermarks()
    model = torch.load("trained_models/watermark_encoder.pt")
    model.eval()

    n = len(watermarks)
    image_inds = np.random.permutation(n)
    watermark_inds = np.random.permutation(n)

    encoded = []
    for ii, wi in zip(image_inds, watermark_inds):
        canvas = mnist_data.data[ii]
        stamp = watermarks[wi][0]
        watermark, _ = model(stamp, canvas)
        enc = canvas + watermark
        encoded.append((enc, mnist_data.targets[ii], watermarks[wi][1]))

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    save_path = os.path.join(save_folder, "encoded_data.pickle")
    with open(save_path, "wb") as f:
        pickle.dump(encoded, f)


if __name__ == "__main__":
    create_encoded_watermarks()
