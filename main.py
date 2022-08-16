import pickle

import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, PILToTensor
from torch.utils.data import DataLoader, Dataset
from mnist_classifier import DigitClassifier
from watermark_encoder import WatermarkClassifier, WatermarkEncoder
from utils import ImagesDataset, create_mnist_watermarks
from mnist_classifier import main as train_mnist_classifier
from watermark_encoder import main as train_watermark_encoder
from watermark_decoder import main as train_watermark_decoder


def model_accuracy(test_model, data):
    correct = 0
    for i, (ims, lbls) in enumerate(data):
        y = test_model(ims.type(torch.FloatTensor))
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum(predictions == lbls)
    return correct / len(data.dataset.labels)


def mnist_classification_accuracy():
    mnist_classifier = torch.load("trained_models/MNIST_classifier.pt")
    dataset = MNIST("data", train=True)
    n = dataset.data.size(0)
    out = mnist_classifier(dataset.data.view(n, 1, *dataset.data.size()[1:]).type(torch.FloatTensor))
    predicted = torch.argmax(out, dim=1)
    labels = dataset.targets
    correct = torch.sum(predicted == labels)
    accuracy = correct / n
    return accuracy


def watermark_decoder_accuracy(is_detect_watermark=True):
    if is_detect_watermark:
        classifier = torch.load("trained_models/watermark_decoder.pt")
        label_ind = 2
    else:
        classifier = torch.load("trained_models/MNIST_classifier.pt")
        label_ind = 1
    classifier.eval()
    data_path = "data/encoded/train_encoded_data.pickle"
    data = ImagesDataset(data_path, label_ind=label_ind)
    data_loader = DataLoader(data, batch_size=1000)
    return model_accuracy(classifier, data_loader)


def plot_encoded_image(ind=None):
    file_name = "data/encoded/train_encoded_data.pickle"
    with open(file_name, "rb") as f:
        encoded = pickle.load(f)
    if not ind:
        ind = np.random.randint(len(encoded))
    image, label, watermark_label = encoded[ind]
    plt.imshow(image[0, :, :], cmap="gray")
    plt.title(f"Label={label}   Watermark={watermark_label}")
    plt.show()


def encode_watermark(image_ind=None, watermark_ind=None):
    mnist_dataset = MNIST("data", train=True)
    images = mnist_dataset.data
    if not image_ind:
        image_ind = np.random.randint(images.size(0))
    image = images[image_ind, :, :]
    image = image.view(1, 1, *image.size())
    watermarks = pickle.load(open("data/watermarks/train_watermarks.pickle", "rb"))
    if not watermark_ind:
        watermark_ind = np.random.randint(len(watermarks))
    watermark = PILToTensor()(watermarks[watermark_ind][0])

    watermark_classifier = torch.load("trained_models/watermark_classifier.pt")
    watermark_classifier.eval()
    w_vector = watermark_classifier(watermark.type(torch.FloatTensor).view(1, *watermark.size()))

    encoder = torch.load("trained_models/watermark_encoder.pt")
    encoder.eval()
    w = encoder(w_vector, image)

    encoded = image + w
    encoded = torch.maximum(encoded, torch.tensor(0))
    encoded = torch.minimum(encoded, torch.tensor(255))

    return encoded[0, 0, :, :].detach(), image[0, 0, :, :], watermark[0, :, :], image_ind, watermark_ind


def plot_original_and_encoded(image_ind=None, watermark_ind=None):
    encoded, image, watermark, image_ind, watermark_ind = encode_watermark()
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image, cmap="gray")
    axs[1].imshow(encoded, cmap="gray")
    plt.show()


def save_image_set(image_inds=None, watermark_inds=None):
    if not image_inds:
        image_inds = [None for _ in range(5)]
    if not watermark_inds:
        watermark_inds = [None for _ in range(5)]
    for ii, wi in zip(image_inds, watermark_inds):
        encoded, image, watermark, image_ind, watermark_ind = encode_watermark(image_ind=ii, watermark_ind=wi)
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(image, cmap="gray")
        axs[1].imshow(encoded, cmap="gray")
        plt.savefig(f"figures/encoded/encoded_ii={image_ind}")
        plt.close(fig)
        # plot watermark
        plt.imshow(watermark, cmap="gray")
        plt.savefig(f"figures/encoded/watermark{watermark_ind}_ii={image_ind}")
        plt.close()


if __name__ == "__main__":
    # train and save MNIST classifier
    train_mnist_classifier(is_train_classifier=True)
    mnist_accuracy = mnist_classification_accuracy()
    print(f"MNIST classification accuracy = {100*mnist_accuracy:.2f}%")

    # create watermarks
    create_mnist_watermarks(train=True)
    create_mnist_watermarks(train=False)

    # train and save watermark encoder
    train_watermark_encoder(is_train_classifier=True, is_train_encoder=True, is_create_encoded=True)
    encoded_accuracy = watermark_decoder_accuracy(is_detect_watermark=False)
    print(f"encoded images accuracy on MNIST classification = {100*encoded_accuracy:.2f}%")

    # train and save watermark decoder
    train_watermark_decoder(is_train_decoder=True)
    decoder_accuracy = watermark_decoder_accuracy(is_detect_watermark=True)
    print(f"watermark decoder accuracy = {100*decoder_accuracy:.2f}%")

    # plot an example of original and encoded images
    plot_original_and_encoded()




