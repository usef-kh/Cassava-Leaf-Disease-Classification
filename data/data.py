import os

import cv2
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

data_dir = r"/projectnb/textconv/ykh/cassava/kaggle"


class CustomDataset(Dataset):
    def __init__(self, names, labels, transform=None, train=True):
        self.image_names = names
        self.labels = labels
        self.transform = transform

        if train:
            self.image_dir = get_path('train_images')
        else:
            self.image_dir = get_path('test_images')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.image_dir, self.image_names[idx])

        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        img = cv2.imread(f'{img_path}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img).half()

        label = torch.tensor(self.labels[idx]).type(torch.long)

        return img, label


def get_path(file):
    return os.path.join(data_dir, file)


def split_data(data: pd.DataFrame()):
    X = data['image_id'].values
    y = data['label'].values

    return train_test_split(X, y, stratify=y, train_size=0.8, random_state=42, shuffle=True)


def prepare_data():
    train = pd.read_csv(get_path('train.csv'))

    xtrain, xval, ytrain, yval = split_data(train)

    # stats obtained from print_stats()
    mu = [0.4314, 0.4973, 0.3140]
    st = [0.2015, 0.2067, 0.1825]

    train_transform = transforms.Compose([
        transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
        transforms.RandomResizedCrop((208, 277)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mu, std=st),
        # transforms.FiveCrop((208, 277)),
        # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((208, 277)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mu, std=st),
        # transforms.TenCrop((208, 277)),
        # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])

    train = CustomDataset(xtrain, ytrain, train_transform)
    val = CustomDataset(xval, yval, val_transform)

    trainloader = DataLoader(train, batch_size=64, shuffle=True, num_workers=2)
    valloader = DataLoader(val, batch_size=64, shuffle=True, num_workers=2)

    return trainloader, valloader


def print_stats():
    train = pd.read_csv(get_path('train.csv'))

    xtrain, xval, ytrain, yval = split_data(train)

    transform = transforms.Compose([
        transforms.Resize((208, 277)),
        transforms.ToTensor(),
    ])

    train = CustomDataset(xtrain, ytrain, transform)
    trainloader = DataLoader(train, batch_size=1024, shuffle=False)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, labels in trainloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print(mean, std)


if __name__ == "__main__":
    print_stats()
