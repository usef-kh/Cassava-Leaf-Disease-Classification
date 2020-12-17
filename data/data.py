import os

import cv2
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

data_dir = r"/projectnb/textconv/ykh/cassava/kaggle/"
data_dir = r"C:\Users\Yousef\Desktop\Projects\Cassava Leaf Disease Classification/kaggle/"

# stats obtained from print_stats()
mu = [0.4314, 0.4977, 0.3149]
st = [0.2061, 0.2110, 0.1873]

Transforms = {
    'train': transforms.Compose([
        # Sheer, Rotation, Translation
        transforms.RandomApply([transforms.RandomRotation(360)], p=0.5),
        transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.5),
        transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=10)], p=0.5),

        # Flips
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),

        # Color
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3)], p=0.4),
        transforms.RandomApply([transforms.ColorJitter(contrast=0.2)], p=0.4),
        transforms.RandomApply([transforms.ColorJitter(saturation=0.3)], p=0.4),

        # Sizing
        transforms.Resize(600),
        transforms.RandomCrop(512),

        transforms.ToTensor(),
    ]),

    'val': transforms.Compose([
        # Sizing
        transforms.Resize(600),
        transforms.RandomCrop(512),

        transforms.ToTensor(),
    ])
}


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]

        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        img = cv2.imread(f'{img_path}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img).half()

        label = torch.tensor(self.labels[idx]).type(torch.long)

        return img, label


def get_path(file, merged=True):
    if merged:
        return os.path.join(data_dir, 'merged_data', file)

    return os.path.join(data_dir, file)


def split_data(data):
    X, y = data
    return train_test_split(X, y, stratify=y, train_size=0.8, random_state=42, shuffle=True)


def prepare_paths(data: pd.DataFrame(), merged=True):
    imgs = data['image_id'].values
    y = data['label'].values.tolist()

    if merged:
        img_dir = os.path.join(data_dir, 'merged_data', 'train_images')

    else:
        img_dir = os.path.join(data_dir, 'train_images')

    X = []
    for img in imgs:
        full_path = os.path.join(img_dir, img)
        X.append(full_path)

    return X, y


def prepare_data(merged=True):
    df_train = pd.read_csv(get_path('train.csv', merged=merged))
    train = prepare_paths(df_train, merged=merged)

    xtrain, xval, ytrain, yval = split_data(train)

    train = CustomDataset(xtrain, ytrain, Transforms['train'])
    val = CustomDataset(xval, yval, Transforms['val'])

    trainloader = DataLoader(train, batch_size=64, shuffle=True, num_workers=2)
    valloader = DataLoader(val, batch_size=32, shuffle=True, num_workers=2)

    return trainloader, valloader


def prepare_folds(k=5, merged=True):
    df_train = pd.read_csv(get_path('train.csv', merged=merged))
    train = prepare_paths(df_train, merged=merged)

    X, y = train

    skf = StratifiedKFold(n_splits=k, random_state=42)

    loaders = []

    for train_idx, test_idx in skf.split(X, y):
        xtrain, ytrain = X[train_idx], y[train_idx]
        xval, yval = X[test_idx], y[test_idx]

        train = CustomDataset(xtrain, ytrain, Transforms['train'])
        val = CustomDataset(xval, yval, Transforms['val'])

        trainloader = DataLoader(train, batch_size=64, shuffle=True, num_workers=2)
        valloader = DataLoader(val, batch_size=32, shuffle=True, num_workers=2)

        loaders.append((trainloader, valloader))

    return loaders


def print_stats(merged=True):
    '''
    This function will iterate through the train loader and print the mean and std of the RGB Channels
    :return:
    '''

    df_train = pd.read_csv(get_path('train.csv', merged=merged))
    train = prepare_paths(df_train, merged=merged)

    xtrain, xval, ytrain, yval = split_data(train)

    transform = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.RandomCrop(512),
        transforms.ToTensor(),
    ])

    train = CustomDataset(xtrain, ytrain, transform)
    trainloader = DataLoader(train, batch_size=5, shuffle=False)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, labels in trainloader:
        print(data.shape)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        data = data.float()

        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print(mean, std)


if __name__ == "__main__":
    print_stats()
