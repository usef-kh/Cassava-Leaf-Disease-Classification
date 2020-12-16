import os

import cv2
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

data_dir = r"/projectnb/textconv/ykh/cassava/kaggle"
data_dir = r"C:\Users\Yousef\Desktop\Projects\Cassava Leaf Disease Classification/kaggle"


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
        transforms.Resize((300, 400)),
        transforms.RandomCrop(224),

        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        # Sizing
        transforms.Resize((300, 400)),
        transforms.TenCrop(224),

        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])

    train = CustomDataset(xtrain, ytrain, train_transform)
    val = CustomDataset(xval, yval, val_transform)

    '''
    counts = [870, 1751, 1909, 10526, 2061]
    total_count = sum(counts)
    weights = [total_count / c for c in counts]
    # sample_weights = [weights[label] for label in ytrain]
    # sampler = WeightedRandomSampler(sample_weights, len(ytrain))
    '''

    trainloader = DataLoader(train, batch_size=64, shuffle=True, num_workers=2)
    valloader = DataLoader(val, batch_size=64, shuffle=True, num_workers=2)

    return trainloader, valloader


def prepare_folds(k=5):
    train_transform = transforms.Compose([
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
        transforms.Resize((300, 400)),
        transforms.RandomCrop(224),

        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        # Sizing
        transforms.Resize((300, 400)),
        transforms.TenCrop(224),

        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])

    train = pd.read_csv(get_path('train.csv'))

    X = train['image_id'].values
    y = train['label'].values

    skf = StratifiedKFold(n_splits=k)

    loaders = []

    for train_idx, test_idx in skf.split(X, y):
        xtrain, ytrain = X[train_idx], y[train_idx]
        xval, yval = X[test_idx], y[test_idx]

        train = CustomDataset(xtrain, ytrain, train_transform)
        val = CustomDataset(xval, yval, val_transform)

        trainloader = DataLoader(train, batch_size=64, shuffle=True, num_workers=2)
        valloader = DataLoader(val, batch_size=64, shuffle=True, num_workers=2)

        loaders.append((trainloader, valloader))

    return loaders


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
    prepare_folds()
