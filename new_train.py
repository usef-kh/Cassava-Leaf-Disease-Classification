import sys
import warnings

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from data.data import prepare_folds
from utils.checkpoint import save
from utils.setup import setup_network, setup_hparams

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(net, dataloader, criterion, optimizer, scaler, epoch):
    net = net.train()
    loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0

    # setup progress bar
    # pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    for step, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        with autocast():
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            scaler.scale(loss).backward()

            # calculate performance metrics
            loss_tr += loss.item()

            _, preds = torch.max(outputs.data, 1)
            correct_count += (preds == labels).sum().item()
            n_samples += labels.size(0)

            if ((step + 1) % 2 == 0) or ((step + 1) == len(dataloader)):
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # if ((step + 1) % 5 == 0) or ((step + 1) == len(dataloader)):
            #     description = f'epoch {epoch} loss: {loss_tr / n_samples:.10f}'
            #     pbar.set_description(description)

    acc = 100 * correct_count / n_samples
    loss = loss_tr / n_samples

    return acc, loss


def evaluate(net, dataloader, criterion, epoch):
    with torch.no_grad():
        net = net.eval()
        loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0

        # setup progress bar
        # pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # calculate performance metrics
            loss_tr += loss.item()

            _, preds = torch.max(outputs.data, 1)
            correct_count += (preds == labels).sum().item()
            n_samples += labels.size(0)

            # if ((step + 1) % 5 == 0) or ((step + 1) == len(dataloader)):
            #     description = f'epoch {epoch} loss: {loss_tr / n_samples:.10f}'
            #     pbar.set_description(description)

        acc = 100 * correct_count / n_samples
        loss = loss_tr / n_samples

    return acc, loss


def run(net):
    folds = prepare_folds()

    trainloader, valloader = folds[int(hps['fold_id'])]

    net = net.to(device)

    scaler = GradScaler()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)
    criterion = nn.CrossEntropyLoss().to(device)
    best_acc_v = 0

    print("Training", hps['name'], hps['fold_id'], "on", device)
    for epoch in range(hps['n_epochs']):

        acc_tr, loss_tr = train(net, trainloader, criterion, optimizer, scaler, epoch + 1)
        acc_v, loss_v = evaluate(net, valloader, criterion, epoch + 1)

        # Update learning rate if plateau
        scheduler.step()

        # Save the best network and print results
        if acc_v > best_acc_v:
            save(net, hps, desc=hps['fold_id'])
            best_acc_v = acc_v

            print('Epoch %2d' % (epoch + 1),
                  'Train Accuracy: %2.2f %%' % acc_tr,
                  'Train Loss: %2.6f' % loss_tr,
                  'Val Accuracy: %2.2f %%' % acc_v,
                  'Val Loss: %2.6f' % loss_v,
                  'Network Saved',
                  sep='\t\t')

        else:
            print('Epoch %2d' % (epoch + 1),
                  'Train Accuracy: %2.2f %%' % acc_tr,
                  'Train Loss: %2.6f' % loss_tr,
                  'Val Accuracy: %2.2f %%' % acc_v,
                  'Val Loss: %2.6f' % loss_v,
                  sep='\t\t')


if __name__ == "__main__":
    # Important parameters
    hps = setup_hparams(sys.argv[1:])
    net = setup_network(hps)

    if hps['fold_id'] is None:
        raise RuntimeError("Please select which fold to train")

    elif hps['fold_id'] not in {'1', '2', '3', '4', '0'}:
        raise RuntimeError("Please select a valid fold_id")

    # Convert to fp16 for faster training
    net.half()
    for layer in net.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()

    run(net)
