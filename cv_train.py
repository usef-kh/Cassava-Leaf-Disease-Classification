import sys
import warnings

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.data import prepare_folds
from loops import train, evaluate
from utils.checkpoint import save
from utils.setup import setup_network, setup_hparams

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run(net):
    folds = prepare_folds()

    trainloader, valloader = folds[int(hps['fold_id'])]

    net = net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=hps['lr'], momentum=0.9, nesterov=True, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    criterion = nn.CrossEntropyLoss()

    best_acc_v = 0

    print("Training", hps['name'], hps['fold_id'], "on", device)
    for epoch in range(hps['n_epochs']):

        acc_tr, loss_tr = train(net, trainloader, criterion, optimizer)
        acc_v, loss_v = evaluate(net, valloader, criterion)

        # Update learning rate if plateau
        scheduler.step(acc_v)

        # Save the best network and print results
        if acc_v > best_acc_v:
            save(net, hps, desc=hps['fold_id'])
            best_acc_v = acc_v

            print('Epoch %2d' % (epoch + 1),
                  'Train Accuracy: %2.2f %%' % acc_tr,
                  'Val Accuracy: %2.2f %%' % acc_v,
                  'Network Saved',
                  sep='\t\t')

        else:
            print('Epoch %2d' % (epoch + 1),
                  'Train Accuracy: %2.2f %%' % acc_tr,
                  'Val Accuracy: %2.2f %%' % acc_v,
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
