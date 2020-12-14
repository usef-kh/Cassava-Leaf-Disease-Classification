import warnings

import torch
import torch.nn as nn

from data.data import prepare_data
from models.vgg import Vgg

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

def train(net, dataloader, criterion, optimizer):
    net = net.train()
    loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        '''        
        # fuse crops and batchsize
        bs, ncrops, c, h, w = inputs.shape
        inputs = inputs.view(-1, c, h, w)

        labels = labels.repeat_interleave(ncrops)
        '''

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # calculate performance metrics
        loss_tr += loss.item()

        _, preds = torch.max(outputs.data, 1)
        correct_count += (preds == labels).sum().item()
        n_samples += labels.size(0)

    acc = 100 * correct_count / n_samples
    loss = loss_tr / n_samples

    return acc, loss


def evaluate(net, dataloader, criterion):
    with torch.no_grad():
        net = net.eval()
        loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0

        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            '''
            # fuse crops and batchsize
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)
            '''

            # forward
            outputs = net(inputs)
            
            '''
            # combine results across the crops
            outputs = outputs.view(bs, ncrops, -1)
            outputs = torch.sum(outputs, dim=1) / ncrops
            '''

            loss = criterion(outputs, labels)

            # calculate performance metrics
            loss_tr += loss.item()

            _, preds = torch.max(outputs.data, 1)
            correct_count += (preds == labels).sum().item()
            n_samples += labels.size(0)

        acc = 100 * correct_count / n_samples
        loss = loss_tr / n_samples

    return acc, loss


def run(net):
    # Create dataloaders
    trainloader, valloader = prepare_data()

    net = net.to(device)

    learning_rate = 0.001
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0.0001)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    criterion = nn.CrossEntropyLoss()

    print("Training on", device)
    for epoch in range(300):
        acc_tr, loss_tr = train(net, trainloader, criterion, optimizer)

        acc_v, loss_v = evaluate(net, valloader, criterion)

        # # Update learning rate if plateau
        # scheduler.step(acc_v)

        print('Epoch %2d' % (epoch + 1),
              'Train Accuracy: %2.2f %%' % acc_tr,
              'Val Accuracy: %2.2f %%' % acc_v,
              sep='\t\t')


if __name__ == "__main__":
    # Important parameters

    net = Vgg()
    net.half()
    for layer in net.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()

    run(net)
