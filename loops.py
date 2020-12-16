import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(net, dataloader, criterion, optimizer):
    net = net.train()
    loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

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


def train_nc(net, dataloader, criterion, optimizer):
    net = net.train()
    loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # fuse crops and batchsize
        bs, ncrops, c, h, w = inputs.shape
        inputs = inputs.view(-1, c, h, w)

        labels = labels.repeat_interleave(ncrops)

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

            # forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # calculate performance metrics
            loss_tr += loss.item()

            _, preds = torch.max(outputs.data, 1)
            correct_count += (preds == labels).sum().item()
            n_samples += labels.size(0)

        acc = 100 * correct_count / n_samples
        loss = loss_tr / n_samples

    return acc, loss


def evaluate_nc(net, dataloader, criterion):
    with torch.no_grad():
        net = net.eval()
        loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0

        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # fuse crops and batchsize
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)

            # forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # calculate performance metrics
            loss_tr += loss.item()

            _, preds = torch.max(outputs.data, 1)
            correct_count += (preds == labels).sum().item()
            n_samples += labels.size(0)

        acc = 100 * correct_count / n_samples
        loss = loss_tr / n_samples

    return acc, loss


def test(net, dataloader, criterion):
    with torch.no_grad():
        net = net.eval()
        loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0

        for data in dataloader:
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

        acc = 100 * correct_count / n_samples
        loss = loss_tr / n_samples

    return acc, loss


def test_nc(net, dataloader, criterion):
    with torch.no_grad():
        net = net.eval()
        loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0

        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # fuse crops and batchsize
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)

            # forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # calculate performance metrics
            loss_tr += loss.item()

            # combine results across the crops
            outputs = outputs.view(bs, ncrops, -1)
            outputs = torch.sum(outputs, dim=1) / ncrops

            _, preds = torch.max(outputs.data, 1)

            correct_count += (preds == labels).sum().item()
            n_samples += labels.size(0)

        acc = 100 * correct_count / n_samples
        loss = loss_tr / n_samples

    return acc, loss
