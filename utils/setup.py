import os

from models import vgg, efficient_net, resnext
from utils.checkpoint import restore

hps = {
    'network': '',  # network type
    'name': '',  # network name
    'model_save_dir': None,  # where will checkpoints be stored (path created automatically using hps[name])
    'restore': False,  # load from checkpoint
    'n_epochs': 300,
    'lr': 0.01,  # starting learning rate
    'fold_id': None,  # Which fold to train
}

nets = {
    'vgg': vgg.Vgg,
    'efn': efficient_net.EfficientNet,
    'resnext': resnext.ResNext,
}


def setup_hparams(args):
    for arg in args:
        key, value = arg.split('=')
        if key not in hps:
            raise ValueError(key + ' is not a valid hyper parameter')
        else:
            hps[key] = value

    # Invalid network check
    if hps['network'] not in nets:
        raise ValueError("Invalid network.\nPossible ones include:\n - " + '\n - '.join(nets.keys()))

    if hps['name'] == '':
        raise ValueError("Please provide a network name")

    # Invalid parameter check
    try:
        hps['n_epochs'] = int(hps['n_epochs'])
        hps['lr'] = float(hps['lr'])

    except Exception as e:
        raise ValueError("Invalid input parameters")

    # Im sure theres a better way to do this, but ill figure it out later
    if hps['restore']:
        if hps['restore'] == 'False':
            hps['restore'] = False

        elif hps['restore'] == 'True':
            hps['restore'] = True

        else:
            raise ValueError("Invalid input parameters")

    # create checkpoint directory
    hps['model_save_dir'] = os.path.join(os.getcwd(), 'checkpoints', hps['name'])

    if not os.path.exists(hps['model_save_dir']):
        os.makedirs(hps['model_save_dir'])

    return hps


def setup_network(hps):
    # Prepare network and logger
    net = nets[hps['network']]()

    # Restore if required
    if hps['restore']:
        restore(net, hps)

    return net
