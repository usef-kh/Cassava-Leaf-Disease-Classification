import os

import torch


def save(net, hps, desc=''):
    # save model params
    params_path = os.path.join(hps['model_save_dir'], hps['network'] + desc)
    torch.save(net.state_dict(), params_path)


def restore(net, hps):
    """ Load back the model from its checkpoint, if available"""

    params_path = os.path.join(hps['model_save_dir'], hps['network'])

    if os.path.exists(params_path):
        try:
            params = torch.load(params_path)
            net.load_state_dict(params)
            print("Network Restored!")

        except Exception as e:
            print("Restore Failed! Creating model from scratch.")
            print(e)

    else:
        print("Restore point unavailable. Creating model from scratch.")


def load_features(model, params):
    """ Load params into all layers of 'model'
        that are compatible, then freeze them"""

    model_dict = model.state_dict()

    imp_params = {k: v for k, v in params.items() if k in model_dict}

    # Load layers
    model_dict.update(imp_params)
    model.load_state_dict(imp_params)

    # Freeze layers
    for name, param in model.named_parameters():
        param.requires_grad = False
