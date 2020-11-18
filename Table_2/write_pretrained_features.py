import os, csv
import argparse
import IPython
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision

from models import model_attributes
from data.data import dataset_attributes, shift_types, prepare_data, log_data
from utils import set_seed, Logger, CSVBatchLogger, log_args

from extractable_resnet import resnet18

def main():
    parser = argparse.ArgumentParser()
    # Settings
    parser.add_argument('-d', '--dataset', choices=dataset_attributes.keys(), required=True)
    parser.add_argument('-s', '--shift_type', choices=shift_types, required=True)
    # Confounders -- this doesn't really matter because we just care about x
    parser.add_argument('-t', '--target_name')
    parser.add_argument('-c', '--confounder_names', nargs='+')

    parser.add_argument('--root_dir', default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument(
        '-m', '--model',
        choices=model_attributes.keys(),
        default='resnet18')
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--layers_to_extract', type=int, default=1)
    parser.add_argument('--get_preds_instead_of_features', action='store_true', default=False)


    args = parser.parse_args()
    assert args.shift_type == 'confounder'
    args.augment_data = False
    if args.root_dir is None:
        args.root_dir = dataset_attributes[args.dataset]['root_dir']

    set_seed(0)

    full_dataset = prepare_data(args, train=False, return_full_dataset=True)
    n_classes = full_dataset.n_classes
    loader_kwargs = {'batch_size':args.batch_size, 'num_workers':4, 'pin_memory':True}
    loader = full_dataset.get_loader(train=False, reweight_groups=None, **loader_kwargs)

    # Initialize model
    if not args.get_preds_instead_of_features:
        if args.model == 'resnet18':
            model = resnet18(
                pretrained=True,
                layers_to_extract=args.layers_to_extract)
        else:
            raise ValueError('Model not recognized.')
    elif args.get_preds_instead_of_features:
        assert args.model_path.endswith('.pth')
        model = torch.load(args.model_path)

    model.eval()
    model = model.cuda()

    n = len(full_dataset)
    idx_check = np.empty(n)
    last_batch = False
    start_pos = 0

    with torch.set_grad_enabled(False):

        for i, (x_batch, y, g) in enumerate(tqdm(loader)):
            x_batch = x_batch.cuda()

            num_in_batch = list(x_batch.shape)[0]
            assert num_in_batch <= args.batch_size
            if num_in_batch < args.batch_size:
                assert last_batch == False
                last_batch = True

            end_pos = start_pos + num_in_batch

            features_batch = model(x_batch).data.cpu().numpy()
            if i == 0:
                d = features_batch.shape[1]
                print(f'Extracting {d} features per example')
                features = np.empty((n, d))
            features[start_pos:end_pos, :] = features_batch

            # idx_check[start_pos:end_pos] = idx_batch.data.numpy()
            start_pos = end_pos

    if not args.get_preds_instead_of_features:
        features_dir = os.path.join(args.root_dir, 'features')
        if not os.path.exists(features_dir):
            os.makedirs(features_dir)
        output_path = os.path.join(
            features_dir,
            f'{args.model}_{args.layers_to_extract}layer.npy')
    else:
        output_path = args.model_path.split('.pth')[0] + '_preds-on_' + args.dataset + '.npy'


    np.save(output_path, features)

    # assert np.all(idx_check == np.arange(n))


if __name__=='__main__':
    main()
