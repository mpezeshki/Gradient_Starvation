# This code is obtained from
# https://github.com/facebookresearch/InvariantRiskMinimization
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--exp', type=str, default='sd')
parser.add_argument('--hidden_dim', type=int, default=300)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--sd_coef', type=float, default=0.0)
parser.add_argument('--n_restarts', type=int, default=1)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=1001)
parser.add_argument('--grayscale_model', action='store_true')
parser.add_argument('--reload', action='store_true')
flags = parser.parse_args()


# Build environments
def make_environment(images, labels, e, gray=False, mean=False):
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()
    def torch_xor(a, b):
        return (a-b).abs() # Assumes both inputs are either 0 or 1
    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
    # labels = torch_xor(labels, torch_bernoulli(0.05, len(labels)))
    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    if mean:
        images = images.float().mean(0, keepdims=True)
        images = images.repeat(len(labels), 1, 1) * 0.0 + images.mean()

    images = torch.stack([images, images], dim=1)
    if not gray:
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0
    else:
        images = images / 2.0
    return {
        'images': (images.float() / 255.).cuda(),
        'labels': labels[:, None].cuda()}


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        if flags.grayscale_model:
            lin1 = nn.Linear(14 * 14, flags.hidden_dim)
        else:
            lin1 = nn.Linear(2 * 14 * 14, flags.hidden_dim)
        lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
        lin3 = nn.Linear(flags.hidden_dim, 1)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

    def forward(self, input):
        if flags.grayscale_model:
            out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
        else:
            out = input.view(input.shape[0], 2 * 14 * 14)
        out = self._main(out)
        return out


# Define loss function helpers
def mean_nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)

def mean_accuracy(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()

def IRM_penalty(logits, y):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

def pretty_print(*values):
    col_width = 13
    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))


print('Flags:')
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

train_accuracies = []
test_accuracies = []
for restart in range(flags.n_restarts):
    print("Restart", restart)

    # Load MNIST, make train/val splits, and shuffle train set examples
    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])

    np.random.seed(1234 + restart)
    torch.manual_seed(1234 + restart)
    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())

    # Define three environments: two for training and one for testing
    envs = [make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
            make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
            make_environment(mnist_val[0], mnist_val[1], 0.9)]

    # Instantiate the model
    mlp = MLP().cuda()
    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

    pretty_print('step', 'train nll', 'train acc', 'test acc')
    for step in range(flags.steps):

        if flags.exp == 'sd':
            # Pile up the two training environments
            # as SD does not require an explicit signal
            # on which domain the data is comming from.
            images = torch.cat([envs[0]['images'], envs[1]['images']])
            logits = mlp(images)
            labels = torch.cat([envs[0]['labels'], envs[1]['labels']])
            train_nll = mean_nll(logits, labels)
            train_acc = mean_accuracy(logits, labels)
            # envs[2] is the test domain
            test_nll = mean_nll(mlp(envs[2]['images']), envs[2]['labels'])
            test_acc = mean_accuracy(mlp(envs[2]['images']), envs[2]['labels'])

            # SD is simply an L2 on networks output
            sd = (logits ** 2).mean()
            # But it is applied with a delay
            if step >= flags.penalty_anneal_iters:
                # loss = 0.00001 * train_nll + flags.sd_coef * sd
                loss = flags.sd_coef * (sd + train_nll)
            else:
                loss = train_nll + flags.sd_coef * sd
                # loss = train_nll

        elif flags.exp == 'erm':
            images = torch.cat([envs[0]['images'], envs[1]['images']])
            logits = mlp(images)
            labels = torch.cat([envs[0]['labels'], envs[1]['labels']])
            train_nll = mean_nll(logits, labels)
            train_acc = mean_accuracy(logits, labels)
            # envs[2] is the test domain
            test_nll = mean_nll(mlp(envs[2]['images']), envs[2]['labels'])
            test_acc = mean_accuracy(mlp(envs[2]['images']), envs[2]['labels'])
            loss = train_nll.clone()

        elif flags.exp == 'irm':
            # Copied from IRM code
            for env in envs:
                logits = mlp(env['images'])
                env['nll'] = mean_nll(logits, env['labels'])
                env['acc'] = mean_accuracy(logits, env['labels'])
                env['penalty'] = IRM_penalty(logits, env['labels'])

            train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
            train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
            train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()
            test_nll = envs[2]['nll']
            test_acc = envs[2]['acc']

            weight_norm = torch.tensor(0.).cuda()
            for w in mlp.parameters():
                weight_norm += w.norm().pow(2)

            loss = train_nll.clone()
            loss += flags.l2_regularizer_weight * weight_norm
            penalty_weight = (
                flags.penalty_weight
                if step >= flags.penalty_anneal_iters else 1.0)
            loss += penalty_weight * train_penalty
            if penalty_weight > 1.0:
                # Rescale the entire loss to keep gradients in a reasonable range
                loss /= penalty_weight

        else:
            raise NotImplementedError()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            pretty_print(
                np.int32(step),
                train_nll.detach().cpu().numpy(),
                train_acc.detach().cpu().numpy(),
                test_acc.detach().cpu().numpy())

    train_accuracies += [train_acc.detach().cpu().numpy()]
    test_accuracies += [test_acc.detach().cpu().numpy()]

print('Average train accuracy over {} runs: {:.2f} % (pm {:.2f})'.format(
    len(train_accuracies),
    100 * np.mean(train_accuracies),
    100 * np.std(train_accuracies)))
print('Average test accuracy over {} runs: {:.2f} % (pm {:.2f})'.format(
    len(test_accuracies),
    100 * np.mean(test_accuracies),
    100 * np.std(test_accuracies)))

# Saving the results for the last restart ...
# Four corners of Fig. X for further investigation
envs = [
    # without color - with shape
    make_environment(mnist_val[0], mnist_val[1], 0.9, gray=True),
    # with color - with shape (the test set above)
    make_environment(mnist_val[0], mnist_val[1], 0.9),
    # without color - without shape (simple blank images)
    make_environment(mnist_val[0], mnist_val[1], 0.9, gray=True, mean=True),
    # with color - without shape (blank images but with color)
    make_environment(mnist_val[0], mnist_val[1], 0.9, mean=True)]

# results contains entropies and accuracies
results = [np.zeros((2, 2)), np.zeros((2, 2))]
for i, env in enumerate(envs):
    logits = mlp(env['images'])
    prob = torch.sigmoid(logits)

    results[0][i // 2, i % 2] = (
        -(prob * torch.log(prob) +
          (1 - prob) * torch.log(1 - prob + 1e-8)).mean()).item()
    results[1][i // 2, i % 2] = mean_accuracy(logits, env['labels']).item()
np.save(flags.exp + '_results', results)
# call plot.py for ploting
