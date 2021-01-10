import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch import FloatTensor as FT
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons


# training for 10 different seeds and then averaging over all runs
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# seeds = [0]
# A two-layer neural netowrk with the following number of hidden units
hidden_dim = 500
n_samples = 300
epochs = 10000

dataset = lambda seed: two_moons_dataset(seed, margin=+0.1)
experiments = [
    {'name': 'No regularization',
     'dataset': dataset},
    {'name': 'weight_decay',
     'dataset': dataset,
     'coef': 0.01},
    {'name': 'Train longer',
     'dataset': dataset},
    {'name': 'adam - smaller lr',
     'dataset': dataset,
     'lr': 1e-4},
    {'name': 'adam - large lr',
     'dataset': dataset,
     'lr': 3e-3},
    {'name': '45 deg Rotation',
     'dataset': lambda seed: two_moons_dataset(seed, margin=+0.1, rotation=45)},
    {'name': 'dropout',
     'dataset': dataset},
    {'name': 'batchnorm',
     'dataset': dataset},
    {'name': 'deeper network (5 layers)',
     'dataset': dataset},
    {'name': 'SD',
     'dataset': dataset,
     'sd_coef': 0.003}]


# Network architecture
class Net(nn.Module):
    def __init__(self, hidden_dim, exp=None):
        super(Net, self).__init__()

        if 'deeper' in exp['name']:
            self.fc1 = nn.Linear(2, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
            self.fc4 = nn.Linear(hidden_dim, hidden_dim)
            self.fc5 = nn.Linear(hidden_dim, 1)

        else:
            if 'larger' in exp['name']:
                hidden_dim = hidden_dim * 10
            elif 'smaller' in exp['name']:
                hidden_dim = hidden_dim // 10

            self.fc1 = nn.Linear(2, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 1)
            if 'dropout' in exp['name']:
                self.dropout_layer = nn.Dropout(p=0.7)
            elif 'batchnorm' in exp['name']:
                self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        if 'deeper' in exp['name']:
            return self.fc5(F.relu(self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))))))
        else:
            if 'dropout' in exp['name']:
                return self.fc2(F.relu(self.dropout_layer(self.fc1(x))))
            elif 'batchnorm' in exp['name']:
                return self.fc2(F.relu(self.bn(self.fc1(x))))
            else:
                return self.fc2(F.relu(self.fc1(x)))


def two_moons_dataset(seed, margin=0.0, rotation=0.0):
    X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=seed)
    y_pm1 = (2.0 * y - 1.0)[:, None]
    move_along_x = 0.5 * np.ones((n_samples, 1))
    move_along_y = y_pm1 * 0.3
    X = X - np.concatenate([move_along_x, move_along_y], 1)
    X[:, 1] = X[:, 1] - X[:, 1].mean()
    alter_sign = np.sign(-X[:, 1] * y_pm1[:, 0])
    X[:, 1] = X[:, 1] * alter_sign
    X[:, 1] = X[:, 1] - y_pm1[:, 0] * margin
    # Rotate data by 90 degrees
    X = X[:, ::-1].copy()

    if rotation != 0.0:
        theta = np.radians(rotation)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        X = np.dot(X, R)
    return X, y

# An n by n grid for the heatmap
n = 100
d1_min = -2
d1_max = 2
d2_min = -2
d2_max = 2
d1, d2 = torch.meshgrid([
    torch.linspace(d1_min, d1_max, n),
    torch.linspace(d2_min, d2_max, n)])
heatmap_plane = torch.stack((d1.flatten(), d2.flatten()), dim=1)
heatmap_avg = np.zeros((heatmap_plane.shape[0], len(experiments)))

for seed in seeds:
    print('Seed: ' + str(seed))

    np.random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    for exp_idx, exp in enumerate(experiments):
        print('Experiment: ' + exp['name'])
        X, y = exp['dataset'](seed)
        X, y = FT(X), FT(2.0 * y - 1.0)

        net = Net(hidden_dim, exp)
        if 'weight_decay' in exp['name']:
            optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=exp['coef'])
        elif 'adam' in exp['name']:
            optimizer = optim.Adam(net.parameters(), lr=exp['lr'])
        else:
            optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)

        if 'longer' in exp['name']:
            epochs_ = 10 * epochs
        else:
            epochs_ = epochs
        for epoch in tqdm(range(epochs_)):
            y_hat = net(X)[:, 0]
            loss = torch.log(1.0 + torch.exp(-y_hat * y)).mean()
            if exp['name'] == 'SD':
                loss += exp['sd_coef'] * (y_hat ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss.item())

        # Average heatmaps over seeds
        net.eval()
        heatmap_avg[:, exp_idx] += net(heatmap_plane).data.cpu().numpy()[:, 0] / len(seeds)

# Plotting
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
cm = ListedColormap(['#C82506', '#0365C0'])
figure = plt.figure(figsize=((len(experiments) // 2) * 2.7, 6))
hmp_x = heatmap_plane[:, 0].data.numpy().reshape(n, n)
hmp_y = heatmap_plane[:, 1].data.numpy().reshape(n, n)

for exp_idx, exp in enumerate(experiments):
    ax = plt.subplot(2, len(experiments) // 2, exp_idx + 1)
    # plot only one of the seeds
    X, y = exp['dataset'](seeds[0])
    hma = heatmap_avg[:, exp_idx].reshape(n, n)
    ax.contourf(hmp_x, hmp_y, hma, 10, cmap=plt.cm.RdBu, alpha=.8)
    ax.contour(hmp_x, hmp_y, hma, 10, antialiased=True, linewidths=0.2, colors='k')
    ax.contour(hmp_x, hmp_y, hma, 0, antialiased=True, linewidths=1.0, colors='k')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm, edgecolors='k', s=18)
    ax.axhline(y=0, ls='--', lw=0.7, color='k', alpha=0.5)
    ax.axvline(x=0, ls='--', lw=0.7, color='k', alpha=0.5)
    ax.set_title(exp['name'])
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

plt.tight_layout()
plt.show()
