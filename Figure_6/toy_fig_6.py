import numpy as np
from scipy.special import lambertw
import torch
from scipy.stats import ortho_group
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import FloatTensor as FT
# blue, orange, green, red, violet, goh, pink
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


np.random.seed(1235)
X = np.random.randn(2, 2)
y = np.array([[+1, +1]]).T
_, _, V = np.linalg.svd(X)
theta = np.radians(-30)
c, s = np.cos(theta), np.sin(theta)
s = 0.5
U = np.array(((np.sqrt(1.0 - s ** 2), -s),
              (s, np.sqrt(1.0 - s ** 2))))

plt.figure(figsize=(8, 3.8))
ax_1 = plt.subplot(1, 2, 1)
ax_2 = plt.subplot(1, 2, 2)


def relu(x):
    return np.maximum(x, 0.0)

colors = ['#3B75AF',
          '#EF8636',
          '#529D3E',
          '#C53932',
          '#85584E']
for ind, l in enumerate([2, 3, 4, 5, 6]):
    print(l)
    #######################################
    C = 100
    Lambda = np.array([l, 2])
    phi = np.dot(U, np.dot(np.diag(Lambda), V))
    K = FT(np.dot((y * phi), (y * phi).T))
    alpha = torch.nn.Parameter(torch.ones((U.shape[0], 1)) * 1e-8)
    optimizer = torch.optim.SGD([alpha], lr=0.000003)
    Zs = []
    for i in range(5000):

        theta = C * torch.mm(FT(y * phi).T, alpha)
        output = torch.mm(FT(phi), theta)
        Zs += [np.dot(U.T, output.data.numpy() * y)[:, 0]]

        H = -(alpha * torch.log(alpha) +
              (1.0 - alpha) * torch.log((1.0 - alpha))).sum()
        rest = C / 2.0 * torch.mm(
            alpha.transpose(0, 1), torch.mm(K, alpha))[0, 0]
        loss = rest - H

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        alpha.data = F.relu(
            -F.relu(-alpha.data + 1.0) + 1.0 - 1e-50) + 1e-50

    Zs = np.vstack(Zs)
    ax_1.plot(np.abs(Zs[:, 0]), np.abs(Zs[:, 1]), lw=2, ls=':', color=colors[ind])
    #######################################
    C = 100
    Lambda = np.array([l, 2])
    phi = np.dot(U, np.dot(np.diag(Lambda), V))
    theta = torch.nn.Parameter(FT(np.zeros((U.shape[0], 1))))
    optimizer = torch.optim.SGD([theta], lr=0.05)
    Zs = []
    for i in range(5000):
        output = torch.mm(FT(phi), theta)
        Zs += [np.dot(U.T, output.data.numpy() * y)[:, 0]]
        loss = torch.log(1.0 + torch.exp(-output * FT(y)))
        loss = loss.sum()
        loss += 0.5 / C * torch.mm(theta.transpose(0, 1), theta).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    Zs = np.vstack(Zs)
    ax_1.plot(np.abs(Zs[:, 0]), np.abs(Zs[:, 1]), lw=2, alpha=0.9, color=colors[ind])
    ax_2.plot(np.abs(Zs[:, 0]), np.abs(Zs[:, 1]), lw=2, alpha=0.9, color=colors[ind])
    #######################################
    C = 100
    Lambda = np.array([l, 2])
    phi = np.dot(U, np.dot(np.diag(Lambda), V))
    theta = torch.nn.Parameter(FT(np.zeros((U.shape[0], 1))))
    optimizer = torch.optim.SGD([theta], lr=0.05)
    Zs = []
    for i in range(5000):
        output = torch.mm(FT(phi), theta)
        Zs += [np.dot(U.T, output.data.numpy() * y)[:, 0]]
        loss = torch.log(1.0 + torch.exp(-output * FT(y)))
        loss = loss.sum()
        loss += 0.1 / C * (output ** 2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    Zs = np.vstack(Zs)
    ax_2.plot(np.abs(Zs[:, 0]), np.abs(Zs[:, 1]), lw=2, ls='--', color=colors[ind])
    #######################################

ax_1.set_ylim([-0.1, 1.75])
ax_2.set_ylim([-0.1, 1.75])
ax_1.set_xlim([-0.5, 10.5])
ax_2.set_xlim([-0.5, 10.5])
for ax in [ax_1, ax_2]:
    # ax.axis('auto')
    ax.set_facecolor('#F9F9F9')
    ax.plot()
    ax.grid(b=True, which='major', linestyle='--', lw=0.7, color='k', alpha=0.3)
plt.tight_layout()
plt.show()
