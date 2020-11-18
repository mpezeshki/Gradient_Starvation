import numpy as np
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib import rc
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

rows = ['digit', 'blank']
columns = ['grayscale', 'colored']

entropy_erm, accuracy_erm = np.load('erm_results.npy')
entropy_sd, accuracy_sd = np.load('sd_results.npy')
entropy_irm, accuracy_irm = np.load('irm_results.npy')

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", show_color_bar=False, **kwargs):
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    if show_color_bar:
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)

        cbar = ax.figure.colorbar(im, cax=cax)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=12)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

plt.figure(figsize=(10, 4))

for i, (name, accuracy, entropy) in enumerate(zip(
        ['ERM', 'SD', 'IRM'],
        [accuracy_erm, accuracy_sd, accuracy_irm],
        [entropy_erm, entropy_sd, entropy_irm])):

    ax_top = plt.subplot(2, 4, i + 2)
    ax_bottom = plt.subplot(2, 4, i + 6)

    im_top = heatmap(100 * accuracy, rows, columns, ax=ax_top,
                     cmap="RdBu", cbarlabel="Accuracy",
                     vmin=0.0, vmax=100.0, show_color_bar=(i == 2))
    annotate_heatmap(im_top, valfmt="{x:.1f} %")
    im_bottom = heatmap(entropy, rows, columns, ax=ax_bottom,
                        cmap="YlGn_r", cbarlabel="Entropy",
                        vmin=0.0, vmax=0.69, show_color_bar=(i == 2))
    annotate_heatmap(im_bottom)
    ax_top.set_title(name, pad=25)

plt.savefig('figure_4.pdf')
