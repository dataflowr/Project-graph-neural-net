import itertools
import os
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="center")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def extractTrainLog(filename):
    with open(filename + ".yaml") as file:
        paramList = yaml.load(file, Loader=yaml.FullLoader)

        n_epoch = paramList["train"]["epoch"]

    metrics_per_epoch = {
        "loss": {"train": [[] for _ in range(n_epoch)], "valid": [[] for _ in range(n_epoch)]},
        "acc": {"train": [[] for _ in range(n_epoch)], "valid": [[] for _ in range(n_epoch)]},
    }
    training_time = 0.0
    n_line_training = 0
    with open(filename + ".log") as file:
        for line in file:
            if line.startswith("Epoch:"):
                line = line.split()
                epoch = int(line[1].split("][")[0][1:])
                i = line.index("Loss")
                metrics_per_epoch["loss"]["train"][epoch].append(float(line[i + 1]))
                metrics_per_epoch["acc"]["train"][epoch].append(float(line[i + 4]))

                i = line.index("Time")
                training_time += float(line[i + 1])
                n_line_training += 1
            elif line.startswith("Validation"):
                line = line.split()
                epoch = int(line[line.index("epoch:") + 1].split("][")[0][1:])
                i = line.index("Loss")
                metrics_per_epoch["loss"]["valid"][epoch].append(float(line[i + 1]))
                metrics_per_epoch["acc"]["valid"][epoch].append(float(line[i + 4]))

    if n_line_training != 0:
        training_time /= n_line_training

    metrics = {
        "loss": {"train_avg": [], "valid_avg": [], "train_std": [], "valid_std": []},
        "acc": {"train_avg": [], "valid_avg": [], "train_std": [], "valid_std": []},
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n_epoch):
            metrics["loss"]["train_avg"].append(np.nanmean(metrics_per_epoch["loss"]["train"][i]))
            metrics["loss"]["train_std"].append(np.nanstd(metrics_per_epoch["loss"]["train"][i]))
            metrics["acc"]["train_avg"].append(np.nanmean(metrics_per_epoch["acc"]["train"][i]))
            metrics["acc"]["train_std"].append(np.nanstd(metrics_per_epoch["acc"]["train"][i]))
            metrics["loss"]["valid_avg"].append(np.nanmean(metrics_per_epoch["loss"]["valid"][i]))
            metrics["loss"]["valid_std"].append(np.nanstd(metrics_per_epoch["loss"]["valid"][i]))
            metrics["acc"]["valid_avg"].append(np.nanmean(metrics_per_epoch["acc"]["valid"][i]))
            metrics["acc"]["valid_std"].append(np.nanstd(metrics_per_epoch["acc"]["valid"][i]))

    return metrics, training_time


def extractEvalLog(filename):
    metrics = {
        "loss": {
            "eval_avg": [],
            "eval_std": [],
        },
        "acc": {"eval_avg": [], "eval_std": []},
    }

    with open(filename + ".log") as file:
        for line in file:
            if line.startswith("Test"):
                line = line.split()
                i = line.index("Loss")
                metrics["loss"]["eval_avg"].append(float(line[i + 1]))
                metrics["acc"]["eval_avg"].append(float(line[i + 4]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics["loss"]["eval_std"] = np.nanstd(metrics["loss"]["eval_avg"])
        metrics["loss"]["eval_avg"] = np.nanmean(metrics["loss"]["eval_avg"])
        metrics["acc"]["eval_std"] = np.nanstd(metrics["acc"]["eval_avg"])
        metrics["acc"]["eval_avg"] = np.nanmean(metrics["acc"]["eval_avg"])

    return metrics


def extractEvalSpectralLog(filename):
    metrics = {
        "acc": {"eval_avg": [], "eval_std": []},
    }

    with open(filename + ".log") as file:
        for line in file:
            if line.startswith("Test"):
                line = line.split()
                i = line.index("Acc")
                metrics["acc"]["eval_avg"].append(float(line[i + 1]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics["acc"]["eval_std"] = np.nanstd(metrics["acc"]["eval_avg"])
        metrics["acc"]["eval_avg"] = np.nanmean(metrics["acc"]["eval_avg"])

    return metrics


def generatePlot(configName):
    filename = os.path.join(
        os.path.join(os.path.join(os.path.dirname(__file__), "configs_computed"), configName),
        configName,
    )
    metrics, _ = extractTrainLog(filename)
    metrics_iter = iter(metrics)
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics))
    for row in axes:
        if not hasattr(row, "__len__"):
            row = [row]
        for col in row:
            metric_name = next(metrics_iter)
            x = np.arange(len(metrics[metric_name]["train_avg"]))
            col.errorbar(
                x,
                metrics[metric_name]["train_avg"],
                yerr=metrics[metric_name]["train_std"],
                label=metric_name + "_train",
            )
            col.errorbar(
                x,
                metrics[metric_name]["valid_avg"],
                yerr=metrics[metric_name]["valid_std"],
                label=metric_name + "_valid",
            )
            col.title.set_text(metric_name)
            col.legend()
    fig.suptitle(configName)
    plt.savefig(filename + ".png")


def generateGrid():
    n_epoch = 15
    n_samples = 10
    x = np.linspace(0.1, 1.0, num=n_samples)
    y = np.array([float(i) / float(n_samples) for i in range(n_samples + 1)])
    acc = np.zeros((n_epoch, n_samples, n_samples + 1), dtype=float)
    for x_i, y_i in itertools.product(range(n_samples), range(n_samples + 1)):
        x_ = x[x_i]
        y_ = y[y_i]
        configName = f"cluster__edge_prob__intra_{x_:.2f}_inter_{y_:.2f}"
        filename = os.path.join(
            os.path.join(os.path.join(os.path.dirname(__file__), "configs_computed"), configName),
            configName,
        )
        metrics, _ = extractTrainLog(filename)
        for i in range(n_epoch):
            acc[i][x_i][y_i] = metrics["acc"]["valid_avg"][i]

    x = np.linspace(0.05, 1.05, num=n_samples + 1)
    y = np.array([(float(i) - 0.5) / float(n_samples) for i in range(n_samples + 2)])

    for i in range(n_epoch):
        levels = MaxNLocator(nbins=15).tick_values(acc.min(), acc.max())
        cmap = plt.get_cmap("RdYlGn")
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        fig, ax = plt.subplots()
        im = ax.pcolormesh(x, y, acc[i].T, cmap=cmap, norm=norm)
        fig.colorbar(im, ax=ax)
        fig.suptitle(f"Valid accuracy for FGNN trained over {i+1} epochs")
        plt.xlabel("Intra cluster edge probabilities")
        plt.ylabel("Inter cluster edge probabilities")
        plt.savefig(
            os.path.join(os.path.dirname(__file__), f"configs_computed/acc_edge_prob_ep_{i}.png")
        )


def generateEvalGrid(imgfilename="acc_local_edge_prob.png"):
    n_samples = 10
    x = np.linspace(0.1, 1.0, num=n_samples)
    y = np.array([float(i) / float(n_samples) for i in range(n_samples + 1)])
    acc = np.zeros((n_samples, n_samples + 1), dtype=float)
    for x_i, y_i in itertools.product(range(n_samples), range(n_samples + 1)):
        x_ = x[x_i]
        y_ = y[y_i]
        configName = f"cluster__eval2__edge_prob__intra_{x_:.2f}_inter_{y_:.2f}"
        filename = os.path.join(
            os.path.join(os.path.join(os.path.dirname(__file__), "configs_computed"), configName),
            configName,
        )
        metrics = extractEvalLog(filename)
        acc[x_i][y_i] = metrics["acc"]["eval_avg"]

    x = np.linspace(0.05, 1.05, num=n_samples + 1)
    y = np.array([(float(i) - 0.5) / float(n_samples) for i in range(n_samples + 2)])

    levels = MaxNLocator(nbins=15).tick_values(acc.min(), acc.max())
    cmap = plt.get_cmap("RdYlGn")
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig, ax = plt.subplots()
    im = ax.pcolormesh(x, y, acc.T, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax)
    fig.suptitle(f"Valid accuracy for a FGNN trained with p_intra=0.9 and p_inter=0.7")
    plt.xlabel("Intra cluster edge probabilities")
    plt.ylabel("Inter cluster edge probabilities")
    plt.savefig(os.path.join(os.path.dirname(__file__), f"configs_computed/{imgfilename}"))


def generateEvalSpectralGrid(imgfilename="acc_spectral.png"):
    n_samples = 10
    x = np.linspace(0.1, 1.0, num=n_samples)
    y = np.array([float(i) / float(n_samples) for i in range(n_samples + 1)])
    acc = np.zeros((n_samples, n_samples + 1), dtype=float)
    for x_i, y_i in itertools.product(range(n_samples), range(n_samples + 1)):
        x_ = x[x_i]
        y_ = y[y_i]
        configName = f"cluster__eval_spectral__edge_prob__intra_{x_:.2f}_inter_{y_:.2f}"
        filename = os.path.join(
            os.path.join(os.path.join(os.path.dirname(__file__), "configs_computed"), configName),
            configName,
        )
        metrics = extractEvalSpectralLog(filename)
        acc[x_i][y_i] = metrics["acc"]["eval_avg"]

    x = np.linspace(0.05, 1.05, num=n_samples + 1)
    y = np.array([(float(i) - 0.5) / float(n_samples) for i in range(n_samples + 2)])

    levels = MaxNLocator(nbins=15).tick_values(acc.min(), acc.max())
    cmap = plt.get_cmap("RdYlGn")
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig, ax = plt.subplots()
    im = ax.pcolormesh(x, y, acc.T, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax)
    fig.suptitle(f"Valid accuracy for spectral clustering")
    plt.xlabel("Intra cluster edge probabilities")
    plt.ylabel("Inter cluster edge probabilities")
    plt.savefig(os.path.join(os.path.dirname(__file__), f"configs_computed/{imgfilename}"))


def generateGridHyperParam():
    n_epoch = 20
    dim_array = np.array([8, 16, 32, 64, 128, 256, 512])
    num_blocks_array = np.array([5, 4, 3, 2, 1])
    n1 = len(dim_array)
    n2 = len(num_blocks_array)
    acc = np.zeros((n1, n2), dtype=float)
    time = np.zeros((n1, n2), dtype=float)
    for x_i, y_i in itertools.product(range(n1), range(n2)):
        x_ = dim_array[x_i]
        y_ = num_blocks_array[y_i]
        configName = f"cluster__emb_dim_{x_}_num_blocks_{y_}"
        filename = os.path.join(
            os.path.join(os.path.join(os.path.dirname(__file__), "configs_computed"), configName),
            configName,
        )
        metrics, time[x_i][y_i] = extractTrainLog(filename)
        acc[x_i][y_i] = metrics["acc"]["valid_avg"][-1]

    fig, ax = plt.subplots()

    im, cbar = heatmap(acc.T, num_blocks_array, dim_array, ax=ax, cmap="YlGn", cbarlabel="Accuracy")
    texts = annotate_heatmap(im, data=time.T, valfmt="{x:.2f}s")

    fig.suptitle(f"Valid accuracy for FGNN trained over {n_epoch} epochs")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Num regular blocks")
    plt.savefig(os.path.join(os.path.dirname(__file__), f"configs_computed/acc_hyperparam.png"))


if len(sys.argv) > 1:
    if sys.argv[1] == "-plot":
        generatePlot(sys.argv[2])
    if sys.argv[1] == "-grid":
        generateGrid()
    if sys.argv[1] == "-gridE":
        if len(sys.argv) > 2:
            generateEvalGrid(sys.argv[2])
        else:
            generateEvalGrid()
    if sys.argv[1] == "-gridES":
        if len(sys.argv) > 2:
            generateEvalSpectralGrid(sys.argv[2])
        else:
            generateEvalSpectralGrid()
    if sys.argv[1] == "-gridH":
        generateGridHyperParam()
