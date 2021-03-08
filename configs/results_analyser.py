import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
import sys

def extractLog(filename):
    with open(filename + ".yaml") as file:
        paramList = yaml.load(file, Loader=yaml.FullLoader)

        n_epoch = paramList["train"]["epoch"]

    metrics_per_epoch = {"loss": {"train": [[] for _ in range(n_epoch)], "valid": [[] for _ in range(n_epoch)]},
    "acc": {"train": [[] for _ in range(n_epoch)], "valid": [[] for _ in range(n_epoch)]}}
    with open(filename + ".log") as file:
        for line in file:
            if line.startswith("Epoch:"):
                line = line.split()
                epoch = int(line[1].split("][")[0][1:])
                i = line.index("Loss")
                metrics_per_epoch["loss"]["train"][epoch].append(float(line[i+1]))
                metrics_per_epoch["acc"]["train"][epoch].append(float(line[i+4]))
            elif line.startswith("Validation"):
                line = line.split()
                epoch = int(line[line.index("epoch:")+1].split("][")[0][1:])
                i = line.index("Loss")
                metrics_per_epoch["loss"]["valid"][epoch].append(float(line[i+1]))
                metrics_per_epoch["acc"]["valid"][epoch].append(float(line[i+4]))
    
    metrics = {"loss": {"train_avg": [], "valid_avg": [], "train_std": [], "valid_std": []},
    "acc": {"train_avg": [], "valid_avg": [], "train_std": [], "valid_std": []}}

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

    return metrics 

def generatePlot(configName):
    filename = os.path.join(os.path.join(os.path.join(os.path.dirname(__file__),"configs_computed"), configName), configName)
    metrics = extractLog(filename)
    metrics_iter = iter(metrics)
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics))
    for row in axes:
        if not hasattr(row, '__len__'):
            row = [row]
        for col in row:
            metric_name = next(metrics_iter)
            x = np.arange(len(metrics[metric_name]["train_avg"]))
            col.errorbar(x, metrics[metric_name]["train_avg"], yerr=metrics[metric_name]["train_std"], label=metric_name+"_train")
            col.errorbar(x, metrics[metric_name]["valid_avg"], yerr=metrics[metric_name]["valid_std"], label=metric_name+"_valid")
            col.title.set_text(metric_name)
            col.legend()
    fig.suptitle(configName)
    plt.savefig(filename + ".png")

if len(sys.argv) > 1:
    if sys.argv[1] == "-plot":
        generatePlot(sys.argv[2])
