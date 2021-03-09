import json
import os
import pathlib
import shutil

import torch
import torch.backends.cudnn as cudnn
from sacred import SETTINGS, Experiment

import trainer as trainer
from loaders.data_generator_label import Generator
from loaders.label_loaders import label_loader
from models import get_model
from toolbox import logger, metrics, utils
from toolbox.losses import *
from toolbox.optimizer import get_optimizer

import numpy as np

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

### BEGIN Sacred setup
ex = Experiment()
ex.add_config("default_cluster.yaml")


@ex.config_hook
def set_experiment_name(config, command_name, logger):
    ex.path = config["name"]
    return config


@ex.config
def update_paths(root_dir, name, train_data, test_data):
    log_dir = "{}/runs/{}/labels_{}_{}_{}_{}_{}/".format(
        root_dir,
        name,
        train_data["graph_1"]["generative_model"],
        train_data["graph_1"]["edge_density_range"],
        train_data["graph_2"]["generative_model"],
        train_data["graph_2"]["edge_density_range"],
        train_data["merge_arg"]["edge_density_range"],
    )
    path_dataset = train_data["path_dataset"]
    # The two keys below are specific to testing
    # These default values are overriden by command line
    model_path = os.path.join(log_dir, "model_best.pth.tar")
    output_filename = "test.json"


@ex.config_hook
def init_observers(config, command_name, logger):
    if command_name == "train":
        neptune = config["observers"]["neptune"]
        if neptune["enable"]:
            from neptunecontrib.monitoring.sacred import NeptuneObserver

            ex.observers.append(NeptuneObserver(project_name=neptune["project"]))
    return config


@ex.post_run_hook
def clean_observer(observers):
    """ Observers that are added in a config_hook need to be cleaned """
    try:
        neptune = observers["neptune"]
        if neptune["enable"]:
            from neptunecontrib.monitoring.sacred import NeptuneObserver

            ex.observers = [obs for obs in ex.observers if not isinstance(obs, NeptuneObserver)]
    except KeyError:
        pass


### END Sacred setup

### Training
@ex.capture
def init_logger(name, _config, _run):
    # set loggers
    exp_logger = logger.Experiment(name, _config, run=_run)
    exp_logger.add_meters("train", metrics.make_meter_matching())
    exp_logger.add_meters("val", metrics.make_meter_matching())
    # exp_logger.add_meters('test', metrics.make_meter_matching())
    exp_logger.add_meters("hyperparams", {"learning_rate": metrics.ValueMeter()})
    return exp_logger


@ex.capture
def setup_env(cpu):
    # Randomness is already controlled by Sacred
    # See https://sacred.readthedocs.io/en/stable/randomness.html
    if not cpu:
        cudnn.benchmark = True


# create necessary folders and config files
@ex.capture
def init_output_env(_config, root_dir, log_dir, path_dataset):
    utils.check_dir(os.path.join(root_dir, "runs"))
    utils.check_dir(log_dir)
    utils.check_dir(path_dataset)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(_config, f)


@ex.capture
def save_checkpoint(state, is_best, log_dir, filename="checkpoint.pth.tar"):
    utils.check_dir(log_dir)
    filename = os.path.join(log_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(log_dir, "model_best.pth.tar"))

    fn = os.path.join(log_dir, "checkpoint_epoch{}.pth.tar")
    torch.save(state, fn.format(state["epoch"]))

    if (state["epoch"] - 1) % 5 != 0:
        # remove intermediate saved models, e.g. non-modulo 5 ones
        if os.path.exists(fn.format(state["epoch"] - 1)):
            os.remove(fn.format(state["epoch"] - 1))

    state["exp_logger"].to_json(log_dir=log_dir, filename="logger.json")


@ex.command
def train(cpu, load_data, train_data, train, arch, log_dir):
    """Main func."""
    global best_score, best_epoch
    best_score, best_epoch = -1, -1
    use_cuda = not cpu and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print("Using device:", device)

    # init random seeds
    setup_env()
    print("Models saved in ", log_dir)

    init_output_env()
    exp_logger = init_logger()

    gene_train = Generator("train", train_data)
    if load_data:
        gene_train.load_dataset()
    else:
        gene_train.create_dataset()
    train_loader = label_loader(gene_train, train["batch_size"], gene_train.constant_n_vertices)
    gene_val = Generator("val", train_data)
    if load_data:
        gene_val.load_dataset()
    else:
        gene_val.create_dataset()
    val_loader = label_loader(gene_val, train["batch_size"], gene_val.constant_n_vertices)

    model = get_model(arch)

    optimizer, scheduler = get_optimizer(train, model)
    if arch["arch"] == "Simple_Node_Embedding":
        criterion = cluster_embedding_loss(device=device)
    elif arch["arch"] == "Similarity_Model":
        criterion = cluster_similarity_loss()

    model.to(device)

    is_best = True
    for epoch in range(train["epoch"]):
        print("Current epoch: ", epoch)
        trainer.train_cluster(
            train_loader,
            model,
            criterion,
            optimizer,
            exp_logger,
            device,
            epoch,
            eval_score=metrics.accuracy_cluster_kmeans,
            print_freq=train["print_freq"],
        )

        acc, loss = trainer.val_cluster(
            val_loader,
            model,
            criterion,
            exp_logger,
            device,
            epoch,
            eval_score=metrics.accuracy_cluster_kmeans,
        )
        scheduler.step(loss)
        # remember best acc and save checkpoint
        is_best = acc > best_score
        best_score = max(acc, best_score)
        if True == is_best:
            best_epoch = epoch

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_score": best_score,
                "best_epoch": best_epoch,
                "exp_logger": exp_logger,
            },
            is_best,
        )


### Testing
@ex.capture
def load_model(model, device, model_path):
    """ Load model. Note that the model_path argument is captured """
    if os.path.exists(model_path):
        print("Reading model from ", model_path)
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint["state_dict"])
        return model
    else:
        raise RuntimeError("Model does not exist!")


def save_to_json(key, acc, loss, filename):
    if os.path.exists(filename):
        with open(filename, "r") as jsonFile:
            data = json.load(jsonFile)
    else:
        data = {}
    data[key] = {"loss": loss, "acc": acc}
    with open(filename, "w") as jsonFile:
        json.dump(data, jsonFile)


@ex.capture
def create_key(log_dir, test_data):
    template = "model_{}data_label_{}_{}_{}_{}_{}"
    key = template.format(
        log_dir,
        test_data["graph_1"]["generative_model"],
        test_data["graph_1"]["edge_density_range"],
        test_data["graph_2"]["generative_model"],
        test_data["graph_2"]["edge_density_range"],
        test_data["merge_arg"]["edge_density_range"],
    )
    return key


@ex.command
def eval(name, cpu, load_data, test_data, train, arch, log_dir, model_path, output_filename, return_result=False):
    use_cuda = not cpu and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print("Using device:", device)

    model = get_model(arch)
    model.to(device)
    model = load_model(model, device)

    if arch["arch"] == "Simple_Node_Embedding":
        criterion = cluster_embedding_loss(device=device)
    elif arch["arch"] == "Similarity_Model":
        criterion = cluster_similarity_loss()
    exp_logger = logger.Experiment(name)
    exp_logger.add_meters("test", metrics.make_meter_matching())

    gene_test = Generator("test", test_data)
    if load_data:
        gene_test.load_dataset()
    else:
        gene_test.create_dataset()
    test_loader = label_loader(gene_test, train["batch_size"], gene_test.constant_n_vertices)
    acc, loss = trainer.val_cluster(
        test_loader,
        model,
        criterion,
        exp_logger,
        device,
        epoch=0,
        eval_score=metrics.accuracy_cluster_kmeans,
        val_test="test",
    )
    if not return_result:
        key = create_key()
        filename_test = os.path.join(log_dir, output_filename)
        print("Saving result at: ", filename_test)
        save_to_json(key, acc, loss, filename_test)
    return acc, loss

@ex.command
def generate_data(test_data):
    print(test_data)
    gene = Generator("test", test_data)
    gene.load_dataset()


@ex.command
def serial_symmetric_evaluation(name, cpu, test_data, train, arch, log_dir, model_path, output_filename):
    acc_t = []
    loss_t = []
    noise_t =np.concatenate(
        ([0], np.exp(np.linspace(np.log(1e-5),np.log(0.5),50)) )
        )
    print(noise_t)
    for noise in noise_t :
        print(noise)
        test_data["graph_2"]["edge_density"]=noise
        acc, loss =  eval(name, cpu, False, test_data, train, arch, log_dir, model_path, output_filename, return_result=True)
        acc_t.append(acc)
        loss_t.append(loss)
    np.save( "sym_result", [noise_t, acc_t, loss_t])

@ex.command
def eval_spectral(name, load_data, train, test_data, log_dir, output_filename):
    exp_logger = logger.Experiment(name)
    exp_logger.add_meters("test", metrics.make_meter_matching())

    gene_test = Generator("test", test_data)
    if load_data:
        gene_test.load_dataset()
    else:
        gene_test.create_dataset()
    test_loader = label_loader(gene_test, train["batch_size"], gene_test.constant_n_vertices)

    exp_logger.reset_meters("test")

    print_freq = 10
    for i, (input, cluster_sizes) in enumerate(test_loader):
        acc, total_n_vertices = metrics.accuracy_spectral_cluster_kmeans(input, cluster_sizes)
        exp_logger.update_meter("test", "acc", acc, n=total_n_vertices)
        if i % print_freq:
            acc = exp_logger.get_meter("test", "acc")
            print("Test set\t" "Acc {acc.avg:.3f} ({acc.val:.3f})".format(acc=acc))


@ex.automain
def main():
    print("Main does nothing")
    pass
