import yaml
import os
import numpy as np


def generateNoiseConfigs(modelConfigFile, noise_array, dumpFolder="./configs/configs_to_run/"):
    with open(modelConfigFile) as file:
        doc = yaml.load(file, Loader=yaml.FullLoader)

        doc["train"]["epoch"] = 10

        for noise in noise_array:
            doc["train_data"]["noise"] = float(noise)
            doc["test_data"]["noise"] = float(noise)
            fileContent = yaml.dump(doc)
            with open(
                dumpFolder + modelConfigFile.split(".yaml")[0] + f"_noise_{noise:.2f}.yaml", "w"
            ) as f:
                f.write(fileContent)


def generateClusterConfigs(
    modelConfigFile, graph_edge_density, merge_edge_density, dumpFolder="./configs/configs_to_run/"
):
    with open(modelConfigFile) as file:
        doc = yaml.load(file, Loader=yaml.FullLoader)

        doc["load_data"] = False
        doc["train"]["epoch"] = 15

        for d1, d2 in zip(graph_edge_density, merge_edge_density):
            doc["train_data"]["graph_1"]["edge_density"] = float(d1)
            doc["train_data"]["graph_2"]["edge_density"] = float(d1)
            doc["train_data"]["merge_arg"]["edge_density"] = float(d2)
            doc["test_data"]["graph_1"]["edge_density"] = float(d1)
            doc["test_data"]["graph_2"]["edge_density"] = float(d1)
            doc["test_data"]["merge_arg"]["edge_density"] = float(d2)
            fileContent = yaml.dump(doc)
            with open(
                dumpFolder + f"cluster__edge_prob__intra_{d1:.2f}_inter_{d2:.2f}.yaml",
                "w",
            ) as f:
                f.write(fileContent)


def runConfigs(n, pythonScript="commander.py"):
    os.system(f"cd configs && make")

    # os.system(f"salloc -N {n} -n {n} mpirun ./configs/multiprocess {pythonScript}")
    # if the previous line don't work, you can try to execute this line instead
    os.system(f"salloc -N {n} -n {n} --mem 40000 mpirun ./configs/multiprocess {pythonScript}")


if __name__ == "__main__":
    n_samples = 10
    n_machines = 30
    # generateNoiseConfigs("default_qap.yaml", np.linspace(0.0, 0.3, num=n_samples))
    generateClusterConfigs(
        "default_cluster.yaml",
        graph_edge_density=np.concatenate(
            (
                np.linspace(0.1, 1.0, num=n_samples),
                np.linspace(0.3, 1.0, num=n_samples),
                np.linspace(0.5, 1.0, num=n_samples),
            ),
        ),
        merge_edge_density=np.concatenate(
            (
                np.linspace(0.0, 1.0, num=n_samples),
                np.array([0.6] * n_samples),
                np.array([0.7] * n_samples),
            )
        ),
    )
    runConfigs(n_machines, "commander_label.py")
