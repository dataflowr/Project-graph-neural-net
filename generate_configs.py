import yaml
import os
import numpy as np


def generateNoiseConfigs(modelConfigFile, noise_array, dumpFolder = "./configs/configs_to_run/"):
    with open(modelConfigFile) as file:
        doc = yaml.load(file, Loader=yaml.FullLoader)

        for noise in noise_array:
            doc["train_data"]["noise"] = float(noise)
            doc["test_data"]["noise"] = float(noise)
            fileContent = yaml.dump(doc)
            with open(dumpFolder + modelConfigFile.split(".yaml")[0] + f"_noise_{noise:.2f}.yaml", "w") as f:
                f.write(fileContent)

def runConfigs(n):
    os.system(f"salloc -N {n} -n {n} mpirun ./configs/multiprocess")

if __name__ == "__main__":
    n = 10
    generateNoiseConfigs("default_qap.yaml", np.linspace(0.0, 0.3, num=n))
    runConfigs(n)
