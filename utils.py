import glob
import re

import pandas as pd

from pathlib import Path
        
def generate_bash_script():
    archs = sorted(glob.glob("workspace/ResNet18/arch/*.yaml"))
    arch_names = [f.replace("workspace/ResNet18/arch/", "").replace(".yaml", "") for f in archs]

    with open("workspace/bash_script.sh", "w") as bash_script:
        bash_script.write("#!/bin/bash")
        bash_script.write("\n\n")
        bash_script.write("chmod -R 777 .")
        bash_script.write("\n\n")
        for arch, arch_name in zip(archs, arch_names):
            Path(f"workspace/ResNet18/output/conf-{arch_name}/").mkdir(mode=777, exist_ok=True)
            Path(f"workspace/ResNet18/output/conf-{arch_name}/").chmod(0o777)
            for i in range(1, 22):
                Path(f"workspace/ResNet18/output/conf-{arch_name}/output{i}/").mkdir(mode=777, exist_ok=True)
                cmd = "timeloop-mapper " + arch.replace("workspace/", "") + " ResNet18/arch/components/*.yaml ResNet18/prob/resnet18_layer"+ str(i) + ".yaml ResNet18/mapper/mapper.yaml ResNet18/constraints/*.yaml -o ./ResNet18/output/conf-" + arch_name + "/output" + str(i)
                bash_script.write(cmd)
                bash_script.write("\n\n")


def get_energy_breakdown_from_stats_txt(file_path):
    data = dict()
    with open(file_path, "r") as f:
        stats = f.read()
        for m in re.findall(r"\b(?!\bMACCs|Total\b)([a-zA-Z<>=]+)\s+=\s+(?=.*[1-9])(\d+.\d+)", stats):
            data[m[0]] = float(m[1])
        f.close()
    return data


def get_area_breakdown_from_stats_txt(file_path):
    data = dict()
    with open(file_path, "r") as f:
        stats = f.read()
        for m in re.findall(r"=== ([a-zA-Z]+) ===.*?Area .*?\s+:\s(\d+.\d+)", stats, re.DOTALL):
            if m[1] != "1.00":
                data[m[0]] = float(m[1])
        f.close()
    return data


def pj_macc_stats(output_path):
    layer_paths = [f for f in glob.glob(output_path + "/*/" + "timeloop-mapper.stats.txt")]
    layers = [int(f.replace(output_path, "")
               .replace("timeloop-mapper.stats.txt", "")
               .replace("/", "")
               .replace("output", "")) for f in layer_paths]

    stats = {layer: get_energy_breakdown_from_stats_txt(layer_path) for layer, layer_path in zip(layers, layer_paths)}
    stats_df = pd.DataFrame(stats).T.sort_index().fillna(0)

    return stats_df, stats_df.sum(axis=0), stats_df.sum(axis=1)


def get_summary_stats(file_path):
    data = dict()
    with open(file_path, "r") as f:
        stats = f.read()
        initial_index = stats.index("Utilization:")
        end_index = stats.index("\n\npJ/MACC")
        cleaned = [c.replace(":", "") for c in stats[initial_index:end_index + 1].split("\n") if c != ""]
        summary = {s.split()[0]: float(s.split()[1]) for s in cleaned}
        
    return summary


def energy_stats(output_path):
    layer_paths = [f for f in glob.glob(output_path + "/*/" + "timeloop-mapper.stats.txt")]
    layers = [int(f.replace(output_path, "")
                .replace("timeloop-mapper.stats.txt", "")
                .replace("/", "")
                .replace("output", "")) for f in layer_paths]

    stats = {layer: {"Energy": get_summary_stats(layer_path)["Energy"]} for layer, layer_path in zip(layers, layer_paths)}
    stats_df = pd.DataFrame(stats).T.sort_index().fillna(0)

    return stats_df, stats_df.sum(axis=0).values.tolist()[0]


def generate_bash_script_by_model(*models):
    pass