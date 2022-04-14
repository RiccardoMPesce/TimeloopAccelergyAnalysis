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
                cmd = "timeloop-mapper " + arch.replace("workspace/", "") + " ResNet18/arch/components/*.yaml ResNet18/prob/resnet18_layer"+ str(i) + ".yaml ResNet18/mapper/mapper.yaml ResNet18/constraints/*.yaml -o ResNet18/output/conf-" + arch_name + "/output" + str(i)
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
        end_index = stats.index("pJ/MACC")
        cleaned = [c.replace(":", "").replace(" =", "") for c in stats[initial_index:end_index].split("\n") if c != ""]
        summary = {s.split()[0]: float(s.split()[1]) for s in cleaned}
        
    return summary


def generate_stats_dict(file_path):
    stats_dict = dict()
    stats_dict_pjmacc = {"pJ/MACC " + element: value for element, value in get_energy_breakdown_from_stats_txt(file_path).items()}
    stats_dict["Area"] = get_area_breakdown_from_stats_txt(file_path)
    
    for stat, value in get_summary_stats(file_path).items():
        stats_dict[stat] = value

    return stats_dict | stats_dict_pjmacc 
    

def generate_stats_by_arch(model_output_path, to_csv=False):
    paths = list(Path(model_output_path).iterdir())
    layers = [int(f.name.replace("output", "")) for f in paths]

    df_dict = {}

    for path, layer in zip(paths, layers):
        df_dict[layer] = generate_stats_dict(path / "timeloop-mapper.stats.txt")

    df_table = pd.DataFrame(df_dict).T.fillna(0.00).sort_index()
    df_summary = pd.DataFrame(df_table.sum(axis=0).round(5), columns=["Total"]).T.fillna(0.00).sort_index()

    df_summary.loc[:, "Utilization"] = df_table["Utilization"].mode().values[0]
    df_summary.loc[:, "Area"] = df_table["Area"].mode().values[0]

    df_summary["Inferences/Second"] = 1000000000 / df_summary["Cycles"].values.tolist()[0]

    if to_csv:
        df_table.to_csv(Path(model_output_path).name + "_stats.csv")
        df_summary.to_csv(Path(model_output_path).name + "_summary.csv")

    return df_table, df_summary

def compare_models(output_path):
    comparison_df = pd.DataFrame()

    for path in sorted(Path(output_path).iterdir()):
        _, summary = generate_stats_by_arch(path)
        comparison_df = pd.concat([comparison_df, summary.rename(index={"Total": path.name})])

    return comparison_df