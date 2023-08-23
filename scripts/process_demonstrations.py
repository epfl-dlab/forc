from src.utils import DemonstrationUtils
import os
import json
import sys


def extract_demonstrations(read_path, save_path):
    demonstration_list = DemonstrationUtils.extract_demonstrations_from_full_file(read_path)
    with open(save_path, 'w') as f:
        json.dump(demonstration_list, f)


if __name__ == "__main__":
    datasets = ['boolq', 'imdb', 'babi', 'nq_cb', 'nq_ob', 'xsum', 'math', 'hellaswag']
    models = ['davinci', 'curie', 'babbage', 'ada']

    for dataset in datasets:
        current_dir = "/".join(os.getcwd().split("/")[:-1])
        save_folder = os.path.join(current_dir, "data/demonstrations_lists/", dataset)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        read_folder = os.path.join(current_dir, "data/raw_completions_datasets/", dataset + "_raw")
        for model in models:
            save_path = os.path.join(save_folder, model + "_demonstrations.json")
            read_path = os.path.join(read_folder, "scenario_state_slim_" + model + ".json")
            extract_demonstrations(read_path, save_path)
