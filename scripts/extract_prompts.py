from src.utils import DemonstrationUtils
import os
import json
import sys

mmlu_datasets = ['mmlu_algebra', 'mmlu_uspolicy', 'mmlu_security', 'mmlu_econometrics', 'mmlu_chemistry']
raft_datasets = ['raft_ade', 'raft_banking', 'raft_neurips', 'raft_onestop', 'raft_overruling',
                 'raft_safety_research', 'raft_semiconductor', 'raft_systematic_review', 'raft_terms_service',
                 'raft_twitter_complaints', 'raft_twitter_hate']
wikifact_datasets = ['wikifact_medical', 'wikifact_birthplace', 'wikifact_currency', 'wikifact_instanceof',
                     'wikifact_discover', 'wikifact_partof', 'wikifact_plaintiff', 'wikifact_position',
                     'wikifact_symptoms', 'wikifact_author']
other_qa_datasets = ['boolq', 'truthfulqa']
sent_analysis_datasets = ['imdb']
language_datasets = ['blimp_binding', 'blimp_irregular_forms', 'blimp_island', 'blimp_quantifiers']
entity_matching_datasets = ['entity_matching_beer', 'entity_matching_buy', 'entity_matching_itunes']
data_imputation_datasets = ['data_imputation_buy', 'data_imputation_restaurant']
reasoning_datasets = ['babi', 'math', 'gsm8k', 'lsat']

datasets = mmlu_datasets + raft_datasets + wikifact_datasets + other_qa_datasets + sent_analysis_datasets + language_datasets + entity_matching_datasets + data_imputation_datasets + reasoning_datasets
models = ['davinci', 'curie', 'babbage', 'ada']


def extract_prompts(read_path, save_path):
    prompt = DemonstrationUtils.extract_prompt_from_full_file(read_path)
    with open(save_path, 'w') as f:
        json.dump(prompt, f)


if __name__ == "__main__":
    for dataset in datasets:
        current_dir = "/".join(os.getcwd().split("/")[:-1])
        save_folder = os.path.join(current_dir, "data/prompts/", dataset)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        read_folder = os.path.join(current_dir, "data/raw_completions_datasets/", dataset + "_raw")
        for model in models:
            save_path = os.path.join(save_folder, model + "_prompt.json")
            read_path = os.path.join(read_folder, "scenario_state_slim_" + model + ".json")
            extract_prompts(read_path, save_path)