from src.utils import DatasetUtils
import json
import os
import pandas as pd
import random
import numpy as np
import torch

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
reasoning_datasets = ['babi', 'math', 'gsm8k', 'lsat', 'legal_support']
toxicity_detection_datasets = ['civil_comments']

datasets = mmlu_datasets + raft_datasets + wikifact_datasets + other_qa_datasets + sent_analysis_datasets + entity_matching_datasets + data_imputation_datasets + reasoning_datasets + toxicity_detection_datasets
models = ['davinci', 'curie', 'babbage', 'ada']
modes = ['accuracy']


def read_default_instances(data, score_func, indicator, discretization, modes, train_trials): #TODO maybe remove completions[0] - take the best one instead?
    queries, prompts, scores, efficiency, completions = [], [], [], [], []
    for obj in data["request_states"]:
        if 'perturbation' in obj["instance"]:
            continue
        if not train_trials:
            if obj['train_trial_index'] != 0:
                continue
        if "output_mapping" in obj:
            output_mapping = obj["output_mapping"]
        else:
            output_mapping = None
        query = obj["instance"]["input"]
        prompt = obj["request"]["prompt"]
        output = obj["result"]["completions"][0]["text"]
        inference_time = obj['result']["request_time"]
        references = obj["instance"]["references"]
        output, gts = DatasetUtils.extract_correct_gts_and_output(references, output_mapping, output)
        scores, efficiency = DatasetUtils.calculate_scores(output, gts, inference_time, modes, scores, efficiency, score_func, indicator, discretization)
        queries.append(query)
        prompts.append(prompt)
        completions.append(output)
    if not train_trials:
        all_scores = {}
        if len(scores) > 0:
            all_scores['accuracy'] = scores
        if len(efficiency) > 0:
            all_scores['efficiency'] = efficiency
    else:
        queries, prompts, all_scores, completions = DatasetUtils.aggregate_scores(queries, prompts, scores, efficiency, completions)

    return queries, prompts, all_scores, completions


def read_grouped_instances(data, score_func, indicator, discretization, modes, train_trials): # todo do not use with efficiency for now, check the output_mapping thing
    queries, scores, prompts, efficiency, outputs, logprobs, completions = [], [], [], [], [], [], []
    query = ""
    prompt = ""
    old_references = ""
    for obj in data["request_states"]:
        gts = []
        if 'perturbation' in obj["instance"]:
            continue
        if not train_trials:
            if obj['train_trial_index'] != 0:
                continue
        if "output_mapping" in obj:
            output_mapping = obj["output_mapping"]
        else:
            output_mapping = None
        output = obj["result"]["completions"][0]["text"]
        logprob = obj["result"]["completions"][0]["logprob"]
        inference_time = obj['result']["request_time"]
        references = obj["instance"]["references"]
        outputs.append(output)
        logprobs.append(logprob)

        if references == old_references:
            prompt = prompt + obj["request"]["prompt"]
            query = query + " or " + output
            for i in range(len(references)):
                if 'correct' in references[i]["tags"]:
                    gts.append(references[i]["output"])
            out = DatasetUtils.find_correct_answer(outputs, logprobs)
            scores, efficiency = DatasetUtils.calculate_scores(out, gts, inference_time, modes, scores, efficiency, score_func, indicator, discretization)
            queries.append(query)
            completions.append(out)
            prompts.append(prompt)
        else:
            query = output
            prompt = obj["request"]["prompt"]
            outputs = [output]
            logprobs = [logprob]
        old_references = references
    if not train_trials:
        all_scores = {}
        if len(scores) > 0:
            all_scores['accuracy'] = scores
        if len(efficiency) > 0:
            all_scores['efficiency'] = efficiency
    else:
        queries, prompts, all_scores, completions = DatasetUtils.aggregate_scores(queries, prompts, scores, efficiency, completions)

    return queries, prompts, all_scores, completions


def read_data(models, path, dataset, discretization=None, modes=['accuracy', 'efficiency'], train_trials=False):
    score_func, indicator = DatasetUtils.get_score_function(dataset)
    df = pd.DataFrame()
    flag = False
    for model in models:
        with open(os.path.join(path, "scenario_state_slim_" + model +".json"), 'r') as f:
            data = json.load(f)
        if dataset.startswith("blimp"):
            queries, prompts, scores, completions = read_grouped_instances(data, score_func, indicator, discretization, modes, train_trials)
        else:
            queries, prompts, scores, completions = read_default_instances(data, score_func, indicator, discretization, modes, train_trials)
        if not flag:
            df['query'] = queries
            # df['prompt'] = prompts
        df['prompt_' + model] = prompts
        for key, val in scores.items():
            df[key + "_" + model] = val
        df['completion_' + model] = completions
        flag = True
    for model in models:
        df = DatasetUtils.add_cost(df, model)
    return df


def process_data(df_list, dataset_list, multimodel=False, append_name=True, mode='accuracy'): # TODO labels not encoded
    train_texts, train_labels, train_prompts, train_models, train_datasets, train_completions, train_total_costs, train_query_costs, train_prompt_costs, train_completion_costs = [], [], [], [], [], [], [], [], [], []
    val_texts, val_labels, val_prompts, val_models, val_datasets, val_completions, val_total_costs, val_query_costs, val_prompt_costs, val_completion_costs = [], [], [], [], [], [], [], [], [], []
    test_texts, test_labels, test_prompts, test_models, test_datasets, test_completions, test_total_costs, test_query_costs, test_prompt_costs, test_completion_costs = [], [], [], [], [], [], [], [], [], []

    append_name = append_name if not multimodel else False
    for df, dat in zip(df_list, dataset_list):
        df_train, df_val, df_test = DatasetUtils.train_val_test_split(df, 0.1, 0.15)
        train_text, train_prompt, train_target, train_model, train_completion, train_total_cost, train_query_cost, train_prompt_cost, train_completion_cost = DatasetUtils.transform_data(df_train, models, append_name, multimodel, mode)
        val_text, val_prompt, val_target, val_model, val_completion, val_total_cost, val_query_cost, val_prompt_cost, val_completion_cost = DatasetUtils.transform_data(
            df_val, models, append_name, multimodel, mode)
        test_text, test_prompt, test_target, test_model, test_completion, test_total_cost, test_query_cost, test_prompt_cost, test_completion_cost = DatasetUtils.transform_data(
            df_test, models, append_name, multimodel, mode)
        train_texts, val_texts, test_texts = train_texts + train_text, val_texts + val_text, test_texts + test_text
        train_labels, val_labels, test_labels = train_labels + train_target, val_labels + val_target, test_labels + test_target
        train_prompts, val_prompts, test_prompts = train_prompts + train_prompt, val_prompts + val_prompt, test_prompts + test_prompt
        train_models, val_models, test_models = train_models + train_model, val_models + val_model, test_models + test_model
        train_datasets, val_datasets, test_datasets = train_datasets + [dat] * len(train_text), val_datasets + [dat] * len(val_text),test_datasets + [dat] * len(test_text)
        train_completions, val_completions, test_completions = train_completions + train_completion, val_completions + val_completion, test_completions + test_completion
        train_total_costs, val_total_costs, test_total_costs = train_total_costs + train_total_cost, val_total_costs + val_total_cost, test_total_costs + test_total_cost
        train_query_costs, val_query_costs, test_query_costs = train_query_costs + train_query_cost, val_query_costs + val_query_cost, test_query_costs + test_query_cost
        train_prompt_costs, val_prompt_costs, test_prompt_costs = train_prompt_costs + train_prompt_cost, val_prompt_costs + val_prompt_cost, test_prompt_costs + test_prompt_cost
        train_completion_costs, val_completion_costs, test_completion_costs = train_completion_costs + train_completion_cost, val_completion_costs + val_completion_cost, test_completion_costs + test_completion_cost

    data = list(
        zip(train_texts, train_labels, train_models, train_datasets, train_prompts, train_completions, train_total_costs,
            train_query_costs, train_completion_costs, train_prompt_costs))
    random.shuffle(data)
    train_texts, train_labels, train_models, train_datasets, train_prompts, train_completions, train_total_costs, train_query_costs, train_completion_costs, train_prompt_costs = zip(
        *data)
    train_texts = list(train_texts)
    train_labels = list(train_labels)
    train_models = list(train_models)
    train_datasets = list(train_datasets)
    train_prompts = list(train_prompts)
    train_completions = list(train_completions)
    train_total_costs = list(train_total_costs)
    train_query_costs = list(train_query_costs)
    train_completion_costs = list(train_completion_costs)
    train_prompt_costs = list(train_prompt_costs)

    # if not multimodel:
    #     train_labels = encoder.fit_transform(np.array(train_labels).reshape(-1, 1))
    #     val_labels = encoder.transform(np.array(val_labels).reshape(-1,1))
    #     test_labels = encoder.transform(np.array(test_labels).reshape(-1,1))
    # else:
    #     train_labels = encoder.fit_transform(np.array(train_labels))
    #     val_labels = encoder.transform(np.array(val_labels))
    #     test_labels = encoder.transform(np.array(test_labels))
    #
    # train_labels = train_labels.todense()
    # train_labels = np.array(train_labels).squeeze()
    # val_labels = val_labels.todense()
    # val_labels = np.array(val_labels).squeeze()
    # test_labels = test_labels.todense()
    # test_labels = np.array(test_labels).squeeze()

    df_train = pd.DataFrame(
        {'query': train_texts, 'prompt': train_prompts, 'label': train_labels, 'completion': train_completions,
         'model': train_models, 'dataset': train_datasets, 'query_cost': train_query_costs,
         'prompt_cost': train_prompt_costs, 'completion_cost': train_completion_costs, 'total_cost': train_total_costs})
    df_val = pd.DataFrame(
        {'query': val_texts, 'prompt': val_prompts, 'label': val_labels, 'completion': val_completions,
         'model': val_models, 'dataset': val_datasets, 'query_cost': val_query_costs,
         'prompt_cost': val_prompt_costs, 'completion_cost': val_completion_costs, 'total_cost': val_total_costs})
    df_test = pd.DataFrame(
        {'query': test_texts, 'prompt': test_prompts, 'label': test_labels, 'completion': test_completions,
         'model': test_models, 'dataset': test_datasets, 'query_cost': test_query_costs,
         'prompt_cost': test_prompt_costs, 'completion_cost': test_completion_costs, 'total_cost': test_total_costs})

    return df_train, df_val, df_test


def save_data(df, save_path):
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    DatasetUtils.set_seed(seed)
    current_dir = "/".join(os.getcwd().split("/")[:-1])
    save_folder = os.path.join(current_dir, "data/processed_data_full/")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    dfs = []
    curr_path = os.path.join(current_dir, "data", "raw_completions_datasets")
    for dataset in datasets:
        print(dataset)
        df = read_data(models, os.path.join(curr_path, dataset + "_raw/"), dataset, modes=modes)
        dfs.append(df)

    df_train, df_val, df_test = process_data(df_list=dfs, dataset_list=datasets)
    save_data(df_train, os.path.join(save_folder, "df_train.csv"))
    save_data(df_val, os.path.join(save_folder, "df_val.csv"))
    save_data(df_test, os.path.join(save_folder, "df_test.csv"))
