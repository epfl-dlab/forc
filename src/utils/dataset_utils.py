import json
import numpy as np
import random
import tiktoken
import pandas as pd

from helm.benchmark.metrics.basic_metrics import quasi_exact_match, get_rouge_function, exact_match, f1_set_match, exact_match_indicator
from helm.benchmark.metrics.basic_metrics import f1_score as f1_scorer
from helm.benchmark.scenarios.math_scenario import is_equiv


ada_enc = tiktoken.encoding_for_model("ada")
babbage_enc = tiktoken.encoding_for_model("babbage")
curie_enc = tiktoken.encoding_for_model("curie")
davinci_enc = tiktoken.encoding_for_model("davinci")

cost_ada = 0.0004
cost_babbage = 0.0005
cost_curie = 0.0020
cost_davinci = 0.0200


class DatasetUtils(object):

    @staticmethod
    def set_seed(seed):
        DatasetUtils.seed = seed

    @staticmethod
    def train_val_test_split(df, frac_val=0.1, frac_test=0.2):
        test = df.sample(frac=frac_test, axis=0, random_state=DatasetUtils.seed)
        train = df.drop(index=test.index)
        val = train.sample(frac=frac_val, axis=0, random_state=DatasetUtils.seed)
        train = train.drop(index=val.index)
        return train, val, test

    @staticmethod
    def transform_data(df, models, append_name=True, multimodel=False, mode='accuracy'):  # TODO
        text, prompt, target, chosen_model, completions, total_costs, prompt_costs, query_costs, completion_costs = [], [], [], [], [], [], [], [], []
        for row in df.iterrows():
            row = row[1]
            scores, tot_costs, q_costs, comp_costs, p_costs, comps, pmt = [], [], [], [], [], [], []
            for model in models:
                if append_name:
                    text.append("<" + model + ">" + row['query'])
                else:
                    text.append(row['query'])
                if not multimodel:
                    if mode != "length":
                        target.append(row[mode + '_' + model])
                    else:
                        target.append(float(len(row['completion_' + model])))
                    chosen_model.append(model)
                    completions.append(row['completion_' + model])
                    total_costs.append(row['total_cost_' + model])
                    prompt_costs.append(row['prompt_cost_' + model])
                    query_costs.append(row['query_cost_' + model])
                    completion_costs.append(row['completion_cost_' + model])
                    prompt.append(row['prompt_' + model])
                else:
                    if mode != "length":
                        scores.append(row[mode + '_' + model])
                    else:
                        scores.append(float(len(row['completion_' + model])))
                    comps.append(row['completion_' + model])
                    tot_costs.append(row['total_cost_' + model])
                    q_costs.append(row['query_cost_' + model])
                    comp_costs.append(row['completion_cost_' + model])
                    p_costs.append(row['prompt_cost_' + model])
                    pmt.append(row['prompt_' + model])
            if multimodel:
                target.append(scores)
                chosen_model.append("all")
                completions.append(comps)
                total_costs.append(tot_costs)
                query_costs.append(q_costs)
                completion_costs.append(comp_costs)
                prompt_costs.append(p_costs)
                prompt.append(pmt)

        data = list(zip(text, prompt, target, chosen_model, completions, total_costs, query_costs, completion_costs))
        random.shuffle(data)
        text, prompt, target, chosen_model, completions, total_costs, query_costs, completion_costs = zip(*data)
        return list(text), list(prompt), list(target), list(chosen_model), list(completions), list(total_costs), list(
            query_costs), list(prompt_costs), list(completion_costs)

    @staticmethod
    def query_cost_ada(query):
        return len(ada_enc.encode(query)) * cost_ada / 1000.

    @staticmethod
    def query_cost_babbage(query):
        return len(babbage_enc.encode(query)) * cost_babbage / 1000.

    @staticmethod
    def query_cost_curie(query):
        return len(curie_enc.encode(query)) * cost_curie / 1000.

    @staticmethod
    def query_cost_davinci(query):
        return len(davinci_enc.encode(query)) * cost_davinci / 1000.

    @staticmethod
    def remove_model_name_prefix(text):
        end_index = text.index('>') + 1
        return text[end_index:]

    @staticmethod
    def add_cost(df, model):
        query_cost_func = {
            'davinci': DatasetUtils.query_cost_davinci,
            'curie': DatasetUtils.query_cost_curie,
            'babbage': DatasetUtils.query_cost_babbage,
            'ada': DatasetUtils.query_cost_ada,
        }

        func = query_cost_func[model]
        df['query_cost_' + model] = df["query"].apply(func)
        df['completion_cost_' + model] = df["completion_" + model].apply(func)
        df['prompt_cost_' + model] = df['prompt_' + model].apply(func)
        df['total_cost_' + model] = df['completion_cost_' + model] + df['prompt_cost_' + model] # total cost is equal to prompt + completion
        return df

    @staticmethod
    def discretize(score, discretization):
        for idx, disc in enumerate(discretization):
            if score <= disc:
                return idx
        return idx + 1

    @staticmethod
    def get_score_function(dataset):
        indicator = None
        if dataset == "boolq" or dataset == "imdb" or dataset == "babi" or dataset == "lsat" or dataset == "legal_support" or dataset.startswith(
                "wikifact") or dataset.startswith("raft") or dataset.startswith(
                "entity_matching") or dataset.startswith("data_imputation") or dataset == "civil_comments":
            score_func = quasi_exact_match
        elif dataset == "nq_cb" or dataset == "nq_ob" or dataset == "narrativeqa" or dataset == "quac":
            score_func = f1_scorer
        elif dataset == "xsum" or dataset == "cnndm":
            score_func = get_rouge_function("rouge2")
        elif dataset == "math":
            score_func = is_equiv
        elif dataset == "hellaswag" or dataset == "openbookqa" or dataset == "truthfulqa" or dataset.startswith(
                "mmlu") or dataset.startswith("blimp"):
            score_func = exact_match
        elif dataset == "synth_easy":
            score_func = f1_set_match
        elif dataset == "gsm8k":
            score_func = exact_match_indicator
            indicator = "The answer is"
        else:
            raise NotImplementedError
        return score_func, indicator

    @staticmethod
    def find_correct_answer(outputs, logprobs):
        max_logprob = np.NINF
        max_out = None
        for out, logpr in zip(outputs, logprobs):
            if logpr > max_logprob:
                max_out = out
                max_logprob = logpr
        return max_out

    @staticmethod
    def extract_correct_gts_and_output(references, output_mapping, output):
        if output_mapping is not None:
            try:
                output = output_mapping[output.strip(" ")]
            except:
                pass
        gts = []
        for i in range(len(references)):
            if 'correct' in references[i]["tags"]:
                gts.append(references[i]["output"])
        return output, gts

    @staticmethod
    def calculate_scores(output, gts, inference_time, modes, scores, efficiency, score_func, indicator, discretization):
        if 'accuracy' in modes:
            score = 0
            for gt in gts:
                if indicator is not None:
                    score_curr = score_func(gt, output, indicator)
                else:
                    score_curr = score_func(gt, output)
                if score_curr > score:
                    score = score_curr
            if discretization is not None:
                score = DatasetUtils.discretize(score, discretization)
            scores.append(float(score))
        if 'efficiency' in modes:
            efficiency.append(float(inference_time))
        return scores, efficiency

    @staticmethod
    def completion_aggregation_function(group, column):
        score = round(group[column].mean())
        return group.loc[group[column] == score, 'completion'].iloc[0]

    @staticmethod
    def aggregate_scores(queries, prompts, scores, efficiency, completions): #TODO clean and fix the prompt thing
        df_dict = {"query": queries, "prompt":prompts, "completion": completions}
        if len(scores) > 0:
            df_dict['accuracy'] = scores
        if len(efficiency) > 0:
            df_dict['efficiency'] = efficiency
        df = pd.DataFrame(df_dict)
        new_df = df.groupby('query', sort=False).apply(DatasetUtils.completion_aggregation_function, ('accuracy')).reset_index()
        new_df.columns = ['query', 'prompt', 'completion']
        accuracy = df.groupby('query', sort=False)['accuracy'].apply(lambda x: x.mode()[0]).reset_index()
        result = pd.merge(new_df, accuracy, on="query")

        if len(efficiency) > 0:
            eff = df.groupby('query', sort=False)['efficiency'].mean().reset_index()
            result = pd.merge(result, eff, on='query')
            return result['query'].tolist(), result['prompt'].tolist(), {'accuracy': result['accuracy'].tolist(),
                                              'efficiency': result['efficiency'].tolist()}, result[
                       'completion'].tolist()
        else:
            return result['query'].tolist(), result['prompt'].tolist(), {'accuracy': result['accuracy'].tolist()}, result['completion'].tolist()

    @staticmethod
    def get_input_output_affixes(sample, params):
        sample["output_prefix"] = params.get("output_prefix", "")
        sample["input_prefix"] = params.get("input_prefix", "")
        sample["output_suffix"] = params.get("output_suffix", "")
        sample["input_suffix"] = params.get("input_suffix", "")
        sample["context_prefix"] = params.get("context_prefix", "")
        sample["context_suffix"] = params.get("context_suffix", "")
        return sample


class DemonstrationUtils(object):

    @staticmethod
    def get_adapter_spec(json_data):
        return json_data["adapter_spec"]

    @staticmethod
    def find_longest_common_substring(strings):
        if not strings:
            return ""

        shortest_string = min(strings, key=len)
        longest_common_substring = ""

        for i in range(len(shortest_string)):
            if all(s.startswith(shortest_string[:i + 1]) for s in strings):
                longest_common_substring = shortest_string[:i + 1]
            else:
                break

        return longest_common_substring

    @staticmethod
    def get_prompt(json_data, i=0):
        return json_data["request_states"][i]["request"]["prompt"] #expecting the same demonstrations for each samples

    @staticmethod
    def extract_substrings(text, prefix, suffix):
        substrings = []
        start = 0
        while True:
            start = text.find(prefix, start)
            if start == -1:
                break
            end = text.find(suffix, start + len(prefix))
            if end == -1:
                break
            substrings.append(text[start + len(prefix): end])
            start = end + len(suffix)
        return substrings

    @staticmethod
    def get_demonstration_list(adapter_spec, prompt):
        inputs = DemonstrationUtils.extract_substrings(prompt, adapter_spec["input_prefix"], adapter_spec["input_suffix"] + adapter_spec["output_prefix"])
        outputs = DemonstrationUtils.extract_substrings(prompt, adapter_spec["input_suffix"] + adapter_spec["output_prefix"], adapter_spec["output_suffix"])
        demonstrations = []
        for dem_input, dem_output in zip(inputs, outputs):
            demonstrations.append({"input":dem_input, "output":dem_output})
        return demonstrations

    @staticmethod
    def extract_prompt_from_full_file(path):
        with open(path, 'r') as f:
            json_data = json.load(f)
        prompts = []
        for i in range(20):
            prompts.append(DemonstrationUtils.get_prompt(json_data, i))
        prompt = DemonstrationUtils.find_longest_common_substring(prompts)
        return prompt

    @staticmethod
    def extract_demonstrations_from_full_file(path):
        with open(path, 'r') as f:
            json_data = json.load(f)
        adapter_spec = DemonstrationUtils.get_adapter_spec(json_data)
        prompt = DemonstrationUtils.get_prompt(json_data)
        demonstration_list = DemonstrationUtils.get_demonstration_list(adapter_spec, prompt)
        return demonstration_list

    @staticmethod
    def process_demonstration_query(query, adapter_spec, sample = None):
        if sample is None:
            return adapter_spec["input_prefix"] + query + adapter_spec["input_suffix"]
        else:
            output_txt = ""
            if "context" in sample:
                output_txt += sample["context_prefix"] + sample["context"] + sample["context_suffix"]
            return output_txt + sample["input_prefix"] + query + sample["input_suffix"] + sample["output_prefix"]

    @staticmethod
    def process_demonstration_target(target, adapter_spec):
        return adapter_spec["output_prefix"] + target + adapter_spec["output_suffix"]
