from models.base_model import BaseModel
from mydatasets.base_dataset import BaseDataset
import os
from typing import Dict, Union
import json
import pandas as pd
from tqdm import tqdm
import re
import importlib

class Agent:
    def __init__(self, config, model=None):
        self.config = config
        self.messages = None
        if model is not None:
            self.model:BaseModel = model
        else:
            module = importlib.import_module(self.config.model.module_name)
            model_class = getattr(module, self.config.model.class_name)
            print("Create model: ", self.config.model.class_name)
            self.model = model_class(self.config.model)
    
    def clean_messages(self):
        self.messages = None
        
    def _predict(self, question, texts=None, images=None, add_to_message = False):
        if not self.config.agent.use_text:
            texts = None
        if not self.config.agent.use_image:
            images = None
        generated_ans, messages = self.model.predict(question, texts, images, self.messages)
        if add_to_message:
            self.messages = messages
        return generated_ans, messages
    
    def predict(self, question, texts=None, images=None, with_sys_prompt=True):
        if with_sys_prompt:
            question = self.config.agent.system_prompt + question
        return self._predict(question, texts, images, add_to_message = True)
    
    def self_reflect(self, prompt=None, add_to_message = True):
        if prompt is None:
            self_reflect_prompt = self.config.agent.self_reflect_prompt
        else:
            self_reflect_prompt = prompt
        
        generated_ans, messages = self._predict(question = self_reflect_prompt)
        if add_to_message:
            self.messages = messages
        
        return generated_ans
    
    def eval(self, question, answer, gt):
        prompt = self.config.agent.eval_system_prompt.format(question=question, answer=answer, gt=gt)
        try:
            generated_ans, _ = self.model.predict(prompt)
            result = extract_evaluation_metrics(generated_ans)
            return result
        except Exception as e:
            print(f"Error evaluating answer: {str(e)}")
            return {"binary_correctness": 0}

    def _load_eval_gold_lookup(self, dataset: BaseDataset):
        sample_path = dataset.config.sample_path
        gt_key = dataset.config.gt_key
        question_key = dataset.config.question_key
        try:
            with open(sample_path, "r", encoding="utf-8") as file_obj:
                gold_samples = json.load(file_obj)
        except Exception as exc:
            print(f"Could not load evaluation gold samples from {sample_path}: {exc}")
            return {}, {}

        lookup = {}
        offsets = {}
        for gold_sample in gold_samples:
            key = (gold_sample.get("doc_id"), gold_sample.get(question_key))
            if key[0] is None or key[1] is None or gt_key not in gold_sample:
                continue
            lookup.setdefault(key, []).append(gold_sample[gt_key])
        return lookup, offsets

    def _resolve_eval_ground_truth(self, dataset: BaseDataset, sample, gold_lookup, gold_offsets):
        gt_key = dataset.config.gt_key
        if gt_key in sample:
            return sample[gt_key]

        key = (sample.get("doc_id"), sample.get(dataset.config.question_key))
        candidates = gold_lookup.get(key, [])
        if not candidates:
            raise KeyError(gt_key)

        offset = gold_offsets.get(key, 0)
        gold_offsets[key] = offset + 1
        if offset < len(candidates):
            return candidates[offset]
        return candidates[-1]

    def eval_dataset(self, dataset: BaseDataset):
        samples, ans_path = dataset.load_latest_results()
        if self.config.truncate_len:
            samples = samples[:self.config.truncate_len]

        gold_lookup, gold_offsets = self._load_eval_gold_lookup(dataset)
        samples_with_answer = []
        for sample in tqdm(samples):
            try:
                question = sample[dataset.config.question_key]
                answer = sample[self.config.ans_key]
                gt = self._resolve_eval_ground_truth(dataset, sample, gold_lookup, gold_offsets)
                result = self.eval(question, answer, gt)
                sample['binary_correctness'] = result.get('binary_correctness', None)
                samples_with_answer.append(sample)
            except Exception as e:
                print(f"Error evaluating sample: {str(e)}")
                
        ans_file_path_name = ans_path[:-5]+"_results.json"
        with open(ans_file_path_name, "w") as file:
            json.dump(samples_with_answer, file, indent=4)
            
        samples_with_answer = pd.DataFrame(samples_with_answer)
        path = os.path.join(dataset.config.result_dir,"results.txt")
        with open(path, "a") as file:
            file.write("\nEvaluation Results Summary:\n")
            file.write(f"Result file: {ans_path}\n")
            file.write(f"Average Binary Correctness: {samples_with_answer['binary_correctness'].mean():.3f}\n")
        
        print(f"Save results to {path}.")

def extract_evaluation_metrics(eval_str: str) -> Dict[str, Union[float, int]]:
    try:
        start_index = eval_str.find('{') 
        end_index = eval_str.rfind('}') + 1 
        eval_str = eval_str[start_index:end_index]
        metrics = json.loads(eval_str)
        return {
            'binary_correctness': int(metrics.get('binary_correctness', 0))
        }
    except json.JSONDecodeError as e:
        return {
            'binary_correctness': 0
        }
    except Exception as e:
        return {
            'binary_correctness': 0
        }
