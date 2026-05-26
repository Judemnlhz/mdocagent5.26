from agents.base_agent import Agent
from mydatasets.base_dataset import BaseDataset
from tqdm import tqdm
import importlib
import json
import torch
from typing import List
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class MultiAgentSystem:
    def __init__(self, config):
        self.config = config
        self.agents: List[Agent] = []
        self.models: dict = {}

        for agent_config in self.config.agents:
            model_key = self._model_cache_key(agent_config.model)

            if model_key not in self.models:
                module = importlib.import_module(agent_config.model.module_name)
                model_class = getattr(module, agent_config.model.class_name)
                print("Create model: ", agent_config.model.class_name)
                self.models[model_key] = model_class(agent_config.model)

            self.add_agent(agent_config, self.models[model_key])

        sum_model_key = self._model_cache_key(config.sum_agent.model)

        if sum_model_key not in self.models:
            module = importlib.import_module(config.sum_agent.model.module_name)
            model_class = getattr(module, config.sum_agent.model.class_name)
            self.models[sum_model_key] = model_class(config.sum_agent.model)

        self.sum_agent = Agent(config.sum_agent, self.models[sum_model_key])

    def _model_cache_key(self, model_config):
        model_name = getattr(model_config, "model", None) or getattr(model_config, "model_id", None)
        base_url = getattr(model_config, "base_url", None)

        return (
            model_config.module_name,
            model_config.class_name,
            model_name,
            base_url,
        )

    def add_agent(self, agent_config, model):
        module = importlib.import_module(agent_config.agent.module_name)
        agent_class = getattr(module, agent_config.agent.class_name)
        agent: Agent = agent_class(agent_config, model)
        self.agents.append(agent)

    def predict(self, question, texts, images):
        """Implement the method in the subclass."""
        pass

    def sum(self, sum_question):
        ans, all_messages = self.sum_agent.predict(sum_question)

        def extract_final_answer(agent_response):
            try:
                response_dict = json.loads(agent_response)
                answer = response_dict.get("Answer", None)
                return answer
            except Exception:
                return agent_response

        final_ans = extract_final_answer(ans)
        return final_ans, all_messages

    def _has_valid_answer(self, sample):
        """
        判断一个样本是否已经有有效预测结果。

        JSON 中的 null 读入 Python 后是 None。
        因此：
        - ans_key 不存在：需要补跑
        - ans_key 存在但值是 None：需要补跑
        - ans_key 存在且值不是 None：跳过
        """
        return (
            self.config.ans_key in sample
            and sample[self.config.ans_key] is not None
        )

    def predict_dataset(self, dataset: BaseDataset, resume_path=None):
        samples = dataset.load_data(use_retreival=True)

        if resume_path:
            assert os.path.exists(resume_path), f"resume_path does not exist: {resume_path}"
            with open(resume_path, "r", encoding="utf-8") as f:
                samples = json.load(f)

        if self.config.truncate_len:
            samples = samples[:self.config.truncate_len]

        num_workers = int(getattr(self.config, "num_workers", 1) or 1)

        if num_workers <= 1:
            self._predict_dataset_serial(dataset, samples, resume_path)
        else:
            self._predict_dataset_parallel(dataset, samples, resume_path, num_workers)

    def _predict_dataset_serial(self, dataset: BaseDataset, samples, resume_path=None):
        sample_no = 0

        for sample in tqdm(samples):
            # 只跳过已有有效答案的样本。
            # 缺失 ans_key 或 ans_key 为 None/null 的样本都会被补跑。
            if self._has_valid_answer(sample):
                continue

            final_ans, final_messages = self._predict_sample_with_retries(dataset, sample)

            sample[self.config.ans_key] = final_ans

            if self.config.save_message:
                sample[self.config.ans_key + "_message"] = final_messages

            torch.cuda.empty_cache()
            self.clean_messages()

            sample_no += 1

            if sample_no % self.config.save_freq == 0:
                path = dataset.dump_reults(samples)
                print(f"Save {sample_no} newly processed results to {path}.")

        path = dataset.dump_reults(samples)
        print(f"Save final results to {path}.")

    def _predict_dataset_parallel(self, dataset: BaseDataset, samples, resume_path, num_workers: int):
        # 只提交缺失答案或答案为 None/null 的样本。
        # 已有有效答案的样本不会重复跑，也不会被覆盖。
        pending_items = [
            (index, sample)
            for index, sample in enumerate(samples)
            if not self._has_valid_answer(sample)
        ]

        print(
            f"Parallel prediction with num_workers={num_workers}, "
            f"total={len(samples)}, pending={len(pending_items)}"
        )

        if len(pending_items) == 0:
            path = dataset.dump_reults(samples)
            print(f"No pending samples. Save final results to {path}.")
            return

        sample_no = 0
        worker_state = threading.local()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_index = {
                executor.submit(
                    self._predict_sample_with_thread_local_worker,
                    worker_state,
                    dataset,
                    sample,
                ): index
                for index, sample in pending_items
            }

            for future in tqdm(as_completed(future_to_index), total=len(future_to_index)):
                index = future_to_index[future]

                try:
                    final_ans, final_messages = future.result()
                except Exception as e:
                    print(f"Error predicting sample {index}: {e}")
                    final_ans, final_messages = None, None

                samples[index][self.config.ans_key] = final_ans

                if self.config.save_message:
                    samples[index][self.config.ans_key + "_message"] = final_messages

                sample_no += 1

                if sample_no % self.config.save_freq == 0:
                    path = dataset.dump_reults(samples)
                    print(f"Save {sample_no} newly processed results to {path}.")

        path = dataset.dump_reults(samples)
        print(f"Save final results to {path}.")

    def _predict_sample_with_thread_local_worker(self, worker_state, dataset: BaseDataset, sample):
        """
        每个线程只创建一次自己的 worker。
        线程之间不共享 MDocAgent / agents / messages。
        每条样本结束后清理 messages，避免同一线程内样本之间上下文污染。
        """
        if not hasattr(worker_state, "worker"):
            worker_state.worker = self.__class__(self.config)

        worker = worker_state.worker

        try:
            return worker._predict_sample_with_retries(dataset, sample)
        finally:
            worker.clean_messages()
            torch.cuda.empty_cache()

    def _predict_sample_with_retries(self, dataset: BaseDataset, sample):
        max_retries = int(getattr(self.config, "max_retries", 1) or 1)

        for attempt in range(max_retries):
            try:
                question, texts, images = dataset.load_sample_retrieval_data(sample)
                return self.predict(question, texts, images)

            except RuntimeError as e:
                print(e)

                if "out of memory" in str(e):
                    torch.cuda.empty_cache()

                if attempt == max_retries - 1:
                    return None, None

            except Exception as e:
                print(f"Prediction failed on attempt {attempt + 1}/{max_retries}: {e}")

                if attempt == max_retries - 1:
                    return None, None

            time.sleep(min(2 ** attempt, 30))

        return None, None

    def clean_messages(self):
        for agent in self.agents:
            agent.clean_messages()

        self.sum_agent.clean_messages()