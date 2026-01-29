# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import json
from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

        self.step_cnt = 0

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        action_or_attn_mask = (
            data.batch["action_mask"] if "action_mask" in data.batch.keys() else data.batch["attention_mask"]
        )
        if "env_reward" in data.batch.keys():
            reward_tensor += data.batch["env_reward"]
            print(
                f" [DEBUG reward] mean={reward_tensor.mean().item()}, min={reward_tensor.min().item()}, max={reward_tensor.max().item()}"
            )

        already_print_data_sources = {}
        # breakpoint()
        # import debugpy
        # debugpy.listen(("0.0.0.0", 5678))
        # print("Waiting for VS Code debugger attach...")
        # debugpy.wait_for_client()
        # print("Debugger attached, continue training")
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            if data_item.batch["tool_cnt"][0].item() > 1.5:
                # breakpoint()
                pass

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
            # if response_str.count('</tool_response>') >=3:
            #     breakpoint()
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            extra_info = extra_info | {"tool_cnt": data_item.batch["tool_cnt"].item()}

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] += reward

            # eos_idx = torch.nonzero(action_or_attn_mask[i, prompt_length: prompt_length + valid_response_length])[-1]
            # reward_tensor[i, eos_idx] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

            self.step_cnt += 1

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


class NaiveLLMRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

        self.step_cnt = 0

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        action_or_attn_mask = (
            data.batch["action_mask"] if "action_mask" in data.batch.keys() else data.batch["attention_mask"]
        )
        if "env_reward" in data.batch.keys():
            reward_tensor += data.batch["env_reward"]
            print(
                f" [DEBUG reward] mean={reward_tensor.mean().item()}, min={reward_tensor.min().item()}, max={reward_tensor.max().item()}"
            )

        already_print_data_sources = {}
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from more_itertools import chunked

        data_records = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            extra_info = extra_info | {"tool_cnt": data_item.batch["tool_cnt"].item()}

            data_records.append(
                {
                    "index": i,
                    "data_source": data_source,
                    "response_str": response_str,
                    "ground_truth": ground_truth,
                    "extra_info": extra_info,
                    "valid_response_length": valid_response_length,
                    "prompt_str": prompt_str,
                }
            )
        import time

        def _worker(record):
            time.sleep(0.1)
            score = self.compute_score(
                data_source=record["data_source"],
                solution_str=record["response_str"],
                ground_truth=record["ground_truth"],
                extra_info=record["extra_info"],
            )
            return score

        import traceback

        import tenacity

        with ThreadPoolExecutor(max_workers=8) as executor:
            index2score = {index: None for index in range(len(data))}
            fut2index = dict()
            futs = []
            for index in range(len(data)):
                fut = executor.submit(_worker, data_records[index])
                fut2index[fut] = index
                futs.append(fut)

            for fut in as_completed(futs):
                index = fut2index[fut]
                try:
                    score = fut.result()
                    index2score[index] = score
                except Exception as e:
                    print(
                        f"[ERROR] Processing item in ThreadPool ({e}): {data_records[index]['data_source']},{data_records[index]['response_str']},{data_records[index]['ground_truth']},{data_records[index]['extra_info']}"
                    )
                    # tenacity: 打印最后一次尝试的真实异常 + traceback
                    if isinstance(e, tenacity.RetryError):
                        last_exc = e.last_attempt.exception()
                        print(f"[ERROR] tenacity last exception: {repr(last_exc)}")
                        print("".join(traceback.format_exception(type(last_exc), last_exc, last_exc.__traceback__)))
                    else:
                        print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
                    target_reward_terms = ["score", "acc", "format", "tool", "cof", "process"]
                    index2score[index] = {k: 0.0 for k in target_reward_terms}
        for i in range(len(data)):
            record = data_records[i]
            data_source, response_str, ground_truth, extra_info, valid_response_length, prompt_str = (
                record["data_source"],
                record["response_str"],
                record["ground_truth"],
                record["extra_info"],
                record["valid_response_length"],
                record["prompt_str"],
            )
            score = index2score[i]

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] += reward

            # eos_idx = torch.nonzero(action_or_attn_mask[i, prompt_length: prompt_length + valid_response_length])[-1]
            # reward_tensor[i, eos_idx] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

            self.step_cnt += 1

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
