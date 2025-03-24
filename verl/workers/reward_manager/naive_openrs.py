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

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.utils.reward_score.math_dapo import verify as dapo_verify, extract_boxed_text
import torch
import math
from functools import partial
from collections import defaultdict
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse as math_parse, verify as math_verify
import re


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    
    def count_tags(text: str) -> float:
        count = 0.0
        # We only count </think> tag, because <think> tag is available in system prompt
        if text.count("\n</think>\n") == 1:
            count += 1.0
        return count

    return [{"reward": count_tags(c)} for c in completions]


def cosine_scaled_reward(
    completions,
    solution,
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    """Reward function that scales based on completion length using a cosine schedule.

    Shorter correct solutions are rewarded more than longer ones.
    Longer incorrect solutions are penalized less than shorter ones.

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    This function is parameterized by the following arguments:
        min_value_wrong: Minimum reward for wrong answers
        max_value_wrong: Maximum reward for wrong answers
        min_value_correct: Minimum reward for correct answers
        max_value_correct: Maximum reward for correct answers
        max_len: Maximum length for scaling
    """
    rewards = []
    for content, sol in zip(completions, solution):
        gold_parsed = math_parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_parsed = math_parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        # equations=True,  # equations is deprecated, as it handled by the parser now
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        is_correct_mv = math_verify(answer_parsed, gold_parsed) if len(gold_parsed) > 0 else False
        is_correct_custom, answer_text = dapo_verify(content, str(sol), strict_box_verify=True)
        is_correct_raw = math_verify(math_parse(answer_text), math_parse(str(sol)))
        # print(sol, answer_parsed, answer_text, is_correct_mv, is_correct_custom, is_correct_raw, content[-50:])

        is_correct = is_correct_mv or is_correct_custom or is_correct_raw  # Just a hack to parse answers multiple ways
        gen_len = len(content)

        # Apply cosine scaling based on length
        progress = gen_len / max_len
        cosine = math.cos(progress * math.pi)

        if is_correct:
            min_value = min_value_correct
            max_value = max_value_correct
        else:
            # Swap min/max for incorrect answers
            min_value = max_value_wrong
            max_value = min_value_wrong

        reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
        rewards.append({
            "reward": float(reward),
            "acc": 1 if is_correct else 0,
            "pred": answer_text,
        })

    return rewards


class NaiveRewardManagerOpenRS:
    """The reward manager.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key='data_source',
        max_resp_len=None,
        overlong_buffer_cfg=None,
        group_reward_cfg=[
            {
                "reward_fn": "format",
                "weight": 1.0,
                "reward_fn_kwargs": {},
            },
            {
                "reward_fn": "cosine",
                "weight": 2.0,
                "reward_fn_kwargs": {
                    "min_value_wrong": -1.0,
                    "max_value_wrong": -0.5,
                    "min_value_correct": 0.5,
                    "max_value_correct": 1.0,
                    "max_len": None,
                },
            },
        ]
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.group_reward_cfg = group_reward_cfg
        for reward_fn_cfg in self.group_reward_cfg:
            if reward_fn_cfg["reward_fn"] == "cosine":
                assert self.max_resp_len is not None, f"max_resp_len must be provided if cosine_scaled_reward is used, but got None"
                assert "reward_fn_kwargs" in reward_fn_cfg, f"reward_fn_kwargs must be provided for cosine_scaled_reward"
                assert "max_len" in reward_fn_cfg["reward_fn_kwargs"], f"max_len must be provided for cosine_scaled_reward"
                reward_fn_cfg["reward_fn_kwargs"]["max_len"] = self.max_resp_len
            elif reward_fn_cfg["reward_fn"] == "format":
                pass
            else:
                raise NotImplementedError(f"Unknown reward_fn: {reward_fn_cfg['reward_fn']}")

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def _compute_rewards(self, solution_strs, ground_truths):
        assert len(solution_strs) == len(ground_truths)
        final_rewards = [0 for _ in range(len(solution_strs))]
        accs = [0 for _ in range(len(solution_strs))]
        preds = ["" for _ in range(len(solution_strs))]

        for group_reward in self.group_reward_cfg:
            reward_fn = {
                "format": format_reward,
                "cosine": cosine_scaled_reward,
            }[group_reward["reward_fn"]]
            weight = group_reward["weight"]
            reward_fn_kwargs = group_reward["reward_fn_kwargs"]
            returns = reward_fn(completions=solution_strs, solution=ground_truths, **reward_fn_kwargs)
            assert len(returns) == len(final_rewards)
            for i in range(len(returns)):
                final_rewards[i] += weight * returns[i]["reward"]
                if "acc" in returns[i]:
                    accs[i] = returns[i]["acc"]  # a bit hacky, but we assume that the accuracy is the same for all reward functions
                if "pred" in returns[i]:
                    preds[i] = returns[i]["pred"]  # a bit hacky, but we assume that the prediction is the same for all reward functions
        
        returns = []
        for reward, pred, acc in zip(final_rewards, preds, accs):
            returns.append({
                "reward": reward,
                "acc": acc,
                "pred": pred,
            })
        assert len(returns) == len(solution_strs)
        return returns

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        reward_extra_metrics = defaultdict(list)

        already_print_data_sources = {}
        per_item_vars = [dict() for _ in range(len(data))]  # each item in the batch has its own dict

        for i in range(len(data)):
            _data_item = data[i]  # DataProtoItem
            _prompt_ids = _data_item.batch['prompts']
            _prompt_length = _prompt_ids.shape[-1]
            _valid_prompt_length = _data_item.batch['attention_mask'][:_prompt_length].sum()
            _valid_prompt_ids = _prompt_ids[-_valid_prompt_length:]

            _response_ids = _data_item.batch['responses']
            _valid_response_length = _data_item.batch['attention_mask'][_prompt_length:].sum()
            _valid_response_ids = _response_ids[:_valid_response_length]
            per_item_vars[i]['valid_response_length'] = _valid_response_length

            # decode
            per_item_vars[i]["prompt_str"] = self.tokenizer.decode(_valid_prompt_ids)
            _response_str = self.tokenizer.decode(_valid_response_ids)
            eos_token = self.tokenizer.eos_token
            if _response_str.endswith(eos_token):
                _response_str = _response_str[:-len(eos_token)]
            per_item_vars[i]["response_str"] = _response_str
            per_item_vars[i]["ground_truth"] = _data_item.non_tensor_batch['reward_model']['ground_truth']
            per_item_vars[i]["data_source"] = _data_item.non_tensor_batch[self.reward_fn_key]
            per_item_vars[i]["extra_info"] = _data_item.non_tensor_batch.get('extra_info', None)

        solution_strs = [per_item_vars[i]["response_str"] for i in range(len(data))]
        ground_truths = [per_item_vars[i]["ground_truth"] for i in range(len(data))]
        reward_prec_acc_list = self._compute_rewards(solution_strs, ground_truths)
        # print(reward_prec_acc_list)

        for i in range(len(data)):
            data_source = per_item_vars[i]["data_source"]
            prompt_str = per_item_vars[i]["prompt_str"]
            ground_truth = per_item_vars[i]["ground_truth"]
            response_str = per_item_vars[i]["response_str"]
            valid_response_length = per_item_vars[i]["valid_response_length"]
            result = reward_prec_acc_list[i]

            reward: float
            if isinstance(result, dict):
                assert "reward" in result
                reward = result["reward"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                reward = result

            final_reward = reward

            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                exceed_len = valid_response_length - (self.max_resp_len - overlong_buffer_len)
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                final_reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)
            reward_extra_metrics["acc"].append(result["acc"])
            reward_extra_metrics["generation_len"].append(valid_response_length)

            reward_tensor[i, valid_response_length - 1] = final_reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[reward]", result)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "reward_extra_metrics": reward_extra_metrics,
            }
        else:
            return reward_tensor
