# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import random
from dataclasses import dataclass
from typing import Dict, List

import einops
import numpy as np
import torch
import tree

from oat.args import OATArgs
from oat.rm import uncertainty
from oat.rm.backbone import RMBackbone
from oat.rm.model import RewardModel
from oat.types import Metric
from oat.utils.data import get_tokenizer, _preprocess_preference_data, zero_pad_sequences
from torch.utils.data import Dataset, DataLoader


@dataclass
class ExplorationResults:
    dueling_candidates: Dict[int, List[str]]
    candidate_features: torch.Tensor
    init_clash: List[bool]
    is_model_data: List[bool]
    all_rewards: torch.Tensor
    info: Metric


class ExplorerBase(abc.ABC):
    @abc.abstractmethod
    def best_of_n(
        self,
        prompts: List[str],
        candidates: Dict[int, List[str]],
    ) -> List[str]:
        """Best-of-N generation given the reward model.

        OATArgs:
            prompts (List[str]): A list of prompt texts, M
            candidates (Dict[int, List[str]]): Lists of responses per prompt, M -> N

        Returns:
            List[str]: A list of the best response per prompt.
        """

    @abc.abstractmethod
    def select(
        self,
        prompts: List[str],
        candidates: Dict[int, List[str]],
    ) -> ExplorationResults:
        """Select dueling responses from candidates.

        OATArgs:
            prompts (List[str]): A list of prompt texts, M
            candidates (Dict[int, List[str]]): Lists of responses per prompt, M -> N

        Returns:
            ExplorationResults: Pair of responses per prompt (and features), M -> 2
        """

    @abc.abstractmethod
    def compare(self, candidate_features: torch.Tensor) -> torch.Tensor:
        """Compare candidates using the reward model.

        OATArgs:
            candidate_features (torch.Tensor): (M, 2, d)

        Returns:
            torch.Tensor: (M,), 1 means the first wins
        """

class CompletionDataset(Dataset):
    def __init__(self, prompt, completions, tokenizer, args):
        self.prompt = prompt
        self.completions = completions
        self.tokenizer = tokenizer
        self.prompt_ids_lens = []
        self.args = args
        self.sequences = []
        for completion in self.completions:
            if self.args.apply_chat_template:
                processed = self.tokenizer.apply_chat_template(
                    [{"content": prompt, "role": "user"},
                    {"content": completion['text'], "role": "assistant"}]
                )
            else:
                processed = (prompt + completion['text']).rstrip("\n")
                if not processed.endswith(self.tokenizer.eos_token):
                    processed += self.tokenizer.eos_token
                self.sequences.append(processed)
        
        self.encoded = self.tokenizer.batch_encode_plus(
            self.sequences, 
            max_length=self.args.prompt_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        prompt_token = self.tokenizer(
            prompt,
            max_length=self.args.prompt_max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
        self.prompt_ids_lens = [prompt_ids_len] * len(self.completions)
    

    def __len__(self):
        return len(self.completions)

    def __getitem__(self, idx):
        return (
            self.encoded['input_ids'][idx],
            self.encoded['attention_mask'][idx],
            self.prompt_ids_lens[idx]
        )
    
    def collate_fn(self, item_list):
        input_ids = []
        att_masks = []
        prompt_ids_lens = []
        extras = {"prompt_ids_lens": [], "same_masks": [], "chosen_ids": []}
        for input_id, att_mask, prompt_ids_len in item_list:
            input_ids.append(input_id)
            att_masks.append(att_mask)
            prompt_ids_lens.append(prompt_ids_len)

        padding_side = "right"
        input_ids = zero_pad_sequences(
            input_ids, side=padding_side, value=self.tokenizer.pad_token_id
        )
        att_masks = zero_pad_sequences(att_masks, side=padding_side)
        return input_ids, att_masks, prompt_ids_lens


class BestOfNExplorer(ExplorerBase):
    def __init__(
        self, ref_model, args:OATArgs
    ):
        # TODO: Add type-hints
        self.ref_model = ref_model
        self.args = args
        self.tokenizer = get_tokenizer(
            args.pretrain,
            ref_model,
            "right",
            use_fast=not args.disable_fast_tokenizer,
        )
        apply_chat_template = args.apply_chat_template
        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(
                self.strategy.args, "tokenizer_chat_template", None
            )
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

    def compare(self, candidate_features: torch.Tensor) -> torch.Tensor:
        # Just a dummy
        rewards = self.reward_model.get_rewards(candidate_features).mean(0)  # (M, 2, 1)
        return (rewards[:, 0] > rewards[:, 1]).squeeze().float().cpu().numpy()
    
    def create_dataloader(self, prompt, candidates):
        # Remember not to shuffle
        dataset = CompletionDataset(prompt, candidates, self.tokenizer, self.args)
        return DataLoader(
            dataset,
            batch_size=self.args.train_batch_size_per_device,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
    
    def get_log_probs(self, model, prompts, candidates):
        """Return log probs of all candidates for all prompts
        """
        all_logps = []
        for i, prompt in enumerate(prompts):
            dataloader = self.create_dataloader(prompt, candidates[i])
            # Batch for each prompt
            logps = []
            for data in dataloader:
                input_ids, att_masks, prompt_id_lens = data
                device = model.model.device
                input_ids, att_masks, prompt_id_lens = input_ids.to(device), att_masks.to(device), prompt_id_lens.to(device)
                output = model(input_ids, attention_mask=att_masks)
                logits = output["logits"]

                # Start from the first position
                labels = input_ids[:, 1:].clone()
                # labels = input_ids.clone()
                logits = logits[:, :-1, :].float()

                loss_masks = att_masks.clone().bool()
                # mask prompts
                for mask, source_len in zip(loss_masks, prompt_id_lens):
                    mask[:source_len] = False
                loss_masks = loss_masks[:, 1:]

                # dummy token; we'll ignore the losses on these tokens later
                labels[loss_masks == False] = 0
                per_token_logps = torch.gather(
                    logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
                ).squeeze(2)

                logp = (per_token_logps * loss_masks).sum(-1)
                logps.extend(logp.cpu().tolist())
            all_logps.append(logps)
            # Create pairing of prompt with candidates
        return torch.Tensor(all_logps)
    
    def best_of_n(
        self,
        prompts: List[str],
        candidates: Dict[int, List[str]],
    ) -> List[str]:
        """Best-of-N generation given the reward model.

        OATArgs:
            prompts (List[str]): A list of prompt texts, M
            candidates (Dict[int, List[str]]): Lists of responses per prompt, M -> N

        Returns:
            List[str]: A list of the best response per prompt.
        """
        print("Trying to get best of n")
        self.ref_model.model.to('cuda:0')
        ref_log_probs_BN = self.get_log_probs(self.ref_model, prompts, candidates)
        model_log_probs_BN = torch.Tensor([[c['logprob'] for c in candidates[i]] for i in range(len(prompts))])
        rew_BN = model_log_probs_BN - ref_log_probs_BN
        rew_B = torch.argmax(rew_BN, dim=1).cpu().tolist()
        best_of_n = []
        for i, prompt in enumerate(prompts):
            best_of_n.append(candidates[i][rew_B[i]]['text'])
        self.ref_model.model.to('cpu')
        return best_of_n
    
    def select(
        self,
        prompts: List[str],
        candidates: Dict[int, List[str]],
        best_running_responses: Dict[str, str]
    ) -> ExplorationResults:
        """Select dueling responses from candidates.

        OATArgs:
            prompts (List[str]): A list of prompt texts, M
            candidates (Dict[int, List[str]]): Lists of responses per prompt, M -> N

        Returns:
            ExplorationResults: Pair of responses per prompt (and features), M -> 2
        """
        best_of_candidates = self.best_of_n(prompts, candidates)
        dueling_candidates = {}
        for i, prompt in enumerate(prompts):
            dueling_candidates[i] = [best_of_candidates[i], best_running_responses[prompt]]

        return ExplorationResults(
            dueling_candidates=dueling_candidates,
            candidate_features=None, # not needed since it is not a learning RM
            init_clash=[False] * len(prompts),
            is_model_data=[False] * len(prompts),
            all_rewards=None, # not needed
            info={},
        )
    


class Explorer(ExplorerBase):
    def __init__(
        self, reward_model: RewardModel, rm_backbone: RMBackbone, args: OATArgs
    ) -> None:
        self.backbone = rm_backbone
        self.reward_model = reward_model

        self.max_length = 2048
        self.source_max_length = 1224
        self.backbone_bs = 8

        self.random_sampling = args.exp_rnd_sample

    def best_of_n(
        self,
        prompts: List[str],
        candidates: Dict[int, List[str]],
    ) -> List[str]:
        """Best-of-N generation given the reward model.

        OATArgs:
            prompts (List[str]): A list of prompt texts, M
            candidates (Dict[int, List[str]]): Lists of responses per prompt, M -> N

        Returns:
            List[str]: A list of the best response per prompt.
        """
        features = self._get_features(prompts, candidates)  # (M, N, d)
        best_response_indices = (
            self.reward_model.get_best_action(features).cpu().squeeze()
        )  # (M,)
        best_responses = [
            candidates[i][sel_idx] for i, sel_idx in enumerate(best_response_indices)
        ]
        return best_responses

    def select(
        self,
        prompts: List[str],
        candidates: Dict[int, List[str]],
    ) -> ExplorationResults:
        """Select dueling responses from candidates.

        OATArgs:
            prompts (List[str]): A list of prompt texts, M
            candidates (Dict[int, List[str]]): Lists of responses per prompt, M -> N

        Returns:
            ExplorationResults: Pair of responses per prompt (and features), M -> 2
        """
        (
            rewards,
            dueling_candidates,
            features,
            selected_candidate_indices,
            init_clash,
            info,
        ) = self._inner_select(prompts, candidates)
        return ExplorationResults(
            dueling_candidates=dueling_candidates,
            candidate_features=(
                torch.stack(
                    [
                        features[i][selected_candidate_indices[i]]
                        for i in range(len(prompts))
                    ]
                )
            ),
            init_clash=init_clash.tolist(),
            is_model_data=[False] * len(prompts),
            all_rewards=rewards,
            info=info,
        )

    def compare(self, candidate_features: torch.Tensor) -> torch.Tensor:
        rewards = self.reward_model.get_rewards(candidate_features).mean(0)  # (M, 2, 1)
        return (rewards[:, 0] > rewards[:, 1]).squeeze().float().cpu().numpy()

    def _inner_select(
        self,
        prompts: List[str],
        candidates: Dict[int, List[str]],
    ):
        features = self._get_features(prompts, candidates)  # (M, N, d)
        rewards, first_indices, second_indices = self.reward_model.get_duel_actions(
            features
        )  # rewards: (E or 2, M, N, 1); indices: both (M, 1)

        init_clash = (second_indices == first_indices).cpu().squeeze()
        rewards_with_agreed_best = rewards[:, init_clash]
        clashed_best_indices = second_indices[init_clash]
        agreed_best_resp_std = np.mean(
            [
                torch.std(rewards_with_agreed_best[:, i, clashed_best_indices[i]]).cpu()
                for i in range(len(clashed_best_indices))
            ]
        )
        rewards_without_agreed_best = rewards[:, ~init_clash]
        not_clashed_best_indices = second_indices[~init_clash]
        not_agreed_best_resp_std = np.mean(
            [
                torch.std(
                    rewards_without_agreed_best[:, i, not_clashed_best_indices[i]]
                ).cpu()
                for i in range(len(not_clashed_best_indices))
            ]
        )
        # In the case where both responses are the same, do random sampling
        if self.random_sampling:
            N = features.shape[1]
            rnd_second_indices = torch.ones_like(second_indices) * -1
            for _ in range(3):
                # Clash prob 1 / N^3
                rand_indices = torch.randint_like(second_indices, N)
                valid_idx = (rand_indices != first_indices) * (rnd_second_indices == -1)
                rnd_second_indices[valid_idx] = rand_indices[valid_idx]
                if -1 not in rnd_second_indices:
                    break

            second_indices = torch.where(
                second_indices == first_indices, rnd_second_indices, second_indices
            )

        selected_candidate_indices = torch.cat(
            [first_indices, second_indices], dim=-1
        ).cpu()
        dueling_candidates = {}
        for i, sel_idx in enumerate(selected_candidate_indices):
            dueling_candidates[i] = [candidates[i][j] for j in sel_idx]

        info = {
            "explorer/agreed_best_resp_std": np.nan_to_num(agreed_best_resp_std),
            "explorer/not_agreed_best_resp_std": np.nan_to_num(
                not_agreed_best_resp_std
            ),
        }
        return (
            rewards,
            dueling_candidates,
            features,
            selected_candidate_indices,
            init_clash,
            info,
        )

    def _get_features(
        self,
        prompts: List[str],
        candidates: Dict[int, List[str]],
    ):
        input_ids = []
        M = len(prompts)
        N = len(candidates[0])
        for i in range(M):
            for j in range(N):
                pair_ids = self.backbone.tokenize_pair(
                    prompt=prompts[i],
                    candidate=candidates[i][j],
                    source_max_length=self.source_max_length,
                    max_length=self.max_length,
                )
                input_ids.append(pair_ids)
        encodings = self.backbone.tokenizer.pad(
            {"input_ids": input_ids},
            return_tensors="pt",
        )

        features = []
        for ndx in range(0, M * N, self.backbone_bs):
            batch_enc = tree.map_structure(
                lambda x: x[ndx : min(ndx + self.backbone_bs, M * N)].to(
                    self.backbone.device
                ),
                encodings,
            )
            features.append(self.backbone.get_feature(**batch_enc))
        features = torch.cat(features, dim=0)  # (M*N, d)
        features = features.view(M, N, -1)
        return features


class ModelBasedExplorer(Explorer):
    """It not only explores based on Thompson sampling, but also synthesizes
    model rollout when it trusts itself to boot sample efficiency."""

    def __init__(
        self, reward_model: RewardModel, rm_backbone: RMBackbone, args: OATArgs
    ) -> None:
        super().__init__(reward_model, rm_backbone, args)
        self.count = 1
        self.burn_in_period = args.burn_in_period
        self.max_model_data_ratio = args.max_model_data_ratio
        self.model_data_selector = getattr(self, f"_{args.model_data_strategy}_select")
        self.pure_model_based = args.pure_model_based

    def _random_select(
        self,
        candidates,
        rewards,
        dueling_candidates,
        selected_candidate_indices,
        is_model_data,
    ):
        reward_margin = rewards - einops.rearrange(rewards, "e m n 1 -> e m 1 n")
        E, M, _, _ = reward_margin.shape
        random_belief_reward_margin = reward_margin[
            torch.randint(E, (M,)), torch.arange(M)
        ]  # M, N, N'
        # mean_rewards = rewards.mean(0)
        max_model_data = int(len(is_model_data) * self.max_model_data_ratio)
        is_model_data[:max_model_data] = 1
        random.shuffle(is_model_data)
        for i, imd in enumerate(is_model_data):
            if imd:
                margin_i = random_belief_reward_margin[i]
                margin_i_abs = torch.abs(margin_i)
                tr_pairs = torch.where(margin_i_abs == margin_i_abs.max())
                sel_idx = np.random.choice(len(tr_pairs[0]))  # break tie
                candidate_1, candidate_2 = tr_pairs[0][sel_idx], tr_pairs[1][sel_idx]
                if margin_i[candidate_1, candidate_2] > 0:
                    rnd_chosen, rnd_rejected = candidate_1, candidate_2
                else:
                    rnd_chosen, rnd_rejected = candidate_2, candidate_1
                dueling_candidates[i] = [
                    candidates[i][rnd_chosen],
                    candidates[i][rnd_rejected],
                ]
                selected_candidate_indices[i] = torch.tensor([rnd_chosen, rnd_rejected])
        return dueling_candidates, selected_candidate_indices, is_model_data

    def select(
        self, prompts: List[str], candidates: Dict[int, List[str]]
    ) -> ExplorationResults:
        # Select the query points using exploration strategies.
        # Be optimistic and reduce uncertainty.
        (
            rewards,  # rewards: (E, M, N, 1)
            dueling_candidates,
            features,
            selected_candidate_indices,
            init_clash,
            info,
        ) = self._inner_select(prompts, candidates)
        # Replace queries that the agent is already confident about the results.
        # Utilize uncertainty to build trust region.
        is_model_data = np.zeros(len(prompts))
        model_chosen_rewards = []
        model_rejected_rewards = []
        model_pred_prob = []
        sel_pair_ep_uct = []
        sel_prompt_ep_uct = []
        uct_mean = 0
        if self.count > self.burn_in_period:
            dueling_candidates, selected_candidate_indices, is_model_data = (
                self.model_data_selector(
                    candidates,
                    rewards,
                    dueling_candidates,
                    selected_candidate_indices,
                    is_model_data,
                )
            )
            mean_rewards = rewards.mean(0)  # (M, N, 1)
            uct = uncertainty.logits_variance(rewards)
            uct_mean = uct.mean().item()

        for i in range(len(prompts)):
            if is_model_data[i]:
                tr_chosen = selected_candidate_indices[i, 0]
                tr_rejected = selected_candidate_indices[i, 1]

                model_chosen_rewards.append(mean_rewards[i, tr_chosen].item())
                model_rejected_rewards.append(mean_rewards[i, tr_rejected].item())
                model_pred_prob.append(
                    (mean_rewards[i, tr_chosen] - mean_rewards[i, tr_rejected])
                    .sigmoid()
                    .item()
                )
                sel_pair_ep_uct.append(uct[i][tr_chosen, tr_rejected].item())
                sel_prompt_ep_uct.append(uct[i].mean().item())
            else:
                if self.pure_model_based:
                    # Disable learning.
                    dueling_candidates[i] = ["dummy", "dummy"]

        self.count += 1

        info.update(
            {
                "explorer/model_chosen_rewards": np.mean(model_chosen_rewards),
                "explorer/model_rejected_rewards": np.mean(model_rejected_rewards),
                "explorer/model_pred_prob_min": (
                    np.min(model_pred_prob) if model_pred_prob else np.nan
                ),
                "explorer/model_pred_prob_max": (
                    np.max(model_pred_prob) if model_pred_prob else np.nan
                ),
                "explorer/model_pred_prob_mean": np.mean(model_pred_prob),
                "explorer/sel_pair_ep_uct_mean": np.mean(sel_pair_ep_uct),
                "explorer/sel_pair_ep_uct_std": np.std(sel_pair_ep_uct),
                "explorer/sel_prompt_ep_uct_mean": np.std(sel_prompt_ep_uct),
                "explorer/sel_prompt_ep_uct_std": np.std(sel_prompt_ep_uct),
                "explorer/all_ep_uct_mean": uct_mean,
                "explorer/model_data_ratio": np.mean(is_model_data),
            }
        )
        return ExplorationResults(
            dueling_candidates=dueling_candidates,
            candidate_features=(
                torch.stack(
                    [
                        features[i][selected_candidate_indices[i]]
                        for i in range(len(prompts))
                    ]
                )
            ),
            init_clash=[False]*len(prompts),
            is_model_data=is_model_data.astype("bool").tolist(),
            all_rewards=rewards,
            info=info,
        )
