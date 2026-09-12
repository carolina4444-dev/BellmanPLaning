#!/usr/bin/env python3
"""
Recursive Bellman-Residue Neural Planner
========================================

A self-contained PyTorch reference implementation of a recursive planner whose
meta-actions are:

    0. CLASSIFY  - terminate with a classification
    1. FORECAST  - terminate with a forecast + latent consequence
    2. RECURSE   - spend computation and explore a deeper node

The planner:
  * represents nodes using action/outcome/consequence latents;
  * predicts Q(s,a), policy pi(a|s), and expected absolute Bellman residue;
  * uses predicted residue to prioritize tree expansion;
  * computes realized Bellman residues after expansion;
  * trains Q/policy/residue heads from Bellman targets;
  * supports a Knowledge Base (KB) as training context;
  * supports context dropout/distillation so evaluation can use a goal/query
    without an explicit KB;
  * uses forecast leaves to create consequence transitions;
  * performs recursive best-first tree search under a computation budget.

This is a research/reference implementation, not a claim that the particular
losses or hyperparameters are optimal.

Dependencies:
    pip install torch

The included synthetic KB environment makes the script runnable without an
external dataset. Replace SyntheticKBTask with your own task adapter.

Example:
    python recursive_bellman_planner.py --demo
    python recursive_bellman_planner.py --train --epochs 10
"""

from __future__ import annotations

import argparse
import copy
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 7):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

CLASSIFY = 0
FORECAST = 1
RECURSE = 2
N_ACTIONS = 3


@dataclass
class KBItem:
    key: int
    value: int


@dataclass
class Example:
    """
    goal:
        Integer query/goal. In the synthetic task it asks for the value of a
        key. In a real system this can be a tokenized query embedding.

    kb:
        Knowledge-base facts available during the first training phase.

    target:
        Ground-truth answer.

    optimal_depth:
        A synthetic supervision hint used only to generate a useful recursive
        training trajectory. Set to 0 when the answer is directly available.
    """
    goal: int
    kb: List[KBItem]
    target: int
    optimal_depth: int


@dataclass
class Node:
    depth: int
    latent: torch.Tensor
    parent_latent: torch.Tensor
    action_latent: torch.Tensor
    outcome_latent: torch.Tensor
    consequence_latent: torch.Tensor
    trace: List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    action: int
    value: float
    residue: float
    output: Optional[int] = None
    consequence: Optional[torch.Tensor] = None
    children: List["SearchResult"] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Synthetic Knowledge Base task
# ---------------------------------------------------------------------------

class SyntheticKBTask:
    """
    Small deterministic task designed to exercise recursive planning.

    A KB contains facts:
        key -> value

    A goal is a key. The target value may be:
        * directly present in KB;
        * one-hop hidden behind a chain;
        * multi-hop hidden behind a chain.

    The planner is trained to decide whether to classify/forecast directly or
    recurse before forecasting.

    This adapter is deliberately simple. Replace it with your real KB/query
    pipeline while preserving the model's interface.
    """

    def __init__(self, n_keys: int = 64, value_mod: int = 10):
        self.n_keys = n_keys
        self.value_mod = value_mod

    def sample(self, max_chain: int = 3) -> Example:
        target = random.randrange(self.value_mod)

        depth = random.randrange(max_chain + 1)
        # Keys are arranged in a deterministic chain:
        # k0 -> k1 -> ... -> target
        chain = random.sample(range(self.n_keys), depth + 1)
        goal = chain[0]

        kb = []
        for i in range(depth):
            # Store a pointer-like value encoded as an integer > value_mod.
            kb.append(KBItem(chain[i], self.value_mod + chain[i + 1]))

        # Final key stores the actual target.
        kb.append(KBItem(chain[-1], target))

        # Add distractors.
        distractor_keys = set(x.key for x in kb)
        for _ in range(max(2, depth * 2)):
            k = random.randrange(self.n_keys)
            if k in distractor_keys:
                continue
            kb.append(KBItem(k, random.randrange(self.value_mod)))

        random.shuffle(kb)
        return Example(goal=goal, kb=kb, target=target, optimal_depth=depth)

    def lookup(self, ex: Example, key: int) -> Optional[int]:
        for item in ex.kb:
            if item.key == key:
                return item.value
        return None

    def resolve(self, ex: Example, key: int, max_hops: int = 32) -> Optional[int]:
        """
        Ground-truth KB resolver. Values >= value_mod encode another key.
        """
        current = key
        for _ in range(max_hops):
            value = self.lookup(ex, current)
            if value is None:
                return None
            if value < self.value_mod:
                return value
            current = value - self.value_mod
        return None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class KBEncoder(nn.Module):
    """
    Encodes a variable-size KB into a fixed context vector.

    For a real application, replace this with a Transformer/RAG encoder.
    """

    def __init__(self, n_keys: int, value_mod: int, d_model: int):
        super().__init__()
        self.key_emb = nn.Embedding(n_keys, d_model)
        self.value_emb = nn.Embedding(value_mod + n_keys, d_model)
        self.proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, kb: List[KBItem]) -> torch.Tensor:
        if not kb:
            return torch.zeros(
                self.key_emb.embedding_dim,
                device=self.key_emb.weight.device,
            )

        keys = torch.tensor(
            [x.key for x in kb],
            dtype=torch.long,
            device=self.key_emb.weight.device,
        )
        vals = torch.tensor(
            [x.value for x in kb],
            dtype=torch.long,
            device=self.key_emb.weight.device,
        )

        x = torch.cat(
            [self.key_emb(keys), self.value_emb(vals)],
            dim=-1,
        )
        x = self.proj(x)
        return x.mean(dim=0)


class BellmanPlannerNet(nn.Module):
    """
    Neural state/action/value/residue architecture.

    The state is explicitly formed from:
        parent latent
        action latent
        outcome latent
        consequence latent
        goal/condition

    Q and policy operate over CLASSIFY/FORECAST/RECURSE.
    Residue predicts E[|Bellman residual|].
    Forecast predicts an answer and consequence latent.
    Recurse produces a child-state latent.
    """

    def __init__(
        self,
        n_keys: int,
        value_mod: int,
        d_model: int = 128,
        gamma: float = 0.95,
    ):
        super().__init__()
        self.d_model = d_model
        self.gamma = gamma

        self.goal_emb = nn.Embedding(n_keys, d_model)
        self.kb_encoder = KBEncoder(n_keys, value_mod, d_model)

        self.action_emb = nn.Embedding(N_ACTIONS, d_model)

        self.node_encoder = nn.Sequential(
            nn.Linear(5 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.policy_head = nn.Linear(d_model, N_ACTIONS)
        self.q_head = nn.Linear(d_model, N_ACTIONS)

        # Positive residue estimate.
        self.residue_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, N_ACTIONS),
            nn.Softplus(),
        )

        self.classify_head = nn.Linear(d_model, value_mod)

        self.forecast_head = nn.Linear(d_model, value_mod)

        self.consequence_head = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.recurse_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.value_head = nn.Linear(d_model, 1)

    def condition(self, goal: int, kb: Optional[List[KBItem]]) -> torch.Tensor:
        device = next(self.parameters()).device
        g = self.goal_emb(
            torch.tensor(goal, dtype=torch.long, device=device)
        )
        if kb is None:
            c_kb = torch.zeros_like(g)
        else:
            c_kb = self.kb_encoder(kb)
        return g + c_kb

    def encode_node(
        self,
        parent: torch.Tensor,
        action_id: int,
        outcome: torch.Tensor,
        consequence: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        device = parent.device
        a = self.action_emb(
            torch.tensor(action_id, dtype=torch.long, device=device)
        )
        x = torch.cat(
            [parent, a, outcome, consequence, condition],
            dim=-1,
        )
        return self.node_encoder(x)

    def evaluate(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.policy_head(z)
        q = self.q_head(z)
        residue = self.residue_head(z)
        return {
            "policy_logits": logits,
            "policy": F.softmax(logits, dim=-1),
            "q": q,
            "residue": residue,
            "value": self.value_head(z).squeeze(-1),
        }

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.classify_head(z)

    def forecast(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        answer_logits = self.forecast_head(z)
        return answer_logits, self.consequence_head(
            torch.cat([z, z], dim=-1)
        )

    def make_recursive_child(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        return self.recurse_head(torch.cat([z, condition], dim=-1))


# ---------------------------------------------------------------------------
# Bellman planner
# ---------------------------------------------------------------------------

class RecursiveBellmanPlanner:
    """
    Controller around BellmanPlannerNet.

    Search is best-first over predicted Bellman residue:
        priority = policy(a|s) * predicted_residue(s,a)

    After a branch is actually evaluated, its realized Bellman residue is
    calculated and used to train the residue head.
    """

    def __init__(
        self,
        model: BellmanPlannerNet,
        value_mod: int,
        gamma: float = 0.95,
        recurse_cost: float = -0.05,
        residue_threshold: float = 0.20,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.value_mod = value_mod
        self.gamma = gamma
        self.recurse_cost = recurse_cost
        self.residue_threshold = residue_threshold
        self.device = device

    def root_node(
        self,
        ex: Example,
        use_kb: bool = True,
    ) -> Tuple[Node, torch.Tensor]:
        c = self.model.condition(
            ex.goal,
            ex.kb if use_kb else None,
        )

        zero = torch.zeros(
            self.model.d_model,
            device=self.device,
        )

        z = self.model.encode_node(
            parent=zero,
            action_id=RECURSE,
            outcome=zero,
            consequence=zero,
            condition=c,
        )

        return (
            Node(
                depth=0,
                latent=z,
                parent_latent=zero,
                action_latent=self.model.action_emb.weight[RECURSE],
                outcome_latent=zero,
                consequence_latent=zero,
                trace=[],
            ),
            c,
        )

    @staticmethod
    def action_name(a: int) -> str:
        return ["CLASSIFY", "FORECAST", "RECURSE"][a]

    def _answer_from_logits(self, logits: torch.Tensor) -> int:
        return int(logits.argmax(dim=-1).item())

    def _reward(self, answer: int, target: int) -> float:
        return 1.0 if answer == target else -1.0

    def _make_transition(
        self,
        node: Node,
        action: int,
        outcome: torch.Tensor,
        consequence: torch.Tensor,
        condition: torch.Tensor,
    ) -> Node:
        z = self.model.encode_node(
            parent=node.latent,
            action_id=action,
            outcome=outcome,
            consequence=consequence,
            condition=condition,
        )
        return Node(
            depth=node.depth + 1,
            latent=z,
            parent_latent=node.latent,
            action_latent=self.model.action_emb.weight[action],
            outcome_latent=outcome,
            consequence_latent=consequence,
            trace=node.trace + [self.action_name(action)],
        )

    @torch.no_grad()
    def inspect_node(
        self,
        node: Node,
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        return self.model.evaluate(node.latent)

    def search(
        self,
        ex: Example,
        budget: int = 8,
        use_kb: bool = True,
        deterministic: bool = True,
    ) -> SearchResult:
        """
        Evaluation-time recursive search.

        For recurse branches, the controller uses the predicted residue to
        decide whether additional computation is worthwhile.

        Forecast leaves produce a consequence latent. The synthetic adapter
        uses the ground-truth KB only for generating an interpretable demo
        consequence; in a real application this should be replaced by the
        learned/environment transition.
        """
        self.model.eval()
        root, condition = self.root_node(ex, use_kb=use_kb)

        def recurse(node: Node, remaining: int) -> SearchResult:
            info = self.model.evaluate(node.latent)
            q = info["q"]
            policy = info["policy"]
            pred_res = info["residue"]

            # Residue-driven priority.
            priority = policy * pred_res

            # If deterministic, pick maximum priority.
            if deterministic:
                action = int(priority.argmax().item())
            else:
                probs = priority / (priority.sum() + 1e-8)
                action = int(
                    torch.multinomial(probs, 1).item()
                )

            trace = node.trace + [self.action_name(action)]

            # CLASSIFY -----------------------------------------------------
            if action == CLASSIFY:
                logits = self.model.classify(node.latent)
                answer = self._answer_from_logits(logits)
                reward = self._reward(answer, ex.target)

                delta = reward - q[CLASSIFY]
                return SearchResult(
                    action=CLASSIFY,
                    value=float(reward),
                    residue=float(abs(delta).item()),
                    output=answer,
                    trace=trace,
                )

            # FORECAST -----------------------------------------------------
            if action == FORECAST:
                logits, consequence = self.model.forecast(node.latent)
                answer = self._answer_from_logits(logits)

                # In a real environment, consequence is supplied by the
                # forecast/transition model. The synthetic reward is direct.
                reward = self._reward(answer, ex.target)

                # A forecast leaf is a transition-bearing terminal.
                delta = reward - q[FORECAST]

                return SearchResult(
                    action=FORECAST,
                    value=float(reward),
                    residue=float(abs(delta).item()),
                    output=answer,
                    consequence=consequence.detach(),
                    trace=trace,
                )

            # RECURSE ------------------------------------------------------
            if remaining <= 0:
                # No budget: fallback to forecast.
                logits, consequence = self.model.forecast(node.latent)
                answer = self._answer_from_logits(logits)
                reward = self._reward(answer, ex.target)
                delta = reward - q[FORECAST]

                return SearchResult(
                    action=FORECAST,
                    value=float(reward),
                    residue=float(abs(delta).item()),
                    output=answer,
                    consequence=consequence.detach(),
                    trace=trace + ["BUDGET_FALLBACK"],
                )

            # If recursion itself has low predicted Bellman inconsistency,
            # don't spend computation blindly.
            if float(pred_res[RECURSE].item()) < self.residue_threshold:
                logits, consequence = self.model.forecast(node.latent)
                answer = self._answer_from_logits(logits)
                reward = self._reward(answer, ex.target)
                delta = reward - q[FORECAST]

                return SearchResult(
                    action=FORECAST,
                    value=float(reward),
                    residue=float(abs(delta).item()),
                    output=answer,
                    consequence=consequence.detach(),
                    trace=trace + ["RESIDUE_STOP"],
                )

            child_z = self.model.make_recursive_child(
                node.latent,
                condition,
            )

            child = Node(
                depth=node.depth + 1,
                latent=child_z,
                parent_latent=node.latent,
                action_latent=self.model.action_emb.weight[RECURSE],
                outcome_latent=torch.zeros_like(node.latent),
                consequence_latent=torch.zeros_like(node.latent),
                trace=trace,
            )

            child_result = recurse(child, remaining - 1)

            target = self.recurse_cost + self.gamma * child_result.value
            delta = target - q[RECURSE]

            return SearchResult(
                action=RECURSE,
                value=float(target),
                residue=float(abs(delta).item()),
                children=[child_result],
                output=child_result.output,
                consequence=child_result.consequence,
                trace=child_result.trace,
            )

        return recurse(root, budget)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs: int = 10
    examples_per_epoch: int = 256
    d_model: int = 128
    lr: float = 3e-4
    gamma: float = 0.95
    recurse_cost: float = -0.05
    residue_threshold: float = 0.20
    context_dropout: float = 0.20
    max_chain: int = 3
    grad_clip: float = 1.0


class PlannerTrainer:
    """
    Generates supervised recursive trajectories from the synthetic KB oracle.

    The target action is:
      * if the answer is directly available as a terminal value: FORECAST;
      * if a pointer chain exists: RECURSE;
      * at the final value: FORECAST.

    A CLASSIFY loss is also included as an auxiliary objective. The model is
    therefore explicitly trained on all three meta-actions.
    """

    def __init__(
        self,
        task: SyntheticKBTask,
        model: BellmanPlannerNet,
        cfg: TrainConfig,
        device: str = "cpu",
    ):
        self.task = task
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        self.target_model = copy.deepcopy(model).to(device)
        for p in self.target_model.parameters():
            p.requires_grad_(False)

        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=1e-4,
        )

    def _target_action(
        self,
        ex: Example,
        key: int,
    ) -> Tuple[int, Optional[int]]:
        value = self.task.lookup(ex, key)
        if value is None:
            return RECURSE, None

        if value >= self.task.value_mod:
            return RECURSE, value - self.task.value_mod

        return FORECAST, value

    def _make_state(
        self,
        goal: int,
        kb: Optional[List[KBItem]],
        parent: torch.Tensor,
        action: int,
        outcome: torch.Tensor,
        consequence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self.model.condition(goal, kb)
        z = self.model.encode_node(
            parent,
            action,
            outcome,
            consequence,
            c,
        )
        return z, c

    def _one_trajectory(
        self,
        ex: Example,
        force_kb: bool,
    ) -> Dict[str, torch.Tensor]:
        """
        Construct a Bellman training trajectory from the ground-truth KB.

        Each step provides:
            state latent
            action
            reward
            next latent
            terminal flag
            observed absolute Bellman residue
        """
        use_kb = force_kb
        if random.random() < self.cfg.context_dropout:
            use_kb = False

        kb = ex.kb if use_kb else None

        zero = torch.zeros(
            self.model.d_model,
            device=self.device,
        )

        parent = zero
        outcome = zero
        consequence = zero
        current_key = ex.goal
        records = []

        max_steps = self.cfg.max_chain + 2

        for step in range(max_steps):
            action, next_key = self._target_action(ex, current_key)

            z, c = self._make_state(
                ex.goal,
                kb,
                parent,
                RECURSE if step > 0 else RECURSE,
                outcome,
                consequence,
            )

            out = self.model.evaluate(z)
            q = out["q"]

            if action == RECURSE:
                # Recursion has a computation cost and transitions to a deeper
                # latent state representing the next KB reasoning step.
                child_key = (
                    next_key if next_key is not None else current_key
                )

                child_goal = torch.tensor(
                    child_key,
                    dtype=torch.long,
                    device=self.device,
                )

                child_condition = self.model.condition(
                    int(child_key),
                    kb,
                )

                child = self.model.make_recursive_child(
                    z,
                    child_condition,
                )

                # The child value is estimated by the target network.
                target_value = (
                    self.cfg.recurse_cost
                    + self.cfg.gamma
                    * self.target_model.value_head(child).squeeze(-1)
                )

                reward = torch.tensor(
                    self.cfg.recurse_cost,
                    device=self.device,
                )

                done = torch.tensor(0.0, device=self.device)
                next_latent = child

                # Ground-truth residual target uses a detached Bellman target.
                bellman_target = (
                    reward
                    + self.cfg.gamma
                    * self.target_model.value_head(child).squeeze(-1)
                ).detach()

            else:
                if action == CLASSIFY:
                    logits = self.model.classify(z)
                else:
                    logits, consequence = self.model.forecast(z)

                answer = int(logits.argmax(dim=-1).item())

                # Reward against ground truth.
                reward_value = self._reward(answer, ex.target)
                reward = torch.tensor(
                    reward_value,
                    dtype=torch.float32,
                    device=self.device,
                )

                done = torch.tensor(1.0, device=self.device)
                next_latent = z.detach()
                bellman_target = reward.detach()

            observed_residue = (
                bellman_target - q[action]
            ).abs().detach()

            records.append({
                "z": z,
                "action": torch.tensor(
                    action, dtype=torch.long, device=self.device
                ),
                "reward": reward,
                "done": done,
                "target": bellman_target,
                "residue": observed_residue,
            })

            if done.item() > 0.5:
                break

            parent = next_latent
            outcome = torch.zeros_like(z)
            consequence = torch.zeros_like(z)
            current_key = (
                next_key if next_key is not None else current_key
            )

        return records

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()

        losses = []
        q_losses = []
        policy_losses = []
        residue_losses = []
        aux_losses = []

        for _ in range(self.cfg.examples_per_epoch):
            ex = self.task.sample(self.cfg.max_chain)

            records = self._one_trajectory(
                ex,
                force_kb=True,
            )

            total = torch.tensor(
                0.0,
                dtype=torch.float32,
                device=self.device,
            )

            for rec in records:
                z = rec["z"]
                action = rec["action"]
                target = rec["target"]
                residue_target = rec["residue"]

                out = self.model.evaluate(z)

                q = out["q"][action]
                q_loss = F.smooth_l1_loss(q, target)

                # Advantage-like policy target: encourage the action selected
                # by the oracle trajectory.
                policy_loss = F.cross_entropy(
                    out["policy_logits"].unsqueeze(0),
                    action.unsqueeze(0),
                )

                residue_pred = out["residue"][action]
                residue_loss = F.smooth_l1_loss(
                    residue_pred,
                    residue_target,
                )

                # Auxiliary answer loss.
                # For terminal steps use the true target. For recursive steps
                # use the eventual answer as a weak auxiliary signal.
                if action.item() == CLASSIFY:
                    logits = self.model.classify(z)
                else:
                    logits, _ = self.model.forecast(z)

                answer_loss = F.cross_entropy(
                    logits.unsqueeze(0),
                    torch.tensor(
                        [ex.target],
                        dtype=torch.long,
                        device=self.device,
                    ),
                )

                step_loss = (
                    q_loss
                    + policy_loss
                    + residue_loss
                    + 0.25 * answer_loss
                )

                total = total + step_loss

                q_losses.append(float(q_loss.detach()))
                policy_losses.append(float(policy_loss.detach()))
                residue_losses.append(float(residue_loss.detach()))
                aux_losses.append(float(answer_loss.detach()))

            total = total / max(1, len(records))

            self.opt.zero_grad(set_to_none=True)
            total.backward()

            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.grad_clip,
            )

            self.opt.step()

            losses.append(float(total.detach()))

        # Soft target-network update.
        tau = 0.02
        with torch.no_grad():
            for p_t, p in zip(
                self.target_model.parameters(),
                self.model.parameters(),
            ):
                p_t.mul_(1.0 - tau).add_(p, alpha=tau)

        return {
            "loss": sum(losses) / len(losses),
            "q_loss": sum(q_losses) / len(q_losses),
            "policy_loss": sum(policy_losses) / len(policy_losses),
            "residue_loss": sum(residue_losses) / len(residue_losses),
            "answer_loss": sum(aux_losses) / len(aux_losses),
        }


# ---------------------------------------------------------------------------
# Evaluation/demo
# ---------------------------------------------------------------------------

def pretty_kb(ex: Example, value_mod: int) -> str:
    parts = []
    for item in ex.kb:
        if item.value >= value_mod:
            parts.append(
                f"{item.key}->key:{item.value - value_mod}"
            )
        else:
            parts.append(f"{item.key}->{item.value}")
    return ", ".join(parts)


def run_demo(
    epochs: int = 5,
    examples_per_epoch: int = 256,
    max_chain: int = 3,
    budget: int = 8,
    d_model: int = 128,
    device: str = "cpu",
):
    value_mod = 10
    n_keys = 64

    task = SyntheticKBTask(
        n_keys=n_keys,
        value_mod=value_mod,
    )

    cfg = TrainConfig(
        epochs=epochs,
        examples_per_epoch=examples_per_epoch,
        d_model=d_model,
        max_chain=max_chain,
    )

    model = BellmanPlannerNet(
        n_keys=n_keys,
        value_mod=value_mod,
        d_model=d_model,
        gamma=cfg.gamma,
    )

    trainer = PlannerTrainer(
        task=task,
        model=model,
        cfg=cfg,
        device=device,
    )

    print(f"Training on {device} ...")
    for epoch in range(1, epochs + 1):
        metrics = trainer.train_epoch(epoch)
        print(
            f"epoch={epoch:03d} "
            f"loss={metrics['loss']:.4f} "
            f"Q={metrics['q_loss']:.4f} "
            f"policy={metrics['policy_loss']:.4f} "
            f"residue={metrics['residue_loss']:.4f} "
            f"answer={metrics['answer_loss']:.4f}"
        )

    planner = RecursiveBellmanPlanner(
        model=trainer.model,
        value_mod=value_mod,
        gamma=cfg.gamma,
        recurse_cost=cfg.recurse_cost,
        residue_threshold=cfg.residue_threshold,
        device=device,
    )

    print("\nEvaluation examples")
    print("=" * 80)

    for i in range(8):
        ex = task.sample(max_chain=max_chain)

        # First show the oracle solution depth.
        oracle = task.resolve(ex, ex.goal)

        # Evaluation without KB demonstrates context-dropout robustness.
        result_no_kb = planner.search(
            ex,
            budget=budget,
            use_kb=False,
        )

        # Evaluation with KB.
        result_kb = planner.search(
            ex,
            budget=budget,
            use_kb=True,
        )

        print(f"\nExample {i + 1}")
        print(f"goal={ex.goal} target={ex.target} oracle={oracle}")
        print(f"KB: {pretty_kb(ex, value_mod)}")
        print(
            f"with KB: answer={result_kb.output} "
            f"value={result_kb.value:.3f} "
            f"residue={result_kb.residue:.3f}"
        )
        print("trace:", " -> ".join(result_kb.trace))
        print(
            f"query only: answer={result_no_kb.output} "
            f"value={result_no_kb.value:.3f} "
            f"residue={result_no_kb.residue:.3f}"
        )
        print("trace:", " -> ".join(result_no_kb.trace))

    return trainer.model


# ---------------------------------------------------------------------------
# Save/load
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: BellmanPlannerNet,
    path: str,
    metadata: Optional[Dict] = None,
):
    payload = {
        "model_state": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(
    model: BellmanPlannerNet,
    path: str,
    device: str = "cpu",
):
    payload = torch.load(
        path,
        map_location=device,
    )
    model.load_state_dict(payload["model_state"])
    print(f"Loaded checkpoint: {path}")
    return payload.get("metadata", {})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Recursive Bellman-residue neural planner"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Train a small synthetic model and run evaluation.",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the synthetic planner.",
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--examples-per-epoch", type=int, default=256)
    parser.add_argument("--max-chain", type=int, default=3)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--context-dropout",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--residue-threshold",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="recursive_bellman_planner.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()
    seed_everything(args.seed)

    if not args.demo and not args.train:
        args.demo = True

    value_mod = 10
    n_keys = 64

    task = SyntheticKBTask(
        n_keys=n_keys,
        value_mod=value_mod,
    )

    cfg = TrainConfig(
        epochs=args.epochs,
        examples_per_epoch=args.examples_per_epoch,
        d_model=args.d_model,
        lr=args.lr,
        context_dropout=args.context_dropout,
        max_chain=args.max_chain,
        residue_threshold=args.residue_threshold,
    )

    model = BellmanPlannerNet(
        n_keys=n_keys,
        value_mod=value_mod,
        d_model=args.d_model,
        gamma=cfg.gamma,
    )

    trainer = PlannerTrainer(
        task,
        model,
        cfg,
        device=args.device,
    )

    print(
        f"device={args.device} "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    for epoch in range(1, cfg.epochs + 1):
        metrics = trainer.train_epoch(epoch)
        print(
            f"epoch={epoch:03d} "
            f"loss={metrics['loss']:.4f} "
            f"Q={metrics['q_loss']:.4f} "
            f"policy={metrics['policy_loss']:.4f} "
            f"residue={metrics['residue_loss']:.4f} "
            f"answer={metrics['answer_loss']:.4f}"
        )

    save_checkpoint(
        trainer.model,
        args.checkpoint,
        metadata={
            "n_keys": n_keys,
            "value_mod": value_mod,
            "d_model": args.d_model,
            "gamma": cfg.gamma,
        },
    )

    planner = RecursiveBellmanPlanner(
        trainer.model,
        value_mod=value_mod,
        gamma=cfg.gamma,
        recurse_cost=cfg.recurse_cost,
        residue_threshold=args.residue_threshold,
        device=args.device,
    )

    print("\nFinal evaluation")
    print("-" * 80)

    correct_kb = 0
    correct_query_only = 0
    n_eval = 100

    for _ in range(n_eval):
        ex = task.sample(args.max_chain)

        r_kb = planner.search(
            ex,
            budget=args.budget,
            use_kb=True,
        )

        r_query = planner.search(
            ex,
            budget=args.budget,
            use_kb=False,
        )

        correct_kb += int(r_kb.output == ex.target)
        correct_query_only += int(r_query.output == ex.target)

    print(
        f"accuracy with KB: "
        f"{correct_kb / n_eval:.3f}"
    )
    print(
        f"accuracy query-only: "
        f"{correct_query_only / n_eval:.3f}"
    )



# ---------------------------------------------------------------------------
# Plain-text Knowledge Base adapter
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryStep:
    state: Optional[str]
    action: str
    consequence: Optional[str]
    next_state: Optional[str]
    consequence_source: str = "observed"


@dataclass
class Trajectory:
    """
    A complete state/action/consequence trajectory.
    """
    initial_state: Optional[str]
    steps: List[TrajectoryStep]
    final_state: Optional[str]


class PlainTextKB:
    """
    Loads a data.txt-style KB and converts contiguous statements into
    state/action/consequence trajectories.

    Example input:

        The bedroom is cold.
        The robot enters the bedroom.
        The robot turns on the heater.
        The bedroom begins warming.
        The bedroom is warm.

    becomes:

        State:
            The bedroom is cold.

        Action:
            The robot enters the bedroom.

        Consequence:
            The robot enters the bedroom.
            [no explicit consequence yet]

        Action:
            The robot turns on the heater.

        Consequence:
            The bedroom begins warming.

        Next state:
            The bedroom begins warming.

        Final state:
            The bedroom is warm.

    The parser deliberately preserves the original natural-language
    statements instead of converting them into artificial key/value pairs.
    """

    ACTION_RE = re.compile(
        r"^The robot\b",
        re.IGNORECASE,
    )

    STATE_RE = re.compile(
        r"^(?!The robot\b).+",
        re.IGNORECASE,
    )

    def __init__(self, path: str):
        self.path = path

        raw = Path(path).read_text(
            encoding="utf-8"
        )

        self.lines = [
            x.strip()
            for x in raw.splitlines()
            if x.strip()
        ]

        self.trajectories = self._build_trajectories()

        # Keep the old name available for compatibility with retrieval code.
        self.episodes = [
            self.trajectory_to_episode(t)
            for t in self.trajectories
        ]

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @classmethod
    def is_action(cls, line: str) -> bool:
        return bool(
            cls.ACTION_RE.match(line)
        )

    @classmethod
    def is_state(cls, line: str) -> bool:
        return bool(
            cls.STATE_RE.match(line)
        )

    def _build_trajectories(self) -> List[Trajectory]:
        """
        Convert a flat sequence of statements into causal trajectories.

        A trajectory starts with a state.

        Robot statements are actions.

        A non-robot statement following an action is treated as the
        observed consequence / resulting state of that action.

        Consecutive robot actions are therefore allowed.

        Example:

            cold
            robot enters bedroom
            robot turns on heater
            begins warming
            warm

        is retained as ONE trajectory rather than split into multiple
        episodes.
        """

        if not self.lines:
            return []

        trajectories = []
        current_states = []
        current_actions = []

        for line in self.lines:

            if self.is_state(line):
                current_states.append(line)

            elif self.is_action(line):
                current_actions.append(line)

        # --------------------------------------------------------------
        # The supplied data format is a single continuous trajectory.
        #
        # For this kind of household/event data, all contiguous statements
        # belong to one causal chain unless there is an explicit delimiter.
        # --------------------------------------------------------------

        initial_state = None
        final_state = None
        steps = []

        states = [
            line
            for line in self.lines
            if self.is_state(line)
        ]

        actions = [
            line
            for line in self.lines
            if self.is_action(line)
        ]

        if states:
            initial_state = states[0]
            final_state = states[-1]

        # --------------------------------------------------------------
        # Build causal transitions.
        #
        # The parser associates the state immediately preceding an action
        # with that action. The first subsequent state is its observed
        # consequence.
        #
        # Consecutive actions are valid:
        #
        #   cold
        #   ENTER
        #   HEATER ON
        #   warming
        #   warm
        #
        # ENTER gets cold -> ENTER -> no explicit state change.
        # HEATER ON gets cold -> HEATER ON -> warming.
        # The final "warm" becomes the terminal state.
        # --------------------------------------------------------------

        state_before = initial_state
        pending_action = None
        observed_states = []

        for line in self.lines:

            if self.is_action(line):

                # If another action is already pending, preserve it as a
                # transition without inventing a consequence.
                if pending_action is not None:
                    steps.append(
                        TrajectoryStep(
                            state=state_before,
                            action=pending_action,
                            consequence=None,
                            next_state=state_before,
                        )
                    )

                pending_action = line

            else:
                # Non-robot statement = observed consequence/state.
                if pending_action is not None:

                    steps.append(
                        TrajectoryStep(
                            state=state_before,
                            action=pending_action,
                            consequence=line,
                            next_state=line,
                        )
                    )

                    state_before = line
                    pending_action = None

                else:
                    # A state with no pending action becomes the current
                    # state. This handles the initial state and any explicit
                    # intermediate state.
                    state_before = line

        # Flush an action at the end of the trajectory.
        if pending_action is not None:
            steps.append(
                TrajectoryStep(
                    state=state_before,
                    action=pending_action,
                    consequence=None,
                    next_state=state_before,
                )
            )

        if steps:
            trajectories.append(
                Trajectory(
                    initial_state=initial_state,
                    steps=steps,
                    final_state=final_state,
                )
            )

        return trajectories

    # ------------------------------------------------------------------
    # Compatibility representation
    # ------------------------------------------------------------------

    @staticmethod
    def trajectory_to_episode(
        trajectory: Trajectory,
    ) -> List[str]:
        """
        Preserve compatibility with the existing retrieval layer.

        The returned episode contains the original statements in causal
        order.
        """

        lines = []

        if trajectory.initial_state:
            lines.append(
                trajectory.initial_state
            )

        for step in trajectory.steps:

            if step.action:
                lines.append(step.action)

            if (
                step.consequence is not None
                and step.consequence not in lines
            ):
                lines.append(
                    step.consequence
                )

        if (
            trajectory.final_state is not None
            and trajectory.final_state not in lines
        ):
            lines.append(
                trajectory.final_state
            )

        return lines

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _tokens(text: str):
        return set(
            re.findall(
                r"[a-z0-9']+",
                text.lower(),
            )
        )

    def retrieve(
        self,
        query: str,
        k: int = 5,
    ):
        """
        Lexical retrieval over complete trajectories.

        Returns:
            [(score, trajectory), ...]
        """

        q = self._tokens(query)
        scored = []

        for trajectory in self.trajectories:

            episode = self.trajectory_to_episode(
                trajectory
            )

            ep_text = " ".join(
                episode
            )

            tokens = self._tokens(
                ep_text
            )

            if not q or not tokens:
                score = 0.0
            else:
                overlap = len(
                    q & tokens
                )

                score = (
                    overlap
                    / math.sqrt(len(tokens))
                )

            scored.append(
                (score, trajectory)
            )

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        return scored[:k]

    def search(
        self,
        query: str,
        k: int = 5,
    ):
        results = self.retrieve(
            query,
            k,
        )

        output = []

        for score, trajectory in results:

            episode = self.trajectory_to_episode(
                trajectory
            )

            output.append(
                {
                    "score": score,
                    "trajectory": trajectory,
                    "episode": episode,
                    "text": " ".join(
                        episode
                    ),
                }
            )

        return output

    def context(
        self,
        query: str,
        k: int = 5,
    ):
        return "\n".join(
            result["text"]
            for result in self.search(
                query,
                k,
            )
        )

    def all_text(self):
        return "\n".join(
            self.lines
        )

    # ------------------------------------------------------------------
    # Causal trajectory access
    # ------------------------------------------------------------------

    def get_trajectory(
        self,
        index: int = 0,
    ) -> Optional[Trajectory]:

        if not self.trajectories:
            return None

        return self.trajectories[index]

    def pretty_trajectory(
        self,
        index: int = 0,
    ) -> str:

        trajectory = self.get_trajectory(
            index
        )

        if trajectory is None:
            return "No trajectory."

        lines = []

        lines.append(
            f"INITIAL STATE: "
            f"{trajectory.initial_state}"
        )

        for i, step in enumerate(
            trajectory.steps,
            start=1,
        ):
            lines.append(
                f"\nSTEP {i}"
            )

            lines.append(
                f"  state:       {step.state}"
            )

            lines.append(
                f"  action:      {step.action}"
            )

            lines.append(
                f"  consequence: "
                f"{step.consequence}"
            )

            lines.append(
                f"  next_state:  "
                f"{step.next_state}"
            )

        lines.append(
            f"\nFINAL STATE: "
            f"{trajectory.final_state}"
        )

        return "\n".join(lines)


def run_text_kb_query(
    kb_path: str,
    query: str,
    k: int = 5,
):
    """
    Demonstration entry point for data.txt.

    This intentionally separates retrieval from neural planning. The retrieved
    text is the conditioning context. To use it with the neural planner, feed
    the retrieved context through a text encoder and use its vector as the
    planner condition.
    """
    kb = PlainTextKB(kb_path)

    print(f"Loaded {len(kb.lines)} statements")
    print(f"Detected {len(kb.episodes)} episodes")
    print(f"\nQUERY: {query}\n")
    print("RETRIEVED KB CONTEXT")
    print("-" * 80)

    for i, result in enumerate(kb.search(query, k), 1):
        print(f"[{i}] score={result['score']:.4f}")
        print(result["text"])
        print()


# ---------------------------------------------------------------------------
# Text-conditioned neural planner bridge
# ---------------------------------------------------------------------------
class TextBellmanPlannerNet(nn.Module):
    """
    Text-native version of the Bellman planner.

    The query and retrieved KB context are converted into a dense condition
    vector. The rest of the architecture remains compatible with the
    recursive Bellman planner.
    """

    def __init__(
        self,
        d_model=128,
        gamma=0.95,
    ):
        super().__init__()

        self.d_model = d_model
        self.gamma = gamma

        self.action_emb = nn.Embedding(
            N_ACTIONS,
            d_model,
        )

        self.node_encoder = nn.Sequential(
            nn.Linear(5 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.policy_head = nn.Linear(
            d_model,
            N_ACTIONS,
        )

        self.q_head = nn.Linear(
            d_model,
            N_ACTIONS,
        )

        self.residue_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, N_ACTIONS),
            nn.Softplus(),
        )

        self.value_head = nn.Linear(
            d_model,
            1,
        )

        self.consequence_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.recurse_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def encode_node(
        self,
        parent,
        action_id,
        outcome,
        consequence,
        condition,
    ):
        device = parent.device

        action = self.action_emb(
            torch.tensor(
                action_id,
                dtype=torch.long,
                device=device,
            )
        )

        x = torch.cat(
            [
                parent,
                action,
                outcome,
                consequence,
                condition,
            ],
            dim=-1,
        )

        return self.node_encoder(x)

    def evaluate(self, z):
        logits = self.policy_head(z)
        q = self.q_head(z)
        residue = self.residue_head(z)

        return {
            "policy_logits": logits,
            "policy": F.softmax(logits, dim=-1),
            "q": q,
            "residue": residue,
            "value": self.value_head(z).squeeze(-1),
        }

    def forecast(self, z):
        consequence = self.consequence_head(
            torch.cat([z, z], dim=-1)
        )

        return consequence

    def make_recursive_child(
        self,
        z,
        condition,
    ):
        return self.recurse_head(
            torch.cat(
                [z, condition],
                dim=-1,
            )
        )

class TextRecursiveBellmanPlanner:
    """
    Recursive Bellman planner operating over natural-language KB episodes.

    Retrieval supplies the evidence.
    The neural model decides whether to recurse or terminate.
    The selected episode supplies the interpretable action sequence.
    """

    def __init__(
        self,
        model,
        encoder,
        kb,
        d_model=128,
        device="cpu",
        residue_threshold=0.20,
    ):
        self.model = model.to(device)
        self.encoder = encoder.to(device)
        self.kb = kb

        self.d_model = d_model
        self.device = device
        self.residue_threshold = residue_threshold

    @staticmethod
    def action_name(action):
        return [
            "CLASSIFY",
            "FORECAST",
            "RECURSE",
        ][action]

    def _condition(
        self,
        query,
        retrieved,
    ):
        context = "\n".join(
            item["text"]
            for item in retrieved
        )

        text = (
            "QUERY:\n"
            + query
            + "\n\n"
            + "KNOWLEDGE BASE:\n"
            + context
        )

        return self.encoder(text).squeeze(0)

    def _root(
        self,
        condition,
    ):
        zero = torch.zeros(
            self.d_model,
            device=self.device,
        )

        return self.model.encode_node(
            parent=zero,
            action_id=RECURSE,
            outcome=zero,
            consequence=zero,
            condition=condition,
        )

    def _select_action(self, z):
        info = self.model.evaluate(z)

        priority = (
            info["policy"]
            * info["residue"]
        )

        action = int(
            priority.argmax().item()
        )

        return action, info

    def _extract_answer(self, trajectory):
        """
        Return the actual causal trajectory rather than treating the episode
        as an unordered bag of robot actions.
        """

        if trajectory is None:
            return None

        steps = []

        for step in trajectory.steps:
            steps.append({
                "state": step.state,
                "action": step.action,
                "consequence": step.consequence,
                "next_state": step.next_state,
            })

        return {
            "initial_state": trajectory.initial_state,
            "steps": steps,
            "final_state": trajectory.final_state,
        }

    def search(
        self,
        query,
        top_k=5,
        budget=4,
    ):
        self.model.eval()
        self.encoder.eval()

        retrieved = self.kb.search(
            query,
            k=top_k,
        )

        if not retrieved:
            return {
                "query": query,
                "answer": None,
                "trace": [],
                "retrieved": [],
            }

        condition = self._condition(
            query,
            retrieved,
        )

        z = self._root(condition)

        trace = []

        for depth in range(budget + 1):

            action, info = self._select_action(z)

            residue = float(
                info["residue"][action].item()
            )

            trace.append({
                "depth": depth,
                "action": self.action_name(action),
                "residue": residue,
            })

            # Terminal action.
            if action in (
                CLASSIFY,
                FORECAST,
            ):
                break

            # Residue says additional recursion
            # is not worthwhile.
            if residue < self.residue_threshold:
                trace.append({
                    "depth": depth,
                    "action": "RESIDUE_STOP",
                    "residue": residue,
                })
                break

            z = self.model.make_recursive_child(
                z,
                condition,
            )

        # Select the strongest retrieved trajectory.
        best = retrieved[0]

        answer = self._extract_answer(
            best["trajectory"]
        )

        return {
            "query": query,
            "answer": answer,
            "trace": trace,
            "retrieved": retrieved,
        }

class TextConditionEncoder(nn.Module):
    """
    Dependency-free text encoder for the supplied data.txt.

    It uses hashed bag-of-words features followed by a learned projection.
    This is deliberately simple and reproducible. For serious experiments,
    replace it with a pretrained Transformer encoder.
    """

    def __init__(self, d_model=128, n_features=4096):
        super().__init__()
        self.n_features = n_features
        self.proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def _hash(self, token):
        # Stable hash, independent of Python's randomized hash seed.
        h = 2166136261
        for ch in token:
            h ^= ord(ch)
            h = (h * 16777619) & 0xffffffff
        return h % self.n_features

    def forward(self, text):
        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)

        x = torch.zeros(
            len(texts),
            self.n_features,
            device=next(self.parameters()).device,
        )

        for row, item in enumerate(texts):
            tokens = re.findall(
                r"[a-z0-9']+",
                item.lower(),
            )
            for token in tokens:
                x[row, self._hash(token)] += 1.0

            # TF normalization.
            if tokens:
                x[row] /= math.sqrt(len(tokens))

        return self.proj(x)




def text_kb_demo(
    kb_path="data.txt",
    query="What becomes warm after the robot turns on the heater?",
    k=5,
    d_model=128,
    device="cpu",
):
    """
    End-to-end KB ingestion demonstration:
        data.txt -> retrieval -> neural context vector.

    The returned vector is the exact object that should condition a
    text-native version of BellmanPlannerNet.
    """
    kb = PlainTextKB(kb_path)

    retrieved = kb.search(query, k=k)
    context = "\n".join(
        x["text"]
        for x in retrieved
    )

    encoder = TextConditionEncoder(
        d_model=d_model
    ).to(device)

    with torch.no_grad():
        condition = encoder(context)

    print(f"Loaded {len(kb.lines)} statements")
    print(f"Detected {len(kb.episodes)} episodes")
    print(f"Retrieved {len(retrieved)} episodes")
    print(f"Condition shape: {tuple(condition.shape)}")
    print("\nRetrieved context:\n")
    print(context)

    return condition, retrieved


# ---------------------------------------------------------------------------
# CLI for a plain data.txt
# ---------------------------------------------------------------------------

def text_kb_cli():
    parser = argparse.ArgumentParser(
        description="Query a plain-text data.txt knowledge base"
    )
    parser.add_argument(
        "--kb",
        default="data.txt",
        help="Path to data.txt",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Natural-language goal/query",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    args = parser.parse_args()

    run_text_kb_query(
        args.kb,
        args.query,
        args.top_k,
    )

def text_planner_cli():
    parser = argparse.ArgumentParser(
        description=(
            "Recursive Bellman planner over a "
            "plain-text knowledge base"
        )
    )

    parser.add_argument(
        "--kb",
        default="data.txt",
        help="Path to the text knowledge base",
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Natural-language question",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    args = parser.parse_args()

    print("=" * 80)
    print("RECURSIVE BELLMAN TEXT PLANNER")
    print("=" * 80)

    kb = PlainTextKB(args.kb)

    print(
        f"Loaded {len(kb.lines)} statements"
    )

    print(
        f"Detected {len(kb.episodes)} episodes"
    )

    encoder = TextConditionEncoder(
        d_model=args.d_model
    ).to(args.device)

    model = TextBellmanPlannerNet(
        d_model=args.d_model
    ).to(args.device)

    planner = TextRecursiveBellmanPlanner(
        model=model,
        encoder=encoder,
        kb=kb,
        d_model=args.d_model,
        device=args.device,
    )

    result = planner.search(
        query=args.query,
        top_k=args.top_k,
        budget=args.budget,
    )

    print("\nQUERY")
    print("-" * 80)
    print(args.query)

    print("\nRETRIEVED TRAJECTORIES")
    print("-" * 80)

    for i, item in enumerate(
        result["retrieved"],
        start=1,
    ):
        print(
            f"\n[{i}] score={item['score']:.4f}"
        )

        for line in item["episode"]:
            print("   ", line)

    print("\nBELLMAN PLANNER TRACE")
    print("-" * 80)

    for step in result["trace"]:
        print(
            f"depth={step['depth']} "
            f"action={step['action']} "
            f"residue={step['residue']:.4f}"
        )

    answer = result["answer"]

    print("\nANSWER")
    print("-" * 80)

    if answer is None:
        print("No relevant trajectory was found.")
        return

    print(
        f"Initial state: "
        f"{answer['initial_state']}"
    )

    for i, step in enumerate(
        answer["steps"],
        start=1,
    ):
        print(f"\nStep {i}")
        print(f"  State:       {step['state']}")
        print(f"  Action:      {step['action']}")
        print(
            f"  Consequence: "
            f"{step['consequence']}"
        )
        print(
            f"  Next state:  "
            f"{step['next_state']}"
        )

    print(
        f"\nFinal state: "
        f"{answer['final_state']}"
    )

# if __name__ == "__main__" and False:
#     # Kept disabled because the main training CLI above remains the default.
#     text_kb_cli()

if __name__ == "__main__":
    text_planner_cli()
    # text_kb_cli()


"""
                         data.txt
                            │
                     parse trajectories
                            │
             ┌──────────────┴──────────────┐
             │                             │
       state/action                   consequence
          latents                        latent
             │                             │
             └──────────────┬──────────────┘
                            │
                       KB encoder
                            │
                         c_KB
                            │
                 ┌──────────▼──────────┐
                 │   Query / Goal      │
                 └──────────┬──────────┘
                            │
                     recursive planner
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
          CLASSIFY       FORECAST        RECURSE
             │              │              │
             │              ▼              │
             │         consequence         │
             │              │              │
             │              └──────┐       │
             │                     ▼       ▼
             │                 Bellman backup
             │                     │
             │                residue δ
             │                     │
             │              ┌──────┴──────┐
             │              │             │
             │          low |δ|       high |δ|
             │              │             │
             ▼              ▼             ▼
            STOP           STOP        RECURSE
"""