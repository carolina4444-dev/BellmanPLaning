#!/usr/bin/env python3
"""
planner.py

Trajectory-aware latent Bellman planner with trajectory-safe windows,
per-step transition supervision, temporal action InfoNCE, a non-collapsing
recursive planner, and an EOS-aware hierarchical decoder.

Requirements:
    pip install tensorflow numpy

Example:
    python planner.py --data data.txt --epochs 150 --depth 4 \
        --state "The bedroom is cold." --goal "The bedroom is warm."

The corpus contains independent trajectories separated by blank lines.
Each non-empty line is one observation/event.
"""

import argparse
import os
import random
import re
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# ============================================================
# CONFIG
# ============================================================
SEED = 42
VOCAB_SIZE = 20000
MAX_SEQ_LEN = 48
D_MODEL = 192
D_ACTION = 160
D_PLAN = 192
D_VALUE = 96
NUM_INCEPTION_BLOCKS = 3
NUM_ACTION_CANDIDATES = 8
PLANNER_DEPTH = 4
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
GAMMA = 0.97
CONTRASTIVE_TEMPERATURE = 0.12
TEMPORAL_ACTION_TEMPERATURE = 0.10
PLANNER_TEMPERATURE = 0.18
PAD_ID = 0
MASK_ID = 1
CLS_ID = 2
EOS_ID = 3
SPECIAL_TOKENS = 4
MAX_SEGMENTS_PER_TRAJECTORY = 128

W_TRANSITION = 1.50
W_RECURSIVE_STATE = 1.75
W_RECURSIVE_TRANSITION = 1.75
W_BELLMAN = 0.20
W_TERMINAL_GOAL = 1.50
W_PLAN_GOAL = 0.65
W_DECODER = 1.00
W_ACTION_LANGUAGE = 1.25
W_TEMPORAL_ACTION_NCE = 0.90
W_STATE_CONTRAST = 0.20
W_CANDIDATE_ACTION = 0.90
W_CANDIDATE_TRANSITION = 1.50
W_CANDIDATE_DIVERSITY = 0.25
W_ACTION_SEPARATION = 0.35
W_RECURSIVE_ACTION = 1.25
W_PLAN_ALIGNMENT = 1.50
W_PLAN_LANGUAGE_ALIGNMENT = 2.0
W_TERMINATION = 1.0
W_DECODER_GROUNDING = 1.25

LABEL_SMOOTHING = 0.04
REPETITION_PENALTY = 1.0
BIGRAM_BLOCKING = True
MIN_GENERATION_TOKENS = 2

# ============================================================
# REPRODUCIBILITY
# ============================================================
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# TEXT / DATASET
# ============================================================
def normalize_segment(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def segment_text(text: str) -> List[str]:
    """Split within a trajectory without treating every sentence as a new trajectory."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Lines are the primary event boundary. Also accept simple arrow-delimited text.
    lines = []
    for line in text.split("\n"):
        line = normalize_segment(line)
        if not line or re.fullmatch(r"[-*_]{3,}", line):
            continue
        # Preserve normal prose containing punctuation; only split explicit arrows.
        parts = re.split(r"\s*(?:→|->)\s*", line)
        lines.extend(parts)
    out = []
    for piece in lines:
        piece = normalize_segment(piece)
        if len(piece) >= 2:
            out.append(piece)
    return out


def read_trajectories(path: str) -> List[List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find corpus: {path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        raise ValueError(f"Corpus is empty: {path}")

    # Blank lines are the trajectory delimiter. Do not accidentally collapse
    # single-line observations into neighboring trajectories.
    blocks = re.split(r"\n\s*\n+", text.strip())
    trajectories = []
    for block in blocks:
        segments = segment_text(block)
        if len(segments) >= 2:
            trajectories.append(segments[:MAX_SEGMENTS_PER_TRAJECTORY])
    if not trajectories:
        raise ValueError("No trajectories containing at least two usable segments.")
    return trajectories


def build_vectorizer(trajectories):
    texts = [s for t in trajectories for s in t]
    vectorizer = layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_SEQ_LEN - 1,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
    )
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(texts).batch(256))
    return vectorizer


def encode_texts(texts, vectorizer):
    x = tf.cast(vectorizer(tf.convert_to_tensor(texts, dtype=tf.string)), tf.int32)
    # TextVectorization reserves 0 for padding and uses 1 for OOV. Shift all
    # vectorizer ids by SPECIAL_TOKENS so our four control ids are disjoint.
    x = tf.where(x > 0, x + SPECIAL_TOKENS, tf.cast(PAD_ID, tf.int32))
    cls = tf.fill([tf.shape(x)[0], 1], tf.cast(CLS_ID, tf.int32))
    return tf.concat([cls, x], axis=1)[:, :MAX_SEQ_LEN]


def augment_text(text: str) -> str:
    words = text.split()
    if len(words) < 6 or random.random() >= 0.35:
        return text
    idxs = list(range(1, len(words) - 1))
    if idxs:
        del words[random.choice(idxs)]
    return " ".join(words)


def make_training_examples(trajectories, depth):
    examples = []
    for trajectory in trajectories:
        n = len(trajectory)
        for start in range(n - 1):
            horizon = min(depth, n - start - 1)
            if horizon < 1:
                continue
            states = list(trajectory[start:start + horizon])
            next_states = list(trajectory[start + 1:start + horizon + 1])
            goal = next_states[-1]
            while len(states) < depth:
                states.append("")
                next_states.append("")
            examples.append({
                "state": trajectory[start],
                "goal": goal,
                "step_states": states,
                "next_states": next_states,
                "action_texts": list(next_states),
                "horizon": horizon,
            })
    if not examples:
        raise ValueError("Could not create trajectory training windows.")
    return examples


def make_dataset(examples, vectorizer, depth, batch_size):
    n = len(examples)
    state_ids = encode_texts([x["state"] for x in examples], vectorizer)
    goal_ids = encode_texts([x["goal"] for x in examples], vectorizer)
    augmented_state_ids = encode_texts(
        [augment_text(x["state"]) for x in examples], vectorizer
    )
    flat_step_states = [text for row in [x["step_states"] for x in examples] for text in row]
    flat_next_states = [text for row in [x["next_states"] for x in examples] for text in row]
    step_state_ids = tf.reshape(
        encode_texts(flat_step_states, vectorizer), [n, depth, MAX_SEQ_LEN]
    )
    next_state_ids = tf.reshape(
        encode_texts(flat_next_states, vectorizer), [n, depth, MAX_SEQ_LEN]
    )
    # Decoder input/target are built from a clean CLS -> content -> EOS sequence.
    action_ids = next_state_ids
    horizons = np.asarray([x["horizon"] for x in examples], dtype=np.float32)
    step_idx = np.arange(depth, dtype=np.float32)[None, :]
    step_mask = (step_idx < horizons[:, None]).astype(np.float32)
    termination_targets = np.zeros(
        (n, depth),
        dtype=np.float32,
    )

    for i, horizon in enumerate(horizons.astype(np.int32)):
        if horizon > 0:
            termination_targets[i, horizon - 1] = 1.0


    ds = tf.data.Dataset.from_tensor_slices({
        "state_ids": state_ids,
        "goal_ids": goal_ids,
        "step_state_ids": step_state_ids,
        "next_state_ids": next_state_ids,
        "action_ids": action_ids,
        "step_mask": step_mask,
        "augmented_state_ids": augmented_state_ids,
        "termination_targets": termination_targets,
    })
    return ds.shuffle(
        min(n, 4096), seed=SEED, reshuffle_each_iteration=True
    ).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


# ============================================================
# ENCODER
# ============================================================
class Inception1DBlock(layers.Layer):
    def __init__(self, d_model, dropout=0.10):
        super().__init__()
        bottleneck = max(d_model // 2, 1)
        self.norm = layers.LayerNormalization()
        self.b1 = layers.Conv1D(bottleneck, 1, padding="same", activation="gelu")
        self.b3 = layers.Conv1D(bottleneck, 3, padding="same", activation="gelu")
        self.b5 = layers.Conv1D(bottleneck, 5, padding="same", activation="gelu")
        self.pool = layers.MaxPooling1D(3, strides=1, padding="same")
        self.bp = layers.Conv1D(bottleneck, 1, padding="same", activation="gelu")
        self.project = layers.Conv1D(d_model, 1, padding="same")
        self.dropout = layers.Dropout(dropout)
        self.post = layers.LayerNormalization()

    def call(self, x, training=False):
        residual = x
        y = self.norm(x)
        y = tf.concat([self.b1(y), self.b3(y), self.b5(y), self.bp(self.pool(y))], axis=-1)
        y = self.project(y)
        y = self.dropout(y, training=training)
        return self.post(residual + y)


class InceptionStateEncoder(layers.Layer):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = layers.Embedding(vocab_size, D_MODEL)
        self.position_embedding = layers.Embedding(MAX_SEQ_LEN, D_MODEL)
        self.blocks = [Inception1DBlock(D_MODEL) for _ in range(NUM_INCEPTION_BLOCKS)]
        self.pool_norm = layers.LayerNormalization()
        self.proj = layers.Dense(D_MODEL, activation="gelu")
        self.state_norm = layers.LayerNormalization()
        self.dropout = layers.Dropout(0.10)
        self.value_head = tf.keras.Sequential([
            layers.Dense(D_VALUE, activation="gelu"), layers.Dense(1)
        ])

    def call(self, ids, training=False):
        ids = tf.cast(ids, tf.int32)
        length = tf.shape(ids)[1]
        pos = tf.range(length)[None, :]
        x = self.token_embedding(ids) + self.position_embedding(pos)
        for block in self.blocks:
            x = block(x, training=training)
        mask = tf.cast(tf.not_equal(ids, PAD_ID), tf.float32)[..., None]
        x = x * mask
        denom = tf.maximum(tf.reduce_sum(mask, axis=1), 1e-6)
        pooled = tf.reduce_sum(x, axis=1) / denom
        pooled = self.pool_norm(pooled)
        pooled = self.dropout(pooled, training=training)
        state = self.state_norm(self.proj(pooled))
        state = tf.math.l2_normalize(state, axis=-1)
        value = self.value_head(state)[..., 0]
        return {"token_states": x, "state": state, "value": value}


# ============================================================
# LATENT MODELS
# ============================================================
class GoalConditioner(layers.Layer):
    def __init__(self):
        super().__init__()

        self.d1 = layers.Dense(
            2 * D_MODEL,
            activation="gelu",
        )
        self.d2 = layers.Dense(
            D_MODEL,
            activation="gelu",
        )
        self.norm = layers.LayerNormalization()

    def call(self, state, goal):

        if state.shape.rank == 3:

            if goal.shape.rank == 2:
                goal = goal[:, None, :]

            goal = tf.broadcast_to(
                goal,
                tf.shape(state),
            )

        elif state.shape.rank == 2:

            if goal.shape.rank == 3:
                goal = goal[:, 0, :]

        else:
            raise ValueError(
                f"Unexpected state rank: {state.shape}"
            )

        x = tf.concat(
            [
                state,
                goal,
                goal - state,
                state * goal,
            ],
            axis=-1,
        )

        x = self.d1(x)
        x = self.d2(x)

        return self.norm(x)


class ActionModel(layers.Layer):
    def __init__(self):
        super().__init__()
        self.conditioner = GoalConditioner()
        self.hidden = tf.keras.Sequential([
            layers.Dense(2 * D_MODEL, activation="gelu"),
            layers.Dense(2 * D_MODEL, activation="gelu"),
            layers.LayerNormalization(),
        ])
        self.action_head = layers.Dense(NUM_ACTION_CANDIDATES * D_ACTION)
        self.score_head = layers.Dense(NUM_ACTION_CANDIDATES)

    def call(self, state, goal):
        context = self.conditioner(state, goal)
        hidden = self.hidden(context)
        actions = tf.reshape(
            self.action_head(hidden),
            [tf.shape(state)[0], NUM_ACTION_CANDIDATES, D_ACTION],
        )
        actions = tf.math.l2_normalize(actions, axis=-1)
        return {"context": context, "actions": actions, "logits": self.score_head(hidden)}


class ObservedActionEncoder(layers.Layer):
    """Infer an action representation from each observed S_t -> S_t+1 transition."""
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential([
            layers.Dense(2 * D_MODEL, activation="gelu"),
            layers.Dense(D_ACTION),
        ])

    def call(self, state, next_state):
        delta = next_state - state
        x = tf.concat([state, next_state, delta, state * next_state], axis=-1)
        return tf.math.l2_normalize(self.net(x), axis=-1)


class TransitionModel(layers.Layer):
    def __init__(self):
        super().__init__()

        self.d1 = layers.Dense(
            2 * D_MODEL,
            activation="gelu",
        )
        self.d2 = layers.Dense(
            2 * D_MODEL,
            activation="gelu",
        )
        self.out = layers.Dense(D_MODEL)
        self.norm = layers.LayerNormalization()

    def call(self, state, action):
        x = tf.concat(
            [state, action],
            axis=-1,
        )

        x = self.d1(x)
        x = self.d2(x)
        delta = self.out(x)

        return tf.math.l2_normalize(
            self.norm(state + delta),
            axis=-1,
        )


class IntrinsicRewardModel(layers.Layer):
    def __init__(self):
        super().__init__()

        self.d1 = layers.Dense(
            D_MODEL,
            activation="gelu",
        )
        self.d2 = layers.Dense(
            D_MODEL // 2,
            activation="gelu",
        )
        self.out = layers.Dense(1)

    def call(self, state, action, next_state, goal):

        # --------------------------------------------------------
        # Make goal have exactly the same leading dimensions as
        # state/next_state.
        #
        # [B,D]     -> [B,T,D]
        # [B,1,D]   -> [B,T,D]
        # [B,T,D]   -> [B,T,D]
        # --------------------------------------------------------

        if state.shape.rank == 3:

            if goal.shape.rank == 2:
                goal = goal[:, None, :]

            goal = tf.broadcast_to(
                goal,
                tf.shape(state),
            )

        elif state.shape.rank == 2:

            if goal.shape.rank == 3:
                goal = goal[:, 0, :]

        else:
            raise ValueError(
                f"Unexpected state rank: {state.shape}"
            )

        # --------------------------------------------------------
        # Explicitly verify the representation dimensions.
        #
        # state      D_MODEL = 192
        # action     D_ACTION = 160
        # next_state D_MODEL = 192
        # goal       D_MODEL = 192
        # delta      D_MODEL = 192
        #
        # concat = 928
        # --------------------------------------------------------

        delta = next_state - state

        progress = (
            tf.reduce_sum(next_state * goal, axis=-1)
            -
            tf.reduce_sum(state * goal, axis=-1)
        )

        x = tf.concat(
            [
                state,
                action,
                next_state,
                goal,
                delta,
            ],
            axis=-1,
        )

        hidden = self.d1(x)
        hidden = self.d2(hidden)

        learned = tf.tanh(
            self.out(hidden)[..., 0]
        )

        return progress + 0.03 * learned


class ValueModel(layers.Layer):
    def __init__(self):
        super().__init__()

        self.conditioner = GoalConditioner()

        self.d1 = layers.Dense(
            2 * D_MODEL,
            activation="gelu",
        )
        self.d2 = layers.Dense(
            D_MODEL,
            activation="gelu",
        )
        self.out = layers.Dense(1)

    def call(self, state, goal):
        x = self.conditioner(state, goal)
        x = self.d1(x)
        x = self.d2(x)
        return self.out(x)[..., 0]

class PlanEmbeddingModel(layers.Layer):

    def __init__(self):
        super().__init__()

        self.state_net = layers.Dense(
            D_PLAN,
            activation="gelu",
        )

        self.action_net = layers.Dense(
            D_PLAN,
            activation="gelu",
        )

        self.transition_net = layers.Dense(
            D_PLAN,
            activation="gelu",
        )

        self.goal_net = layers.Dense(
            D_PLAN,
            activation="gelu",
        )

        self.q_net = layers.Dense(
            D_PLAN,
            activation="gelu",
        )

        self.fusion = tf.keras.Sequential([
            layers.Dense(2 * D_PLAN, activation="gelu"),
            layers.Dense(D_PLAN),
            layers.LayerNormalization(),
        ])

        self.action_gate = layers.Dense(
            D_PLAN,
            activation="sigmoid",
        )

    def call(
        self,
        state,
        action,
        next_state,
        goal,
        q,
    ):
        state_x = self.state_net(state)
        action_x = self.action_net(action)

        goal_x = self.goal_net(goal)

        action_gate = self.action_gate(
            tf.concat([state_x, goal_x], axis=-1)
        )

        action_x = action_x * action_gate

        transition_x = self.transition_net(
            next_state - state
        )

        next_state_x = self.transition_net(
            next_state
        )

        

        # Always present Q as [N, 1].
        q = tf.reshape(q, [-1, 1])
        q_x = self.q_net(q)

        x = tf.concat(
            [
                state_x,
                action_x,
                transition_x,
                next_state_x,
                goal_x,
                q_x,
            ],
            axis=-1,
        )

        x = self.fusion(x)

        return tf.math.l2_normalize(
            x,
            axis=-1,
        )

class PlanSentenceProjection(layers.Layer):
    """
    Converts each grounded transition/action plan embedding into the
    representation consumed by the hierarchical language decoder.

    The projection is independently applied to every planning step, so:

        [P1, P2, ..., Pn]

    remains an ordered sequence.
    """

    def __init__(self):
        super().__init__()

        self.d1 = layers.Dense(
            2 * D_MODEL,
            activation="gelu",
        )

        self.d2 = layers.Dense(
            D_MODEL,
        )

        self.norm = layers.LayerNormalization()

        self.gate = layers.Dense(
            D_MODEL,
            activation="sigmoid",
        )

    def call(self, plan):
        shape = tf.shape(plan)

        flat = tf.reshape(
            plan,
            [-1, D_PLAN],
        )

        x = self.d1(flat)
        x = self.d2(x)
        x = self.norm(x)

        # Learned information gate.
        gate = self.gate(x)

        x = x * gate

        x = tf.math.l2_normalize(
            x,
            axis=-1,
        )

        return tf.reshape(
            x,
            tf.concat(
                [
                    shape[:-1],
                    [D_MODEL],
                ],
                axis=0,
            ),
        )

# ============================================================
# RECURSIVE PLANNER
# ============================================================
class RecursiveBellmanPlanner(layers.Layer):
    """
    Trajectory-grounded recursive Bellman planner.

    Each planning step explicitly represents:

        current_state
            |
            +-- candidate action
                    |
                    v
              predicted next state
                    |
                    +-- reward / value / goal progress
                    |
                    v
                 Q value

    The selected tuple

        (state_t, action_t, next_state_t, reward_t, q_t, goal)

    is converted into a plan embedding.

    During training, observed transitions can be supplied as anchors. This
    teaches the planner that plan step t corresponds to the actual transition
    S_t -> S_{t+1}.

    During inference, the planner is fully autoregressive:

        S_0 -> A_0 -> S_1 -> A_1 -> S_2 -> ...

    Therefore the decoder receives a sequence of embeddings that explicitly
    contains the transition/action information required to generate the
    natural-language plan.
    """

    def __init__(
        self,
        action_model,
        transition_model,
        reward_model,
        value_model,
        plan_embedding_model,
    ):
        super().__init__()

        self.action_model = action_model
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.value_model = value_model
        self.plan_embedding_model = plan_embedding_model

        # Additional trajectory-level projection.
        #
        # This lets the planner combine:
        #   state
        #   action
        #   transition
        #   reward
        #   Q
        #   goal
        #
        # into a richer representation before producing the final plan token.
        self.trajectory_projection = tf.keras.Sequential([
            layers.Dense(2 * D_PLAN, activation="gelu"),
            layers.Dense(D_PLAN),
            layers.LayerNormalization(),
        ])

        # Used to determine how strongly an observed action should anchor
        # candidate selection during training.
        self.anchor_temperature = 0.08

        self.termination_head = tf.keras.Sequential([
            layers.Dense(D_MODEL, activation="gelu"),
            layers.Dense(1),
        ])

    # ------------------------------------------------------------
    # Candidate evaluation
    # ------------------------------------------------------------

    def evaluate_candidates(self, state, goal):
        """
        Generate and evaluate K candidate actions from the current state.

        Returns:
            actions       [B, K, D_ACTION]
            next_states   [B, K, D_MODEL]
            rewards       [B, K]
            values        [B, K]
            q_values      [B, K]
            logits        [B, K]
        """

        proposal = self.action_model(state, goal)

        actions = proposal["actions"]
        proposal_logits = proposal["logits"]

        b = tf.shape(state)[0]
        k = tf.shape(actions)[1]
        d = tf.shape(state)[1]

        flat_state = tf.reshape(
            tf.broadcast_to(
                state[:, None, :],
                [b, k, d],
            ),
            [-1, d],
        )

        flat_goal = tf.reshape(
            tf.broadcast_to(
                goal[:, None, :],
                [b, k, tf.shape(goal)[1]],
            ),
            [-1, tf.shape(goal)[1]],
        )

        flat_actions = tf.reshape(
            actions,
            [-1, tf.shape(actions)[2]],
        )

        flat_next = self.transition_model(
            flat_state,
            flat_actions,
        )

        flat_reward = self.reward_model(
            flat_state,
            flat_actions,
            flat_next,
            flat_goal,
        )

        flat_value = self.value_model(
            flat_next,
            flat_goal,
        )

        next_states = tf.reshape(
            flat_next,
            [b, k, d],
        )

        rewards = tf.reshape(
            flat_reward,
            [b, k],
        )

        values = tf.reshape(
            flat_value,
            [b, k],
        )

        # Explicit goal progress.
        goal_similarity = tf.reduce_sum(
            next_states * goal[:, None, :],
            axis=-1,
        )

        # Bellman candidate value.
        q_values = (
            rewards
            + GAMMA * values
            + 0.20 * goal_similarity
            + 0.03 * proposal_logits
        )

        return (
            actions,
            next_states,
            rewards,
            values,
            q_values,
            proposal_logits,
        )

    # ------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------

    def select_candidate(
        self,
        q_values,
        actions,
        candidate_states=None,
        previous_action=None,
        observed_action=None,
        observed_next_state=None,
        training=False,
    ):
        adjusted_q = q_values

        if previous_action is not None:
            previous_similarity = tf.reduce_sum(
                actions * previous_action[:, None, :],
                axis=-1,
            )

            adjusted_q -= 0.40 * tf.nn.relu(
                previous_similarity - 0.20
            )

        if training and observed_action is not None:
            observed_action = tf.stop_gradient(
                tf.math.l2_normalize(observed_action, axis=-1)
            )

            action_similarity = tf.reduce_sum(
                tf.math.l2_normalize(actions, axis=-1)
                * observed_action[:, None, :],
                axis=-1,
            )

            adjusted_q += 0.45 * action_similarity

        if (
            training
            and observed_next_state is not None
            and candidate_states is not None
        ):
            observed_next_state = tf.stop_gradient(
                tf.math.l2_normalize(observed_next_state, axis=-1)
            )

            transition_similarity = tf.reduce_sum(
                tf.math.l2_normalize(candidate_states, axis=-1)
                * observed_next_state[:, None, :],
                axis=-1,
            )

            adjusted_q += 0.75 * transition_similarity

        weights = tf.nn.softmax(
            adjusted_q / PLANNER_TEMPERATURE,
            axis=-1,
        )

        index = tf.argmax(
            adjusted_q,
            axis=-1,
            output_type=tf.int32,
        )

        hard_weights = tf.one_hot(
            index,
            depth=tf.shape(q_values)[1],
            dtype=q_values.dtype,
        )

        weights = (
            tf.stop_gradient(hard_weights - weights)
            + weights
        )

        selected_action = tf.reduce_sum(
            actions * weights[..., None],
            axis=1,
        )

        selected_q = tf.reduce_sum(
            q_values * weights,
            axis=1,
        )

        return selected_action, selected_q, weights, index

    # ------------------------------------------------------------
    # One recursive planning step
    # ------------------------------------------------------------

    def recursive_step(
        self,
        state,
        goal,
        previous_action=None,
        observed_action=None,
        observed_next_state=None,
        training=False,
    ):
        """
        Perform one complete planning transition.

        Returns a fully grounded plan step.
        """

        (
            candidates,
            candidate_states,
            rewards,
            values,
            q_values,
            proposal_logits,
        ) = self.evaluate_candidates(
            state,
            goal,
        )

        (
            selected_action,
            selected_q,
            weights,
            selected_index,
        ) = self.select_candidate(
            q_values=q_values,
            actions=candidates,
            candidate_states=candidate_states,
            previous_action=previous_action,
            observed_action=observed_action,
            observed_next_state=observed_next_state,
            training=training,
        )

        # ------------------------------------------------------------
        # Predict the selected transition.
        # ------------------------------------------------------------

        predicted_next_state = self.transition_model(
            state,
            selected_action,
        )

        predicted_reward = self.reward_model(
            state,
            selected_action,
            predicted_next_state,
            goal,
        )

        predicted_value = self.value_model(
            predicted_next_state,
            goal,
        )

        predicted_q = (
            predicted_reward
            + GAMMA * predicted_value
            + 0.20 * tf.reduce_sum(
                predicted_next_state * goal,
                axis=-1,
            )
        )

        # Explicitly keep Q as [B].
        predicted_q = tf.reshape(
            predicted_q,
            [-1],
        )

        # ------------------------------------------------------------
        # Transition anchoring for the PLAN EMBEDDING only.
        #
        # The recursive state remains model-generated. The observed
        # next state is only used as a supervised representation anchor.
        # ------------------------------------------------------------

        if training and observed_next_state is not None:
            embedding_next_state = (
                0.75 * predicted_next_state
                + 0.25 * tf.stop_gradient(observed_next_state)
            )

            embedding_next_state = tf.math.l2_normalize(
                embedding_next_state,
                axis=-1,
            )
        else:
            embedding_next_state = predicted_next_state

        # ------------------------------------------------------------
        # Explicit plan embedding:
        #
        # P_t = f(
        #     S_t,
        #     A_t,
        #     S_{t+1},
        #     G,
        #     Q_t
        # )
        # ------------------------------------------------------------

        plan_embedding = self.plan_embedding_model(
            state,
            selected_action,
            embedding_next_state,
            goal,
            predicted_q,
        )

        # ------------------------------------------------------------
        # Additional trajectory-level representation.
        # ------------------------------------------------------------

        trajectory_features = tf.concat(
            [
                state,
                selected_action,
                embedding_next_state,
                goal,
                embedding_next_state - state,
                predicted_reward[..., None],
                predicted_q[..., None],
            ],
            axis=-1,
        )

        trajectory_embedding = self.trajectory_projection(
            trajectory_features
        )

        # Fuse the explicit transition/action representation with
        # the trajectory representation.
        plan_embedding = tf.math.l2_normalize(
            plan_embedding + trajectory_embedding,
            axis=-1,
        )

        termination_features = tf.concat(
            [
                state,
                predicted_next_state,
                selected_action,
                predicted_reward[..., None],
                predicted_q[..., None],
                goal,
            ],
            axis=-1,
        )

        termination_logit = self.termination_head(
            termination_features
        )[..., 0]

        return {
            "state": state,
            "action": selected_action,
            "next_state": predicted_next_state,
            "embedding_next_state": embedding_next_state,
            "reward": predicted_reward,
            "value": predicted_value,
            "q_value": predicted_q,
            "plan": plan_embedding,

            "candidate_actions": candidates,
            "candidate_states": candidate_states,
            "candidate_rewards": rewards,
            "candidate_values": values,
            "candidate_q_values": q_values,
            "candidate_logits": proposal_logits,
            "candidate_weights": weights,
            "selected_index": selected_index,

            "termination_logit": termination_logit,
        }

    # ------------------------------------------------------------
    # Full recursive trajectory
    # ------------------------------------------------------------

    def plan(
        self,
        state,
        goal,
        depth,
        observed_actions=None,
        observed_next_states=None,
        step_mask=None,
        training=False,
        hard=False,
        teacher_forcing=True
    ):
        """
        Construct an ordered trajectory of plan embeddings.

        Training:
            observed_actions / observed_next_states may be supplied to anchor
            each recursive step to the actual trajectory.

        Inference:
            both are None and the planner rolls forward autonomously.

        Returns:
            plan              [B, depth, D_PLAN]
            states            [B, depth, D_MODEL]
            actions           [B, depth, D_ACTION]
            next_states       [B, depth, D_MODEL]
            rewards           [B, depth]
            q_values          [B, depth]
            candidate_weights [B, depth, K]
        """

        current_state = state

        previous_action = None

        plans = []
        states = []
        actions = []
        next_states = []
        rewards = []
        values = []
        q_values = []
        termination_logits = []
        candidate_weights = []
        candidate_q_values = []
        selected_indices = []
        candidate_actions = []

        for t in range(depth):
            source_state = current_state

            observed_action_t = None
            observed_next_state_t = None

            if observed_actions is not None:
                observed_action_t = observed_actions[:, t, :]

            if observed_next_states is not None:
                observed_next_state_t = observed_next_states[:, t, :]

            result = self.recursive_step(
                state=current_state,
                goal=goal,
                previous_action=previous_action,
                observed_action=observed_action_t,
                observed_next_state=observed_next_state_t,
                training=training,
            )

            plans.append(result["plan"])

            states.append(result["state"])
            actions.append(result["action"])
            next_states.append(result["next_state"])

            rewards.append(result["reward"])
            values.append(result["value"])
            q_values.append(result["q_value"])
            termination_logits.append(result["termination_logit"])

            candidate_weights.append(
                result["candidate_weights"]
            )

            candidate_q_values.append(
                result["candidate_q_values"]
            )

            selected_indices.append(
                result["selected_index"]
            )

            candidate_actions.append(
                result["candidate_actions"]
            )

            current_state = result["next_state"]

            previous_action = result["action"]

        return {
            "plan": tf.stack(plans, axis=1),
            "states": tf.stack(states, axis=1),
            "actions": tf.stack(actions, axis=1),
            "next_states": tf.stack(next_states, axis=1),

            "rewards": tf.stack(rewards, axis=1),
            "values": tf.stack(values, axis=1),
            "q_values": tf.stack(q_values, axis=1),

            "candidate_weights": tf.stack(
                candidate_weights,
                axis=1,
            ),

            "candidate_q_values": tf.stack(
                candidate_q_values,
                axis=1,
            ),

            "selected_indices": tf.stack(
                selected_indices,
                axis=1,
            ),

            "final_state": current_state,

            "termination_logits": tf.stack(
                termination_logits,
                axis=1,
            ),
        }

def sample_token(
    logits,
    temperature=0.65,
    top_k=40,
    top_p=0.92,
):
    """
    Temperature + top-k + nucleus sampling.

    Much more stable than sampling directly from the full vocabulary.
    """

    logits = logits / max(float(temperature), 1e-5)

    vocab = tf.shape(logits)[-1]

    # ------------------------------------------------------------
    # Top-k
    # ------------------------------------------------------------

    k = tf.minimum(
        tf.cast(top_k, tf.int32),
        vocab,
    )

    top_values, top_indices = tf.math.top_k(
        logits,
        k=k,
    )

    # ------------------------------------------------------------
    # Top-p / nucleus filtering
    # ------------------------------------------------------------

    sorted_probs = tf.nn.softmax(top_values, axis=-1)

    cumulative = tf.cumsum(
        sorted_probs,
        axis=-1,
    )

    keep = cumulative <= top_p

    # Always retain at least the best token.
    keep = tf.concat(
        [
            tf.ones_like(keep[:, :1], dtype=tf.bool),
            keep[:, 1:],
        ],
        axis=1,
    )

    filtered_values = tf.where(
        keep,
        top_values,
        tf.fill(
            tf.shape(top_values),
            tf.cast(-1e9, top_values.dtype),
        ),
    )

    sampled = tf.random.categorical(
        filtered_values,
        num_samples=1,
    )[:, 0]

    return tf.gather(
        top_indices,
        sampled,
        batch_dims=1,
    )

# ============================================================
# HIERARCHICAL DECODER
# ============================================================
class HierarchicalPlanDecoder(layers.Layer):
    """
    Hierarchical autoregressive decoder.

    Level 1:
        plan embeddings -> ordered sentence contexts

    Level 2:
        sentence context + previous sentence context
        -> autoregressive token generation

    Training and inference share the same GRUCell transition.
    """

    def __init__(self, vocab_size, max_depth):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_depth = max_depth

        # --------------------------------------------------------
        # Sentence / plan level
        # --------------------------------------------------------

        self.step_embedding = layers.Embedding(
            max_depth,
            D_MODEL,
        )

        self.plan_norm = layers.LayerNormalization()

        self.step_gru = layers.GRU(
            D_MODEL,
            return_sequences=True,
        )

        # Previous sentence -> current sentence conditioning.
        self.previous_sentence_projection = layers.Dense(
            D_MODEL,
            activation="gelu",
        )

        self.sentence_fusion = tf.keras.Sequential([
            layers.Dense(
                2 * D_MODEL,
                activation="gelu",
            ),
            layers.Dense(D_MODEL),
            layers.LayerNormalization(),
        ])

        # --------------------------------------------------------
        # Token level
        # --------------------------------------------------------

        self.token_embedding = layers.Embedding(
            vocab_size,
            D_MODEL,
        )

        self.token_context_fusion = tf.keras.Sequential([
            layers.Dense(
                2 * D_MODEL,
                activation="gelu",
            ),
            layers.Dense(D_MODEL),
            layers.LayerNormalization(),
        ])

        self.token_gru_cell = layers.GRUCell(
            D_MODEL,
        )

        self.output_norm = layers.LayerNormalization()

        self.output_head = layers.Dense(
            vocab_size,
        )

        # Learnable EOS preference.
        self.eos_bias = self.add_weight(
            name="eos_bias",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.35),
            trainable=True,
        )

        # Controls how strongly previous sentence information enters.
        self.previous_sentence_gate = layers.Dense(
            D_MODEL,
            activation="sigmoid",
        )

    # ============================================================
    # SENTENCE LEVEL
    # ============================================================

    def compute_step_contexts(
        self,
        plan_sequence,
        training=False,
    ):
        depth = tf.shape(plan_sequence)[1]

        positions = tf.range(depth)

        position_embedding = self.step_embedding(
            positions
        )[None, :, :]

        x = self.plan_norm(
            plan_sequence + position_embedding
        )

        return self.step_gru(
            x,
            training=training,
        )

    # ============================================================
    # TOKEN INPUT
    # ============================================================

    def _prepare_token_input(
        self,
        token_ids,
        context,
    ):
        token_x = self.token_embedding(
            token_ids
        )

        if context.shape.rank == 2:
            context_tokens = context[:, None, :]
        else:
            context_tokens = context

        context_tokens = tf.broadcast_to(
            context_tokens,
            tf.shape(token_x),
        )

        return self.token_context_fusion(
            tf.concat(
                [
                    token_x,
                    context_tokens,
                ],
                axis=-1,
            )
        )

    # ============================================================
    # TEACHER FORCING
    # ============================================================

    def _teacher_force_logits(
        self,
        step_contexts,
        decoder_input_ids,
        training=False,
    ):
        """
        decoder_input_ids:

            [B, depth, T]

        Returns:

            [B, depth, T, vocab]
        """

        b = tf.shape(step_contexts)[0]
        depth = tf.shape(step_contexts)[1]
        t = tf.shape(decoder_input_ids)[2]

        # --------------------------------------------------------
        # Flatten sentence dimension.
        # --------------------------------------------------------

        flat_context = tf.reshape(
            step_contexts,
            [b * depth, D_MODEL],
        )

        flat_ids = tf.reshape(
            decoder_input_ids,
            [b * depth, t],
        )

        # --------------------------------------------------------
        # Process each sentence independently at token level.
        # --------------------------------------------------------

        token_x = self._prepare_token_input(
            flat_ids,
            flat_context,
        )

        state = flat_context

        outputs = []

        for i in range(MAX_SEQ_LEN - 1):

            yi, states = self.token_gru_cell(
                token_x[:, i, :],
                [state],
                training=training,
            )

            state = states[0]

            outputs.append(
                yi
            )

        y = tf.stack(
            outputs,
            axis=1,
        )

        y = self.output_norm(y)

        logits = self.output_head(y)

        return tf.reshape(
            logits,
            [
                b,
                depth,
                MAX_SEQ_LEN - 1,
                self.vocab_size,
            ],
        )

    # ============================================================
    # TRAINING CALL
    # ============================================================

    def call(
        self,
        plan_sequence,
        decoder_input_ids,
        training=False,
    ):
        contexts = self.compute_step_contexts(
            plan_sequence,
            training=training,
        )

        return self._teacher_force_logits(
            contexts,
            decoder_input_ids,
            training=training,
        )

    # ============================================================
    # REPETITION CONTROL
    # ============================================================

    def _apply_repetition_penalty(
        self,
        logits,
        generated,
        penalty=1.0,
    ):
        if not generated:
            return logits

        logits = tf.identity(logits)

        # Penalize all previously generated tokens.
        history = tf.stack(
            generated,
            axis=1,
        )

        history = tf.cast(
            history,
            tf.int32,
        )

        one_hot = tf.one_hot(
            history,
            self.vocab_size,
        )

        seen = tf.reduce_max(
            one_hot,
            axis=1,
        )

        logits = logits - (
            seen * penalty
        )

        return logits

    # ============================================================
    # BIGRAM BLOCKING
    # ============================================================

    def _apply_bigram_blocking(
        self,
        logits,
        generated,
    ):
        if len(generated) < 2:
            return logits

        batch_size = int(
            tf.shape(logits)[0].numpy()
        )

        sequences = tf.stack(
            generated,
            axis=1,
        ).numpy()

        masks = np.zeros(
            [
                batch_size,
                self.vocab_size,
            ],
            dtype=np.float32,
        )

        for b in range(batch_size):

            seq = sequences[b].tolist()

            previous = seq[-1]

            forbidden = set()

            for i in range(
                len(seq) - 1
            ):
                if seq[i] == previous:
                    forbidden.add(
                        seq[i + 1]
                    )

            for token_id in forbidden:
                if (
                    0 <= token_id
                    < self.vocab_size
                ):
                    masks[
                        b,
                        token_id,
                    ] = 1.0

        return logits - (
            tf.convert_to_tensor(
                masks,
                dtype=logits.dtype,
            )
            * 1e9
        )

    # ============================================================
    # GENERATION
    # ============================================================

    def generate(
        self,
        plan_sequence,
        max_length=MAX_SEQ_LEN,
        temperature=0.60,
        top_k=40,
        top_p=0.92,
        greedy=False,
        min_tokens=2,
    ):
        """
        Generate one sentence per planning step.

        Unlike the original implementation, each sentence is conditioned
        on the representation of the previous sentence. This gives:

            step 1 -> step 2 -> step 3 -> ...

        language continuity.
        """

        plan_sequence = tf.convert_to_tensor(
            plan_sequence,
            dtype=tf.float32,
        )

        contexts = self.compute_step_contexts(
            plan_sequence,
            training=False,
        )

        batch_size = tf.shape(
            plan_sequence
        )[0]

        depth = plan_sequence.shape[1]

        if depth is None:
            depth = int(
                tf.shape(plan_sequence)[1].numpy()
            )

        all_steps = []

        # --------------------------------------------------------
        # Previous sentence semantic state.
        # --------------------------------------------------------

        previous_sentence_state = tf.zeros(
            [
                batch_size,
                D_MODEL,
            ],
            dtype=tf.float32,
        )

        for step_idx in range(depth):

            plan_context = contexts[
                :,
                step_idx,
                :,
            ]

            # ----------------------------------------------------
            # Hierarchical sentence context.
            # ----------------------------------------------------

            previous_projected = (
                self.previous_sentence_projection(
                    previous_sentence_state
                )
            )

            gate = self.previous_sentence_gate(
                tf.concat(
                    [
                        plan_context,
                        previous_projected,
                    ],
                    axis=-1,
                )
            )

            previous_projected = (
                previous_projected * gate
            )

            context = self.sentence_fusion(
                tf.concat(
                    [
                        plan_context,
                        previous_projected,
                    ],
                    axis=-1,
                )
            )

            # ----------------------------------------------------
            # Token generation.
            # ----------------------------------------------------

            state = context

            current = tf.fill(
                [batch_size],
                tf.cast(
                    CLS_ID,
                    tf.int32,
                ),
            )

            generated = []

            finished = tf.zeros(
                [batch_size],
                dtype=tf.bool,
            )

            for t in range(
                max_length - 1
            ):

                token_x = (
                    self._prepare_token_input(
                        current[:, None],
                        context,
                    )[:, 0, :]
                )

                output, states = (
                    self.token_gru_cell(
                        token_x,
                        [state],
                        training=False,
                    )
                )

                state = states[0]

                logits = self.output_head(
                    self.output_norm(
                        output
                    )
                )

                logits = tf.identity(
                    logits
                )

                # ------------------------------------------------
                # Never generate control tokens.
                # ------------------------------------------------

                for token_id in (
                    PAD_ID,
                    MASK_ID,
                    CLS_ID,
                ):
                    ids = tf.fill(
                        [batch_size],
                        tf.cast(
                            token_id,
                            tf.int32,
                        ),
                    )

                    indices = tf.stack(
                        [
                            tf.range(
                                batch_size
                            ),
                            ids,
                        ],
                        axis=1,
                    )

                    logits = (
                        tf.tensor_scatter_nd_update(
                            logits,
                            indices,
                            tf.fill(
                                [batch_size],
                                tf.cast(
                                    -1e9,
                                    logits.dtype,
                                ),
                            ),
                        )
                    )

                # ------------------------------------------------
                # EOS.
                # ------------------------------------------------

                eos_ids = tf.fill(
                    [batch_size],
                    tf.cast(
                        EOS_ID,
                        tf.int32,
                    ),
                )

                eos_indices = tf.stack(
                    [
                        tf.range(
                            batch_size
                        ),
                        eos_ids,
                    ],
                    axis=1,
                )

                if t < min_tokens:
                    logits = (
                        tf.tensor_scatter_nd_update(
                            logits,
                            eos_indices,
                            tf.fill(
                                [batch_size],
                                tf.cast(
                                    -1e9,
                                    logits.dtype,
                                ),
                            )
                        )
                    )
                else:
                    logits = (
                        tf.tensor_scatter_nd_add(
                            logits,
                            eos_indices,
                            tf.fill(
                                [batch_size],
                                tf.cast(
                                    self.eos_bias,
                                    logits.dtype,
                                ),
                            )
                        )
                    )

                # ------------------------------------------------
                # Repetition penalty.
                # ------------------------------------------------

                logits = (
                    self._apply_repetition_penalty(
                        logits,
                        generated,
                        penalty=REPETITION_PENALTY,
                    )
                )

                # ------------------------------------------------
                # Bigram blocking.
                # ------------------------------------------------

                if BIGRAM_BLOCKING:
                    logits = (
                        self._apply_bigram_blocking(
                            logits,
                            generated,
                        )
                    )

                # ------------------------------------------------
                # Decode.
                # ------------------------------------------------

                if greedy:
                    next_token = tf.argmax(
                        logits,
                        axis=-1,
                        output_type=tf.int32,
                    )

                else:
                    next_token = sample_token(
                        logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )

                # Already-finished examples remain EOS.
                next_token = tf.where(
                    finished,
                    tf.fill(
                        [batch_size],
                        tf.cast(
                            EOS_ID,
                            tf.int32,
                        ),
                    ),
                    next_token,
                )

                generated.append(
                    next_token
                )

                finished = tf.logical_or(
                    finished,
                    tf.equal(
                        next_token,
                        EOS_ID,
                    ),
                )

                current = next_token

                if bool(
                    tf.reduce_all(
                        finished
                    ).numpy()
                ):
                    break

            # ----------------------------------------------------
            # Pad the generated sequence.
            # ----------------------------------------------------

            while len(generated) < max_length - 1:
                generated.append(
                    tf.fill(
                        [batch_size],
                        tf.cast(
                            EOS_ID,
                            tf.int32,
                        ),
                    )
                )

            generated_tensor = tf.stack(
                generated[
                    :max_length - 1
                ],
                axis=1,
            )

            all_steps.append(
                generated_tensor
            )

            # ----------------------------------------------------
            # IMPORTANT:
            #
            # Feed a semantic representation of the generated
            # sentence into the next sentence.
            #
            # We use the final GRU state rather than averaging tokens,
            # because it contains the autoregressive sentence history.
            # ----------------------------------------------------

            previous_sentence_state = state

        return tf.stack(
            all_steps,
            axis=1,
        )


################################################################################

# ============================================================
# GROUNDED HIERARCHICAL DECODER
# ============================================================

class GroundedHierarchicalPlanDecoder(layers.Layer):
    """
    Grounded hierarchical autoregressive decoder.

    Every planning step has an explicit grounding record:

        S_t
        A_t
        predicted S_{t+1}
        G
        reward_t
        Q_t
        plan_t

    The decoder creates several grounding tokens from these quantities
    and uses cross-attention at EVERY generated language token.

    This is substantially stronger than passing a single fused plan vector.

    Training:

        grounding_t + teacher-forced tokens -> target sentence

    Inference:

        grounding_t + autoregressive tokens -> generated sentence

    Sentence t also receives a semantic summary of sentence t-1, giving
    hierarchical continuity without allowing previous language to replace
    the actual state/action grounding.
    """

    def __init__(self, vocab_size, max_depth):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_depth = max_depth

        # --------------------------------------------------------
        # Step / temporal embeddings
        # --------------------------------------------------------

        self.step_embedding = layers.Embedding(
            max_depth,
            D_MODEL,
        )

        # --------------------------------------------------------
        # Individual grounding projections
        # --------------------------------------------------------

        self.state_projection = layers.Dense(
            D_MODEL,
            activation="gelu",
        )

        self.action_projection = layers.Dense(
            D_MODEL,
            activation="gelu",
        )

        self.next_state_projection = layers.Dense(
            D_MODEL,
            activation="gelu",
        )

        self.goal_projection = layers.Dense(
            D_MODEL,
            activation="gelu",
        )

        self.plan_projection = layers.Dense(
            D_MODEL,
            activation="gelu",
        )

        self.reward_projection = layers.Dense(
            D_MODEL,
            activation="gelu",
        )

        self.q_projection = layers.Dense(
            D_MODEL,
            activation="gelu",
        )

        # --------------------------------------------------------
        # Grounding token type embeddings
        #
        # These tell attention whether a vector represents state,
        # action, goal, etc.
        # --------------------------------------------------------

        self.grounding_type_embedding = layers.Embedding(
            7,
            D_MODEL,
        )

        self.grounding_norm = layers.LayerNormalization()

        # --------------------------------------------------------
        # Grounding attention
        # --------------------------------------------------------

        self.grounding_attention = layers.MultiHeadAttention(
            num_heads=6,
            key_dim=D_MODEL // 6,
            dropout=0.10,
        )

        self.grounding_attention_norm = layers.LayerNormalization()

        self.grounding_ffn = tf.keras.Sequential([
            layers.Dense(
                2 * D_MODEL,
                activation="gelu",
            ),
            layers.Dense(D_MODEL),
        ])

        self.grounding_ffn_norm = layers.LayerNormalization()

        # --------------------------------------------------------
        # Sentence-level hierarchy
        # --------------------------------------------------------

        self.sentence_gru = layers.GRU(
            D_MODEL,
            return_sequences=True,
        )

        self.previous_sentence_projection = layers.Dense(
            D_MODEL,
            activation="gelu",
        )

        self.previous_sentence_gate = layers.Dense(
            D_MODEL,
            activation="sigmoid",
        )

        self.sentence_fusion = tf.keras.Sequential([
            layers.Dense(
                2 * D_MODEL,
                activation="gelu",
            ),
            layers.Dense(D_MODEL),
            layers.LayerNormalization(),
        ])

        # --------------------------------------------------------
        # Token embedding
        # --------------------------------------------------------

        self.token_embedding = layers.Embedding(
            vocab_size,
            D_MODEL,
        )

        self.token_fusion = tf.keras.Sequential([
            layers.Dense(
                2 * D_MODEL,
                activation="gelu",
            ),
            layers.Dense(D_MODEL),
            layers.LayerNormalization(),
        ])

        # --------------------------------------------------------
        # Token GRU
        # --------------------------------------------------------

        self.token_gru_cell = layers.GRUCell(
            D_MODEL,
        )

        self.output_norm = layers.LayerNormalization()

        self.output_head = layers.Dense(
            vocab_size,
        )

        # --------------------------------------------------------
        # Learned grounding gate
        #
        # Prevents the language model from completely ignoring the
        # physical/latent transition representation.
        # --------------------------------------------------------

        self.grounding_gate = layers.Dense(
            D_MODEL,
            activation="sigmoid",
        )

        # --------------------------------------------------------
        # EOS
        # --------------------------------------------------------

        self.eos_bias = self.add_weight(
            name="grounded_eos_bias",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.25),
            trainable=True,
        )

        self.sentence_grounding_head = tf.keras.Sequential([
            layers.Dense(
                2 * D_MODEL,
                activation="gelu",
            ),
            layers.Dense(D_MODEL),
            layers.LayerNormalization(),
        ])

    # ============================================================
    # GROUNDING REPRESENTATION
    # ============================================================

    def build_grounding_tokens(
        self,
        states,
        actions,
        next_states,
        goal,
        rewards,
        q_values,
        plan,
    ):
        """
        Construct explicit grounding tokens.

        Inputs:

            states       [B,T,D_MODEL]
            actions      [B,T,D_ACTION]
            next_states  [B,T,D_MODEL]
            goal         [B,D_MODEL]
            rewards      [B,T]
            q_values     [B,T]
            plan         [B,T,D_PLAN]

        Returns:

            grounding_tokens [B,T,7,D_MODEL]
        """

        b = tf.shape(states)[0]
        t = tf.shape(states)[1]

        # --------------------------------------------------------
        # Broadcast goal over planning steps.
        # --------------------------------------------------------

        goal = tf.broadcast_to(
            goal[:, None, :],
            [b, t, D_MODEL],
        )

        # --------------------------------------------------------
        # Individual semantic projections.
        # --------------------------------------------------------

        state_x = self.state_projection(states)

        action_x = self.action_projection(actions)

        next_state_x = self.next_state_projection(
            next_states
        )

        goal_x = self.goal_projection(
            goal
        )

        reward_x = self.reward_projection(
            rewards[..., None]
        )

        q_x = self.q_projection(
            q_values[..., None]
        )

        plan_x = self.plan_projection(
            plan
        )

        tokens = tf.stack(
            [
                state_x,
                action_x,
                next_state_x,
                goal_x,
                reward_x,
                q_x,
                plan_x,
            ],
            axis=2,
        )

        # --------------------------------------------------------
        # Add token-type information.
        # --------------------------------------------------------

        type_ids = tf.range(
            7,
            dtype=tf.int32,
        )

        type_x = self.grounding_type_embedding(
            type_ids
        )

        type_x = type_x[None, None, :, :]

        tokens = tokens + type_x

        return self.grounding_norm(tokens)



    def sentence_grounding_embedding(
        self,
        grounding_tokens,
    ):
        """
        Produce one representation per grounded planning step.

        [B,T,7,D] -> [B,T,D]
        """

        x = tf.reduce_mean(
            grounding_tokens,
            axis=2,
        )

        return tf.math.l2_normalize(
            self.sentence_grounding_head(x),
            axis=-1,
        )
    # ============================================================
    # GROUNDING SELF-ORGANIZATION
    # ============================================================

    def process_grounding(
        self,
        grounding_tokens,
        training=False,
    ):
        """
        Let the seven grounding components interact before the
        token decoder consumes them.
        """

        b = tf.shape(grounding_tokens)[0]
        t = tf.shape(grounding_tokens)[1]

        flat = tf.reshape(
            grounding_tokens,
            [
                b * t,
                7,
                D_MODEL,
            ],
        )

        attended = self.grounding_attention(
            query=flat,
            value=flat,
            key=flat,
            training=training,
        )

        flat = self.grounding_attention_norm(
            flat + attended
        )

        ff = self.grounding_ffn(
            flat,
            training=training,
        )

        flat = self.grounding_ffn_norm(
            flat + ff
        )

        return tf.reshape(
            flat,
            [
                b,
                t,
                7,
                D_MODEL,
            ],
        )

    # ============================================================
    # SENTENCE CONTEXT
    # ============================================================

    def compute_sentence_contexts(
        self,
        grounding_tokens,
        plan_sequence,
        training=False,
    ):
        """
        Produce one high-level context per planning step.

        The sentence context is obtained from the entire grounding
        tuple, not merely from the plan embedding.
        """

        # Mean over the seven grounding channels.
        grounding_summary = tf.reduce_mean(
            grounding_tokens,
            axis=2,
        )

        step_count = tf.shape(plan_sequence)[1]

        positions = tf.range(
            step_count
        )

        step_x = self.step_embedding(
            positions
        )[None, :, :]

        x = grounding_summary + step_x

        x = self.sentence_gru(
            x,
            training=training,
        )

        return self.sentence_fusion(
            tf.concat(
                [
                    x,
                    plan_sequence,
                ],
                axis=-1,
            )
        )

    # ============================================================
    # TOKEN + GROUNDING ATTENTION
    # ============================================================

    def decode_token(
        self,
        token_ids,
        token_state,
        sentence_context,
        grounding_tokens,
        training=False,
    ):
        """
        Decode one token.
        """

        # GRUCell state is always [batch, hidden].
        if token_state.shape.rank == 3:
            token_state = tf.squeeze(token_state, axis=1)

        token_x = self.token_embedding(token_ids)

        token_x = self.token_fusion(
            tf.concat(
                [
                    token_x,
                    tf.squeeze(sentence_context, axis=1),
                ],
                axis=-1,
            )
        )

        # --------------------------------------------------------
        # Cross-attend token state to grounded transition.
        # --------------------------------------------------------

        query = tf.expand_dims(token_state, axis=1)

        attended = self.grounding_attention(
            query=query,
            key=grounding_tokens,
            value=grounding_tokens,
            training=training,
        )

        attended = attended[:, 0, :]

        # --------------------------------------------------------
        # Learned grounding gate.
        # --------------------------------------------------------

        gate = self.grounding_gate(
            tf.concat(
                [
                    token_x,
                    attended,
                ],
                axis=-1,
            )
        )

        grounded = attended * gate

        # --------------------------------------------------------
        # Token input.
        # --------------------------------------------------------

        x = token_x + grounded

        output, states = self.token_gru_cell(
            x,
            [token_state],
            training=training,
        )

        return output, states[0]
    # ============================================================
    # TEACHER FORCING
    # ============================================================

    def teacher_force(
        self,
        plan_sequence,
        grounding_tokens,
        decoder_input_ids,
        training=False,
    ):
        """
        decoder_input_ids:

            [B,T,L]

        Returns:

            [B,T,L,vocab]
        """

        b = tf.shape(plan_sequence)[0]
        depth = tf.shape(plan_sequence)[1]
        length = tf.shape(decoder_input_ids)[2]

        sentence_contexts = self.compute_sentence_contexts(
            grounding_tokens,
            plan_sequence,
            training=training,
        )

        outputs = []

        # Previous sentence state is maintained across planning steps.
        previous_sentence_state = tf.zeros(
            [b, D_MODEL],
            dtype=tf.float32,
        )

        for step in range(self.max_depth):

            # ----------------------------------------------------
            # Current sentence grounding.
            # ----------------------------------------------------

            current_grounding = grounding_tokens[
                :,
                step,
                :,
                :
            ]

            current_sentence_context = sentence_contexts[
                :,
                step,
                :
            ]

            # ----------------------------------------------------
            # Hierarchical previous-sentence conditioning.
            # ----------------------------------------------------

            previous_x = self.previous_sentence_projection(
                previous_sentence_state
            )

            gate = self.previous_sentence_gate(
                tf.concat(
                    [
                        current_sentence_context,
                        previous_x,
                    ],
                    axis=-1,
                )
            )

            previous_x = previous_x * gate

            if current_sentence_context.shape.rank == 2:
                current_sentence_context = tf.expand_dims(
                    current_sentence_context, axis=1
                )

            if previous_x.shape.rank == 2:
                previous_x = tf.expand_dims(
                    previous_x, axis=1
                )

            current_sentence_context = self.sentence_fusion(
                tf.concat(
                    [
                        current_sentence_context,
                        previous_x,
                    ],
                    axis=-1,
                )
            )

            # ----------------------------------------------------
            # Token decoding.
            # ----------------------------------------------------

            state = current_sentence_context

            step_ids = decoder_input_ids[
                :,
                step,
                :
            ]

            sentence_outputs = []

            for token_idx in range(MAX_SEQ_LEN - 1):

                token_ids = step_ids[
                    :,
                    token_idx
                ]

                output, state = self.decode_token(
                    token_ids,
                    state,
                    current_sentence_context,
                    current_grounding,
                    training=training,
                )

                sentence_outputs.append(
                    output
                )

            sentence_outputs = tf.stack(
                sentence_outputs,
                axis=1,
            )

            sentence_outputs = self.output_norm(
                sentence_outputs
            )

            logits = self.output_head(
                sentence_outputs
            )

            outputs.append(logits)

            # The final autoregressive hidden state becomes the
            # previous sentence representation.
            previous_sentence_state = state

        return tf.stack(
            outputs,
            axis=1,
        )

    # ============================================================
    # TRAINING CALL
    # ============================================================

    def call(
        self,
        plan_sequence,
        grounding,
        decoder_input_ids,
        training=False,
    ):
        """
        grounding is a dictionary containing:

            state
            action
            next_state
            goal
            reward
            q_value
            plan
        """

        grounding_tokens = self.build_grounding_tokens(
            states=grounding["state"],
            actions=grounding["action"],
            next_states=grounding["next_state"],
            goal=grounding["goal"],
            rewards=grounding["reward"],
            q_values=grounding["q_value"],
            plan=grounding["plan"],
        )

        grounding_tokens = self.process_grounding(
            grounding_tokens,
            training=training,
        )

        return self.teacher_force(
            plan_sequence=plan_sequence,
            grounding_tokens=grounding_tokens,
            decoder_input_ids=decoder_input_ids,
            training=training,
        )

    # ============================================================
    # GENERATION
    # ============================================================

    def generate(
        self,
        plan_sequence,
        grounding,
        max_length=MAX_SEQ_LEN,
        temperature=0.60,
        top_k=40,
        top_p=0.92,
        greedy=False,
        min_tokens=2,
    ):
        """
        Grounded autoregressive generation.

        Every token generated at step t can attend to:

            S_t
            A_t
            S_{t+1}
            G
            R_t
            Q_t
            P_t
        """

        grounding_tokens = self.build_grounding_tokens(
            states=grounding["state"],
            actions=grounding["action"],
            next_states=grounding["next_state"],
            goal=grounding["goal"],
            rewards=grounding["reward"],
            q_values=grounding["q_value"],
            plan=grounding["plan"],
        )

        grounding_tokens = self.process_grounding(
            grounding_tokens,
            training=False,
        )

        contexts = self.compute_sentence_contexts(
            grounding_tokens,
            plan_sequence,
            training=False,
        )

        batch_size = tf.shape(
            plan_sequence
        )[0]

        depth = plan_sequence.shape[1]

        if depth is None:
            depth = int(
                tf.shape(plan_sequence)[1].numpy()
            )

        all_steps = []

        previous_sentence_state = tf.zeros(
            [batch_size, D_MODEL],
            dtype=tf.float32,
        )

        for step in range(depth):

            current_grounding = grounding_tokens[
                :,
                step,
                :,
                :
            ]

            current_context = contexts[
                :,
                step,
                :
            ]

            # ----------------------------------------------------
            # Previous sentence conditioning
            # ----------------------------------------------------

            previous_x = self.previous_sentence_projection(
                previous_sentence_state
            )

            gate = self.previous_sentence_gate(
                tf.concat(
                    [
                        current_context,
                        previous_x,
                    ],
                    axis=-1,
                )
            )

            previous_x = previous_x * gate

            current_context = self.previous_sentence_fusion(
                tf.concat(
                    [
                        current_context,
                        previous_x,
                    ],
                    axis=-1,
                )
            )

            # ----------------------------------------------------
            # Autoregressive token generation
            # ----------------------------------------------------

            state = current_context

            current = tf.fill(
                [batch_size],
                tf.cast(CLS_ID, tf.int32),
            )

            generated = []

            finished = tf.zeros(
                [batch_size],
                dtype=tf.bool,
            )

            for token_idx in range(max_length - 1):

                output, state = self.decode_token(
                    current,
                    state,
                    current_context,
                    current_grounding,
                    training=False,
                )

                logits = self.output_head(
                    self.output_norm(output)
                )

                logits = tf.identity(logits)

                # ------------------------------------------------
                # Never emit PAD/MASK/CLS.
                # ------------------------------------------------

                for token_id in (
                    PAD_ID,
                    MASK_ID,
                    CLS_ID,
                ):
                    ids = tf.fill(
                        [batch_size],
                        tf.cast(token_id, tf.int32),
                    )

                    indices = tf.stack(
                        [
                            tf.range(batch_size),
                            ids,
                        ],
                        axis=1,
                    )

                    logits = tf.tensor_scatter_nd_update(
                        logits,
                        indices,
                        tf.fill(
                            [batch_size],
                            tf.cast(
                                -1e9,
                                logits.dtype,
                            ),
                        ),
                    )

                # ------------------------------------------------
                # EOS control
                # ------------------------------------------------

                eos_ids = tf.fill(
                    [batch_size],
                    tf.cast(EOS_ID, tf.int32),
                )

                eos_indices = tf.stack(
                    [
                        tf.range(batch_size),
                        eos_ids,
                    ],
                    axis=1,
                )

                if token_idx < min_tokens:

                    logits = tf.tensor_scatter_nd_update(
                        logits,
                        eos_indices,
                        tf.fill(
                            [batch_size],
                            tf.cast(
                                -1e9,
                                logits.dtype,
                            ),
                        ),
                    )

                else:

                    logits = tf.tensor_scatter_nd_add(
                        logits,
                        eos_indices,
                        tf.fill(
                            [batch_size],
                            tf.cast(
                                self.eos_bias,
                                logits.dtype,
                            ),
                        ),
                    )

                # ------------------------------------------------
                # Repetition penalty
                # ------------------------------------------------

                if generated:

                    history = tf.stack(
                        generated,
                        axis=1,
                    )

                    one_hot = tf.one_hot(
                        history,
                        self.vocab_size,
                    )

                    seen = tf.reduce_max(
                        one_hot,
                        axis=1,
                    )

                    logits -= (
                        seen * REPETITION_PENALTY
                    )

                # ------------------------------------------------
                # Bigram blocking
                # ------------------------------------------------

                if BIGRAM_BLOCKING and len(generated) >= 2:

                    sequences = tf.stack(
                        generated,
                        axis=1,
                    ).numpy()

                    masks = np.zeros(
                        [
                            int(batch_size.numpy()),
                            self.vocab_size,
                        ],
                        dtype=np.float32,
                    )

                    for b_idx in range(
                        sequences.shape[0]
                    ):
                        seq = sequences[b_idx].tolist()
                        previous = seq[-1]

                        forbidden = set()

                        for j in range(
                            len(seq) - 1
                        ):
                            if seq[j] == previous:
                                forbidden.add(
                                    seq[j + 1]
                                )

                        for token_id in forbidden:
                            if 0 <= token_id < self.vocab_size:
                                masks[
                                    b_idx,
                                    token_id,
                                ] = 1.0

                    logits -= (
                        tf.convert_to_tensor(
                            masks,
                            dtype=logits.dtype,
                        )
                        * 1e9
                    )

                # ------------------------------------------------
                # Decode
                # ------------------------------------------------

                if greedy:

                    next_token = tf.argmax(
                        logits,
                        axis=-1,
                        output_type=tf.int32,
                    )

                else:

                    next_token = sample_token(
                        logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )

                # Already-finished rows stay EOS.
                next_token = tf.where(
                    finished,
                    tf.fill(
                        [batch_size],
                        tf.cast(
                            EOS_ID,
                            tf.int32,
                        ),
                    ),
                    next_token,
                )

                generated.append(
                    next_token
                )

                finished = tf.logical_or(
                    finished,
                    tf.equal(
                        next_token,
                        EOS_ID,
                    ),
                )

                current = next_token

                if bool(
                    tf.reduce_all(
                        finished
                    ).numpy()
                ):
                    break

            while len(generated) < max_length - 1:
                generated.append(
                    tf.fill(
                        [batch_size],
                        tf.cast(EOS_ID, tf.int32),
                    )
                )

            all_steps.append(
                tf.stack(
                    generated[:max_length - 1],
                    axis=1,
                )
            )

            previous_sentence_state = state

        return tf.stack(
            all_steps,
            axis=1,
        )

# ============================================================
# TOKEN UTILITIES
# ============================================================
def add_eos(ids):
    """Return CLS/content/EOS targets without moving EOS onto a padding tail.

    Input ids already begin with CLS. We locate the first PAD and replace it
    with EOS; if the sequence is full, the final position becomes EOS. Positions
    after EOS are padded. This gives a clean teacher-forcing pair:
    [CLS, w1, ..., EOS, PAD, ...] -> [CLS, w1, ..., EOS, PAD, ...].
    """
    ids = tf.cast(ids, tf.int32)
    width = tf.shape(ids)[-1]
    non_pad = tf.not_equal(ids, PAD_ID)
    lengths = tf.reduce_sum(tf.cast(non_pad, tf.int32), axis=-1)
    eos_pos = tf.minimum(tf.maximum(lengths, 1), width - 1)
    positions = tf.range(width)[None, :]
    before_eos = positions < eos_pos[..., None]
    eos_here = tf.equal(positions, eos_pos[..., None])
    # Preserve real content before EOS, force EOS at eos_pos, pad afterward.
    return tf.where(eos_here, tf.cast(EOS_ID, tf.int32), tf.where(before_eos, ids, tf.cast(PAD_ID, tf.int32)))


def make_decoder_io(action_ids):
    """Construct decoder input/target with EOS supervision at every valid step."""
    full = add_eos(action_ids)
    return full[:, :, :-1], full[:, :, 1:]


def ids_to_text(ids, vectorizer):
    vocabulary = vectorizer.get_vocabulary()
    output = []
    for token_id in np.asarray(ids).reshape(-1):
        token_id = int(token_id)
        if token_id == EOS_ID:
            break
        if token_id in (PAD_ID, MASK_ID, CLS_ID):
            continue
        vocab_id = token_id - SPECIAL_TOKENS
        if 0 <= vocab_id < len(vocabulary):
            token = vocabulary[vocab_id]
            if token not in ("", "[UNK]"):
                output.append(token)
    return " ".join(output)

def candidate_action_alignment_loss_per_step(
    candidate_actions,
    observed_actions,
    step_mask,
):
    """
    Candidate action supervision for every recursive planning step.

    candidate_actions:
        [B, K, D_ACTION]

    observed_actions:
        [B, T, D_ACTION]
    """

    candidate_actions = tf.math.l2_normalize(
        candidate_actions,
        axis=-1,
    )

    observed_actions = tf.math.l2_normalize(
        tf.stop_gradient(observed_actions),
        axis=-1,
    )

    # candidate_actions are generated from the current state only.
    # This function is intended to be called once per recursive step.
    similarity = tf.einsum(
        "bkd,bd->bk",
        candidate_actions,
        observed_actions,
    )

    labels = tf.stop_gradient(
        tf.argmax(similarity, axis=-1)
    )

    loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels,
        similarity / 0.08,
        from_logits=True,
    )

    return masked_mean(
        loss,
        step_mask,
    )


# ============================================================
# FULL MODEL
# ============================================================
class BellmanLatentPlanner(tf.keras.Model):
    def __init__(self, vocab_size, depth):
        super().__init__()
        self.vocab_size = vocab_size
        self.depth = depth
        self.encoder = InceptionStateEncoder(vocab_size)
        self.observed_action_encoder = ObservedActionEncoder()
        self.action_model = ActionModel()
        self.transition_model = TransitionModel()
        self.reward_model = IntrinsicRewardModel()
        self.value_model = ValueModel()
        self.plan_embedding_model = PlanEmbeddingModel()
        self.plan_sentence_projection = PlanSentenceProjection()

        self.action_plan_projection = tf.keras.Sequential([
            layers.Dense(D_PLAN, activation="gelu"),
            layers.LayerNormalization(),
        ])

        # self.decoder = HierarchicalPlanDecoder(vocab_size, depth)
        self.decoder = GroundedHierarchicalPlanDecoder(
            vocab_size,
            depth,
        )
        self.planner = RecursiveBellmanPlanner(
            self.action_model,
            self.transition_model,
            self.reward_model,
            self.value_model,
            self.plan_embedding_model,
        )

        self.sentence_fusion = tf.keras.Sequential([
            layers.Dense(
                2 * D_MODEL,
                activation="gelu",
            ),
            layers.Dense(D_MODEL),
            layers.LayerNormalization(),
        ])

        self.previous_sentence_fusion = tf.keras.Sequential([
            layers.Dense(
                2 * D_MODEL,
                activation="gelu",
            ),
            layers.Dense(D_MODEL),
            layers.LayerNormalization(),
        ])

    def encode(self, ids, training=False):
        return self.encoder(ids, training=training)

    def call(self, inputs, training=False):
        state_ids = inputs["state_ids"]
        goal_ids = inputs["goal_ids"]
        step_state_ids = inputs["step_state_ids"]
        next_state_ids = inputs["next_state_ids"]
        step_mask = tf.cast(inputs["step_mask"], tf.float32)

        state = self.encode(state_ids, training=training)["state"]
        goal = self.encode(goal_ids, training=training)["state"]

        flat_step_states = tf.reshape(step_state_ids, [-1, MAX_SEQ_LEN])
        flat_next_states = tf.reshape(next_state_ids, [-1, MAX_SEQ_LEN])
        step_out = self.encode(flat_step_states, training=training)["state"]
        next_out = self.encode(flat_next_states, training=training)["state"]
        observed_states = tf.reshape(step_out, [tf.shape(step_state_ids)[0], self.depth, D_MODEL])
        observed_next_states = tf.reshape(next_out, [tf.shape(next_state_ids)[0], self.depth, D_MODEL])

        # Every observed transition has an explicit action embedding.
        observed_actions = self.observed_action_encoder(observed_states, observed_next_states)
        observed_action = observed_actions[:, 0, :]

        # ------------------------------------------------------------
        # Observed trajectory plan embeddings
        # ------------------------------------------------------------

        observed_q = (
            self.reward_model(
                observed_states,
                observed_actions,
                observed_next_states,
                goal,
            )
            + GAMMA * self.value_model(
                observed_next_states,
                goal,
            )
        )

        observed_plan = self.plan_embedding_model(
            tf.reshape(observed_states, [-1, D_MODEL]),
            tf.reshape(observed_actions, [-1, D_ACTION]),
            tf.reshape(observed_next_states, [-1, D_MODEL]),
            tf.reshape(
                tf.broadcast_to(
                    goal[:, None, :],
                    tf.shape(observed_next_states),
                ),
                [-1, D_MODEL],
            ),
            tf.reshape(observed_q, [-1]),
        )

        observed_plan = tf.reshape(
            observed_plan,
            [
                tf.shape(state)[0],
                self.depth,
                D_PLAN,
            ],
        )

        proposal = self.action_model(state, goal) # One-step prediction for the observed action. 
        predicted_next_state = self.transition_model( state, observed_action, ) 
        reward = self.reward_model( state, observed_action, predicted_next_state, goal, ) 
        state_value = self.value_model( state, goal, ) 
        next_value = self.value_model( predicted_next_state, goal, ) 

        recursive = self.planner.plan(
            state=state,
            goal=goal,
            depth=self.depth,
            observed_actions=observed_actions,
            observed_next_states=observed_next_states,
            step_mask=step_mask,
            training=training,
        )

        plan_sentence = self.plan_sentence_projection(
            recursive["plan"]
        )

        # Project recursive actions into the language-decoder space.
        recursive_action_plan = self.action_plan_projection(
            recursive["actions"]
        )

        action_sentence = recursive_action_plan

        sentence_plan = self.sentence_fusion(
            tf.concat(
                [plan_sentence, action_sentence],
                axis=-1,
            )
        )

        sentence_plan = tf.math.l2_normalize(
            sentence_plan,
            axis=-1,
        )

        decoder_full = add_eos(inputs["action_ids"])
        decoder_inputs = decoder_full[:, :, :-1]
        decoder_targets = decoder_full[:, :, 1:]

        # decoder_logits = self.decoder(
        #     sentence_plan,
        #     decoder_inputs,
        #     training=training,
        # )
        # ============================================================
        # EXPLICIT DECODER GROUNDING
        # ============================================================

        decoder_grounding = {
            # Recursive state at planning step t.
            "state": recursive["states"],

            # Selected latent action A_t.
            "action": recursive["actions"],

            # Model-predicted next state S_{t+1}.
            "next_state": recursive["next_states"],

            # Global goal G.
            "goal": goal,

            # Intrinsic/goal-directed reward.
            "reward": recursive["rewards"],

            # Bellman Q estimate.
            "q_value": recursive["q_values"],

            # Explicit plan representation.
            "plan": recursive["plan"],
        }

        decoder_logits = self.decoder(
            plan_sequence=sentence_plan,
            grounding=decoder_grounding,
            decoder_input_ids=decoder_inputs,
            training=training,
        )

        augmented_state = self.encode(
            inputs["augmented_state_ids"],
            training=training,
        )["state"]

        return {
            "state": state,
            "goal": goal,
            "observed_states": observed_states,
            "observed_next_states": observed_next_states,
            "observed_actions": observed_actions,
            "observed_action": observed_action,
            "observed_plan": observed_plan,

            "candidate_actions": proposal["actions"],
            "candidate_logits": proposal["logits"],

            "predicted_next_state": predicted_next_state,
            "reward": reward,
            "state_value": state_value,
            "next_value": next_value,

            "recursive_plan": recursive,
            "recursive_action_plan": recursive_action_plan,

            "sentence_plan": sentence_plan,
            "decoder_logits": decoder_logits,
            "decoder_targets": decoder_targets,
            "step_mask": step_mask,
            "augmented_state": augmented_state,

            "termination_logits": recursive["termination_logits"],
            "termination_targets": inputs["termination_targets"],

            "decoder_grounding": decoder_grounding,
        }

    def generate_plan(
        self,
        state_ids,
        goal_ids,
        depth,
        greedy=False,
        temperature=0.60,
        top_k=40,
        top_p=0.92,
    ):
        """
        Autoregressively generate a latent plan from state -> goal.
        """

        state = self.encode(
            state_ids,
            training=False,
        )["state"]

        goal = self.encode(
            goal_ids,
            training=False,
        )["state"]

        # --------------------------------------------------------
        # Recursive latent planning
        # --------------------------------------------------------

        recursive = self.planner.plan(
            state=state,
            goal=goal,
            depth=depth,
            observed_actions=None,
            observed_next_states=None,
            training=False,
        )

        # --------------------------------------------------------
        # Convert plan + action representations into decoder space
        # --------------------------------------------------------

        plan_sentence = self.plan_sentence_projection(
            recursive["plan"]
        )

        recursive_action_plan = self.action_plan_projection(
            recursive["actions"]
        )

        sentence_plan = self.sentence_fusion(
            tf.concat(
                [
                    plan_sentence,
                    recursive_action_plan,
                ],
                axis=-1,
            )
        )

        sentence_plan = tf.math.l2_normalize(
            sentence_plan,
            axis=-1,
        )

        # --------------------------------------------------------
        # Generate language for every planning step
        # --------------------------------------------------------

        # generated_ids = self.decoder.generate(
        #     sentence_plan,
        #     max_length=MAX_SEQ_LEN,
        #     temperature=temperature,
        #     top_k=top_k,
        #     top_p=top_p,
        #     greedy=greedy,
        #     min_tokens=MIN_GENERATION_TOKENS,
        # )

        # --------------------------------------------------------
        # EXPLICIT GENERATION GROUNDING
        # --------------------------------------------------------

        decoder_grounding = {
            "state": recursive["states"],
            "action": recursive["actions"],
            "next_state": recursive["next_states"],
            "goal": goal,
            "reward": recursive["rewards"],
            "q_value": recursive["q_values"],
            "plan": recursive["plan"],
        }

        # --------------------------------------------------------
        # Grounded language generation
        # --------------------------------------------------------

        generated_ids = self.decoder.generate(
            plan_sequence=sentence_plan,
            grounding=decoder_grounding,
            max_length=MAX_SEQ_LEN,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
            min_tokens=MIN_GENERATION_TOKENS,
        )

# ============================================================
# LOSSES
# ============================================================
def cosine_similarity(a, b):
    return tf.reduce_sum(
        tf.math.l2_normalize(a, axis=-1) * tf.math.l2_normalize(b, axis=-1), axis=-1
    )


def masked_mean(values, mask):
    mask = tf.cast(mask, values.dtype)
    return tf.reduce_sum(values * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def transition_loss(predicted, observed):
    return tf.reduce_mean(1.0 - cosine_similarity(predicted, tf.stop_gradient(observed)))


def recursive_state_loss(predicted_states, observed_states, step_mask):
    sim = cosine_similarity(predicted_states, tf.stop_gradient(observed_states))
    # State at planner step t corresponds to observed S_{t+1}; this is the
    # missing alignment that prevents the recursive planner from drifting.
    return masked_mean(1.0 - sim, step_mask)


def recursive_transition_loss(predicted_states, observed_next_states, step_mask):
    sim = cosine_similarity(predicted_states, tf.stop_gradient(observed_next_states))
    return masked_mean(1.0 - sim, step_mask)


def bellman_loss(state_value, reward, next_value):
    target = tf.stop_gradient(reward + GAMMA * next_value)
    return tf.reduce_mean(tf.square(state_value - target))


def terminal_goal_loss(final_state, goal):
    return tf.reduce_mean(1.0 - cosine_similarity(final_state, tf.stop_gradient(goal)))


def plan_goal_loss(predicted_states, goal, step_mask):
    depth = tf.shape(predicted_states)[1]
    weights = tf.cast(tf.range(1, depth + 1), tf.float32)
    weights /= tf.reduce_sum(weights)
    similarity = tf.reduce_sum(predicted_states * goal[:, None, :], axis=-1)
    weighted = (1.0 - similarity) * step_mask * weights[None, :]
    return tf.reduce_sum(weighted) / tf.maximum(tf.reduce_sum(step_mask * weights[None, :]), 1.0)


def _smoothed_sparse_ce(targets, logits, smoothing):
    vocab = tf.cast(tf.shape(logits)[-1], tf.float32)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot = tf.one_hot(tf.cast(targets, tf.int32), tf.shape(logits)[-1])
    one_hot = one_hot * (1.0 - smoothing) + smoothing / vocab
    return -tf.reduce_sum(one_hot * log_probs, axis=-1)


def hierarchical_decoder_loss(logits, targets, step_mask):
    loss = _smoothed_sparse_ce(targets, logits, LABEL_SMOOTHING)
    token_mask = tf.cast(tf.not_equal(targets, PAD_ID), tf.float32)
    # EOS is non-padding and therefore receives explicit loss.
    mask = token_mask * step_mask[:, :, None]
    return tf.reduce_sum(loss * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def info_nce_loss(queries, positives, temperature=CONTRASTIVE_TEMPERATURE):
    queries = tf.math.l2_normalize(queries, axis=-1)
    positives = tf.math.l2_normalize(positives, axis=-1)
    logits = tf.matmul(queries, positives, transpose_b=True) / temperature
    labels = tf.range(tf.shape(logits)[0], dtype=tf.int32)
    fwd = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    bwd = tf.keras.losses.sparse_categorical_crossentropy(labels, tf.transpose(logits), from_logits=True)
    return 0.5 * (tf.reduce_mean(fwd) + tf.reduce_mean(bwd))


def masked_temporal_action_info_nce(predicted_actions, observed_actions, step_mask):
    """Symmetric InfoNCE over valid (trajectory-window, time-step) pairs.

    Positives are matched by the exact trajectory window and exact time step.
    All other valid steps in the batch are negatives. Importantly, padding
    steps are removed before the logits matrix is built, so an empty window can
    never become a false positive/negative.
    """
    mask = tf.cast(step_mask, tf.bool)
    q = tf.boolean_mask(predicted_actions, mask)
    k = tf.boolean_mask(observed_actions, mask)
    count = tf.shape(q)[0]

    def compute():
        return info_nce_loss(q, tf.stop_gradient(k), TEMPORAL_ACTION_TEMPERATURE)

    # With one valid transition there are no meaningful in-batch negatives.
    return tf.cond(count > 1, compute, lambda: tf.constant(0.0, tf.float32))


def stronger_state_contrastive_loss(state, next_state, augmented_state):
    return 0.75 * info_nce_loss(state, augmented_state) + 0.25 * info_nce_loss(state, next_state)


def candidate_action_alignment_loss(candidate_actions, observed_action):
    sim = tf.einsum(
        "bkd,bd->bk",
        tf.math.l2_normalize(candidate_actions, axis=-1),
        tf.math.l2_normalize(tf.stop_gradient(observed_action), axis=-1),
    )
    labels = tf.stop_gradient(tf.argmax(sim, axis=-1))
    return tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(labels, sim / 0.08, from_logits=True)
    )


def candidate_transition_loss(state, candidate_actions, observed_next_state, transition_model):
    b = tf.shape(state)[0]
    k = tf.shape(candidate_actions)[1]
    d = tf.shape(state)[1]
    flat_state = tf.reshape(tf.broadcast_to(state[:, None, :], [b, k, d]), [-1, d])
    flat_action = tf.reshape(candidate_actions, [-1, tf.shape(candidate_actions)[-1]])
    predicted = tf.reshape(transition_model(flat_state, flat_action), [b, k, d])
    sim = cosine_similarity(predicted, tf.stop_gradient(observed_next_state)[:, None, :])
    top = tf.stop_gradient(tf.argmax(sim, axis=-1))
    ce = tf.keras.losses.sparse_categorical_crossentropy(top, sim / 0.08, from_logits=True)
    best_distance = 1.0 - tf.reduce_max(sim, axis=-1)
    return tf.reduce_mean(best_distance + 0.20 * ce)


def candidate_diversity_loss(actions, margin=0.20):
    actions = tf.math.l2_normalize(actions, axis=-1)
    sim = tf.matmul(actions, actions, transpose_b=True)
    k = tf.shape(actions)[1]
    eye = tf.eye(k, batch_shape=[tf.shape(actions)[0]])
    violation = tf.nn.relu(sim * (1.0 - eye) - margin)
    return tf.reduce_mean(tf.square(violation))


def recursive_action_separation_loss(actions, step_mask, margin=0.10):
    actions = tf.math.l2_normalize(actions, axis=-1)
    if actions.shape[1] is not None and actions.shape[1] <= 1:
        return tf.constant(0.0, tf.float32)
    sim = tf.reduce_sum(actions[:, :-1, :] * actions[:, 1:, :], axis=-1)
    valid = step_mask[:, :-1] * step_mask[:, 1:]
    violation = tf.nn.relu(sim - margin)
    return masked_mean(tf.square(violation), valid)


def recursive_action_alignment_loss(
    predicted_actions,
    observed_actions,
    step_mask,
):
    similarity = tf.reduce_sum(
        tf.math.l2_normalize(
            predicted_actions,
            axis=-1,
        )
        *
        tf.math.l2_normalize(
            tf.stop_gradient(observed_actions),
            axis=-1,
        ),
        axis=-1,
    )

    return masked_mean(
        1.0 - similarity,
        step_mask,
    )

def recursive_action_language_loss(recursive_actions, observed_actions, step_mask):
    sim = tf.reduce_sum(
        tf.math.l2_normalize(recursive_actions, axis=-1)
        * tf.math.l2_normalize(tf.stop_gradient(observed_actions), axis=-1),
        axis=-1,
    )
    return masked_mean(1.0 - sim, step_mask)

def recursive_transition_consistency_loss(
    states,
    actions,
    observed_next_states,
    step_mask,
    transition_model,
):
    """
    Explicitly checks:

        T(S_t, A_t) ~= observed S_{t+1}
    """

    b = tf.shape(states)[0]
    depth = tf.shape(states)[1]

    d_state = tf.shape(states)[2]

    flat_states = tf.reshape(
        states,
        [-1, d_state],
    )

    flat_actions = tf.reshape(
        actions,
        [-1, tf.shape(actions)[2]],
    )

    predicted = transition_model(
        flat_states,
        flat_actions,
    )

    predicted = tf.reshape(
        predicted,
        [b, depth, d_state],
    )

    similarity = cosine_similarity(
        predicted,
        tf.stop_gradient(observed_next_states),
    )

    return masked_mean(
        1.0 - similarity,
        step_mask,
    )

def recursive_plan_alignment_loss(
    predicted_plan,
    observed_plan,
    step_mask,
):
    similarity = cosine_similarity(
        predicted_plan,
        tf.stop_gradient(observed_plan),
    )

    return masked_mean(
        1.0 - similarity,
        step_mask,
    )


def plan_action_alignment_loss(
    plan_embeddings,
    action_embeddings,
    step_mask,
):
    sim = cosine_similarity(
        plan_embeddings,
        tf.stop_gradient(action_embeddings),
    )

    return masked_mean(
        1.0 - sim,
        step_mask,
    )


def termination_loss(logits, targets, step_mask):
    targets = tf.cast(targets, tf.float32)
    step_mask = tf.cast(step_mask, tf.float32)

    positive_weight = 3.0

    loss = tf.nn.weighted_cross_entropy_with_logits(
        labels=targets,
        logits=logits,
        pos_weight=positive_weight,
    )

    return masked_mean(loss, step_mask)


def decoder_grounding_alignment_loss(
    decoder_grounding,
    observed_plan,
    step_mask,
):
    similarity = cosine_similarity(
        decoder_grounding,
        tf.stop_gradient(observed_plan),
    )

    return masked_mean(
        1.0 - similarity,
        step_mask,
    )

def compute_total_loss(outputs, transition_model):
    l_termination = termination_loss(
        outputs["recursive_plan"]["termination_logits"],
        outputs["termination_targets"],
        outputs["step_mask"],
    )
    l_transition = transition_loss(
        outputs["predicted_next_state"], outputs["observed_next_states"][:, 0, :]
    )
    l_recursive_state = recursive_state_loss(
        outputs["recursive_plan"]["next_states"],
        outputs["observed_next_states"],
        outputs["step_mask"],
    )
    l_recursive_action = recursive_action_alignment_loss(
        outputs["recursive_plan"]["actions"],
        outputs["observed_actions"],
        outputs["step_mask"],
    )
    l_recursive_transition = recursive_transition_consistency_loss(
        outputs["recursive_plan"]["states"],
        outputs["recursive_plan"]["actions"],
        outputs["observed_next_states"],
        outputs["step_mask"],
        transition_model,
    )
    l_bellman = bellman_loss(outputs["state_value"], outputs["reward"], outputs["next_value"])
    l_terminal = terminal_goal_loss(outputs["recursive_plan"]["final_state"], outputs["goal"])
    l_plan_goal = plan_goal_loss(
        outputs["recursive_plan"]["states"], outputs["goal"], outputs["step_mask"]
    )
    l_decoder = hierarchical_decoder_loss(
        outputs["decoder_logits"], outputs["decoder_targets"], outputs["step_mask"]
    )
    l_action_language = recursive_action_language_loss(
        outputs["recursive_plan"]["actions"], outputs["observed_actions"], outputs["step_mask"]
    )
    l_temporal_nce = masked_temporal_action_info_nce(
        outputs["recursive_plan"]["actions"], outputs["observed_actions"], outputs["step_mask"]
    )
    l_candidate_action = candidate_action_alignment_loss(
        outputs["candidate_actions"], outputs["observed_action"]
    )
    l_candidate_transition = candidate_transition_loss(
        outputs["state"], outputs["candidate_actions"], outputs["observed_next_states"][:, 0, :], transition_model
    )
    l_candidate_diversity = candidate_diversity_loss(outputs["candidate_actions"])
    l_action_separation = recursive_action_separation_loss(
        outputs["recursive_plan"]["actions"], outputs["step_mask"]
    )
    l_contrast = stronger_state_contrastive_loss(
        outputs["state"], outputs["observed_next_states"][:, 0, :], outputs["augmented_state"]
    )

    l_plan_alignment = recursive_plan_alignment_loss(
        outputs["recursive_plan"]["plan"],
        outputs["observed_plan"],
        outputs["step_mask"],
    )

    l_text_alignment_loss = plan_action_alignment_loss(
        outputs["recursive_plan"]["plan"],
        outputs["recursive_action_plan"],
        outputs["step_mask"],
    )

    l_decoder_grounding = decoder_grounding_alignment_loss(
        outputs["decoder_grounding"]["plan"],
        outputs["observed_plan"],
        outputs["step_mask"],
    )

    total = (
        W_TRANSITION * l_transition
        + W_RECURSIVE_STATE * l_recursive_state
        + W_RECURSIVE_TRANSITION * l_recursive_transition
        + W_BELLMAN * l_bellman
        + W_TERMINAL_GOAL * l_terminal
        + W_PLAN_GOAL * l_plan_goal
        + W_DECODER * l_decoder
        + W_ACTION_LANGUAGE * l_action_language
        + W_TEMPORAL_ACTION_NCE * l_temporal_nce
        + W_STATE_CONTRAST * l_contrast
        + W_CANDIDATE_ACTION * l_candidate_action
        + W_CANDIDATE_TRANSITION * l_candidate_transition
        + W_CANDIDATE_DIVERSITY * l_candidate_diversity
        + W_ACTION_SEPARATION * l_action_separation
        + W_RECURSIVE_ACTION * l_recursive_action
        + W_PLAN_ALIGNMENT * l_plan_alignment
        + W_PLAN_LANGUAGE_ALIGNMENT * l_text_alignment_loss
        + W_TERMINATION * l_termination
        + W_DECODER_GROUNDING * l_decoder_grounding
    )

    return total, {
        "transition": l_transition,
        "recursive_state": l_recursive_state,
        "recursive_transition": l_recursive_transition,
        "bellman": l_bellman,
        "terminal_goal": l_terminal,
        "plan_goal": l_plan_goal,
        "decoder": l_decoder,
        "action_language": l_action_language,
        "temporal_action_nce": l_temporal_nce,
        "contrast": l_contrast,
        "candidate_action": l_candidate_action,
        "candidate_transition": l_candidate_transition,
        "candidate_diversity": l_candidate_diversity,
        "action_separation": l_action_separation,
        "termination": l_termination,
    }


# ============================================================
# TRAIN STEP
# ============================================================
@tf.function(reduce_retracing=True)
def train_step(model, batch, optimizer):
    with tf.GradientTape() as tape:
        outputs = model(batch, training=True)

        total, losses = compute_total_loss(
            outputs,
            model.transition_model,
        )

    gradients = tape.gradient(total, model.trainable_variables)

    pairs = []
    for g, v in zip(gradients, model.trainable_variables):
        if g is not None:
            pairs.append(
                (tf.clip_by_norm(g, GRAD_CLIP_NORM), v)
            )

    if pairs:
        optimizer.apply_gradients(pairs)

    return total, losses


# ============================================================
# SAVE
# ============================================================
def save_vocabulary(vectorizer, path):
    with open(path, "w", encoding="utf-8") as f:
        for token in vectorizer.get_vocabulary():
            f.write(token.replace("\n", " ") + "\n")


def save_planner(model, path):
    model.save_weights(path)
    print(f"Saved model weights: {path}")


# ============================================================
# TRAINING
# ============================================================
def train_from_file(data_path, epochs, batch_size, depth):
    print(f"Loading trajectory corpus: {data_path}")
    trajectories = read_trajectories(data_path)
    print(f"Trajectories: {len(trajectories)}")
    print("Trajectory lengths:", [len(x) for x in trajectories[:20]])

    vectorizer = build_vectorizer(trajectories)
    base_vocab_size = len(vectorizer.get_vocabulary())
    vocabulary_size = base_vocab_size + SPECIAL_TOKENS
    print(f"Base vocabulary: {base_vocab_size}")
    print(f"Model vocabulary: {vocabulary_size}")

    examples = make_training_examples(trajectories, depth)
    print(f"Training windows: {len(examples)}")
    dataset = make_dataset(examples, vectorizer, depth, batch_size)

    model = BellmanLatentPlanner(vocabulary_size, depth)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, clipnorm=GRAD_CLIP_NORM
    )
    first_batch = next(iter(dataset))
    _ = model(first_batch, training=False)
    print("Model initialized.")
    print("Trainable parameters:", model.count_params())

    metric_names = [
        "total", "transition", "recursive_state", "recursive_transition", "bellman",
        "terminal_goal", "plan_goal", "decoder", "action_language", "temporal_action_nce",
        "contrast", "candidate_action", "candidate_transition", "candidate_diversity",
        "action_separation", "termination",
    ]

    for epoch in range(epochs):
        metrics = {name: tf.keras.metrics.Mean() for name in metric_names}
        for batch in dataset:
            total, losses = train_step(model, batch, optimizer)
            metrics["total"].update_state(total)
            for name, value in losses.items():
                metrics[name].update_state(value)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        for name in metric_names:
            print(f"  {name:24s}: {float(metrics[name].result()):.6f}")
        random_infonce = np.log(max(batch_size, 2))
        contrast_value = float(metrics["contrast"].result())
        print(f"  contrast random baseline: {random_infonce:.6f}")
        if contrast_value < random_infonce * 0.85:
            print("  contrastive status: learning signal is substantially better than random.")

    return model, vectorizer, trajectories


# ============================================================
# GENERATION / DIAGNOSTICS
# ============================================================
def generate_plan(
    model,
    vectorizer,
    state_text,
    goal_text,
    depth,
    greedy=False,
    temperature=0.60,
):
    state_ids = encode_texts(
        [state_text],
        vectorizer,
    )

    goal_ids = encode_texts(
        [goal_text],
        vectorizer,
    )

    output = model.generate_plan(
        state_ids,
        goal_ids,
        depth,
        greedy=greedy,
        temperature=temperature,
        top_k=40,
        top_p=0.92,
    )

    generated_ids = output["generated_ids"][0]

    sentences = []

    for step in generated_ids:
        sentence = ids_to_text(
            step.numpy(),
            vectorizer,
        )

        sentences.append(
            sentence
            if sentence
            else "[empty decoded step]"
        )

    return {
        "sentences": sentences,
        "actions": output["actions"][0].numpy(),
        "q_values": output["q_values"][0].numpy(),
        "predicted_states": output["predicted_states"][0].numpy(),
        "termination_logits": output["termination_logits"][0].numpy(),
        "final_state": output["final_state"][0].numpy(),
        "goal": output["goal"][0].numpy(),
    }


def print_plan_diagnostics(result):
    actions = result["actions"]
    states = result["predicted_states"]
    goal = result["goal"]
    goal = goal / max(np.linalg.norm(goal), 1e-8)
    print("\n" + "=" * 70)
    print("PLAN DIAGNOSTICS")
    print("=" * 70)
    print("Q values:")
    print(np.array2string(result["q_values"], precision=4, suppress_small=True))
    print("\nAction norms:")
    print(np.array2string(np.linalg.norm(actions, axis=-1), precision=4))
    print("\nGoal similarity by predicted step:")
    for i, state in enumerate(states, start=1):
        state = state / max(np.linalg.norm(state), 1e-8)
        print(f"  step {i}: {float(np.dot(state, goal)):.4f}")
    print("\nAdjacent action similarities:")
    for i in range(len(actions) - 1):
        a = actions[i] / max(np.linalg.norm(actions[i]), 1e-8)
        b = actions[i + 1] / max(np.linalg.norm(actions[i + 1]), 1e-8)
        print(f"  A{i + 1} <-> A{i + 2}: {float(np.dot(a, b)):.4f}")


# ============================================================
# ARGUMENTS / MAIN
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Trajectory-aware latent Bellman planner")
    parser.add_argument("--data", default="data.txt")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--depth", type=int, default=PLANNER_DEPTH)
    parser.add_argument("--state", default="The bedroom is cold.")
    parser.add_argument("--goal", default="The bedroom is warm.")
    parser.add_argument("--weights", default="bellman_planner.weights.h5")
    parser.add_argument("--vocabulary", default="bellman_planner.vocab.txt")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use deterministic greedy decoding.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.60,
        help="Sampling temperature.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.depth < 1:
        raise ValueError("--depth must be >= 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 2:
        raise ValueError("--batch-size should be >= 2 for meaningful in-batch contrastive negatives.")

    print("=" * 70)
    print("CORRECTED TRAJECTORY-AWARE BELLMAN LATENT PLANNER")
    print("=" * 70)
    print(f"TensorFlow: {tf.__version__}")
    device = args.device or ("/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0")
    print(f"Device: {device}")

    with tf.device(device):
        model, vectorizer, _ = train_from_file(
            data_path=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            depth=args.depth,
        )
        save_planner(model, args.weights)
        save_vocabulary(vectorizer, args.vocabulary)
        print(f"Saved vocabulary: {args.vocabulary}")
        result = generate_plan(
            model,
            vectorizer,
            args.state,
            args.goal,
            args.depth,
            greedy=args.greedy,
            temperature=args.temperature,
        )

    print("\n" + "=" * 70)
    print("PLANNING QUERY")
    print("=" * 70)
    print(f"STATE: {args.state}")
    print(f"GOAL:  {args.goal}")
    print("\n" + "=" * 70)
    print("GENERATED PLAN")
    print("=" * 70)
    for i, sentence in enumerate(result["sentences"], start=1):
        print(f"{i}. {sentence}")
    print_plan_diagnostics(result)

    final_state = result["final_state"] / max(np.linalg.norm(result["final_state"]), 1e-8)
    goal = result["goal"] / max(np.linalg.norm(result["goal"]), 1e-8)
    similarity = float(np.dot(final_state, goal))
    print("\n" + "=" * 70)
    print("FINAL STATE / GOAL SIMILARITY")
    print("=" * 70)
    print(f"Final cosine similarity: {similarity:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()

"""
For evaluation, I recommend:

python planner.py \
    --data data.txt \
    --epochs 150 \
    --depth 4 \
    --state "The bedroom is cold." \
    --goal "The bedroom is warm." \
    --greedy

For more varied plans:

python planner.py \
    --data data.txt \
    --epochs 150 \
    --depth 4 \
    --state "The bedroom is cold." \
    --goal "The bedroom is warm." \
    --temperature 0.65

"""