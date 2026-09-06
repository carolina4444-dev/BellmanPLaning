#!/usr/bin/env python3
"""
planner.py - Latent Bellman Planner with Modular Multi-Stage Training Protocol.
"""

import argparse
import os
import random
import re
from typing import List, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# ============================================================
# CONFIG & HYPERPARAMETERS
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
PAD_ID = 0
MASK_ID = 1
CLS_ID = 2
EOS_ID = 3
SPECIAL_TOKENS = 4
MAX_SEGMENTS_PER_TRAJECTORY = 128

# Loss Weights
W_TRANSITION = 1.50
W_BELLMAN = 0.20
W_DECODER = 1.00
W_STATE_CONTRAST = 0.20
W_CANDIDATE_DIVERSITY = 0.25
W_RECURSIVE_STATE = 1.75
W_TERMINATION = 1.0

# Reproducibility
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# TEXT & DATASET HELPERS
# ============================================================
def normalize_segment(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def read_data_lines(path: str) -> List[str]:
    """Canonical loader for line-based corpus."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find corpus: {path}")

    with open(path, "r", encoding="utf-8") as f:
        lines = [normalize_segment(line) for line in f if normalize_segment(line)]

    if not lines:
        raise ValueError(f"Corpus is empty: {path}")

    return lines


def read_trajectories_from_lines(all_data_lines: List[str]):
    """
    Builds trajectories directly referencing indices in all_data_lines.
    Ensures 1:1 mapping between global line target IDs and candidates.
    """
    line_to_id = {line: idx for idx, line in enumerate(all_data_lines)}
    trajectories = []
    
    current_traj = []
    current_rows = []

    for line in all_data_lines:
        if re.fullmatch(r"[-*_]{3,}", line):
            if len(current_traj) >= 2:
                trajectories.append({"segments": current_traj, "row_ids": current_rows})
            current_traj, current_rows = [], []
            continue

        parts = [normalize_segment(p) for p in re.split(r"\s*(?:→|->)\s*", line) if normalize_segment(p)]
        for piece in parts:
            if piece in line_to_id:
                current_traj.append(piece)
                current_rows.append(line_to_id[piece])

    if len(current_traj) >= 2:
        trajectories.append({"segments": current_traj, "row_ids": current_rows})

    if not trajectories:
        trajectories = [{"segments": all_data_lines, "row_ids": list(range(len(all_data_lines)))}]

    return trajectories


def generate_synthetic_corpus():
    lines = [
        "The bedroom is cold.",
        "Turn on the thermostat.",
        "The heater starts warming up.",
        "The bedroom is warm.",
        "The kitchen light is off.",
        "Walk to the light switch.",
        "Flip the switch on.",
        "The kitchen light is bright.",
        "The front door is unlocked.",
        "Reach for the door key.",
        "Turn key to the right.",
        "The front door is locked."
    ]
    return lines, read_trajectories_from_lines(lines)


def build_vectorizer(texts):
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

    for traj in trajectories:
        trajectory = traj["segments"]
        row_ids = traj["row_ids"]

        n = len(trajectory)

        for start in range(n - 1):
            horizon = min(depth, n - start - 1)

            if horizon < 1:
                continue

            states = list(trajectory[start:start + horizon])
            next_states = list(trajectory[start + 1:start + horizon + 1])
            target_rows = list(row_ids[start + 1:start + horizon + 1])
            goal = next_states[-1]

            while len(states) < depth:
                states.append("")
                next_states.append("")
                target_rows.append(-1)

            examples.append({
                "state": trajectory[start],
                "goal": goal,
                "step_states": states,
                "next_states": next_states,
                "target_row_ids": target_rows,
                "horizon": horizon,
            })

    return examples


def make_dataset(examples, all_line_ids, vectorizer, depth, batch_size):
    n = len(examples)
    state_ids = encode_texts([x["state"] for x in examples], vectorizer)
    goal_ids = encode_texts([x["goal"] for x in examples], vectorizer)
    augmented_state_ids = encode_texts([augment_text(x["state"]) for x in examples], vectorizer)
    flat_next_states = [text for row in [x["next_states"] for x in examples] for text in row]
    next_state_ids = tf.reshape(encode_texts(flat_next_states, vectorizer), [n, depth, MAX_SEQ_LEN])

    horizons = np.asarray([x["horizon"] for x in examples], dtype=np.float32)
    step_idx = np.arange(depth, dtype=np.float32)[None, :]
    step_mask = (step_idx < horizons[:, None]).astype(np.float32)
    termination_targets = np.zeros((n, depth), dtype=np.float32)

    for i, horizon in enumerate(horizons.astype(np.int32)):
        if horizon > 0:
            termination_targets[i, horizon - 1] = 1.0

    target_row_ids = np.asarray([x["target_row_ids"] for x in examples], dtype=np.int32)

    # Broadcast line_ids across dataset batches
    line_ids_batch = tf.tile(all_line_ids[None, ...], [n, 1, 1])

    ds = tf.data.Dataset.from_tensor_slices({
        "state_ids": state_ids,
        "goal_ids": goal_ids,
        "next_state_ids": next_state_ids,
        "step_mask": step_mask,
        "augmented_state_ids": augmented_state_ids,
        "termination_targets": termination_targets,
        "target_row_ids": target_row_ids,
        "line_ids": line_ids_batch,
    })
    return ds.shuffle(min(n, 4096), seed=SEED, reshuffle_each_iteration=True).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


# ============================================================
# NEURAL BLOCKS & DECODER DEFINITIONS
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
        self.value_head = tf.keras.Sequential([layers.Dense(D_VALUE, activation="gelu"), layers.Dense(1)])

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


class GoalConditioner(layers.Layer):
    def __init__(self):
        super().__init__()
        self.d1 = layers.Dense(2 * D_MODEL, activation="gelu")
        self.d2 = layers.Dense(D_MODEL, activation="gelu")
        self.norm = layers.LayerNormalization()

    def call(self, state, goal):
        x = tf.concat([state, goal, goal - state, state * goal], axis=-1)
        return self.norm(self.d2(self.d1(x)))


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
        actions = tf.reshape(self.action_head(hidden), [tf.shape(state)[0], NUM_ACTION_CANDIDATES, D_ACTION])
        actions = tf.math.l2_normalize(actions, axis=-1)
        return {"context": context, "actions": actions, "logits": self.score_head(hidden)}


class ObservedActionEncoder(layers.Layer):
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
        self.d1 = layers.Dense(2 * D_MODEL, activation="gelu")
        self.d2 = layers.Dense(2 * D_MODEL, activation="gelu")
        self.out = layers.Dense(D_MODEL)
        self.norm = layers.LayerNormalization()

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        delta = self.out(self.d2(self.d1(x)))
        return tf.math.l2_normalize(self.norm(state + delta), axis=-1)


class IntrinsicRewardModel(layers.Layer):
    def __init__(self):
        super().__init__()
        self.d1 = layers.Dense(D_MODEL, activation="gelu")
        self.d2 = layers.Dense(D_MODEL // 2, activation="gelu")
        self.out = layers.Dense(1)

    def call(self, state, action, next_state, goal):
        delta = next_state - state
        progress = tf.reduce_sum(next_state * goal, axis=-1) - tf.reduce_sum(state * goal, axis=-1)
        x = tf.concat([state, action, next_state, goal, delta], axis=-1)
        learned = tf.tanh(self.out(self.d2(self.d1(x)))[..., 0])
        return progress + 0.03 * learned


class ValueModel(layers.Layer):
    def __init__(self):
        super().__init__()
        self.conditioner = GoalConditioner()
        self.d1 = layers.Dense(2 * D_MODEL, activation="gelu")
        self.d2 = layers.Dense(D_MODEL, activation="gelu")
        self.out = layers.Dense(1)

    def call(self, state, goal):
        x = self.conditioner(state, goal)
        return self.out(self.d2(self.d1(x)))[..., 0]


class PlanEmbeddingModel(layers.Layer):
    def __init__(self):
        super().__init__()
        self.state_net = layers.Dense(D_PLAN, activation="gelu")
        self.action_net = layers.Dense(D_PLAN, activation="gelu")
        self.transition_net = layers.Dense(D_PLAN, activation="gelu")
        self.goal_net = layers.Dense(D_PLAN, activation="gelu")
        self.q_net = layers.Dense(D_PLAN, activation="gelu")
        self.fusion = tf.keras.Sequential([
            layers.Dense(2 * D_PLAN, activation="gelu"),
            layers.Dense(D_PLAN),
            layers.LayerNormalization(),
        ])
        self.action_gate = layers.Dense(D_PLAN, activation="sigmoid")

    def call(self, state, action, next_state, goal, q):
        state_x = self.state_net(state)
        action_x = self.action_net(action)
        goal_x = self.goal_net(goal)
        action_gate = self.action_gate(tf.concat([state_x, goal_x], axis=-1))
        action_x = action_x * action_gate
        transition_x = self.transition_net(next_state - state)
        next_state_x = self.transition_net(next_state)

        q = tf.reshape(q, [-1, 1])
        q_x = self.q_net(q)

        x = tf.concat([state_x, action_x, transition_x, next_state_x, goal_x, q_x], axis=-1)
        return tf.math.l2_normalize(self.fusion(x), axis=-1)


class PlanSentenceProjection(layers.Layer):
    def __init__(self):
        super().__init__()
        self.d1 = layers.Dense(2 * D_MODEL, activation="gelu")
        self.d2 = layers.Dense(D_MODEL)
        self.norm = layers.LayerNormalization()
        self.gate = layers.Dense(D_MODEL, activation="sigmoid")

    def call(self, plan):
        shape = tf.shape(plan)
        flat = tf.reshape(plan, [-1, D_PLAN])
        x = self.norm(self.d2(self.d1(flat)))
        x = x * self.gate(x)
        x = tf.math.l2_normalize(x, axis=-1)
        return tf.reshape(x, tf.concat([shape[:-1], [D_MODEL]], axis=0))


class RecursiveBellmanPlanner(layers.Layer):
    def __init__(self, action_model, transition_model, reward_model, value_model, plan_embedding_model):
        super().__init__()
        self.action_model = action_model
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.value_model = value_model
        self.plan_embedding_model = plan_embedding_model

        self.termination_head = tf.keras.Sequential([
            layers.Dense(D_MODEL, activation="gelu"),
            layers.Dense(1),
        ])

    def recursive_step(self, state, goal, training=False):
        proposal = self.action_model(state, goal)
        actions = proposal["actions"]
        b, k = tf.shape(state)[0], tf.shape(actions)[1]

        flat_state = tf.reshape(tf.broadcast_to(state[:, None, :], [b, k, tf.shape(state)[1]]), [-1, tf.shape(state)[1]])
        flat_goal = tf.reshape(tf.broadcast_to(goal[:, None, :], [b, k, tf.shape(goal)[1]]), [-1, tf.shape(goal)[1]])
        flat_actions = tf.reshape(actions, [-1, tf.shape(actions)[2]])

        flat_next = self.transition_model(flat_state, flat_actions)
        flat_reward = self.reward_model(flat_state, flat_actions, flat_next, flat_goal)
        flat_value = self.value_model(flat_next, flat_goal)

        q_values = tf.reshape(flat_reward + GAMMA * flat_value, [b, k])
        index = tf.argmax(q_values, axis=-1, output_type=tf.int32)

        selected_action = tf.gather(actions, index, batch_dims=1)
        predicted_next_state = self.transition_model(state, selected_action)
        predicted_reward = self.reward_model(state, selected_action, predicted_next_state, goal)
        predicted_value = self.value_model(predicted_next_state, goal)
        predicted_q = predicted_reward + GAMMA * predicted_value

        plan_embedding = self.plan_embedding_model(state, selected_action, predicted_next_state, goal, predicted_q)
        term_logit = self.termination_head(tf.concat([state, predicted_next_state, selected_action, goal], axis=-1))[..., 0]

        return {
            "state": state,
            "action": selected_action,
            "next_state": predicted_next_state,
            "plan_embedding": plan_embedding,
            "termination_logit": term_logit
        }


class HierarchicalPlanDecoder(layers.Layer):
    """
    Row-Selection Plan Decoder with Temperature-Scaled Softmax.
    """
    def __init__(self, d_model=D_MODEL, max_depth=MAX_SEGMENTS_PER_TRAJECTORY):
        super().__init__()
        self.step_embedding = layers.Embedding(max_depth, d_model)
        self.step_gru = layers.GRU(d_model, return_sequences=True)
        self.plan_norm = layers.LayerNormalization()

        self.query_proj = layers.Dense(d_model)
        self.line_proj = layers.Dense(d_model)
        # After
        self.temperature = self.add_weight(
            name="temperature",
            shape=(),
            initializer=tf.constant_initializer(0.07),
            trainable=True
        )

    def compute_step_contexts(self, plan_sequence, training=False):
        depth = tf.shape(plan_sequence)[1]
        positions = tf.range(depth)[None, :]
        pos_embed = self.step_embedding(positions)
        x = self.plan_norm(plan_sequence + pos_embed)
        return self.step_gru(x, training=training)

    def call(self, plan_sequence, line_embeddings, training=False):
        step_contexts = self.compute_step_contexts(plan_sequence, training=training)

        queries = tf.math.l2_normalize(self.query_proj(step_contexts), axis=-1)
        keys = tf.math.l2_normalize(self.line_proj(line_embeddings), axis=-1)

        row_logits = tf.matmul(queries, keys, transpose_b=True) / tf.maximum(self.temperature, 1e-4)

        return {"row_logits": row_logits}

    def predict_rows(self, plan_sequence, line_embeddings):
        out = self(plan_sequence, line_embeddings, training=False)
        return tf.argmax(out["row_logits"], axis=-1, output_type=tf.int32)


# ============================================================
# TRAINER MODULE
# ============================================================
class ModularPlannerTrainer(tf.keras.Model):
    def __init__(
        self,
        state_encoder,
        action_encoder,
        action_model,
        transition_model,
        reward_model,
        value_model,
        plan_embedding_model,
        plan_projection,
        planner,
        decoder,
        learning_rate=2e-4
    ):
        super().__init__()
        self.encoder = state_encoder
        self.action_encoder = action_encoder
        self.action_model = action_model
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.value_model = value_model
        self.plan_embedding_model = plan_embedding_model
        self.plan_projection = plan_projection
        self.planner = planner
        self.decoder = decoder

        self.opt_phase1 = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=WEIGHT_DECAY)
        self.opt_phase2 = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=WEIGHT_DECAY)
        self.opt_phase3 = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=WEIGHT_DECAY)

    def train_phase_1(self, batch):
        with tf.GradientTape() as tape:
            s_enc = self.encoder(batch["state_ids"], training=True)
            g_enc = self.encoder(batch["goal_ids"], training=True)
            s_aug_enc = self.encoder(batch["augmented_state_ids"], training=True)

            sim_matrix = tf.matmul(s_enc["state"], s_aug_enc["state"], transpose_b=True) / CONTRASTIVE_TEMPERATURE
            labels = tf.range(tf.shape(sim_matrix)[0])
            loss_state_contrast = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=sim_matrix)
            )

            proposal = self.action_model(s_enc["state"], g_enc["state"])
            actions = proposal["actions"]
            actions_norm = tf.math.l2_normalize(actions, axis=-1)
            similarity = tf.matmul(actions_norm, actions_norm, transpose_b=True)
            eye = tf.eye(NUM_ACTION_CANDIDATES)[None, :, :]
            loss_action_diversity = tf.reduce_mean(tf.square(similarity * (1.0 - eye)))

            loss_p1 = (W_STATE_CONTRAST * loss_state_contrast) + (W_CANDIDATE_DIVERSITY * loss_action_diversity)

        vars_phase1 = (
            self.encoder.trainable_variables +
            self.action_encoder.trainable_variables +
            self.action_model.trainable_variables
        )
        grads = tape.gradient(loss_p1, vars_phase1)
        grads_and_vars = [(g, v) for g, v in zip(grads, vars_phase1) if g is not None]

        if grads_and_vars:
            grads, trainable_variables = zip(*grads_and_vars)
            grads, _ = tf.clip_by_global_norm(grads, GRAD_CLIP_NORM)
            self.opt_phase1.apply_gradients(zip(grads, trainable_variables))

        return {"loss_p1": loss_p1}

    def train_phase_2(self, batch):
        with tf.GradientTape() as tape:
            s_enc = tf.stop_gradient(self.encoder(batch["state_ids"], training=False)["state"])
            g_enc = tf.stop_gradient(self.encoder(batch["goal_ids"], training=False)["state"])
            s_next_true = tf.stop_gradient(self.encoder(batch["next_state_ids"][:, 0, :], training=False)["state"])

            a_latent = tf.stop_gradient(self.action_encoder(s_enc, s_next_true))
            s_next_pred = self.transition_model(s_enc, a_latent)
            loss_transition = tf.reduce_mean(1.0 - tf.reduce_sum(s_next_pred * s_next_true, axis=-1))

            val_current = self.value_model(s_enc, g_enc)
            val_next = self.value_model(s_next_pred, g_enc)
            reward_pred = self.reward_model(s_enc, a_latent, s_next_pred, g_enc)

            bellman_target = tf.stop_gradient(reward_pred + GAMMA * val_next)
            loss_bellman = tf.reduce_mean(tf.square(val_current - bellman_target))

            loss_p2 = (W_TRANSITION * loss_transition) + (W_BELLMAN * loss_bellman)

        vars_phase2 = (
            self.transition_model.trainable_variables +
            self.reward_model.trainable_variables +
            self.value_model.trainable_variables
        )
        grads = tape.gradient(loss_p2, vars_phase2)
        grads_and_vars = [(g, v) for g, v in zip(grads, vars_phase2) if g is not None]

        if grads_and_vars:
            grads, trainable_variables = zip(*grads_and_vars)
            grads, _ = tf.clip_by_global_norm(grads, GRAD_CLIP_NORM)
            self.opt_phase2.apply_gradients(zip(grads, trainable_variables))

        return {"loss_p2": loss_p2}

    def train_phase_3(self, batch):
        with tf.GradientTape() as tape:
            s_enc = self.encoder(batch["state_ids"], training=True)["state"]
            g_enc = self.encoder(batch["goal_ids"], training=True)["state"]

            horizon_depth = tf.shape(batch["next_state_ids"])[1]
            batch_size = tf.shape(s_enc)[0]

            curr_state = s_enc
            plan_steps, term_logits = [], []
            loss_recursive_drift = 0.0

            # Unroll planner self-consistently
            for t in range(PLANNER_DEPTH):
                is_active = tf.cast(t < horizon_depth, tf.float32)
                t_idx = tf.minimum(t, horizon_depth - 1)

                obs_next = tf.stop_gradient(self.encoder(batch["next_state_ids"][:, t_idx, :], training=False)["state"])

                step_out = self.planner.recursive_step(
                    state=curr_state,
                    goal=g_enc,
                    training=True
                )

                plan_steps.append(step_out["plan_embedding"])
                term_logits.append(step_out["termination_logit"])

                step_drift = (1.0 - tf.reduce_sum(step_out["next_state"] * obs_next, axis=-1))
                loss_recursive_drift += tf.reduce_mean(step_drift * batch["step_mask"][:, t_idx] * is_active)

                curr_state = step_out["next_state"]

            loss_recursive_drift = tf.reduce_mean(loss_recursive_drift)
            term_logits = tf.stack(term_logits, axis=1)

            num_unrolled_steps = tf.shape(term_logits)[1]
            target_labels = batch["termination_targets"][:, :num_unrolled_steps]

            loss_termination = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=target_labels, logits=term_logits
                )
            )

            plan_seq = tf.stack(plan_steps, axis=1)[:, :horizon_depth, :]
            decoder_input = self.plan_projection(plan_seq)

            # Global Corpus Line Encoding (Allow gradient flow to encoder)
            flat_line_ids = tf.reshape(batch["line_ids"], [-1, MAX_SEQ_LEN])
            encoded_lines = self.encoder(flat_line_ids, training=True)["state"]
            num_lines = tf.shape(batch["line_ids"])[1]
            line_embeddings = tf.reshape(encoded_lines, [batch_size, num_lines, D_MODEL])

            dec_out = self.decoder(decoder_input, line_embeddings, training=True)

            num_unrolled_steps = tf.shape(dec_out["row_logits"])[1]
            target_rows = batch["target_row_ids"][:, :num_unrolled_steps]
            valid_mask = batch["step_mask"][:, :num_unrolled_steps]

            safe_target_rows = tf.maximum(target_rows, 0)
            decoder_ce = tf.keras.losses.sparse_categorical_crossentropy(
                safe_target_rows,
                dec_out["row_logits"],
                from_logits=True
            )
            decoder_ce *= valid_mask

            loss_decoder = tf.reduce_sum(decoder_ce) / tf.maximum(tf.reduce_sum(valid_mask), 1.0)

            loss_p3 = (
                (W_RECURSIVE_STATE * loss_recursive_drift) +
                (W_TERMINATION * loss_termination) +
                (W_DECODER * loss_decoder)
            )

        # Include self.encoder variables so Phase 3 shapes candidate line embeddings
        vars_phase3 = (
            self.encoder.trainable_variables +
            self.plan_embedding_model.trainable_variables +
            self.plan_projection.trainable_variables +
            self.planner.trainable_variables +
            self.decoder.trainable_variables
        )
        grads = tape.gradient(loss_p3, vars_phase3)
        grads_and_vars = [(g, v) for g, v in zip(grads, vars_phase3) if g is not None]

        if grads_and_vars:
            grads, trainable_variables = zip(*grads_and_vars)
            grads, _ = tf.clip_by_global_norm(grads, GRAD_CLIP_NORM)
            self.opt_phase3.apply_gradients(zip(grads, trainable_variables))

        return {"loss_p3": loss_p3}


# ============================================================
# INFERENCE & PLAN PRINTING
# ============================================================
def decode_and_print_plan(decoder, plan_projection, plan_step_embeddings, encoder, vectorizer, all_data_lines):
    line_token_ids = encode_texts(all_data_lines, vectorizer)
    encoded_candidates = encoder(line_token_ids, training=False)["state"]
    candidate_embeds = tf.expand_dims(encoded_candidates, axis=0)

    plan_seq = tf.stack(plan_step_embeddings, axis=1)
    decoder_input = plan_projection(plan_seq)

    predicted_row_ids = decoder.predict_rows(decoder_input, candidate_embeds)
    predicted_row_ids = predicted_row_ids[0].numpy()

    print("\n--- Decoded Plan Steps ---")
    for step_idx, row_idx in enumerate(predicted_row_ids, 1):
        clamped_idx = max(0, min(row_idx, len(all_data_lines) - 1))
        matched_action = all_data_lines[clamped_idx]
        print(f" Step {step_idx} (Row {clamped_idx}): {matched_action}")


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Self-Supervised Bellman Planner")
    parser.add_argument("--data", type=str, default=None, help="Path to data corpus text file.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--depth", type=int, default=PLANNER_DEPTH, help="Planning horizon depth.")
    parser.add_argument("--state", type=str, default="The kitchen is dirty.", help="Initial state query.")
    parser.add_argument("--goal", type=str, default="The kitchen is clean.", help="Goal state query.")
    args = parser.parse_args()

    # 1. Dataset & Canonical Lines Initialization
    if args.data and os.path.exists(args.data):
        all_data_lines = read_data_lines(args.data)
        trajectories = read_trajectories_from_lines(all_data_lines)
    else:
        all_data_lines, trajectories = generate_synthetic_corpus()

    vectorizer = build_vectorizer(all_data_lines)
    actual_vocab_size = len(vectorizer.get_vocabulary()) + SPECIAL_TOKENS + 10

    all_line_ids = encode_texts(all_data_lines, vectorizer)

    examples = make_training_examples(trajectories, args.depth)
    dataset = make_dataset(examples, all_line_ids, vectorizer, args.depth, args.batch_size)

    # 2. Instantiate Network Components
    state_encoder = InceptionStateEncoder(actual_vocab_size)
    action_encoder = ObservedActionEncoder()
    action_model = ActionModel()
    transition_model = TransitionModel()
    reward_model = IntrinsicRewardModel()
    value_model = ValueModel()
    plan_embedding_model = PlanEmbeddingModel()
    plan_projection = PlanSentenceProjection()

    planner = RecursiveBellmanPlanner(
        action_model, transition_model, reward_model, value_model, plan_embedding_model
    )

    decoder = HierarchicalPlanDecoder(
        d_model=D_MODEL,
        max_depth=args.depth + 1
    )

    trainer = ModularPlannerTrainer(
        state_encoder=state_encoder,
        action_encoder=action_encoder,
        action_model=action_model,
        transition_model=transition_model,
        reward_model=reward_model,
        value_model=value_model,
        plan_embedding_model=plan_embedding_model,
        plan_projection=plan_projection,
        planner=planner,
        decoder=decoder,
        learning_rate=LEARNING_RATE
    )

    # 3. Training Loop
    print(f"\n--- Starting Multi-Stage Training ({args.epochs} Epochs) ---")
    for epoch in range(1, args.epochs + 1):
        p1_losses, p2_losses, p3_losses = [], [], []

        for batch in dataset:
            p1_out = trainer.train_phase_1(batch)
            p2_out = trainer.train_phase_2(batch)
            p3_out = trainer.train_phase_3(batch)

            p1_losses.append(p1_out["loss_p1"])
            p2_losses.append(p2_out["loss_p2"])
            p3_losses.append(p3_out["loss_p3"])

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"Phase 1: {np.mean(p1_losses):.4f} | "
                f"Phase 2: {np.mean(p2_losses):.4f} | "
                f"Phase 3: {np.mean(p3_losses):.4f}"
            )

    # 4. Inferred Self-Supervised Plan Output
    print("\n==================================================")
    print("            INFERRED PLAN OUTPUT                  ")
    print("==================================================")
    print(f"Initial State: \"{args.state}\"")
    print(f"Goal State:    \"{args.goal}\"")
    print("--------------------------------------------------")

    s_ids = encode_texts([args.state], vectorizer)
    g_ids = encode_texts([args.goal], vectorizer)

    s_init = state_encoder(s_ids, training=False)["state"]
    g_init = state_encoder(g_ids, training=False)["state"]

    curr_state = s_init
    plan_step_embeddings = []

    for step in range(args.depth):
        step_out = planner.recursive_step(curr_state, g_init, training=False)
        plan_step_embeddings.append(step_out["plan_embedding"])
        curr_state = step_out["next_state"]

    decode_and_print_plan(
        decoder=decoder,
        plan_projection=plan_projection,
        plan_step_embeddings=plan_step_embeddings,
        encoder=trainer.encoder,
        vectorizer=vectorizer,
        all_data_lines=all_data_lines
    )
    print("==================================================\n")


if __name__ == "__main__":
    main()