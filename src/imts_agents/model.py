"""Causal time-patch recurrence with explicit T/S/ST role memories.

This is a small, trainable research prototype, not a reproduction of TimeFilter
or TarMAC. Each relation role has a persistent channel-wise memory when outer
recurrence is enabled. Learned query/key/value messages exchange pooled role
states. Role scopes restrict direct token reads; received messages may convey
information from other scopes. The three roles are therefore interacting neural
modules, not evidence by themselves that decentralised agency is necessary.

All variants use the same token encoder, chronological patch construction and
masked role/channel/patch pooled classification head. Removing outer recurrence
resets memories each patch, but retains causal prefix attention, intentionally
isolating state recurrence from access to earlier input. Single-ALL controls have
three updates per patch, while fixed specialised controls update T/S/ST once.

State-only variants replace old input tokens by role-specific memory captured
before this patch's initial write. The persistent state also includes which
channels have been seen and the pooled readout accumulator; time-gap features are
prepared by the causal data adapter. Selective writing changes the initial input
GRU write only. It does not hide current input from subsequent relation attention
or policy, nor skip that GRU's computation. Independent sigmoid gates allow several
roles to write the same input, unlike a mutually exclusive softmax router.

Hard dynamic actions train with a REINFORCE term supplied by the runner. Only the
selected relation update executes for a sample; policy/message computation has
additional cost and must not be represented as free. Training samples actions;
evaluation uses argmax. COMMIT is prohibited before the first update in a patch.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from .variants import VariantSpec, get_variant


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def relation_mask(scope: str, n_patches: int, n_channels: int, device: Any = None) -> Tensor:
    """Allowed direct reads for current-patch queries against a causal prefix.

    T: same channel, current or earlier patches.
    S: different channels in the current patch.
    ST: different channels in strictly earlier patches.
    ALL: every channel in current or earlier patches.
    """
    if n_patches < 1 or n_channels < 1:
        raise ValueError("A relation mask requires at least one patch and channel")
    query_channel = torch.arange(n_channels, device=device)[:, None]
    key_channel = torch.arange(n_channels, device=device).repeat(n_patches)[None, :]
    key_patch = torch.arange(n_patches, device=device).repeat_interleave(n_channels)[None, :]
    same_channel = query_channel == key_channel
    current_patch = key_patch == n_patches - 1
    if scope == "T":
        return same_channel
    if scope == "S":
        return (~same_channel) & current_patch
    if scope == "ST":
        return (~same_channel) & (~current_patch)
    if scope == "ALL":
        return torch.ones(n_channels, n_patches * n_channels, dtype=torch.bool, device=device)
    raise ValueError(f"Unknown relation scope: {scope}")


def _masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weight = mask.to(values.dtype).unsqueeze(-1)
    return (values * weight).sum(dim) / weight.sum(dim).clamp_min(1)


class RelationUpdate(nn.Module):
    """A role-specific attention read followed by a channel memory update."""

    def __init__(self, hidden: int, capacity_width: int = 0) -> None:
        super().__init__()
        self.hidden = hidden
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.cell = nn.GRUCell(2 * hidden, hidden)
        self.capacity = (
            nn.Sequential(nn.Linear(hidden, capacity_width), nn.GELU(), nn.Linear(capacity_width, hidden))
            if capacity_width else None
        )

    def forward(self, state: Tensor, tokens: Tensor, mask: Tensor, allowed: Tensor, message: Tensor) -> Tensor:
        batch, channels, hidden = state.shape
        flattened = tokens.flatten(1, 2)
        valid = allowed.unsqueeze(0) & mask.flatten(1, 2).unsqueeze(1)
        scores = torch.bmm(self.query(state), self.key(flattened).transpose(1, 2)) / math.sqrt(hidden)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = scores.softmax(-1) * valid.to(scores.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1)
        context = self.norm(self.output(torch.bmm(weights, self.value(flattened))))
        # A role without a direct neighbour can still receive another role's
        # message; the missing attention context itself must be exactly zero.
        context = context * valid.any(-1, keepdim=True).to(context.dtype)
        if self.capacity is not None:
            context = context + self.capacity(context)
        received = message[:, None, :].expand(batch, channels, hidden)
        return self.cell(
            torch.cat((context, received), -1).reshape(-1, 2 * hidden),
            state.reshape(-1, hidden),
        ).reshape(batch, channels, hidden)


class RoleCommunication(nn.Module):
    """Targeted messages among role summaries, excluding each role itself."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, hidden)

    def forward(self, summaries: Tensor) -> Tensor:
        n_roles = summaries.shape[1]
        if n_roles == 1:
            return torch.zeros_like(summaries)
        scores = torch.bmm(self.query(summaries), self.key(summaries).transpose(1, 2)) / math.sqrt(self.hidden)
        self_mask = torch.eye(n_roles, device=summaries.device, dtype=torch.bool)[None]
        weights = scores.masked_fill(self_mask, torch.finfo(scores.dtype).min).softmax(-1)
        return self.output(torch.bmm(weights, self.value(summaries)))


class AgentClassifier(nn.Module):
    """Classification model with auditable core factors and policy controls.

    Inputs are X[B,P,C,F] and observed-token mask[B,P,C]. Features masked out by
    the input mask are never read, even if they contain arbitrary sentinel values.
    No-time-feature ablation is applied by the data adapter using the registry's
    time_features flag; this module does not guess feature-column meanings.
    """

    def __init__(
        self, n_features: int, n_channels: int, n_classes: int,
        variant: str | VariantSpec = "core_both", hidden: int = 32,
        max_inner_steps: int = 3,
    ) -> None:
        super().__init__()
        if min(n_features, n_channels, n_classes, hidden, max_inner_steps) < 1:
            raise ValueError("Feature/channel/class/hidden/step counts must be positive")
        self.variant = get_variant(variant)
        if self.variant.train_policy and len(self.variant.role_scopes) != 3:
            raise ValueError("Dynamic action names require three role memories")
        if not self.variant.train_policy and max_inner_steps != 3:
            raise ValueError("Fixed controls require max_inner_steps=3 to preserve their matched update budget")
        self.n_features = n_features
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.hidden = hidden
        self.max_inner_steps = max_inner_steps
        self.scopes = self.variant.role_scopes
        self.n_roles = len(self.scopes)
        self.encoder = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.channel_embedding = nn.Embedding(n_channels, hidden)
        self.input_cell = nn.GRUCell(hidden, hidden)
        self.roles = nn.ModuleList(RelationUpdate(hidden) for _ in self.scopes)
        self.communication = RoleCommunication(hidden) if self.n_roles > 1 else None
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, n_classes))
        self.parameter_match_target: int | None = None

        if self.variant.parameter_matched:
            # Active ALL-role feedforward capacity approximately matches the
            # specialised fixed reference. This matches parameters, NOT FLOPs.
            one_role = count_parameters(self.roles[0])
            communication_params = 4 * (hidden * hidden + hidden)
            extra_target = 2 * one_role + communication_params
            width = max(1, round((extra_target - hidden) / (2 * hidden + 1)))
            self.parameter_match_target = count_parameters(self) + extra_target
            self.roles[0].capacity = nn.Sequential(
                nn.Linear(hidden, width), nn.GELU(), nn.Linear(width, hidden),
            )

        if self.variant.train_policy:
            self.step_embedding = nn.Embedding(max_inner_steps, hidden)
            if self.variant.routing == "centralized":
                self.router_cell = nn.GRUCell(self.n_roles * 3 * hidden, hidden)
                self.router_head = nn.Linear(hidden, 4)
            else:
                self.local_policies = nn.ModuleList(
                    nn.Sequential(nn.Linear(3 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 4))
                    for _ in self.scopes
                )

        # Construct only for new variants and after all existing modules so old
        # variants retain their parameter names and seeded initialization.
        if self.variant.input_write == "static":
            self.write_logits = nn.Parameter(torch.zeros(self.n_roles))
        elif self.variant.input_write == "content":
            self.write_role_embedding = nn.Embedding(self.n_roles, hidden)
            self.write_gate = nn.Sequential(
                nn.Linear(3 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1),
            )

    def _input_gates(self, current_tokens: Tensor, states: Tensor) -> Tensor:
        """Independent initial-write gates [B,R,C], before validity masking."""
        if self.variant.input_write == "broadcast":
            return current_tokens.new_ones(current_tokens.shape[:3])
        if self.variant.input_write == "static":
            return self.write_logits.sigmoid()[None, :, None].expand(current_tokens.shape[:3])
        identity = self.write_role_embedding.weight[None, :, None, :].expand_as(states)
        return self.write_gate(torch.cat((current_tokens, states, identity), -1)).squeeze(-1).sigmoid()

    def _write_input(self, states: Tensor, current_tokens: Tensor, current_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Gate the initial GRU write; closed gates retain the old memory."""
        fused = self.input_cell(
            current_tokens.reshape(-1, self.hidden), states.reshape(-1, self.hidden),
        ).reshape_as(states)
        gates = self._input_gates(current_tokens, states)
        valid = current_mask[:, None, :].expand_as(gates)
        if self.variant.input_write == "broadcast":
            # Preserve the old arithmetic exactly for old experiment results.
            written = fused
        else:
            written = states + gates[..., None] * (fused - states)
        return torch.where(valid[..., None], written, states), gates, valid

    def _relation_sources(
        self, role: int, tokens: Tensor, mask: Tensor,
        prior_states: Tensor | None = None, prior_seen: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build immutable direct-read sources for one role in this patch.

        In state mode, tokens must contain ONLY the current patch. The first
        source block is that role's frozen previous-patch channel memory; the
        second is current input. ST therefore reads cross-channel old memory,
        including when this channel has no current observation.
        """
        if self.variant.history_access == "state":
            if tokens.shape[1] != 1 or prior_states is None or prior_seen is None:
                raise ValueError("State-only reads require current-patch tokens and frozen prior states/seen mask")
            tokens = torch.cat((prior_states[:, role, None], tokens), 1)
            mask = torch.cat((prior_seen[:, None], mask), 1)
        return tokens, mask, relation_mask(self.scopes[role], tokens.shape[1], self.n_channels, tokens.device)

    def _role_summaries(self, states: Tensor, channel_mask: Tensor) -> Tensor:
        return _masked_mean(states, channel_mask[:, None, :], 2)

    def _messages(self, states: Tensor, channel_mask: Tensor) -> Tensor:
        summaries = self._role_summaries(states, channel_mask)
        if self.communication is None or not self.variant.communication:
            return torch.zeros_like(summaries)
        return self.communication(summaries)

    def _policy_observations(
        self, states: Tensor, tokens: Tensor, mask: Tensor, channel_mask: Tensor,
        prior_states: Tensor | None = None, prior_seen: Tensor | None = None,
    ) -> Tensor:
        summaries = self._role_summaries(states, channel_mask)
        messages = self._messages(states, channel_mask)
        scoped_contexts = []
        for role in range(self.n_roles):
            source_tokens, source_mask, allowed = self._relation_sources(role, tokens, mask, prior_states, prior_seen)
            flattened = source_tokens.flatten(1, 2)
            flat_mask = source_mask.flatten(1, 2)
            # Each role observes the union of its directly readable tokens.
            readable = flat_mask & allowed.any(0)[None]
            scoped_contexts.append(_masked_mean(flattened, readable, 1))
        return torch.cat((summaries, messages, torch.stack(scoped_contexts, 1)), -1)

    def _policy_scores(self, observation: Tensor, step: int, router_state: Tensor) -> tuple[Tensor, Tensor]:
        step_signal = self.step_embedding.weight[step]
        observation = observation + torch.cat((step_signal, torch.zeros_like(step_signal), torch.zeros_like(step_signal)))[None, None]
        if self.variant.routing == "centralized":
            next_router = self.router_cell(observation.flatten(1), router_state)
            return self.router_head(next_router), next_router
        local_scores = torch.stack([
            policy(observation[:, role]) for role, policy in enumerate(self.local_policies)
        ], 1)
        return local_scores.mean(1), router_state

    def _update_role(
        self, states: Tensor, role: int, tokens: Tensor, mask: Tensor, channel_mask: Tensor, selected: Tensor,
        prior_states: Tensor | None = None, prior_seen: Tensor | None = None,
    ) -> Tensor:
        # Actual sub-batch dispatch: unselected role updates are not evaluated.
        indices = selected.nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            return states
        subset = states.index_select(0, indices)
        subset_channels = channel_mask.index_select(0, indices)
        message = self._messages(subset, subset_channels)[:, role]
        source_tokens, source_mask, allowed = self._relation_sources(role, tokens, mask, prior_states, prior_seen)
        proposal = self.roles[role](
            subset[:, role], source_tokens.index_select(0, indices), source_mask.index_select(0, indices), allowed, message,
        )
        proposal = torch.where(subset_channels[:, :, None], proposal, subset[:, role])
        next_role = states[:, role].index_copy(0, indices, proposal)
        result = states.clone()
        result[:, role] = next_role
        return result

    def forward(self, x: Tensor, mask: Tensor, *, return_trace: bool = False, return_actions: bool = False) -> dict[str, Tensor]:
        if x.ndim != 4 or mask.shape != x.shape[:3]:
            raise ValueError("Expected x[B,P,C,F] and mask[B,P,C]")
        if x.shape[2:] != (self.n_channels, self.n_features) or x.shape[1] == 0:
            raise ValueError("Input channel/feature counts must match the model; at least one patch is required")
        mask = mask.to(device=x.device, dtype=torch.bool)
        # torch.where also discards NaNs in masked input, unlike multiplication.
        safe_x = torch.where(mask[..., None], x, torch.zeros_like(x))
        if not torch.isfinite(safe_x).all():
            raise ValueError("Observed input features must be finite")
        tokens = self.encoder(safe_x) + self.channel_embedding.weight[None, None]
        tokens = torch.where(mask[..., None], tokens, torch.zeros_like(tokens))
        batch, patches, channels, hidden = tokens.shape
        states = tokens.new_zeros(batch, self.n_roles, channels, hidden)
        router_state = tokens.new_zeros(batch, hidden)
        updates = tokens.new_zeros(batch)
        log_prob = tokens.new_zeros(batch)
        entropy = tokens.new_zeros(batch)
        action_counts = tokens.new_zeros(batch, 4)
        decision_counts = tokens.new_zeros(batch)
        input_gate_sum = tokens.new_zeros(batch, self.n_roles)
        input_gate_squared_sum = tokens.new_zeros(batch, self.n_roles)
        input_gate_count = tokens.new_zeros(batch, self.n_roles)
        seen_channels = mask.new_zeros(batch, channels)
        patch_summaries = []
        patch_states = []
        policy_trace = []
        action_trace = []
        gate_trace = []
        gate_valid_trace = []
        before_write_trace = []
        after_write_trace = []
        prior_seen_trace = []

        for patch in range(patches):
            current_mask = mask[:, patch]
            active_patch = current_mask.any(-1)
            if not self.variant.outer_recurrent:
                states = torch.zeros_like(states)
                router_state = torch.zeros_like(router_state)
            prior_states, prior_seen = states, seen_channels
            current_tokens = tokens[:, patch, None].expand(-1, self.n_roles, -1, -1)
            states, gates, gate_valid = self._write_input(states, current_tokens, current_mask)
            # Gate diagnostics are detached: they are measurements, not an
            # extra loss or a way to retain the training graph unnecessarily.
            measured_gates = gates.detach() * gate_valid
            input_gate_sum = input_gate_sum + measured_gates.sum(-1)
            input_gate_squared_sum = input_gate_squared_sum + measured_gates.square().sum(-1)
            input_gate_count = input_gate_count + gate_valid.sum(-1).to(tokens.dtype)
            seen_channels = prior_seen | current_mask
            if self.variant.history_access == "state":
                read_tokens, read_mask = tokens[:, patch:patch + 1], mask[:, patch:patch + 1]
                source_states, source_seen = prior_states, prior_seen
            else:
                read_tokens, read_mask = tokens[:, :patch + 1], mask[:, :patch + 1]
                source_states = source_seen = None
            if return_trace:
                gate_trace.append(gates)
                gate_valid_trace.append(gate_valid)
                before_write_trace.append(prior_states)
                after_write_trace.append(states)
                prior_seen_trace.append(prior_seen)
            active = active_patch.clone()
            initial_observation = (
                self._policy_observations(states, read_tokens, read_mask, seen_channels, source_states, source_seen)
                if self.variant.routing == "preplanned" else None
            )
            patch_policy = []
            patch_actions = []

            for step in range(self.max_inner_steps):
                if self.variant.train_policy:
                    observation = initial_observation if initial_observation is not None else self._policy_observations(states, read_tokens, read_mask, seen_channels, source_states, source_seen)
                    scores, next_router = self._policy_scores(observation, step, router_state)
                    router_state = torch.where(active[:, None], next_router, router_state)
                    if step == 0 or not self.variant.allow_commit:
                        scores = scores.clone()
                        scores[:, 3] = torch.finfo(scores.dtype).min
                    distribution = torch.distributions.Categorical(logits=scores)
                    action = distribution.sample() if self.training else scores.argmax(-1)
                    # Finished/empty samples contribute no policy loss or cost.
                    log_prob = log_prob + distribution.log_prob(action) * active.to(tokens.dtype)
                    entropy = entropy + distribution.entropy() * active.to(tokens.dtype)
                    decision_counts = decision_counts + active.to(tokens.dtype)
                    action_counts = action_counts + nn.functional.one_hot(action, 4).to(tokens.dtype) * active[:, None]
                    if return_trace:
                        patch_policy.append(scores)
                    recorded_action = torch.where(active, action, torch.full_like(action, -1))
                    for role in range(self.n_roles):
                        selected = active & (action == role)
                        states = self._update_role(states, role, read_tokens, read_mask, seen_channels, selected, source_states, source_seen)
                    updated = active & (action != 3)
                    updates = updates + updated.to(tokens.dtype)
                    active = active & (action != 3)
                else:
                    role = step if self.n_roles == 3 else 0
                    states = self._update_role(states, role, read_tokens, read_mask, seen_channels, active, source_states, source_seen)
                    updates = updates + active.to(tokens.dtype)
                    action_counts[:, role] = action_counts[:, role] + active.to(tokens.dtype)
                    recorded_action = torch.where(active, torch.full((batch,), role, device=x.device, dtype=torch.long), torch.full((batch,), -1, device=x.device, dtype=torch.long))
                if return_trace or return_actions:
                    patch_actions.append(recorded_action)

            summary = self._role_summaries(states, seen_channels).mean(1)
            patch_summaries.append(summary * active_patch[:, None])
            if return_trace:
                patch_states.append(states)
                if patch_policy:
                    policy_trace.append(torch.stack(patch_policy, 1))
            if return_trace or return_actions:
                action_trace.append(torch.stack(patch_actions, 1))

        observed_patches = mask.any(-1).sum(-1).clamp_min(1).to(tokens.dtype)
        representation = torch.stack(patch_summaries, 1).sum(1) / observed_patches[:, None]
        result = {
            "logits": self.head(representation),
            "policy_log_prob": log_prob,
            "policy_entropy": entropy,
            "updates": updates,
            "action_counts": action_counts,
            "effective_steps": decision_counts,
            "input_gate_sum": input_gate_sum,
            "input_gate_count": input_gate_count,
            "input_gate_squared_sum": input_gate_squared_sum,
        }
        if return_trace:
            result["patch_states"] = torch.stack(patch_states, 1)
            result["input_gates"] = torch.stack(gate_trace, 1)
            result["input_gate_valid"] = torch.stack(gate_valid_trace, 1)
            result["input_states_before"] = torch.stack(before_write_trace, 1)
            result["input_states_after"] = torch.stack(after_write_trace, 1)
            result["prior_seen"] = torch.stack(prior_seen_trace, 1)
            if policy_trace:
                result["policy_scores"] = torch.stack(policy_trace, 1)
        if return_trace or return_actions:
            result["actions"] = torch.stack(action_trace, 1)
        return result


# A descriptive alias for callers that prefer a model name over its role label.
TimePatchClassifier = AgentClassifier
