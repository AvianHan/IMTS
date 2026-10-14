"""Mechanism-level checks for causal access and the promised ablations."""

from dataclasses import fields
from unittest.mock import patch

import pytest
import torch

from imts_agents.model import AgentClassifier, count_parameters, relation_mask
from imts_agents.variants import GROUPS, VARIANTS, get_variant, variant_names


@pytest.fixture(autouse=True)
def _small_cpu_models():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(13)
    yield
    torch.set_num_threads(previous)


def _model(variant="core_both", channels=3):
    return AgentClassifier(12, channels, 2, variant=variant, hidden=8).eval()


def test_core_design_is_a_two_by_two_and_control_flags_are_distinct():
    assert {(VARIANTS[name].outer_recurrent, VARIANTS[name].specialized_roles)
            for name in GROUPS["core"]} == {(False, False), (True, False), (False, True), (True, True)}
    factor_fields = [field.name for field in fields(get_variant("core_both"))
                     if field.name not in {"name", "group", "description", "reference"}]
    signatures = [tuple(getattr(VARIANTS[key], name) for name in factor_fields) for key in GROUPS["all"]]
    assert len(set(signatures)) == len(signatures)
    assert variant_names("pilot") == variant_names("core")
    assert all(spec.reference in VARIANTS for spec in VARIANTS.values() if spec.reference)


def test_new_experiment_groups_deduplicate_references_and_keep_old_groups_stable():
    assert len(GROUPS["all"]) == 12
    assert len(GROUPS["ablation"]) == 10
    assert len(set(GROUPS["ablation"])) == 10
    assert GROUPS["write"][0] == GROUPS["interaction"][0] == "history_state"
    assert all(VARIANTS[name].specialized_roles for name in GROUPS["ablation"])
    assert set(GROUPS["ablation"]) == set(GROUPS["history"] + GROUPS["write"] + GROUPS["interaction"])


@pytest.mark.parametrize("old,new", [("core_both", "history_both"), ("core_inner", "history_reread")])
def test_history_aliases_keep_old_seeded_parameters_and_outputs(old, new):
    torch.manual_seed(133)
    original = _model(old)
    torch.manual_seed(133)
    renamed = _model(new)
    assert original.state_dict().keys() == renamed.state_dict().keys()
    for key in original.state_dict():
        torch.testing.assert_close(original.state_dict()[key], renamed.state_dict()[key], rtol=0, atol=0)
    x = torch.randn(2, 3, 3, 12)
    mask = torch.rand(2, 3, 3) > 0.3
    with torch.no_grad():
        left, right = original(x, mask, return_trace=True), renamed(x, mask, return_trace=True)
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_relation_scopes_cover_causal_prefix_without_overlap():
    # Query channel 1, with flattened keys (p0:c0,c1,c2,p1:c0,c1,c2).
    temporal = relation_mask("T", 2, 3)
    spatial = relation_mask("S", 2, 3)
    spatiotemporal = relation_mask("ST", 2, 3)
    assert temporal[1].tolist() == [False, True, False, False, True, False]
    assert spatial[1].tolist() == [False, False, False, True, False, True]
    assert spatiotemporal[1].tolist() == [True, False, True, False, False, False]
    assert torch.equal(temporal.int() + spatial.int() + spatiotemporal.int(), relation_mask("ALL", 2, 3).int())
    assert not relation_mask("ST", 1, 3).any()


@pytest.mark.parametrize("name", list(VARIANTS))
def test_padded_values_are_ignored_and_empty_samples_have_no_actions(name):
    model = _model(name)
    x = torch.randn(2, 3, 3, 12)
    mask = torch.tensor([[[1, 1, 0], [0, 0, 0], [1, 0, 1]], [[0, 0, 0]] * 3], dtype=torch.bool)
    poisoned = x.clone()
    poisoned[~mask] = float("nan")
    with torch.no_grad():
        normal = model(x, mask)
        padded = model(poisoned, mask)
    assert normal["logits"].shape == (2, 2)
    assert all(torch.isfinite(value).all() for value in padded.values())
    torch.testing.assert_close(normal["logits"], padded["logits"])
    assert padded["updates"][1] == 0
    assert padded["action_counts"][1].sum() == 0
    assert padded["policy_log_prob"][1] == 0
    assert padded["policy_entropy"][1] == 0


@pytest.mark.parametrize("name", ["core_none", "core_both", "dynamic_feedback", "preplanned_path", "centralized_router"] + list(GROUPS["ablation"]))
def test_future_input_cannot_change_earlier_patch_state_or_decisions(name):
    model = _model(name)
    x = torch.randn(1, 4, 3, 12)
    mask = torch.ones(1, 4, 3, dtype=torch.bool)
    changed = x.clone()
    changed[:, 2:] = 200 * torch.randn_like(changed[:, 2:])
    with torch.no_grad():
        original = model(x, mask, return_trace=True)
        perturbed = model(changed, mask, return_trace=True)
        prefix = model(x[:, :2], mask[:, :2], return_trace=True)
    torch.testing.assert_close(original["patch_states"][:, :2], perturbed["patch_states"][:, :2])
    torch.testing.assert_close(original["patch_states"][:, :2], prefix["patch_states"])
    torch.testing.assert_close(original["actions"][:, :2], perturbed["actions"][:, :2])


@pytest.mark.parametrize("name", ["core_inner", "core_both"])
def test_outer_factor_controls_persistence_of_each_role_memory(name):
    model = _model(name)
    incoming = []
    hook = model.input_cell.register_forward_pre_hook(lambda _module, args: incoming.append(args[1].detach().clone()))
    with torch.no_grad():
        result = model(torch.randn(1, 2, 3, 12), torch.ones(1, 2, 3, dtype=torch.bool), return_trace=True)
    hook.remove()
    assert incoming[0].abs().sum() == 0
    if name == "core_both":
        torch.testing.assert_close(incoming[1].reshape(1, 3, 3, 8), result["patch_states"][:, 0])
        assert incoming[1].abs().sum() > 0
    else:
        assert incoming[1].abs().sum() == 0
    assert not torch.allclose(result["patch_states"][:, 0, 0], result["patch_states"][:, 0, 1])


@pytest.mark.parametrize("name", ["core_none", "core_outer", "core_inner", "core_both", "no_commit"])
def test_fixed_budget_and_no_commit_execute_three_updates_per_nonempty_patch(name):
    model = _model(name)
    mask = torch.tensor([[[1, 1, 1], [0, 0, 0], [1, 0, 0]]], dtype=torch.bool)
    with torch.no_grad():
        result = model(torch.randn(1, 3, 3, 12), mask, return_trace=True)
    assert result["updates"].item() == 6
    assert result["action_counts"][0, 3].item() == 0
    assert torch.equal(result["actions"][:, 1], torch.full((1, 3), -1))


def test_commit_requires_one_update_and_stops_later_updates():
    model = _model("dynamic_feedback")
    for policy in model.local_policies:
        with torch.no_grad():
            policy[-1].weight.zero_()
            policy[-1].bias.copy_(torch.tensor([1., 0., 0., 100.]))
    with torch.no_grad():
        result = model(torch.randn(1, 2, 3, 12), torch.ones(1, 2, 3, dtype=torch.bool), return_trace=True)
    assert result["actions"].tolist() == [[[0, 3, -1], [0, 3, -1]]]
    assert result["updates"].item() == 2
    assert result["action_counts"].tolist() == [[2., 0., 0., 2.]]


def test_no_communication_keeps_order_and_blocks_cross_role_gradient():
    standard, no_messages = _model("core_both"), _model("no_communication")
    no_messages.load_state_dict(standard.state_dict())
    tokens = torch.randn(1, 2, 3, 8)
    mask = torch.ones(1, 2, 3, dtype=torch.bool)
    channels = mask.any(1)
    for model, has_gradient in [(standard, True), (no_messages, False)]:
        states = torch.randn(1, 3, 3, 8, requires_grad=True)
        updated = model._update_role(states, 0, tokens, mask, channels, torch.tensor([True]))
        gradient = torch.autograd.grad(updated[:, 0].square().sum(), states)[0]
        assert bool(gradient[:, 1:].abs().sum() > 0) == has_gradient
    x = torch.randn(1, 2, 3, 12)
    with torch.no_grad():
        no_comm_result = no_messages(x, mask, return_trace=True)
        with patch.object(standard, "_messages", side_effect=lambda states, channel_mask: torch.zeros_like(standard._role_summaries(states, channel_mask))):
            zeroed_result = standard(x, mask, return_trace=True)
    torch.testing.assert_close(no_comm_result["logits"], zeroed_result["logits"])
    assert no_comm_result["actions"].tolist() == [[[0, 1, 2], [0, 1, 2]]]


@pytest.mark.parametrize("name,should_change", [("dynamic_feedback", True), ("preplanned_path", False)])
def test_preplanned_policy_cannot_see_intermediate_role_outputs(name, should_change):
    model = _model(name)
    for policy in model.local_policies:
        with torch.no_grad():
            policy[-1].bias.copy_(torch.tensor([100., 0., 0., -100.]))
    x, mask = torch.randn(1, 1, 3, 12), torch.ones(1, 1, 3, dtype=torch.bool)
    with torch.no_grad():
        normal = model(x, mask, return_trace=True)
        with patch.object(model.roles[0], "forward", side_effect=lambda state, *args: state + 5):
            intervened = model(x, mask, return_trace=True)
    torch.testing.assert_close(normal["policy_scores"][:, :, 0], intervened["policy_scores"][:, :, 0])
    later_equal = torch.allclose(normal["policy_scores"][:, :, 1:, :3], intervened["policy_scores"][:, :, 1:, :3])
    assert later_equal != should_change


def test_policies_receive_reinforce_gradients_and_chosen_roles_receive_classification_gradients():
    model = _model("dynamic_feedback").train()
    result = model(torch.randn(4, 2, 3, 12), torch.ones(4, 2, 3, dtype=torch.bool))
    ce = torch.nn.functional.cross_entropy(result["logits"], torch.tensor([0, 1, 0, 1]), reduction="none")
    detached_cost = ce.detach() + 0.01 * result["updates"].detach()
    loss = ce.mean() + (detached_cost * result["policy_log_prob"]).mean() - 0.01 * result["policy_entropy"].mean()
    loss.backward()
    policy_grad = sum(parameter.grad.abs().sum() for parameter in model.local_policies.parameters() if parameter.grad is not None)
    role_grad = sum(parameter.grad.abs().sum() for parameter in model.roles.parameters() if parameter.grad is not None)
    assert policy_grad > 0 and role_grad > 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_parameter_and_specialisation_controls_are_honest():
    specialised = _model("core_both")
    homogeneous = _model("homogeneous_roles")
    matched = _model("single_matched")
    assert homogeneous.scopes == ("ALL", "ALL", "ALL")
    assert count_parameters(homogeneous) == count_parameters(specialised)
    assert homogeneous.roles[0].query.weight.data_ptr() != homogeneous.roles[1].query.weight.data_ptr()
    assert abs(count_parameters(matched) - count_parameters(specialised)) / count_parameters(specialised) < 0.01
    assert matched.parameter_match_target == count_parameters(specialised)
    x, mask = torch.randn(1, 2, 3, 12), torch.ones(1, 2, 3, dtype=torch.bool)
    matched(x, mask)["logits"].sum().backward()
    assert matched.roles[0].capacity[0].weight.grad.abs().sum() > 0


def test_inserting_an_empty_patch_does_not_change_predictions():
    x, mask = torch.randn(1, 2, 3, 12), torch.ones(1, 2, 3, dtype=torch.bool)
    inserted_x = torch.cat((x[:, :1], torch.full_like(x[:, :1], float("nan")), x[:, 1:]), 1)
    inserted_mask = torch.cat((mask[:, :1], torch.zeros_like(mask[:, :1]), mask[:, 1:]), 1)
    for name in ("core_none", "core_both", "dynamic_feedback", "centralized_router", *GROUPS["ablation"]):
        model = _model(name)
        with torch.no_grad():
            original, inserted = model(x, mask), model(inserted_x, inserted_mask)
        torch.testing.assert_close(original["logits"], inserted["logits"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(original["updates"], inserted["updates"])


def test_state_only_relation_sources_are_frozen_old_role_memory_not_raw_history():
    model = _model("history_state")
    captured = [[] for _ in range(3)]
    hooks = []
    for role in range(3):
        def capture(_module, args, role=role):
            # state, source tokens, source mask, allowed, message
            captured[role].append(tuple(item.detach().clone() for item in args))
        hooks.append(model.roles[role].register_forward_pre_hook(capture))
    x = torch.randn(1, 3, 3, 12)
    mask = torch.tensor([[[True, False, False], [False, True, False], [True, False, True]]])
    with torch.no_grad():
        result = model(x, mask, return_trace=True)
    for hook in hooks:
        hook.remove()
    for role in range(3):
        for step, args in enumerate(captured[role]):
            source_tokens, source_mask, allowed = args[1:4]
            assert source_tokens.shape == (1, 2, 3, 8)
            torch.testing.assert_close(source_tokens[:, 0], result["input_states_before"][:, step, role])
            torch.testing.assert_close(source_mask[:, 0], result["prior_seen"][:, step])
            torch.testing.assert_close(source_mask[:, 1], mask[:, step])
            torch.testing.assert_close(allowed, relation_mask(model.scopes[role], 2, 3))
    # At patch 1 the newly observed channel 1 is not in ST's old memory mask.
    st_second = captured[2][1]
    assert st_second[2][:, 0].tolist() == [[True, False, False]]
    assert not st_second[3][:, 3:].any()
    assert not captured[2][0][2][:, 0].any()
    torch.testing.assert_close(result["input_states_before"][:, 1:], result["patch_states"][:, :-1])


@pytest.mark.parametrize("name", ["interaction_dynamic", "interaction_preplanned", "interaction_centralized"])
def test_state_only_policy_cannot_be_passed_a_raw_input_prefix(name):
    model = _model(name)
    states = torch.randn(1, 3, 3, 8)
    prior = torch.randn_like(states)
    mask = torch.ones(1, 2, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="current-patch tokens"):
        model._policy_observations(states, torch.randn(1, 2, 3, 8), mask, mask.any(1), prior, mask[:, 0])
    observations = model._policy_observations(
        states, torch.randn(1, 1, 3, 8), mask[:, :1], mask.any(1), prior, mask[:, 0],
    )
    # ST's direct context is only the old ST memory. It does not depend on
    # current-patch raw inputs, even though states/messages may contain them.
    changed_input = model._policy_observations(
        states, torch.randn(1, 1, 3, 8) * 100, mask[:, :1], mask.any(1), prior, mask[:, 0],
    )
    torch.testing.assert_close(observations[:, 2, -8:], changed_input[:, 2, -8:])
    torch.testing.assert_close(observations[:, 2, -8:], prior[:, 2].mean(1))


@pytest.mark.parametrize("name", ["write_static", "write_content"])
def test_closed_input_gate_preserves_memory_and_open_gate_matches_broadcast(name):
    gated = _model(name)
    broadcast = _model("history_state")
    missing, unexpected = gated.load_state_dict(broadcast.state_dict(), strict=False)
    assert not unexpected
    assert all(key.startswith("write_") for key in missing)
    old = torch.randn(2, 3, 3, 8)
    current = torch.randn_like(old)
    valid = torch.tensor([[True, False, True], [False, True, True]])
    with patch.object(gated, "_input_gates", return_value=torch.zeros(2, 3, 3)):
        kept, _, _ = gated._write_input(old, current, valid)
    torch.testing.assert_close(kept, old, rtol=0, atol=0)
    with patch.object(gated, "_input_gates", return_value=torch.ones(2, 3, 3)):
        opened, _, _ = gated._write_input(old, current, valid)
    standard, _, _ = broadcast._write_input(old, current, valid)
    torch.testing.assert_close(opened, standard)
    torch.testing.assert_close(opened[:, :, 1][~valid[:, 1]], old[:, :, 1][~valid[:, 1]])


@pytest.mark.parametrize("name", ["write_static", "write_content"])
def test_input_gates_have_gradients_and_report_only_observed_writes(name):
    model = _model(name).train()
    x = torch.randn(2, 3, 3, 12)
    mask = torch.tensor([[[1, 0, 1], [0, 0, 0], [1, 1, 0]], [[1, 1, 1], [0, 1, 0], [0, 0, 1]]], dtype=torch.bool)
    result = model(x, mask, return_trace=True)
    gate = result["input_gates"]
    valid = result["input_gate_valid"]
    assert gate.shape == valid.shape == (2, 3, 3, 3)
    assert (gate > 0).all() and (gate < 1).all()
    torch.testing.assert_close(valid, mask[:, :, None].expand_as(valid))
    torch.testing.assert_close(result["input_gate_sum"], (gate * valid).sum((1, 3)))
    torch.testing.assert_close(result["input_gate_squared_sum"], (gate.square() * valid).sum((1, 3)))
    torch.testing.assert_close(result["input_gate_count"], valid.sum((1, 3)).float())
    if name == "write_static":
        # Three independent gates initially 0.5: their sum is 1.5, not 1.
        torch.testing.assert_close(gate.sum(2), torch.full((2, 3, 3), 1.5))
    else:
        assert gate[valid].std() > 0
        assert not torch.allclose(gate[:, :, 0], gate[:, :, 1])
    torch.nn.functional.cross_entropy(result["logits"], torch.tensor([0, 1])).backward()
    parameters = [(key, value) for key, value in model.named_parameters() if key.startswith("write_")]
    assert all(value.grad is not None and value.grad.abs().sum() > 0 for _, value in parameters)


def test_action_only_diagnostics_omit_large_state_and_policy_traces():
    model = _model("interaction_dynamic")
    x, mask = torch.randn(2, 3, 3, 12), torch.ones(2, 3, 3, dtype=torch.bool)
    with torch.no_grad():
        small, full = model(x, mask, return_actions=True), model(x, mask, return_trace=True)
    torch.testing.assert_close(small["actions"], full["actions"])
    assert "patch_states" not in small and "input_gates" not in small and "policy_scores" not in small


@pytest.mark.parametrize("name", ["interaction_dynamic", "interaction_preplanned", "interaction_centralized"])
def test_state_only_policy_and_role_weights_receive_training_gradients(name):
    model = _model(name).train()
    result = model(torch.randn(4, 2, 3, 12), torch.ones(4, 2, 3, dtype=torch.bool))
    ce = torch.nn.functional.cross_entropy(result["logits"], torch.tensor([0, 1, 0, 1]), reduction="none")
    loss = ce.mean() + ((ce.detach() + .01 * result["updates"].detach()) * result["policy_log_prob"]).mean()
    loss.backward()
    policies = model.router_head if name == "interaction_centralized" else model.local_policies
    assert sum(p.grad.abs().sum() for p in policies.parameters() if p.grad is not None) > 0
    assert sum(p.grad.abs().sum() for p in model.roles.parameters() if p.grad is not None) > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
