"""Declarative experiment registry; deliberately independent of torch.

Every control names its comparison reference. The four core experiments change
only outer recurrence and relation specialisation. Dynamic routing is a later
motivation experiment, not silently included in either core factor.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantSpec:
    name: str
    group: str
    outer_recurrent: bool
    specialized_roles: bool
    routing: str = "fixed"
    communication: bool = True
    allow_commit: bool = False
    time_features: bool = True
    homogeneous_roles: bool = False
    parameter_matched: bool = False
    reference: str | None = None
    description: str = ""
    history_access: str = "prefix"
    input_write: str = "broadcast"

    def __post_init__(self) -> None:
        if self.history_access not in {"prefix", "state"}:
            raise ValueError("history_access must be 'prefix' or 'state'")
        if self.input_write not in {"broadcast", "static", "content"}:
            raise ValueError("input_write must be 'broadcast', 'static', or 'content'")

    @property
    def train_policy(self) -> bool:
        return self.routing != "fixed"

    @property
    def role_scopes(self) -> tuple[str, ...]:
        if self.specialized_roles:
            return ("T", "S", "ST")
        if self.homogeneous_roles:
            return ("ALL", "ALL", "ALL")
        return ("ALL",)


VARIANTS: dict[str, VariantSpec] = {
    "core_none": VariantSpec(
        "core_none", "core", False, False,
        description="Reset one ALL-relation memory each patch; three updates.",
    ),
    "core_outer": VariantSpec(
        "core_outer", "core", True, False, reference="core_none",
        description="Carry one ALL-relation memory across patches; three updates.",
    ),
    "core_inner": VariantSpec(
        "core_inner", "core", False, True, reference="core_none",
        description="Reset T/S/ST memories each patch; fixed T then S then ST.",
    ),
    "core_both": VariantSpec(
        "core_both", "core", True, True, reference="core_none",
        description="Carry T/S/ST memories across patches; fixed T then S then ST.",
    ),
    "no_communication": VariantSpec(
        "no_communication", "motivation", True, True,
        communication=False, reference="core_both",
        description="Same fixed order and memory updates; zero inter-role messages.",
    ),
    "dynamic_feedback": VariantSpec(
        "dynamic_feedback", "motivation", True, True,
        routing="distributed_feedback", allow_commit=True, reference="core_both",
        description="Local policies use current memories/messages; average-logit consensus selects each next action.",
    ),
    "preplanned_path": VariantSpec(
        "preplanned_path", "motivation", True, True,
        routing="preplanned", allow_commit=True, reference="dynamic_feedback",
        description="Same policy architecture, but all action distributions use the patch-start observation, without intermediate feedback.",
    ),
    "centralized_router": VariantSpec(
        "centralized_router", "motivation", True, True,
        routing="centralized", allow_commit=True, reference="dynamic_feedback",
        description="A single recurrent controller receives all role observations and selects each next action.",
    ),
    "no_commit": VariantSpec(
        "no_commit", "motivation", True, True,
        routing="distributed_feedback", allow_commit=False,
        reference="core_both",
        description="Feedback-based role choice with exactly three updates: compare core_both for dynamic selection at equal update count, and dynamic_feedback for the effect of COMMIT.",
    ),
    "no_time": VariantSpec(
        "no_time", "motivation", True, True,
        time_features=False, reference="core_both",
        description="Fixed T/S/ST model with within-patch time and time-gap features zeroed by the data pipeline; chronological patch structure remains.",
    ),
    "homogeneous_roles": VariantSpec(
        "homogeneous_roles", "motivation", True, False,
        homogeneous_roles=True, reference="core_both",
        description="Three independently parameterised ALL-relation memories, with the same communication and fixed update budget.",
    ),
    "single_matched": VariantSpec(
        "single_matched", "motivation", True, False,
        parameter_matched=True, reference="core_both",
        description="One ALL-relation memory with added active capacity to approximately match core_both parameter count; still three updates.",
    ),
}

# These named aliases intentionally preserve old controls while giving tonight's
# experiment families explicit, independently resumable identities. All retain
# T/S/ST: their purpose is history, input writing, or interaction, not role count.
VARIANTS.update({
    "history_both": VariantSpec(
        "history_both", "history", True, True,
        description="Carry T/S/ST states and reread causal input history; fixed T/S/ST updates.",
    ),
    "history_reread": VariantSpec(
        "history_reread", "history", False, True, reference="history_both",
        description="Reset T/S/ST states each patch while retaining causal input rereads.",
    ),
    "history_state": VariantSpec(
        "history_state", "history", True, True, history_access="state", reference="history_both",
        description="Carry T/S/ST states; direct history reads use frozen previous-patch memory, never old input tokens.",
    ),
    "write_static": VariantSpec(
        "write_static", "write", True, True, history_access="state", input_write="static", reference="history_state",
        description="State-only fixed T/S/ST with one learned, input-independent sigmoid write gate per role.",
    ),
    "write_content": VariantSpec(
        "write_content", "write", True, True, history_access="state", input_write="content", reference="write_static",
        description="State-only fixed T/S/ST with independent role-aware initial-write gates using current input and old role state; later attention still sees current input.",
    ),
    "interaction_no_communication": VariantSpec(
        "interaction_no_communication", "interaction", True, True, history_access="state",
        communication=False, reference="history_state",
        description="State-only fixed T/S/ST with inter-role messages zeroed; direct current-input/state reads remain.",
    ),
    "interaction_dynamic": VariantSpec(
        "interaction_dynamic", "interaction", True, True, history_access="state",
        routing="distributed_feedback", allow_commit=True, reference="history_state",
        description="State-only local feedback policies with average-logit action consensus and optional COMMIT.",
    ),
    "interaction_preplanned": VariantSpec(
        "interaction_preplanned", "interaction", True, True, history_access="state",
        routing="preplanned", allow_commit=True, reference="interaction_dynamic",
        description="State-only policy chooses from patch-start observations without intermediate-update feedback.",
    ),
    "interaction_centralized": VariantSpec(
        "interaction_centralized", "interaction", True, True, history_access="state",
        routing="centralized", allow_commit=True, reference="interaction_dynamic",
        description="State-only recurrent centralized controller receives all role observations.",
    ),
    "interaction_no_commit": VariantSpec(
        "interaction_no_commit", "interaction", True, True, history_access="state",
        routing="distributed_feedback", allow_commit=False, reference="interaction_dynamic",
        description="State-only feedback-based role choice with exactly three updates; no COMMIT.",
    ),
})

GROUPS: dict[str, tuple[str, ...]] = {
    "core": tuple(name for name, spec in VARIANTS.items() if spec.group == "core"),
    "motivation": tuple(name for name, spec in VARIANTS.items() if spec.group == "motivation"),
}
GROUPS["pilot"] = GROUPS["core"]
GROUPS["all"] = GROUPS["core"] + GROUPS["motivation"]
GROUPS["history"] = ("history_both", "history_reread", "history_state")
GROUPS["write"] = ("history_state", "write_static", "write_content")
GROUPS["interaction"] = ("history_state",) + tuple(
    name for name, spec in VARIANTS.items() if spec.group == "interaction"
)
GROUPS["ablation"] = tuple(dict.fromkeys(
    GROUPS["history"] + GROUPS["write"] + GROUPS["interaction"]
))


def get_variant(name: str | VariantSpec) -> VariantSpec:
    if isinstance(name, VariantSpec):
        return name
    try:
        return VARIANTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown variant {name!r}; choose from {', '.join(VARIANTS)}") from exc


def variant_names(group: str = "core") -> tuple[str, ...]:
    try:
        return GROUPS[group]
    except KeyError as exc:
        raise ValueError(f"Unknown experiment group {group!r}; choose from {', '.join(GROUPS)}") from exc


def variants_for_group(group: str = "core") -> tuple[VariantSpec, ...]:
    return tuple(VARIANTS[name] for name in variant_names(group))
