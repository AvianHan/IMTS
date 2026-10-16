"""CLI for one audited fit, experiment listing, and protocol-aware summaries."""
import argparse
from dataclasses import asdict
import json
import sys


def parser():
    from .variants import GROUPS, VARIANTS
    from .runner import RunConfig
    defaults = RunConfig()
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run one fresh dataset / variant / seed fit")
    run.add_argument("--variant", choices=sorted(VARIANTS), default=defaults.variant)
    run.add_argument("--dataset", default=defaults.dataset)
    run.add_argument("--device", choices=("cpu", "cuda"), default=defaults.device)
    for field in ("seed", "epochs", "patience", "batch_size", "hidden", "n_patches", "max_inner_steps", "threads"):
        run.add_argument("--" + field.replace("_", "-"), type=int, default=getattr(defaults, field))
    for field in ("learning_rate", "weight_decay", "cost_weight", "policy_weight", "entropy_weight", "gradient_clip"):
        run.add_argument("--" + field.replace("_", "-"), type=float, default=getattr(defaults, field))
    run.add_argument("--output-dir", default="results/method-pilot")
    run.add_argument("--cache-dir")
    run.add_argument("--smoke", action="store_true", help="One diagnostic epoch; never an official result")
    run.add_argument("--evaluate-test", action="store_true", help="Explicitly evaluate official test after locking the configuration")
    listing = commands.add_parser("list", help="List variants and what each intervention changes")
    listing.add_argument("--group", choices=sorted(GROUPS), default="all")
    summary = commands.add_parser("summary", help="Keep compatible outcomes together and show failed/running attempts")
    summary.add_argument("--output-dir", default="results/method-pilot")
    evaluation = commands.add_parser("evaluate-checkpoint", help="Evaluate a locked validation checkpoint on official test once, without retraining")
    evaluation.add_argument("--run-dir", required=True)
    evaluation.add_argument("--device", choices=("cpu", "cuda"))
    evaluation.add_argument("--cache-dir")
    comparison = commands.add_parser("compare", help="Compare official-test coverage and scores against all six baselines")
    comparison.add_argument("--method-root", default="results/method-ablation-full")
    comparison.add_argument("--baseline-root", required=True, help="Baseline result directory, e.g. ../IMTS/results/jv-first-full")
    comparison.add_argument("--output", help="Write aggregate CSV with explicit missing/partial coverage")
    comparison.add_argument("--recovery-policy", help="Explicit audited Geolife source-compatibility policy")
    return root


def main(argv=None):
    args = vars(parser().parse_args(argv))
    command = args.pop("command")
    try:
        if command == "run":
            from .runner import execute
            execute(**args)
        elif command == "list":
            from .variants import variants_for_group
            print(json.dumps([asdict(spec) for spec in variants_for_group(args["group"])], indent=2))
        elif command == "summary":
            from .runner import summarize
            print(json.dumps(summarize(args["output_dir"]), indent=2))
        elif command == "compare":
            from .report import compare_results, format_coverage
            print(format_coverage(compare_results(**args)))
        else:
            from .runner import evaluate_checkpoint
            evaluate_checkpoint(**args)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
