# Geolife vocabulary recovery approval

The original method source is `4ce2704ecb1037b5031abe9ce099e93269e2a10d9bc591981594fb091858f8e0`
(tree of user commit `bb6136d5e93f646cad9826396fe8512800e55e17`).
The correction declares class identities from the official non-test pool, keeps
the existing inner train/validation split, and never uses official test labels
to define classes. All learned feature statistics still use inner training.
Previously successful runs already contained all classes in inner training;
the corrected encoding is identical for those runs. Geolife seed 0 previously
failed before fitting because its inner training omitted a rare class.

`geolife-recovery-policy.json` is a reviewable, explicit approval for this
correction only. It records both real source hashes, the exact old text and
old/new SHA-256 of every changed hashed file. The validator substitutes the old
text into the current source tree and requires the reconstructed whole-tree
hash to equal the fixed legacy hash. Thus unrelated changes in any source or
configuration file invalidate the policy. Allowed changed files are exactly
`src/imts_agents/data.py`, `src/imts_agents/report.py`, and
`src/imts_agents/__main__.py` (the latter two implement report approval).

No approval is automatic. Without `--recovery-policy`, launchers and reports
retain strict source-version behavior. With this policy, only the two recorded
source hashes are accepted, and a legacy GeolifeSupervised seed 0 success is
explicitly forbidden. Unknown source hashes, changed environments/configs,
damaged artifacts, incompatible splits/protocols, and duplicate successful
seeds remain errors. Existing result files and their real hashes are never
rewritten. The report retains each actual source hash and includes approval
metadata. Every submitted method manifest embeds the policy and its digest;
the validation helper is frozen beside the worker with its own SHA-256.

After reviewing and deploying the exact approved tree, a recovery command is:

```bash
.venv/bin/python -u scripts/submit_method_pilot.py \
  --group ablation --all-datasets --seeds 0 1 2 --evaluate-test \
  --output-dir results/method-ablation-full \
  --time 3-00:00:00 --max-user-jobs 60 \
  --recovery-policy docs/geolife-recovery-policy.json \
  --retry-failed --submit --watch
```

This checks all 540 fit identities, reuses compatible successes (500 in the
October 16 snapshot), and retries only missing successful identities (40 in
that snapshot). It never resumes a partly trained checkpoint or changes the
training budget. Slurm walltime is increased independently of model settings.

```bash
.venv/bin/python -m imts_agents compare \
  --method-root results/method-ablation-full \
  --baseline-root "$HOME/IMTS/results/jv-first-full" \
  --recovery-policy docs/geolife-recovery-policy.json \
  --output results/accounting/recovered-comparison.csv
```

The Python interface also accepts `compare_results(..., recovery_policy=path)`.
CSV output must be a new path, never an overwrite.

For reviewers regenerating the policy after finishing these exact approved
source edits, use a new output path:

```bash
python scripts/build_recovery_policy.py --output /tmp/geolife-recovery-policy-review.json
```

The generator requires the fixed legacy Git object locally and refuses any
unlisted source/configuration changes. It does not commit, push, submit jobs,
change Git history, or rewrite experiment results. Regeneration is an explicit
review step, not an instruction to broaden the policy for future method edits.
