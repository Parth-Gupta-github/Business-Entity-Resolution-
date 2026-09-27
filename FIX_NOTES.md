# Why the last submission scored 0.111 — and what changed

## Root cause

`matching_results.tsv` from the previous run had 1,732,544 Source-1 test
entities, but only **127,784 (7.4%)** received any predicted match — the
rest were blank. `candidate_pairs.tsv` showed blocking was working fine
(60–130 real candidates per entity), so the failure was in the classifier
stage.

`code/business_entity_resolution/src/build_submission.py` — the script
that actually produced the submitted files — trained its "production"
model via `train_and_save_ensemble()`, which called
`benchmark_models.load_benchmark_sample()`. That function was written for
quick model-architecture comparisons and reads:

- only the **first 20,000 rows** (`nrows=20000`) of `train_source2.tsv`,
  not a random sample,
- stops scanning ground truth after the **first 400** matched S1 anchors,
- adds only **150** singleton anchors.

So the real model was fit and threshold-calibrated on ~550 S1 entities
against a 40k-record target pool sliced from the start of the files —
nothing like the real ~2.2M × 10.3M distribution. The learned probabilities
and the swept threshold (0.40–0.95) were calibrated to that unrepresentative
slice, so almost every real test-set candidate scored below threshold and
fell back to "no match." That is the entire story behind 0.111.

`pipeline.py` was already built correctly (stratified random sample of
`config.SAMPLE_TRAIN_ENTITIES` S1 anchors drawn from the full pool,
blocked against the full S2+S3 target pool) — it just wasn't the script
that generated the submitted output.

## What changed

1. **`build_submission.py::train_and_save_ensemble()`** now reuses
   `pipeline.py`'s full-scale, randomly-sampled training procedure
   (`load_and_preprocess` → `create_validation_split` → `run_blocking` →
   `extract_features_with_labels`) instead of `load_benchmark_sample()`.
   This is the fix that actually matters.
2. **`postprocess.py`** (score-gap filtering + a per-entity match cap) was
   fully written but never called anywhere — it's now wired into
   `run_full_submission_generation()`. Since F₀.₅ weights precision 2×,
   dropping weak secondary candidates that just barely cross the threshold
   should help.
3. **Candidate-set size**: `postal_code_blocking` / `name_prefix_blocking`
   block-size caps were reduced (200→60, 300→80), and
   `MultiPassBlocker.generate_candidates` now trims the merged candidate
   list to `config.MAX_CANDIDATES_PER_ENTITY` (default 40), keeping the
   highest-scoring (real TF-IDF) hits first. The problem statement ranks
   smaller candidate sets higher beyond the leaderboard score, and the
   previous version unioned three passes with no final cap at all.

## What you need to do

I don't have your dataset or the compute this needs (the README says
~2.2M × 10.3M records) so I can't regenerate `matching_results.tsv` /
`candidate_pairs.tsv` myself — you'll need to rerun this on your own
machine:

```bash
# delete stale checkpoints from the broken run so training reruns properly
python code/business_entity_resolution/src/build_submission.py --team <your_team_name>
```

If `checkpoints/ensemble_model.pkl` and `checkpoints/best_threshold.txt`
from the old broken run still exist, **delete them first** — otherwise
`train_and_save_ensemble()` will just reload the bad model instead of
retraining:

```bash
rm -f checkpoints/ensemble_model.pkl checkpoints/best_threshold.txt
rm -f checkpoints/lightgbm_model.pkl checkpoints/catboost_model.pkl checkpoints/xgboost_model.pkl
```

Then validate locally before spending a submission:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## On the 0.99 target

I want to set expectations honestly: 0.99 macro F₀.₅ is not a realistic
target on this kind of noisy, real-world entity-resolution task, even with
a correctly trained model. F₀.₅ punishes any false merge hard, and with
typo'd names, partial addresses, and transliteration variants across three
sources, some genuine ambiguity is unavoidable. Fixing the training bug
above should take you from 0.111 to something dramatically better (this
was a "model never actually learned anything real" bug, not a fine-tuning
gap), but treat 0.99 as out of reach — focus on maximizing precision at a
sensible recall trade-off instead. If you have time after this fix, the
next-highest-leverage levers are: (a) tuning `POST_MAX_MATCHES` /
`POST_SCORE_GAP` in `build_submission.py` against your own validation F0.5,
and (b) increasing `config.SAMPLE_TRAIN_ENTITIES` if your machine has
headroom, since the model still only trains on a sample, not all ~2.2M
anchors.
