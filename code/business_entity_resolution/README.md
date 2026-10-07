# Business entity resolution — Tensor Titans

The whole pipeline is one file, `src/er_pipeline.py`. It links every Source 2/3 record
to at most one Source 1 record (or to none) and writes the two challenge outputs,
`matching_results.tsv` and `candidate_pairs.tsv`. It uses classical ML only: string
normalization, an inverted-index retriever written in numba, and LightGBM matchers
(MIT license). It uses no external data, APIs or services.

This is the code behind the final submission **v14l** (leaderboard 0.987221). Running
`build_final` from its saved predictions reproduces both files in `output/` byte for byte:

| file | SHA-256 |
|---|---|
| `matching_results.tsv` | `3d36d526aed0444582e735b250e364047aad5f9aaedc454c291dcd58f4000afc` |
| `candidate_pairs.tsv` | `24d0fa4d6878a0320f56de287e73eff061fa8becd2be825aa57ccf52525f31ad` |

## Environment

Python 3.12. An Apple M1 with 16 GB RAM was enough. The caches need about 80 GB of
disk.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Three environment variables set the paths:

| variable | holds |
|---|---|
| `ER_DATA` | the challenge `dataset/` folder, with `train/` and `test/` TSVs |
| `ER_WORK` | caches, features, models and predictions (created as needed) |
| `ER_OUT` | where `matching_results.tsv` and `candidate_pairs.tsv` are written |

## Reproducing the submission end to end

```bash
export ER_DATA=/path/to/student_resource/dataset ER_WORK=/path/to/work ER_OUT=/path/to/output
.venv/bin/python src/er_pipeline.py run_all
```

`run_all` runs the steps below in order. Each step is `python src/er_pipeline.py <step> <args>`,
run in a fresh process with the listed settings, so every step can also be run on its own.

| # | step (args) | settings | what it does |
|---|---|---|---|
| 0 | `to_parquet` | | TSV → parquet (tab-separated, no quoting); explodes the train ground truth into pairs |
| 1 | `translit` | | learns the Indic-script → English word dictionary from aligned train pairs |
| 2 | `prep train test` | | normalizes every name and address |
| 3 | `run_retrieve train "" _a`, `run_retrieve test "" _a` | | **blocking**: one joint inverted index per country over name words, name character 4-grams and address keys; top 20 by summed cosine plus each channel's top 3; adaptive second pass with larger caps |
| 4 | `stage0 _a`; `build_features {train,test} _a _s`; `add_unsup … _s _su`; `add_group … _su _sug`; `build_loco _sug _sgloco` | `ER_CHUNK=3000000 ER_STAGE0=_a ER_T0=0.001 ER_KMIN=3 ER_K0=7` | stage-0 pruning model on retrieval signals, pair features with cross-fitted word and legal-form risk, label-free token statistics, group features (built, then left out of every model), leave-one-country-out risks for the France model |
| 5 | `train_model t1`, `t2`, `tfr`, `tfr2`; `score_v7` | `ER_MAX_TRAIN=10000000 ER_DROP_FEATS='g_*,q_ncand,s_ncand'`, `ER_FEAT=_sug` (US/India) or `_sgloco` (France), `ER_STAGE2=<stage-1 tag>` for stage 2 | 2-fold cross-fitted LightGBM, stage 1 and stage 2 (context from stage-1 scores) for US/India and the leave-one-country-out France model; test scoring |
| 6 | `france_selftrain tfr2 _sug _sgloco t2 v8fr` | `ER_DROP_FEATS` as above | France self-training round 1 on confident pseudo-labels |
| 7 | `train_model t1s{n}`, `t2s{n}`, `score_seed {n}` for n = 1, 2 | `ER_SEED=n ER_MAX_TRAIN=12000000`, `ER_DROP_FEATS` as above | two more copies of the US/India stage-1/2 chain (new row sample and LightGBM seeds) |
| 8 | `avg_preds test t2,t2s1,t2s2 t2avg`; `france_noise_fix v8fr fr_r1`; `merge_preds t2avg fr_r1 France r1all`; `france_selftrain r1all _sug _sgloco t2avg fr_r2_raw`; `france_noise_fix fr_r2_raw fr_r2`; `france_combine fr_r1 fr_r2 fr_final` | `ER_DROP_FEATS` as above | French noise-word rule, self-training round 2 from it, rule again, combination of the two rounds |
| 9 | as step 7 for n = 3, 4 | also `ER_LR=0.04 ER_ROUNDS=1600` | two more US/India copies at a lower learning rate |
| 10 | `build_final`: `avg_preds test t2,t2s1,t2s2,t2s3,t2s4 t2avg5`; `france_mean fr_final fr_final t2avg5 v14l max`; `cand_filter v14l 0.0005`; `predict v14l 0.7 France=0.6` | `ER_CAND_KEEP=$ER_WORK/cand_keep.npy ER_EXPF=US,India` for `predict` | 5-model US/India mean; French legal-form conflict cap; last filtering stage; many-to-one assignment, per-entity expected-F0.5 rule for US/India, threshold 0.6 for France; writes both files |

From the saved predictions in `ER_WORK`, `python src/er_pipeline.py build_final` runs only
step 10 and rebuilds both output files in under a minute.

Validate the output with the challenge's checker, from `student_resource/`:

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

## Candidate set and the last filtering stage

Stage 1 scores all 43.4M stage-0 test pairs. Its scores feed the stage-2 context
features and the French pseudo-labels. `cand_filter` then keeps a pair for the final
decision only if its stage-1 probability is ≥ 0.0005 or it fits the French noise-word
rule. That leaves 8.3M pairs, 4.8 per Source 1 entity instead of 25.
`candidate_pairs.tsv` is exactly this set. No final match falls outside it, and the
matches are byte-identical to deciding over all 43.4M pairs.

## Layout of `src/er_pipeline.py`

The file is organised in the order the data flows. Each section is one stage and starts
with a banner comment. The header docstring maps them:

| section | role |
|---|---|
| `config` | paths from `ER_DATA`, `ER_WORK`, `ER_OUT` |
| `textnorm`, `translit`, `to_parquet`, `prep` | parsing and name/address normalization (transliteration, accents, OCR digits, aliases, websites, legal forms, street types, states) |
| `retrieve`, `run_retrieve` | numba multi-channel inverted index and search (blocking) |
| `labels`, `metric` | train labels; many-to-one assignment and macro F0.5 |
| `tokrisk`, `features`, `stage0`, `build_features` | pair features, per-query context, cross-fitted token/legal-form risk, stage-0 pruning |
| `unsup`, `add_unsup`, `groupfeat`, `add_group`, `build_loco` | label-free token statistics, group features (not used by the models), leave-one-country-out risks |
| `stage2`, `train_model` | stage-2 context features; cross-fitted LightGBM training |
| `decide`, `predict`, `merge_preds`, `score_v7` | expected-F0.5 decision, scoring, merging the country models, writing the outputs |
| `france_noise_fix`, `france_selftrain`, `france_combine`, `france_mean` | France (no labels): noise-word rule, self-training, combination of the two rounds, French legal-form conflict cap |
| `cand_filter`, `avg_preds` | last filtering stage (defines `candidate_pairs.tsv`); averaging of the five US/India models |
| orchestration | `run_all`, `build_final`, and the command-line entry point |
