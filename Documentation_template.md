# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Tensor Titans  
**Team Members:** Aman Rajput, Aditya Chaturvedi, Arpit Solanki  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

Each Source 2/3 record is linked to at most one Source 1 business, or to none. A joint
name + address inverted index, written in numba, retrieves 98.9% of the true links. A
cross-fitted LightGBM matcher with 82 pair features then scores the candidates, and a
stage-2 model adds 9 competition-context features. The largest gains came from measuring how
the data was generated: a learned Indic-script dictionary, legal-form conflicts as the
signature of "sibling" businesses, and a leave-one-country-out model for France, which
has no training labels.

---

## 2. Methodology

### 2.1 Problem Analysis

Measured on the full training labels:

- **One owner per record.** Each Source 2/3 record matches at most one Source 1 record,
  and 26% match none. So the task is: for each Source 2/3 record, pick its business or
  "none".
- **Indic scripts are a word-level dictionary.** 18% of India Source 2/3 names are in
  Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam, Odia or Gurmukhi. They
  come from a fixed map of about 1,350 words, and 97% of the mappings are one-to-one
  (प्राइवेट → private). Positional alignment of training pairs learns the map; it covers
  96–98% of Indic word occurrences in both train and test. Indic text in addresses is
  only the 16 state names.
- **Noise in true copies.** Case changes, accents, OCR digits (5ervices, c0m, 8lue),
  bracket and symbol decorations, honorifics (Sri, Smt, M/S), word drops and reorders,
  typos, website forms (`berrykochavmarlborough.com`), aliases
  (`<made-up word> dba|f/k/a|née|t/a|formerly: <real name>`), phone numbers, generic
  additions (Center, Services, Partners), legal-suffix churn, abbreviated or reordered
  address components, "null" fillers, zero-padded or prefixed house numbers, and blank
  addresses (3%).
- **Hard negatives ("siblings").** Near-copies of a Source 1 business that are different
  businesses: the same name on the same street with a *different legal form* (US Inc vs
  LLC; India Pvt Ltd vs LLP at the same office), a nearby house number, or an inserted
  category word (Traders, Stores, Holdings, (India)). Among uncertain candidate pairs, a
  US legal-form conflict is a true match only 1.6% of the time. True copies also get
  house-number noise, so the number alone is weak evidence.
- **France (test only)** follows the same sibling mechanism with French words (Holding,
  Groupe, Développement, Participations, International, Distribution, (France)) at a
  nearby house number, and French legal forms (SARL, SAS, EURL…). Its names are short
  and repetitive, so the address and legal form carry most of the signal.
- **Noise words vs sibling markers, told apart without labels.** Measured on all
  candidate pairs on one street whose names differ by one added word:
  - True-copy noise words keep the house number: US/India Center/Services/Partners in
    77–83% of pairs, 99.8% true when it agrees.
  - Sibling markers almost never do: US Holdings/Group 1%, 0% true.
  - In France, Fils / Associés / Compagnie keep it in 86–89% of pairs, exactly like
    "Services", so they are the French noise words.
  - Holding / Participations / Distribution / International keep it in 0.2–0.3%.
  - Groupe / Développement / (France) are on both lists.
  - Noise words also repeat across copies of one business, while sibling decoys never do.

### 2.2 Solution Strategy

**Approach Type:** Blocking + classifier (two-stage gradient boosting), with many-to-one
assignment.  
**Core Innovation:**
1. Retrieval from the noisy side with joint multi-channel scoring, instead of capped
   per-Source-1 blocks.
2. Features that target the generator's sibling mechanism: cross-fitted token and
   legal-form risk, and label-free house-number statistics for one-word name differences.
3. An unseen-country model trained with leave-one-country-out risks, validated by
   training on US and scoring India as if it were unlabeled.
4. A label-free diagnosis of the unseen country. France's predicted matches per business
   fit the train truth distribution thinned to a 91.5% keep rate (US 97.5%), which
   pointed to missed true links. These were traced to French noise words the model had
   never seen and fixed with a measured rule plus a second self-training round.

---

## 3. Candidate Generation (Blocking)

- **Normalization:** Indic dictionary transliteration, unidecode, OCR-digit repair,
  alias and website extraction, legal-form canonicalization, street-type and state
  canonicalization (US state names ↔ codes, Indian states, French departments →
  regions).
- **Blocking keys (three channels, one inverted index per country):**
  1. core-name word unigrams and adjacent bigrams;
  2. character 4-grams of the joined core name (catches typos, joined words, website
     domains);
  3. address tokens, numbers, and number|token combinations for the first two numbers.
- **Scoring:** IDF-weighted cosine per channel. Every Source 2/3 record keeps the top 20
  Source 1 records by the sum of the three cosines, plus each channel's own top 3.
  Postings longer than a cap are skipped. Records whose best combined score is below
  1.5 are searched again with 10× larger caps.
- **Candidate pairs generated:** 233M on train and 226M on test (22.6 per record). A
  stage-0 LightGBM re-ranker on retrieval signals only (per-channel cosines, combined
  score/rank, gaps to the record's best) keeps a pair if its probability ≥ 0.001 or it
  is in the combined top 3. France additionally keeps the top 7 plus channel extras. That
  leaves 35.7M train and 43.4M test pairs for stage 1. The re-ranker is cross-fitted
  by record halves. It keeps 1.6 pairs per record at 98.81% of true links, vs 9.6 pairs
  at 98.15% for a fixed top-7.
- **Last filtering stage:** the stage-1 matcher scores these pairs. Only pairs with
  stage-1 probability ≥ 0.0005, or that fit the French noise-word rule, go on to the
  final matchers (stage 2, French self-trained models). That is 8.3M test pairs:
  **4.8 candidates per Source 1 entity** (US 4.7, India 4.6, France 5.6), down from 25.
  No final match is lost, and the matching output is byte-identical to deciding over
  all 43.4M pairs. The stage-2 context features and the French pseudo-labels are
  computed from the stage-1 scores of all 43.4M pairs. `candidate_pairs.tsv` contains
  exactly the 8.3M pairs that reach the final decision.
- **How true matches were kept:** recall is measured on all 7.6M training links.

| blocking | true links retrieved | oracle macro F0.5 |
|---|---|---|
| per-Source-1 blocks capped at 150 (earlier baseline) | 93.7% | 0.978 |
| joint scoring | 98.40% | 0.995 |
| + adaptive second pass | 98.86% | 0.997 |

---

## 4. Matching Model

**Features used (82 in stage 1, 91 in stage 2):**
- **Name:** IDF-weighted token coverage both ways; shared IDF mass; max IDF of extra and
  missing tokens; rapidfuzz ratio / token-set / token-sort / partial ratio on the core
  name; Jaro-Winkler, Levenshtein and partial ratio on the joined name or domain; full
  name ratio; first-token agreement; share of made-up (never-seen) words.
- **Learned token risk:** for each word, the log-odds of appearing as an *extra* or
  *missing* word in false vs true pairs. It is cross-fitted across Source 1 folds, so a
  pair never sees its own label.
- **Legal form:** relation (none, one-sided, agree, conflict) and cross-fitted risk of
  the (query, Source 1) legal-form pair.
- **Address:** IDF coverage, token-set / sort / partial ratios, state agreement.
  House-number features: exact, present anywhere, digit edit distance,
  relative/absolute difference, prefix relation.
- **Retrieval and competition:** channel cosines, combined score and rank, and each
  candidate's margin and rank against the record's other candidates.
- **Label-free statistics:** house-number agreement when a word is the only name
  difference (noise words keep the address, sibling markers change it).
- **Stage 2:** from out-of-fold stage-1 probabilities — best competing probability for
  the record, how many other records already pick this business confidently, and their
  probability mass.
- **Left out of every final model:** group-consistency features (how many other records
  share a candidate's house number, name or added word) and candidate-count features.
  They are still computed, but their distribution shifts between train and test (French
  groups of 600 records vs 109 in train), so they are dropped (`ER_DROP_FEATS`).

**Model type:** LightGBM (binary, 255 leaves), 2-fold cross-fitted by Source 1 entity.
Stage 2 is trained on out-of-fold stage-1 probabilities. The France model uses the same
features, but token and legal risks are computed only from the *other* training country
plus a fallback for unseen words.  
**Ensembling:** five US/India stage-1/stage-2 chains are averaged. They differ in row
sample, bagging/feature-sampling seeds and learning rate (0.08, or 0.04 with 1,600
rounds). Out-of-fold: 0.98833 for one model, 0.98877 for the average with the
expected-F0.5 rule.  
**French legal-form conflicts:** "& Cie" is normalized to the legal token "co", which
outranks the French forms and hid SAS-vs-SARL conflicts. Pairs whose French legal forms
differ are capped at p = 0.01, as explicit French conflicts already are.  
**France:** leave-one-country-out model → self-training on confident pseudo-labels.
A noise-word rule gives p = max(p, 0.97) when all of these hold:
- same house number, same address up to an omitted component or a single typo;
- the query only adds or swaps in French noise words;
- no legal-form conflict;
- only one Source 1 record fits.

A second self-training round runs from those labels. The two rounds are combined by
maximum, except for swaps to "centre"/"service", which are category words in France.
French matches per business rise from 3.17 to 3.32 (US 3.38).  
**Threshold selection method:** each Source 2/3 record keeps its best candidate, and a
per-entity expected-F0.5 rule decides how many of each Source 1 entity's assigned
candidates to keep. It is used for US and India, where it gains +0.0001 over the global
threshold 0.7 out-of-fold. France, which has no labels to check the rule on, keeps a
threshold of 0.6 on its self-trained probabilities. On the leaderboard, removing doubtful
French links helped (the legal-form conflict cap) and adding borderline ones hurt.
Probabilities are well calibrated: predicted p matches observed precision in every
decile.

---

## 5. Results & Error Analysis

- **Leaderboard:** **0.987221**, the final submission (v14l). Earlier: 0.985 (v7, v8),
  0.987145 (v11, after the French noise-word fix), 0.987161 (v12a); 0.987221 after the
  French legal-form conflict cap and a fifth US/India model.
- **F_0.5 Score (macro), out-of-fold over all 2.2M train entities:** **0.9888** with five
  averaged US/India models and the expected-F0.5 rule (single model 0.9883)
  (v1 0.9774 → v2 0.9836 → v4 0.9863 → v6 0.9886 → v7 0.9883 after removing
  features that shift between train and test). A per-entity expected-F0.5 decision adds
  about +0.0001.
- **Unseen-country proxy (train US, score India as unlabeled):** 0.868 with naive
  handling → 0.927 with the word-risk fallback → 0.946 with leave-one-country-out
  training. Self-training on confident pseudo-labels adds a further +0.5–0.9 points on
  the proxy and is used for France.
- **Common false positives:** siblings with the same name and legal form at a nearby
  house number; blank-address records whose name exactly matches several businesses;
  made-up-name distractors at a real business's exact address.
- **Common false negatives:** true copies whose house number was perturbed and a word
  inserted; blank-address records of common names (ties); heavy typos with no address.

---

## 6. Conclusion

Measuring the generator paid off more than model capacity did: the transliteration
dictionary, legal-form conflicts, cross-fitted word risks and label-free house-number
statistics each fix a specific failure. Joint retrieval from the noisy side lifted blocking recall
from 93.7% to 98.9%. A leave-one-country-out design handled France without any French
labels.

---

## Appendix

### A. Code Artefacts

The whole pipeline is one file, `code/business_entity_resolution/src/er_pipeline.py`.
The `README.md` next to it lists every step with its exact arguments and settings. The
file's sections follow the data flow:
- parsing and normalization (`to_parquet`, `translit`, `prep`);
- blocking (`retrieve`, `run_retrieve`);
- stage-0 pruning and pair features (`stage0`, `build_features`, `add_unsup`,
  `build_loco`);
- LightGBM stage 1 and stage 2 (`train_model`, `stage2`);
- France self-training and rules (`france_selftrain`, `france_noise_fix`,
  `france_combine`, `france_mean`);
- the last filtering stage (`cand_filter`);
- assignment and decision (`predict`, `decide`).

`python src/er_pipeline.py run_all` runs everything from the TSVs.
`python src/er_pipeline.py build_final` rebuilds both output files from the saved
predictions, byte for byte.
Outputs: `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### B. Additional Results

The measurements behind each design choice are given in sections 2–5 above.

---
