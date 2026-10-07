"""Business entity resolution (ML Challenge 2026), team Tensor Titans: the complete
pipeline that produced the final submission v14l (leaderboard 0.987221) in one file.

Links every Source 2/3 record to at most one Source 1 record (or none) and writes
output/matching_results.tsv and output/candidate_pairs.tsv. Classical ML only: string
normalization, an inverted-index retriever written in numba, and LightGBM matchers
(MIT license). No external data, APIs or services.

The file is organised in the order the data flows. Each section is one stage of the
pipeline and is marked by a banner comment:

  config                paths (ER_DATA, ER_WORK, ER_OUT environment variables)
  textnorm, translit,   TSV -> parquet, Indic-script word dictionary learned from train
  to_parquet, prep      pairs, name/address normalization
  retrieve,             blocking: joint name-word / name-character / address inverted
  run_retrieve          index per country with an adaptive second pass
  labels, metric        ground-truth rows, many-to-one assignment, macro F0.5
  tokrisk, features,    pair features, cross-fitted word and legal-form risk, stage-0
  stage0,               candidate pruning model, label-free token statistics,
  build_features,       group features (built; dropped from the models via
  unsup, add_unsup,     ER_DROP_FEATS), leave-one-country-out risks for France
  groupfeat, add_group,
  build_loco
  stage2, train_model   2-fold cross-fitted LightGBM, stage-2 context features
  decide, predict       per-entity expected-F0.5 rule, assignment, output writing
  merge_preds, score_v7 test scoring of the stage-1/2 and France models
  france_noise_fix,     France (no labels): noise-word rule, self-training on
  france_selftrain,     pseudo-labels, round combination, French legal-form
  france_combine,       conflict cap
  france_mean
  cand_filter           last filtering stage = candidate_pairs.tsv
  avg_preds             5-model US/India average
  orchestration         the exact sequence of steps and settings behind v14l

Usage (Python 3.12, packages pinned in requirements.txt):

  ER_DATA=<dataset dir with train/ and test/> ER_WORK=<cache dir> ER_OUT=<output dir> \
      python er_pipeline.py run_all        # data -> blocking -> matching -> output (hours)
  python er_pipeline.py build_final        # final stage from the saved predictions in
                                           # ER_WORK (minutes): writes the two files
  python er_pipeline.py <step> [args ...]  # one step, e.g. `predict v14l 0.7 France=0.6`

Each step runs in a fresh Python process with its own ER_* settings, exactly as the
original per-stage scripts did; the settings are listed in run_all() below.
"""

from collections import Counter, defaultdict
from multiprocessing import Pool
import gc
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
import unicodedata

from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from unidecode import unidecode
import lightgbm as lgb
import numba as nb
import numpy as np
import polars as pl


# ========================================================================================
# config.py
# ========================================================================================

DATA = os.environ.get('ER_DATA', '/Users/arpit/projects/ML/student_resource/dataset')
WORK = os.environ.get('ER_WORK', '/Users/arpit/projects/ML-claude/work')
OUT = os.environ.get('ER_OUT', '/Users/arpit/projects/ML-claude/student_resource/output')


# ========================================================================================
# textnorm.py
#
# String normalization for business names and addresses.
#
# Everything here is a pure function of one record, so it runs in worker
# processes. Indic-script words are mapped back to English through a
# dictionary learned from the training pairs (see translit.py); leftovers go
# through unidecode.
# ========================================================================================

textnorm_INDIC_RE = re.compile(r'[ऀ-෿]')
NONASCII_RE = re.compile(r'[^\x00-\x7f]')

# ---------------------------------------------------------------- names

# "<made-up name> <marker> <real name>": keep the part after the marker.
ALIAS_RE = re.compile(
    r'\s(?:d\.?b\.?a\.?:?|d/b/a|doing business as|formerly:?|formerly known as|'
    r'f/k/a|f\.k\.a\.?|fka|a/k/a|a\.k\.a\.?|aka|also known as|known as|'
    r'trading as|t/a|n[e]e)\s', re.I)
WEB_RE = re.compile(r'(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|co\.in|in|net|org|co|biz|info|fr|us)\b')
PHONE_RE = re.compile(r'\+?\d[\d\- ]{8,}\d')
ORDINAL_RE = re.compile(r'^\d+(?:st|nd|rd|th)$')

LEET = str.maketrans({'0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '6': 'g', '7': 't', '8': 'b', '9': 'g'})

LEGAL_CANON = {
    'incorporated': 'inc', 'lnc': 'inc', 'inc': 'inc',
    'corporation': 'corp', 'corp': 'corp',
    'company': 'co', 'co': 'co', 'cie': 'co',
    'limited': 'ltd', 'ltd': 'ltd', 'ltda': 'ltd',
    'private': 'pvt', 'pvt': 'pvt',
    'llc': 'llc', 'llp': 'llp', 'lp': 'lp', 'pc': 'pc', 'pllc': 'pllc', 'plc': 'plc', 'pa': 'pa',
    'sarl': 'sarl', 'sas': 'sas', 'sasu': 'sasu', 'eurl': 'eurl', 'sci': 'sci', 'sa': 'sa', 'snc': 'snc',
    'selarl': 'selarl', 'scop': 'scop', 'scp': 'scp',
    'etablissements': 'ets', 'ets': 'ets', 'centre': 'center', 'center': 'center',
    '&': 'and', 'and': 'and', 'et': 'and',
}
LEGAL_TOKENS = {'inc', 'corp', 'co', 'ltd', 'pvt', 'llc', 'llp', 'lp', 'pc', 'pllc', 'plc', 'pa',
                'sarl', 'sas', 'sasu', 'eurl', 'sci', 'sa', 'snc', 'selarl', 'scop', 'scp'}
HONORIFIC = {'dr', 'mr', 'mrs', 'ms', 'smt', 'sri', 'shri', 'shree', 'sree', 'm/s', 'messrs'}
STOP = {'the', 'a', 'an', 'of', 'and', 'de', 'des', 'du', 'la', 'le', 'les', 'l', 'd', 'en', 'et'}


def strip_accents(s):
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    if NONASCII_RE.search(s):
        s = unidecode(s)
    return s


def map_indic(s, indic_map):
    if not textnorm_INDIC_RE.search(s):
        return s
    out = []
    for w in s.split():
        out.append(indic_map.get(w, w))
    return ' '.join(out)


def _tokens(s):
    # drop dots inside acronyms (l.l.c. -> llc) before splitting
    s = re.sub(r'(?<=\b[a-z])\.(?=[a-z]\b)', '', s)
    s = s.replace('.', ' ')
    s = s.replace('&', ' & ').replace('+', ' & ')
    toks = re.findall(r'[a-z0-9]+|&', s)
    out = []
    for t in toks:
        if t != '&' and not t.isalpha() and not t.isdigit() and not ORDINAL_RE.match(t):
            t = t.translate(LEET)  # 5ervices -> services, c0m -> com
        out.append(LEGAL_CANON.get(t, t))
    return out


def norm_name(raw, indic_map):
    """Return (tokens, core_tokens, web_string, flags)."""
    has_indic = bool(textnorm_INDIC_RE.search(raw))
    s = map_indic(raw, indic_map)
    s = strip_accents(s).lower()
    flags = 0
    if has_indic:
        flags |= 1
    parts = [p.strip() for p in s.split('|') if p.strip()]
    main = parts[0] if parts else ''
    m = ALIAS_RE.search(' ' + main + ' ')
    if m and m.start() > 0:
        main = (' ' + main + ' ')[m.end():].strip()
        flags |= 2
    web = ''
    for p in [main] + parts[1:]:
        w = WEB_RE.search(p)
        if w:
            web = w.group(1).replace('-', '')
            if p is main:
                flags |= 4
                main = (main[:w.start()] + ' ' + main[w.end():]).strip()
            break
    main = PHONE_RE.sub(' ', main)
    main = re.sub(r'\bm/s\b', ' ', main)
    toks = _tokens(main)
    if not toks and web:
        toks = [web]
    core = [t for t in toks if t not in LEGAL_TOKENS and t not in HONORIFIC and t not in STOP and t != '&']
    if len(parts) > 1:
        flags |= 8
    return toks, core, web, flags


# ------------------------------------------------------------- addresses

INDIC_STATES = {
    'महाराष्ट्र': 'maharashtra', 'दिल्ली': 'delhi', 'उत्तर प्रदेश': 'uttar pradesh', 'ಕರ್ನಾಟಕ': 'karnataka',
    'தமிழ்நாடு': 'tamil nadu', 'পশ্চিমবঙ্গ': 'west bengal', 'ગુજરાત': 'gujarat', 'తెలంగాణ': 'telangana',
    'हरियाणा': 'haryana', 'राजस्थान': 'rajasthan', 'കേരളം': 'kerala', 'बिहार': 'bihar',
    'मध्य प्रदेश': 'madhya pradesh', 'ఆంధ్రప్రదేశ్': 'andhra pradesh', 'ਪੰਜਾਬ': 'punjab', 'ଓଡ଼ିଶା': 'odisha',
}

US_STATES = {
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca', 'colorado': 'co',
    'connecticut': 'ct', 'delaware': 'de', 'florida': 'fl', 'georgia': 'ga', 'hawaii': 'hi', 'idaho': 'id',
    'illinois': 'il', 'indiana': 'in', 'iowa': 'ia', 'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la',
    'maine': 'me', 'maryland': 'md', 'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn',
    'mississippi': 'ms', 'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv',
    'new hampshire': 'nh', 'new jersey': 'nj', 'new mexico': 'nm', 'new york': 'ny', 'north carolina': 'nc',
    'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok', 'oregon': 'or', 'pennsylvania': 'pa',
    'rhode island': 'ri', 'south carolina': 'sc', 'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx',
    'utah': 'ut', 'vermont': 'vt', 'virginia': 'va', 'washington': 'wa', 'west virginia': 'wv',
    'wisconsin': 'wi', 'wyoming': 'wy', 'district of columbia': 'dc', 'puerto rico': 'pr',
}
IN_STATES = {
    'maharashtra': 'mh', 'delhi': 'dl', 'new delhi': None, 'uttar pradesh': 'up', 'karnataka': 'ka',
    'tamil nadu': 'tn', 'west bengal': 'wb', 'gujarat': 'gj', 'telangana': 'tg', 'haryana': 'hr',
    'rajasthan': 'rj', 'kerala': 'kl', 'bihar': 'br', 'madhya pradesh': 'mp', 'andhra pradesh': 'ap',
    'punjab': 'pb', 'odisha': 'od', 'orissa': 'od', 'goa': 'ga', 'assam': 'as', 'jharkhand': 'jh',
    'chhattisgarh': 'cg', 'uttarakhand': 'uk', 'himachal pradesh': 'hp', 'jammu and kashmir': 'jk',
    'chandigarh': 'ch', 'puducherry': 'py', 'pondicherry': 'py',
}
IN_CODES = {'mh', 'dl', 'up', 'ka', 'tn', 'wb', 'gj', 'tg', 'ts', 'hr', 'rj', 'kl', 'br', 'mp', 'ap', 'pb',
            'od', 'or', 'ga', 'as', 'jh', 'cg', 'ct', 'uk', 'hp', 'jk', 'ch', 'py'}
FR_REGIONS = {
    'hauts de france': 'hdf', 'nord': 'hdf', 'pas de calais': 'hdf',
    'nouvelle aquitaine': 'naq', 'gironde': 'naq',
    'pays de la loire': 'pdl', 'loire atlantique': 'pdl',
}

STREET_CANON = {
    'street': 'st', 'str': 'st', 'st': 'st', 'road': 'rd', 'rd': 'rd', 'roda': 'rd', 'avenue': 'ave', 'ave': 'ave',
    'av': 'ave', 'avn': 'ave', 'drive': 'dr', 'dr': 'dr', 'lane': 'ln', 'ln': 'ln', 'court': 'ct', 'ct': 'ct',
    'boulevard': 'blvd', 'blvd': 'blvd', 'bd': 'blvd', 'bvd': 'blvd', 'boul': 'blvd', 'place': 'pl', 'pl': 'pl',
    'circle': 'cir', 'cir': 'cir', 'highway': 'hwy', 'hwy': 'hwy', 'parkway': 'pkwy', 'pkwy': 'pkwy',
    'trail': 'trl', 'trl': 'trl', 'terrace': 'ter', 'ter': 'ter', 'trace': 'trce', 'trce': 'trce',
    'way': 'way', 'wy': 'way', 'square': 'sq', 'sq': 'sq', 'point': 'pt', 'pt': 'pt', 'crossing': 'xing',
    'xing': 'xing', 'north': 'n', 'south': 's', 'east': 'e', 'west': 'w', 'n': 'n', 's': 's', 'e': 'e', 'w': 'w',
    'northeast': 'ne', 'northwest': 'nw', 'southeast': 'se', 'southwest': 'sw',
    'rue': 'rue', 'r': 'rue', 'allee': 'all', 'all': 'all', 'chemin': 'ch', 'ch': 'ch', 'che': 'ch',
    'impasse': 'imp', 'imp': 'imp', 'route': 'rte', 'rte': 'rte', 'quai': 'qu', 'qu': 'qu', 'cours': 'crs',
    'crs': 'crs', 'faubourg': 'fg', 'fbg': 'fg', 'fg': 'fg', 'sentier': 'sen', 'sente': 'sen',
    'apartment': 'apt', 'apt': 'apt', 'appt': 'apt', 'appartement': 'apt', 'suite': 'ste', 'ste': 'ste',
    'unit': 'unit', 'floor': 'fl', 'flr': 'fl', 'fl': 'fl', 'building': 'bldg', 'bldg': 'bldg',
    'saint': 'st', 'sainte': 'ste', 'mount': 'mt', 'mt': 'mt', 'fort': 'ft', 'ft': 'ft',
    'nagar': 'nagar', 'ngr': 'nagar', 'marg': 'marg', 'mg': 'marg', 'sector': 'sec', 'sec': 'sec',
    'opp': 'opp', 'opposite': 'opp', 'nr': 'near', 'near': 'near', 'behind': 'behind', 'bh': 'behind',
    'post': 'po', 'po': 'po', 'dist': 'dist', 'district': 'dist', 'tq': 'taluk', 'taluk': 'taluk',
    'taluka': 'taluk', 'tal': 'taluk',
}
# tokens that carry no address identity
ADDR_NOISE = {'null', 'none', 'na', 'no', 'nos', 'number', 'num', 'door', 'plot', 'flat', 'house',
              'hno', 'pmb', 'box', 'cdp', 'city', 'town', 'village', 'township', 'of', 'the', 'and'}
ADDR_NOISE_FR = ADDR_NOISE | {'de', 'du', 'des', 'la', 'le', 'les', 'l', 'd', 'et', 'au', 'aux'}
NUM_RE = re.compile(r'\d+')


def _addr_phrase_map(s, table):
    for k, v in table.items():
        if k in s:
            s = re.sub(r'\b' + k + r'\b', ' ' + v + ' ', s)
    return s


def norm_addr(raw, country):
    """Return (tokens, numbers, state, flags)."""
    s = raw
    if textnorm_INDIC_RE.search(s):
        for k, v in INDIC_STATES.items():
            if k in s:
                s = s.replace(k, v)
    s = re.sub(r'[nN]\s*[°º]', ' no ', s).replace('°', ' ').replace('º', ' ')
    s = strip_accents(s).lower()
    s = re.sub(r'\b(?:null|n/a|none)\b', ' ', s)
    s = re.sub(r'\b(\d+)\s*(?:st|nd|rd|th)\b', r'\1', s)  # 13th -> 13, 1st -> 1
    comps = [c.strip() for c in s.split(',')]
    comps = [c for c in comps if c and c not in ('null', 'na', 'n/a')]
    s2 = ' , '.join(comps)
    s2 = re.sub(r"[\-'/\.]", ' ', s2)
    state = ''
    if country == 'US':
        for k, v in US_STATES.items():
            if k in s2:
                s2, n = re.subn(r'\b' + k + r'\b', ' ' + v + ' ', s2)
    elif country == 'India':
        for k, v in IN_STATES.items():
            if v and k in s2:
                s2 = re.sub(r'\b' + k + r'\b', ' ' + v + ' ', s2)
    elif country == 'France':
        for k, v in FR_REGIONS.items():
            if k in s2:
                s2 = re.sub(r'\b' + k + r'\b', ' ' + v + ' ', s2)
    toks_raw = re.findall(r'[a-z]+|\d+', s2)
    noise = ADDR_NOISE_FR if country == 'France' else ADDR_NOISE
    toks = []
    nums = []
    for t in toks_raw:
        if t.isdigit():
            n = t.lstrip('0') or '0'
            nums.append(n)
            continue
        t = STREET_CANON.get(t, t)
        if t in noise:
            continue
        toks.append(t)
    # state guess: last state-looking token
    if country == 'US':
        for t in reversed(toks):
            if len(t) == 2 and t in US_CODES:
                state = t
                break
    elif country == 'India':
        for t in reversed(toks):
            if t in IN_CODES_ALL:
                state = IN_ALIAS.get(t, t)
                break
    elif country == 'France':
        for t in reversed(toks):
            if t in ('hdf', 'naq', 'pdl'):
                state = t
                break
    return toks, nums, state


US_CODES = set(US_STATES.values())
IN_ALIAS = {'ts': 'tg', 'or': 'od', 'ct': 'cg'}
IN_CODES_ALL = IN_CODES


# ========================================================================================
# translit.py
#
# Learn the Indic-script -> English word map from the training pairs.
#
# In training, a Source 2/3 name written in an Indic script has the same word
# count as its Source 1 name in 99.99% of pairs, and each Indic word maps to
# one English word 97% of the time (the rest are spelling variants such as
# laxmi/lakshmi). So a positional word alignment recovers the dictionary.
# ========================================================================================

translit_INDIC_RE = re.compile(r'[ऀ-෿]')


def _latin(w):
    w = unicodedata.normalize('NFKD', w.lower())
    w = ''.join(c for c in w if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9&]', '', w)


def translit_learn():
    s1 = pl.read_parquet(f'{WORK}/train_s1.parquet').filter(pl.col('country') == 'India')
    sx = pl.concat([pl.read_parquet(f'{WORK}/train_s2.parquet'),
                    pl.read_parquet(f'{WORK}/train_s3.parquet')]).filter(pl.col('country') == 'India')
    sx = sx.filter(pl.col('business_name').str.contains(r'[ऀ-෿]'))
    pairs = pl.read_parquet(f'{WORK}/train_pairs.parquet')
    j = (pairs.join(sx.select('entity_id', 'business_name'), left_on='sx', right_on='entity_id')
         .join(s1.select('entity_id', pl.col('business_name').alias('n1')), left_on='s1', right_on='entity_id'))
    mp = defaultdict(Counter)
    for a, b in zip(j['n1'].to_list(), j['business_name'].to_list()):
        wa, wb = a.split(), b.split()
        if len(wa) != len(wb):
            continue
        for x, y in zip(wa, wb):
            if translit_INDIC_RE.search(y):
                lx = _latin(x)
                if lx:
                    mp[y][lx] += 1
    word_map = {}
    variants = defaultdict(Counter)
    for y, c in mp.items():
        tot = sum(c.values())
        top, k = c.most_common(1)[0]
        word_map[y] = top
        for x, kk in c.items():
            if kk >= 0.1 * tot:
                variants[top][x] += kk
    # English spelling variants that share one Indic spelling (laxmi/lakshmi):
    # union them and map every member to the alphabetically first spelling.
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    skip = {'limited', 'ltd', 'private', 'pvt', 'al', 'all'}
    for top, c in variants.items():
        for x in c:
            if x != top and x not in skip and top not in skip:
                ra, rb = find(x), find(top)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    canon = {a: find(a) for a in list(parent) if find(a) != a}
    return word_map, canon


def translit_cli():
    """`python translit.py ...` of the original pipeline (reads sys.argv)."""
    wm, canon = translit_learn()
    json.dump({'word_map': wm, 'canon': canon}, open(f'{WORK}/indic_map.json', 'w'), ensure_ascii=False, indent=0)
    print('indic words', len(wm), 'english variant canon', len(canon))
    print(sorted(canon.items())[:80])


# ========================================================================================
# to_parquet.py
#
# Step 0: read the challenge TSVs (tab-separated, no quoting) into parquet
# caches, and explode the train ground truth into (s1, sx) pairs.
# ========================================================================================

def to_parquet_main():
    for split in ('train', 'test'):
        for s in (1, 2, 3):
            t = time.time()
            df = pl.read_csv(f'{DATA}/{split}/{split}_source{s}.tsv', separator='\t', quote_char=None,
                             infer_schema=False, encoding='utf8')
            df = df.with_columns([pl.col(c).fill_null('') for c in df.columns])
            df.write_parquet(f'{WORK}/{split}_s{s}.parquet')
            print(split, s, df.shape, round(time.time() - t, 1), flush=True)
    gt = pl.read_csv(f'{DATA}/train/train_ground_truth.tsv', separator='\t', quote_char=None,
                     infer_schema=False).with_columns(pl.col('matched_entity_ids').fill_null(''))
    gt.write_parquet(f'{WORK}/train_gt.parquet')
    pairs = (gt.with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids')
             .filter(pl.col('matched_entity_ids') != '')
             .rename({'source1_entity_id': 's1', 'matched_entity_ids': 'sx'}))
    pairs.write_parquet(f'{WORK}/train_pairs.parquet')
    print('pairs', pairs.shape)


def to_parquet_cli():
    """`python to_parquet.py ...` of the original pipeline (reads sys.argv)."""
    to_parquet_main()


# ========================================================================================
# prep.py
#
# Normalize every record of a split and cache the result as parquet.
#
# Output per split: WORK/norm_{split}.parquet with one row per record of all
# three sources:
#   id, src (1/2/3), country, n_toks, n_core, n_web, n_flags, a_toks, a_nums, a_state
# ========================================================================================

_MAP = None
_CANON = None


def _init():
    global _MAP, _CANON
    d = json.load(open(f'{WORK}/indic_map.json'))
    _MAP = d['word_map']
    _CANON = d['canon']


def _work(rows):
    out = []
    for eid, name, addr, country in rows:
        toks, core, web, flags = norm_name(name, _MAP)
        if _CANON:
            toks = [_CANON.get(t, t) for t in toks]
            core = [_CANON.get(t, t) for t in core]
        at, an, st = norm_addr(addr, country)
        out.append((eid, toks, core, web, flags, at, an, st))
    return out


def prep_run(split, nproc=8):
    t0 = time.time()
    frames = []
    for s in (1, 2, 3):
        df = pl.read_parquet(f'{WORK}/{split}_s{s}.parquet')
        frames.append(df.with_columns(pl.lit(s, dtype=pl.Int8).alias('src')))
    df = pl.concat(frames)
    rows = list(zip(df['entity_id'].to_list(), df['business_name'].to_list(),
                    df['business_address'].to_list(), df['country'].to_list()))
    chunks = [rows[i:i + 20000] for i in range(0, len(rows), 20000)]
    res = []
    with Pool(nproc, initializer=_init) as p:
        for k, r in enumerate(p.imap(_work, chunks, chunksize=1)):
            res.extend(r)
            if k % 100 == 0:
                print(f'{split}: {len(res)}/{len(rows)} {time.time() - t0:.0f}s', flush=True)
    cols = list(zip(*res))
    out = pl.DataFrame({
        'id': cols[0],
        'n_toks': cols[1], 'n_core': cols[2], 'n_web': cols[3], 'n_flags': cols[4],
        'a_toks': cols[5], 'a_nums': cols[6], 'a_state': cols[7],
    }, schema={'id': pl.Utf8, 'n_toks': pl.List(pl.Utf8), 'n_core': pl.List(pl.Utf8), 'n_web': pl.Utf8,
               'n_flags': pl.Int16, 'a_toks': pl.List(pl.Utf8), 'a_nums': pl.List(pl.Utf8), 'a_state': pl.Utf8})
    out = pl.concat([df.select(pl.col('entity_id').alias('id2'), 'src', 'country', 'business_name',
                               'business_address'), out], how='horizontal')
    assert (out['id'] == out['id2']).all()
    out = out.drop('id2')
    out.write_parquet(f'{WORK}/norm_{split}.parquet')
    print(f'{split}: done {out.height} rows in {time.time() - t0:.0f}s', flush=True)


def prep_cli():
    """`python prep.py ...` of the original pipeline (reads sys.argv)."""
    for split in sys.argv[1:]:
        prep_run(split)


# ========================================================================================
# retrieve.py
#
# Candidate retrieval: for every Source 2/3 record, the top Source 1
# records inside one country.
#
# Channels are sparse bags of hashed features with IDF weights (Source 1 is
# the index side). One joint pass per query accumulates each channel's dot
# product; the query keeps the top-K by the weighted sum of per-channel
# cosines plus each channel's own top-KC. Postings longer than a channel's
# cap are not traversed (those features still count in the query norm).
# ========================================================================================

@nb.njit(parallel=True, cache=True)
def _topk_multi(q_ptr, q_feat, q_norm, fid_ch, idf, p_ptr, p_idx, s_norm, wts, n_index, k, kc):
    C = q_norm.shape[1]
    nq = q_ptr.shape[0] - 1
    M = k + C * kc
    out_idx = np.full((nq, M), -1, dtype=np.int32)
    out_sc = np.zeros((nq, M, C), dtype=np.float32)
    out_comb = np.zeros((nq, M), dtype=np.float32)
    nblocks = 64
    bs = (nq + nblocks - 1) // nblocks
    for b in nb.prange(nblocks):
        acc = np.zeros(n_index * C, dtype=np.float32)
        seen = np.zeros(n_index, dtype=np.uint8)
        touched = np.empty(n_index, dtype=np.int32)
        top_i = np.empty(k, dtype=np.int32)
        top_s = np.empty(k, dtype=np.float32)
        ch_i = np.empty((C, kc), dtype=np.int32)
        ch_s = np.empty((C, kc), dtype=np.float32)
        ckk = np.zeros(C, dtype=np.int32)
        cs = np.zeros(C, dtype=np.float32)
        lo = b * bs
        hi = min(nq, lo + bs)
        for i in range(lo, hi):
            nt = 0
            for c in range(C):
                ckk[c] = 0
            for a in range(q_ptr[i], q_ptr[i + 1]):
                f = q_feat[a]
                c = fid_ch[f]
                w = idf[f]
                w2 = w * w
                for t in range(p_ptr[f], p_ptr[f + 1]):
                    j = p_idx[t]
                    if seen[j] == 0:
                        seen[j] = 1
                        touched[nt] = j
                        nt += 1
                    acc[j * C + c] += w2
            kk = 0
            for t in range(nt):
                j = touched[t]
                comb = 0.0
                for c in range(C):
                    v = acc[j * C + c]
                    x = v / (q_norm[i, c] * s_norm[j * C + c]) if v > 0 else 0.0
                    cs[c] = x
                    comb += wts[c] * x
                    if x > 0:
                        if ckk[c] < kc:
                            pos = ckk[c]
                            ckk[c] += 1
                        elif x > ch_s[c, kc - 1]:
                            pos = kc - 1
                        else:
                            pos = -1
                        if pos >= 0:
                            while pos > 0 and ch_s[c, pos - 1] < x:
                                ch_s[c, pos] = ch_s[c, pos - 1]
                                ch_i[c, pos] = ch_i[c, pos - 1]
                                pos -= 1
                            ch_s[c, pos] = x
                            ch_i[c, pos] = j
                if kk < k:
                    pos = kk
                    kk += 1
                elif comb > top_s[k - 1]:
                    pos = k - 1
                else:
                    continue
                while pos > 0 and top_s[pos - 1] < comb:
                    top_s[pos] = top_s[pos - 1]
                    top_i[pos] = top_i[pos - 1]
                    pos -= 1
                top_s[pos] = comb
                top_i[pos] = j
            n = 0
            for t in range(kk):
                out_idx[i, n] = top_i[t]
                n += 1
            for c in range(C):
                for t in range(ckk[c]):
                    j = ch_i[c, t]
                    dup = False
                    for u in range(n):
                        if out_idx[i, u] == j:
                            dup = True
                            break
                    if not dup:
                        out_idx[i, n] = j
                        n += 1
            for u in range(n):
                j = out_idx[i, u]
                comb = 0.0
                for c in range(C):
                    v = acc[j * C + c]
                    x = v / (q_norm[i, c] * s_norm[j * C + c]) if v > 0 else 0.0
                    out_sc[i, u, c] = x
                    comb += wts[c] * x
                out_comb[i, u] = comb
            for t in range(nt):
                j = touched[t]
                seen[j] = 0
                for c in range(C):
                    acc[j * C + c] = 0.0
    return out_idx, out_sc, out_comb


class MultiIndex:
    """Packed inverted index over Source 1 for C channels."""

    def __init__(self, s1_frames, caps, n1):
        self.C = len(s1_frames)
        self.n1 = n1
        self.max_idf = float(np.log((n1 + 1) / 0.5))
        vocabs, counts, pidx = [], [], []
        s_norm = np.full((n1, self.C), 1e9, dtype=np.float32)
        off = 0
        for c, e1 in enumerate(s1_frames):
            e1 = e1.unique(['r', 'f'])
            v = e1.group_by('f').agg(pl.len().alias('df')).with_row_index('fid')
            v = v.with_columns((pl.col('fid').cast(pl.Int32) + off).alias('fid'),
                               ((n1 + 1) / (pl.col('df') + 0.5)).log().cast(pl.Float32).alias('idf'),
                               pl.lit(c, dtype=pl.Int8).alias('ch'))
            e1 = e1.join(v.select('f', 'fid', 'idf'), on='f').sort(['fid', 'r'])
            counts.append(np.bincount(e1['fid'].to_numpy() - off, minlength=v.height))
            pidx.append(e1['r'].to_numpy().astype(np.int32))
            sn = e1.group_by('r').agg((pl.col('idf') ** 2).sum().sqrt().alias('n'))
            s_norm[sn['r'].to_numpy(), c] = sn['n'].to_numpy()
            vocabs.append(v)
            off += v.height
            del e1, sn
        self.vocab = pl.concat(vocabs)
        self.caps = list(caps)
        cnt = np.concatenate(counts)
        self.p_ptr = np.zeros(len(cnt) + 1, dtype=np.int64)
        np.cumsum(cnt, out=self.p_ptr[1:])
        self.p_idx = np.concatenate(pidx)
        self.s_norm = s_norm.ravel()
        self.fid_ch = self.vocab.sort('fid')['ch'].to_numpy().astype(np.int8)
        self.idf = self.vocab.sort('fid')['idf'].to_numpy().astype(np.float32)

    def query_arrays(self, q_frames, nq, caps=None):
        """q_frames: per channel exploded (r, f) frames for a batch of queries.
        Features whose Source 1 document frequency exceeds the channel cap are
        not traversed (they still count in the query norm)."""
        caps = caps or self.caps
        q_norm = np.zeros((nq, self.C), dtype=np.float32)
        ents = []
        for c, eq in enumerate(q_frames):
            eq = eq.unique(['r', 'f']).join(
                self.vocab.filter(pl.col('ch') == c).select('f', 'fid', 'idf', 'df'), on='f', how='left')
            eq = eq.with_columns(pl.col('idf').fill_null(self.max_idf))
            qn = eq.group_by('r').agg((pl.col('idf') ** 2).sum().sqrt().alias('n'))
            q_norm[qn['r'].to_numpy(), c] = qn['n'].to_numpy()
            ents.append(eq.filter(pl.col('df').is_not_null() & (pl.col('df') <= caps[c])).select('r', 'fid'))
        q_norm[q_norm == 0] = 1.0
        e = pl.concat(ents).sort('r')
        qc = np.bincount(e['r'].to_numpy(), minlength=nq)
        q_ptr = np.zeros(nq + 1, dtype=np.int64)
        np.cumsum(qc, out=q_ptr[1:])
        return q_ptr, e['fid'].to_numpy().astype(np.int32), q_norm

    def search(self, q_ptr, q_feat, q_norm, wts, k, kc):
        return _topk_multi(q_ptr, q_feat, q_norm, self.fid_ch, self.idf, self.p_ptr, self.p_idx, self.s_norm,
                           np.asarray(wts, dtype=np.float32), self.n1, k, kc)


# ------------------------------------------------------------ feature sets
# Every function takes a frame with column r plus normalized columns and
# returns an exploded (r, f:uint64) frame.

def _explode_list(df, col, prefix):
    return (df.select(pl.col('r'), pl.col(col).alias('t')).explode('t').drop_nulls('t')
            .filter(pl.col('t') != '')
            .select('r', (pl.lit(prefix) + pl.col('t')).hash(7).alias('f')))


def feats_name_words(df):
    """Core-name unigrams and adjacent bigrams."""
    core = pl.when(pl.col('n_core').list.len() > 0).then(pl.col('n_core')).otherwise(pl.col('n_toks'))
    d = df.select('r', core.alias('c'))
    uni = _explode_list(d, 'c', 'w:')
    ex = d.explode('c').drop_nulls('c').with_columns(pl.int_range(pl.len()).over('r').alias('p'))
    bi = (ex.join(ex.with_columns(pl.col('p') - 1), on=['r', 'p'], suffix='2')
          .select('r', (pl.lit('b:') + pl.col('c') + ' ' + pl.col('c2')).hash(7).alias('f')))
    return pl.concat([uni, bi])


def joined_name(df):
    """Core tokens joined without spaces; website names use the domain."""
    core = pl.when(pl.col('n_core').list.len() > 0).then(pl.col('n_core')).otherwise(pl.col('n_toks'))
    j = core.list.join('')
    return pl.when((pl.col('n_web') != '') & ((pl.col('n_flags') & 4) > 0)).then(pl.col('n_web')).otherwise(j)


def feats_name_chars(df, n=4):
    d = df.select('r', ('#' + joined_name(df) + '#').alias('s'))
    d = d.with_columns(pl.int_ranges(0, (pl.col('s').str.len_chars().cast(pl.Int32) - n + 1).clip(1, None)).alias('p')).explode('p')
    return d.select('r', (pl.lit('c:') + pl.col('s').str.slice(pl.col('p'), n)).hash(7).alias('f'))


def feats_address(df):
    """Address tokens, numbers, and number|token combos for the first two numbers."""
    toks = _explode_list(df, 'a_toks', 'a:')
    nums = _explode_list(df, 'a_nums', 'n:')
    d = df.select('r', pl.col('a_nums').list.slice(0, 2).alias('n'), 'a_toks').explode('n').drop_nulls('n')
    d = d.explode('a_toks').drop_nulls('a_toks')
    combo = d.select('r', (pl.lit('x:') + pl.col('n') + '|' + pl.col('a_toks')).hash(7).alias('f'))
    return pl.concat([toks, nums, combo])


# ========================================================================================
# run_retrieve.py
#
# Run candidate retrieval for one split and save the candidate table.
#
# Rows are integer row numbers in WORK/norm_{split}.parquet (q = query
# record from Source 2/3, s = Source 1 record). One joint search per country
# scores name words, name character 4-grams and address keys together; each
# query keeps the top-K by the summed cosines plus each channel's own top-KC.
#
# Output: WORK/cand_{split}{tag}.npz with q, s, sc (n x C), comb, rk
# (rank by combined score, 99 for channel-only extras).
# ========================================================================================

CHANNELS = [
    ('nw', feats_name_words, int(os.environ.get('ER_CAP_NW', '5000'))),
    ('nc', feats_name_chars, int(os.environ.get('ER_CAP_NC', '2000'))),
    ('ad', feats_address, int(os.environ.get('ER_CAP_AD', '2000'))),
]
WTS = [float(x) for x in os.environ.get('ER_WTS', '1,1,1').split(',')]
K = int(os.environ.get('ER_K', '20'))
KC = int(os.environ.get('ER_KC', '3'))
BATCH = int(os.environ.get('ER_QBATCH', '400000'))
# Adaptive second pass: queries whose best combined score is below ADAPT_THR
# are searched again with the much larger caps CAPS2 (0 disables).
ADAPT_THR = float(os.environ.get('ER_ADAPT', '1.5'))
CAPS2 = [int(x) for x in os.environ.get('ER_CAPS2', '50000,20000,20000').split(',')]
COLS = ['src', 'country', 'n_toks', 'n_core', 'n_web', 'n_flags', 'a_toks', 'a_nums']


def run_country(split, country, t0):
    lf = (pl.scan_parquet(f'{WORK}/norm_{split}.parquet').select(COLS)
          .with_row_index('row').filter(pl.col('country') == country))
    d = lf.collect()
    s1 = d.filter(pl.col('src') == 1).with_columns(pl.int_range(pl.len(), dtype=pl.Int32).alias('r'))
    qx = d.filter(pl.col('src') != 1).with_columns(pl.int_range(pl.len(), dtype=pl.Int32).alias('r'))
    del d
    s1_rows = s1['row'].to_numpy().astype(np.int32)
    q_rows = qx['row'].to_numpy().astype(np.int32)
    n1, nq = s1.height, qx.height
    print(f'[{time.time()-t0:.0f}s] {country}: S1 {n1} queries {nq}', flush=True)
    idx_ = MultiIndex([fn(s1) for _, fn, _ in CHANNELS], [cap for _, _, cap in CHANNELS], n1)
    del s1
    gc.collect()
    print(f'[{time.time()-t0:.0f}s]   index: {len(idx_.p_idx)} postings, vocab {idx_.vocab.height}', flush=True)
    outs = {'q': [], 's': [], 'sc': [], 'comb': [], 'rk': []}
    t_search = 0.0
    n_adapt = 0
    for lo in range(0, nq, BATCH):
        b = qx.slice(lo, BATCH).with_columns(pl.int_range(pl.len(), dtype=pl.Int32).alias('r'))
        nb_ = b.height
        q_ptr, q_feat, q_norm = idx_.query_arrays([fn(b) for _, fn, _ in CHANNELS], nb_)
        t1 = time.time()
        idx, sc, comb = idx_.search(q_ptr, q_feat, q_norm, WTS, K, KC)
        if ADAPT_THR > 0:
            # second pass with much larger caps for queries without a strong candidate
            sel = np.where(comb[:, 0] < ADAPT_THR)[0]
            if len(sel):
                bs_ = b[sel].with_columns(pl.int_range(pl.len(), dtype=pl.Int32).alias('r'))
                p2, f2, n2 = idx_.query_arrays([fn(bs_) for _, fn, _ in CHANNELS], len(sel), CAPS2)
                i2, s2, c2 = idx_.search(p2, f2, n2, WTS, K, KC)
                idx[sel], sc[sel], comb[sel] = i2, s2, c2
                n_adapt += len(sel)
        t_search += time.time() - t1
        ok = idx >= 0
        cnt = ok.sum(1)
        M = idx.shape[1]
        rk = np.broadcast_to(np.where(np.arange(M) < K, np.arange(M), 99).astype(np.int8), idx.shape)
        outs['q'].append(np.repeat(q_rows[lo:lo + nb_], cnt))
        outs['s'].append(s1_rows[idx[ok]])
        outs['sc'].append(sc[ok])
        outs['comb'].append(comb[ok])
        outs['rk'].append(rk[ok])
        del idx, sc, comb, ok, q_ptr, q_feat, q_norm
    out = {k: np.concatenate(v) for k, v in outs.items()}
    print(f'[{time.time()-t0:.0f}s]   {country}: search {t_search:.0f}s, {len(out["q"])} pairs '
          f'({len(out["q"])/nq:.1f}/query), second pass for {n_adapt/nq:.3f} of queries', flush=True)
    return out


def run_retrieve_main(split, countries=None, tag=''):
    t0 = time.time()
    if not countries:
        countries = sorted(pl.scan_parquet(f'{WORK}/norm_{split}.parquet').select('country').unique().collect()['country'].to_list())
    parts = []
    for country in countries:
        parts.append(run_country(split, country, t0))
        gc.collect()
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez(f'{WORK}/cand_{split}{tag}.npz', channels=np.array([c[0] for c in CHANNELS]), **out)
    print(f'[{time.time()-t0:.0f}s] saved {len(out["q"])} pairs', flush=True)


def run_retrieve_cli():
    """`python run_retrieve.py ...` of the original pipeline (reads sys.argv)."""
    split = sys.argv[1]
    countries = sys.argv[2].split(',') if len(sys.argv) > 2 and sys.argv[2] else None
    tag = sys.argv[3] if len(sys.argv) > 3 else ''
    run_retrieve_main(split, countries, tag)


# ========================================================================================
# labels.py
#
# Integer labels for the train split.
#
# true_s[row] = norm row of the matching Source 1 record for a Source 2/3
# row, or -1. Each Source 2/3 record matches at most one Source 1 record.
# ========================================================================================

def true_s1(split='train'):
    norm = pl.read_parquet(f'{WORK}/norm_{split}.parquet', columns=['id']).with_columns(
        pl.int_range(pl.len(), dtype=pl.Int32).alias('row'))
    pairs = pl.read_parquet(f'{WORK}/train_pairs.parquet')
    p = (pairs.join(norm.rename({'id': 's1', 'row': 's'}), on='s1')
         .join(norm.rename({'id': 'sx', 'row': 'q'}), on='sx'))
    t = np.full(norm.height, -1, dtype=np.int32)
    t[p['q'].to_numpy()] = p['s'].to_numpy()
    return t


# ========================================================================================
# metric.py
#
# Macro F0.5 over Source 1 entities, with many-to-one assignment.
# ========================================================================================

def assign(q, s, p):
    """Best candidate per query. Returns (uq, best_s, best_p)."""
    order = np.lexsort((-p, q))
    qo = q[order]
    first = np.ones(len(qo), dtype=bool)
    first[1:] = qo[1:] != qo[:-1]
    idx = order[first]
    return q[idx], s[idx], p[idx]


def macro_f05(pred_q, pred_s, ts, s1_mask):
    """pred_q/pred_s: predicted pairs (each query at most once).
    ts: true Source 1 row per record (-1 none). s1_mask: which rows are the
    Source 1 entities to average over."""
    n = len(ts)
    tp = np.bincount(pred_s[ts[pred_q] == pred_s], minlength=n)
    npred = np.bincount(pred_s, minlength=n)
    ntrue = np.bincount(ts[ts >= 0], minlength=n)
    tp, npred, ntrue = tp[s1_mask], npred[s1_mask], ntrue[s1_mask]
    P = np.where(npred > 0, tp / np.maximum(npred, 1), 0.0)
    R = np.where(ntrue > 0, tp / np.maximum(ntrue, 1), 0.0)
    f = np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0)
    f = np.where(ntrue == 0, (npred == 0).astype(float), f)
    return f.mean(), f


# ========================================================================================
# tokrisk.py
#
# Learned risk of unmatched name tokens.
#
# For each (country, token): how often it appears as an *extra* token (in the
# query name, not in the Source 1 name) or a *missing* token (in Source 1, not
# in the query) among true pairs versus false candidate pairs. Some extras
# are generator noise ("center", "services"); others mark a different
# business ("traders", "stores" inserted into a sibling's name).
#
# Counts are cross-fitted: pairs of fold f get risks computed from fold 1-f,
# so a pair's own label never feeds its feature. Test uses all train pairs.
# ========================================================================================

def fold_of(s):
    return ((s.astype(np.int64) * 2654435761) >> 7) % 2


@nb.njit(cache=True)
def _tokrisk_count(qi, si, y, ptr, tok, cnt):
    for p in range(qi.shape[0]):
        a0, a1 = ptr[qi[p]], ptr[qi[p] + 1]
        b0, b1 = ptr[si[p]], ptr[si[p] + 1]
        lab = y[p]
        for a in range(a0, a1):
            t = tok[a]
            found = False
            for b in range(b0, b1):
                if tok[b] == t:
                    found = True
                    break
            if not found:
                cnt[t, lab] += 1
        for b in range(b0, b1):
            t = tok[b]
            found = False
            for a in range(a0, a1):
                if tok[a] == t:
                    found = True
                    break
            if not found:
                cnt[t, 2 + lab] += 1


def tokrisk_counts(st, q, s, y):
    cnt = np.zeros((len(st.nm_idf), 4), dtype=np.int64)
    _tokrisk_count(q, s, y.astype(np.int64), st.nm_ptr, st.nm_tok, cnt)
    return cnt


def tokrisk_risk(cnt, n_pos, n_neg):
    """log P(token as extra | false pair) - log P(... | true pair), smoothed."""
    r_extra = np.log((cnt[:, 0] + 1.0) / (n_neg + 1.0)) - np.log((cnt[:, 1] + 1.0) / (n_pos + 1.0))
    r_miss = np.log((cnt[:, 2] + 1.0) / (n_neg + 1.0)) - np.log((cnt[:, 3] + 1.0) / (n_pos + 1.0))
    # tokens never seen as extra/missing in either class carry no evidence
    r_extra[(cnt[:, 0] + cnt[:, 1]) == 0] = 0.0
    r_miss[(cnt[:, 2] + cnt[:, 3]) == 0] = 0.0
    return r_extra.astype(np.float32), r_miss.astype(np.float32)


@nb.njit(parallel=True, cache=True)
def _tokrisk_pair(qi, si, ptr, tok, r_extra, r_miss):
    n = qi.shape[0]
    out = np.zeros((n, 5), dtype=np.float32)
    for p in nb.prange(n):
        a0, a1 = ptr[qi[p]], ptr[qi[p] + 1]
        b0, b1 = ptr[si[p]], ptr[si[p] + 1]
        mx_e = -99.0
        sm_e = 0.0
        mx_m = -99.0
        sm_m = 0.0
        ne = 0
        for a in range(a0, a1):
            t = tok[a]
            found = False
            for b in range(b0, b1):
                if tok[b] == t:
                    found = True
                    break
            if not found:
                ne += 1
                v = r_extra[t]
                sm_e += v
                if v > mx_e:
                    mx_e = v
        for b in range(b0, b1):
            t = tok[b]
            found = False
            for a in range(a0, a1):
                if tok[a] == t:
                    found = True
                    break
            if not found:
                v = r_miss[t]
                sm_m += v
                if v > mx_m:
                    mx_m = v
        out[p, 0] = mx_e
        out[p, 1] = sm_e
        out[p, 2] = mx_m
        out[p, 3] = sm_m
        out[p, 4] = ne
    return out


def tokrisk_pair_features(st, q, s, r_extra, r_miss):
    o = _tokrisk_pair(q, s, st.nm_ptr, st.nm_tok, r_extra, r_miss)
    return {'risk_extra_max': o[:, 0], 'risk_extra_sum': o[:, 1], 'risk_miss_max': o[:, 2],
            'risk_miss_sum': o[:, 3], 'n_extra': o[:, 4]}


# ------------------------------------------------ legal-form pair risk

NLEG = 32


def _rel_grid():
    a = np.arange(NLEG)[:, None]
    b = np.arange(NLEG)[None, :]
    return np.select([(a == 0) & (b == 0), a == 0, b == 0, a == b], [0, 1, 2, 3], 4)


def leg_counts(lq, ls, y):
    code = lq.astype(np.int64) * NLEG + ls.astype(np.int64)
    pos = np.bincount(code[y == 1], minlength=NLEG * NLEG)
    neg = np.bincount(code[y == 0], minlength=NLEG * NLEG)
    return pos, neg


def leg_risk(pos, neg):
    """log-odds that a (query form, Source 1 form) combination is a false
    pair. Combinations never seen (e.g. French forms) get the pooled value
    of their relation class (none / one-sided / agree / conflict)."""
    npos, nneg = pos.sum(), neg.sum()
    r = np.log((neg + 1.0) / (nneg + 1.0)) - np.log((pos + 1.0) / (npos + 1.0))
    rel = _rel_grid().ravel()
    unseen = (pos + neg) == 0
    for c in range(5):
        m = rel == c
        pooled = np.log((neg[m].sum() + 1.0) / (nneg + 1.0)) - np.log((pos[m].sum() + 1.0) / (npos + 1.0))
        r[m & unseen] = pooled
    return r.astype(np.float32)


def leg_pair(lq, ls, r):
    return r[lq.astype(np.int64) * NLEG + ls.astype(np.int64)]


def to_table(st, r_extra, r_miss):
    """Risk keyed by (country, token) so test can reuse it."""
    return st.nm_vocab.with_columns(pl.Series('r_extra', r_extra), pl.Series('r_miss', r_miss))


def from_table(st, table, fallback=True):
    """Map train risks onto this split's vocabulary.

    A token with no statistics for its own country (every French word, since
    train has no France) first borrows the same word's risk from the other
    countries ("services", "center" stay harmless), and otherwise gets the
    median risk of an ordinary frequent word. Leaving it at 0 would make an
    unknown word look like known generator noise.
    """
    v = st.nm_vocab.join(table.select('country', 't', 'r_extra', 'r_miss'), on=['country', 't'], how='left')
    if fallback:
        seen = table['country'].unique().to_list()
        pooled = (table.filter(pl.col('df') >= 20).group_by('t')
                  .agg(pl.col('r_extra').mean().alias('pe'), pl.col('r_miss').mean().alias('pm')))
        freq = table.filter(pl.col('df') >= 100)
        de, dm = float(freq['r_extra'].median()), float(freq['r_miss'].median())
        new = ~pl.col('country').is_in(seen)
        v = v.join(pooled, on='t', how='left').with_columns(
            pl.when(new).then(pl.coalesce('pe', pl.lit(de))).otherwise(pl.col('r_extra')).alias('r_extra'),
            pl.when(new).then(pl.coalesce('pm', pl.lit(dm))).otherwise(pl.col('r_miss')).alias('r_miss'))
    v = v.sort('tid')
    return (v['r_extra'].fill_null(0.0).to_numpy().astype(np.float32),
            v['r_miss'].fill_null(0.0).to_numpy().astype(np.float32))


# ========================================================================================
# features.py
#
# Pair features for (query record, Source 1 record) candidates.
#
# RecordStore holds per-record arrays for one split (all three sources):
# token-id CSR arrays with per-country IDF, number lists, flags, and the
# strings rapidfuzz needs. pair_features() turns aligned index arrays into a
# feature frame.
# ========================================================================================

# Priority order for a record's legal form (first match wins).
LEGAL_ORDER = ['llp', 'pvt', 'ltd', 'pllc', 'llc', 'inc', 'corp', 'co', 'lp', 'pc', 'pa', 'plc',
               'sasu', 'sas', 'sarl', 'eurl', 'sci', 'sa', 'snc', 'ei', 'selarl', 'scop', 'scp']


def _csr(lists_len, flat):
    ptr = np.zeros(len(lists_len) + 1, dtype=np.int64)
    np.cumsum(lists_len, out=ptr[1:])
    return ptr, flat


class RecordStore:
    def __init__(self, split):
        t0 = time.time()
        d = pl.read_parquet(f'{WORK}/norm_{split}.parquet').drop('business_name', 'business_address')
        self.n = d.height
        # fixed encoding so train and test agree: US=0, India=1, anything else=2
        self.country = d['country'].replace_strict({'US': 0, 'India': 1}, default=2, return_dtype=pl.Int8).to_numpy()
        self.src = d['src'].to_numpy().astype(np.int8)
        self.flags = d['n_flags'].to_numpy().astype(np.int16)
        core = pl.when(pl.col('n_core').list.len() > 0).then(pl.col('n_core')).otherwise(pl.col('n_toks'))
        d = d.with_columns(core.alias('core'), pl.int_range(pl.len(), dtype=pl.Int32).alias('row'))
        # strings for rapidfuzz, kept as Arrow-backed series; lists are made per chunk
        isweb = (pl.col('n_flags') & 4) > 0
        strs = d.select(
            pl.col('core').list.join(' ').alias('core'),
            pl.col('n_toks').list.join(' ').alias('full'),
            pl.when(isweb & (pl.col('n_web') != '')).then(pl.col('n_web')).otherwise(pl.col('core').list.join('')).alias('join'),
            pl.col('a_toks').list.join(' ').alias('addr'),
        )
        self.s_core, self.s_full, self.s_join, self.s_addr = strs['core'], strs['full'], strs['join'], strs['addr']
        del strs
        # token ids per (country, token) with IDF from Source 1 document frequency
        self.nm_ptr, self.nm_tok, self.nm_idf, self.nm_vocab = self._tok_csr(d, 'core')
        self.ad_ptr, self.ad_tok, self.ad_idf, _ = self._tok_csr(d, 'a_toks')
        # name tokens never seen in any Source 1 name of the country (made-up names)
        self.nm_oov = (self.nm_vocab['df'].to_numpy() == 0).astype(np.uint8)
        # legal form: first legal token by priority, 0 = none
        leg = {t: i + 1 for i, t in enumerate(LEGAL_ORDER)}
        self.leg = d.select(pl.col('n_toks').list.eval(pl.element().replace_strict(leg, default=99, return_dtype=pl.Int8))
                            .list.min().fill_null(99).alias('l'))['l'].to_numpy().astype(np.int8)
        self.leg[self.leg == 99] = 0
        nums = d.select('row', pl.col('a_nums').list.eval(pl.element().str.slice(0, 12).cast(pl.Int64)).alias('nums'))
        lens = nums['nums'].list.len().to_numpy()
        self.num_ptr, self.num_val = _csr(lens, nums['nums'].explode().drop_nulls().to_numpy().astype(np.int64))
        st = d['a_state'].fill_null('')
        sv = sorted(st.unique().to_list())
        self.state = st.replace_strict(sv, list(range(len(sv))), return_dtype=pl.Int32).to_numpy()
        self.state_empty = sv.index('') if '' in sv else -1
        # how many Source 1 records in the country share this exact core name
        dup = (d.filter(pl.col('src') == 1).group_by(['country', pl.col('core').list.join(' ').alias('k')])
               .len().rename({'len': 'dup'}))
        dd = d.select('row', 'country', pl.col('core').list.join(' ').alias('k')).join(dup, on=['country', 'k'], how='left')
        self.name_dup = dd.sort('row')['dup'].fill_null(0).to_numpy().astype(np.int32)
        print(f'RecordStore({split}) {self.n} rows in {time.time()-t0:.0f}s', flush=True)

    def _tok_csr(self, d, col):
        e = d.select('row', 'country', pl.col(col).alias('t')).explode('t').drop_nulls('t').filter(pl.col('t') != '')
        e = e.with_columns(pl.int_range(pl.len()).over('row').alias('pos'))
        vocab = e.select('country', 't').unique().with_row_index('tid')
        e = e.join(vocab, on=['country', 't']).sort(['row', 'pos'])
        s1rows = d.filter(pl.col('src') == 1)
        n1 = s1rows.group_by('country').len().rename({'len': 'n1'})
        dfc = (e.join(d.select('row', 'src'), on='row').filter(pl.col('src') == 1)
               .unique(['row', 'tid']).group_by('tid').len().rename({'len': 'df'}))
        vocab = vocab.join(dfc, on='tid', how='left').join(n1, on='country').with_columns(pl.col('df').fill_null(0))
        vocab = vocab.with_columns(((pl.col('n1') + 1) / (pl.col('df') + 0.5)).log().alias('idf')).sort('tid')
        idf = vocab['idf'].to_numpy().astype(np.float32)
        lens = np.bincount(e['row'].to_numpy(), minlength=d.height)
        ptr, tok = _csr(lens, e['tid'].to_numpy().astype(np.int32))
        return ptr, tok, idf, vocab.select('tid', 'country', 't', 'df')

    @staticmethod
    def gather(series: pl.Series, idx: np.ndarray):
        return series.gather(idx).to_list()


# ------------------------------------------------------------ numba kernels

@nb.njit(parallel=True, cache=True)
def _tok_overlap(qi, si, ptr, tok, idf):
    """IDF overlap features for token sets (duplicates inside a record ignored)."""
    n = qi.shape[0]
    out = np.zeros((n, 9), dtype=np.float32)
    for p in nb.prange(n):
        a0, a1 = ptr[qi[p]], ptr[qi[p] + 1]
        b0, b1 = ptr[si[p]], ptr[si[p] + 1]
        tq = 0.0
        ts = 0.0
        sh = 0.0
        nsh = 0
        mx_extra = 0.0
        mx_miss = 0.0
        for a in range(a0, a1):
            t = tok[a]
            dup = False
            for a2 in range(a0, a):
                if tok[a2] == t:
                    dup = True
                    break
            if dup:
                continue
            w = idf[t]
            tq += w
            found = False
            for b in range(b0, b1):
                if tok[b] == t:
                    found = True
                    break
            if found:
                sh += w
                nsh += 1
            elif w > mx_extra:
                mx_extra = w
        for b in range(b0, b1):
            t = tok[b]
            dup = False
            for b2 in range(b0, b):
                if tok[b2] == t:
                    dup = True
                    break
            if dup:
                continue
            w = idf[t]
            ts += w
            found = False
            for a in range(a0, a1):
                if tok[a] == t:
                    found = True
                    break
            if not found and w > mx_miss:
                mx_miss = w
        out[p, 0] = sh / tq if tq > 0 else -1.0
        out[p, 1] = sh / ts if ts > 0 else -1.0
        out[p, 2] = sh
        out[p, 3] = nsh
        out[p, 4] = mx_extra
        out[p, 5] = mx_miss
        out[p, 6] = a1 - a0
        out[p, 7] = b1 - b0
        # first-token agreement
        out[p, 8] = 1.0 if (a1 > a0 and b1 > b0 and tok[a0] == tok[b0]) else 0.0
    return out


@nb.njit(cache=True)
def _ndigits(x):
    if x == 0:
        return 1
    c = 0
    while x > 0:
        c += 1
        x //= 10
    return c


@nb.njit(cache=True)
def _digit_ed(x, y):
    """Levenshtein distance between the decimal strings of two ints."""
    dx = np.empty(20, dtype=np.int64)
    dy = np.empty(20, dtype=np.int64)
    nx = 0
    if x == 0:
        dx[0] = 0
        nx = 1
    while x > 0 and nx < 20:
        dx[nx] = x % 10
        x //= 10
        nx += 1
    ny = 0
    if y == 0:
        dy[0] = 0
        ny = 1
    while y > 0 and ny < 20:
        dy[ny] = y % 10
        y //= 10
        ny += 1
    prev = np.arange(ny + 1)
    cur = np.zeros(ny + 1, dtype=np.int64)
    for i in range(1, nx + 1):
        cur[0] = i
        for j in range(1, ny + 1):
            c = 0 if dx[nx - i] == dy[ny - j] else 1
            v = prev[j - 1] + c
            if prev[j] + 1 < v:
                v = prev[j] + 1
            if cur[j - 1] + 1 < v:
                v = cur[j - 1] + 1
            cur[j] = v
        for j in range(ny + 1):
            prev[j] = cur[j]
    return prev[ny]


@nb.njit(cache=True)
def _is_prefix(a, b):
    """Decimal string of a is a proper prefix of b's."""
    la = _ndigits(a)
    lb = _ndigits(b)
    if la >= lb:
        return False
    for _ in range(lb - la):
        b //= 10
    return a == b


@nb.njit(parallel=True, cache=True)
def _num_feats(qi, si, ptr, val):
    n = qi.shape[0]
    out = np.full((n, 11), -1.0, dtype=np.float32)
    for p in nb.prange(n):
        a0, a1 = ptr[qi[p]], ptr[qi[p] + 1]
        b0, b1 = ptr[si[p]], ptr[si[p] + 1]
        out[p, 0] = a1 - a0
        out[p, 1] = b1 - b0
        if a1 == a0 or b1 == b0:
            continue
        # s first number present anywhere in q
        s0 = val[b0]
        q0 = val[a0]
        anyeq = 0.0
        best_ed = 99.0
        best_rel = 99.0
        best_abs = 1e12
        pref = 0.0
        for a in range(a0, a1):
            if val[a] == s0:
                anyeq = 1.0
            ed = _digit_ed(val[a], s0)
            if ed < best_ed:
                best_ed = ed
            d = abs(val[a] - s0) / max(1.0, float(s0))
            if d < best_rel:
                best_rel = d
            da = abs(val[a] - s0)
            if da < best_abs:
                best_abs = da
            if _is_prefix(val[a], s0) or _is_prefix(s0, val[a]):
                pref = 1.0
        out[p, 2] = anyeq
        out[p, 3] = 1.0 if q0 == s0 else 0.0
        out[p, 4] = best_ed
        out[p, 5] = best_rel
        out[p, 8] = math.log1p(best_abs)
        out[p, 9] = pref
        out[p, 10] = math.log1p(abs(q0 - s0))
        # jaccard of number sets
        inter = 0
        for b in range(b0, b1):
            for a in range(a0, a1):
                if val[a] == val[b]:
                    inter += 1
                    break
        out[p, 6] = inter / ((a1 - a0) + (b1 - b0) - inter)
        out[p, 7] = _ndigits(s0) - _ndigits(q0)
    return out


@nb.njit(parallel=True, cache=True)
def _oov(qi, ptr, tok, oov):
    """Share and count of a record's name tokens that no Source 1 name uses."""
    n = qi.shape[0]
    out = np.zeros((n, 2), dtype=np.float32)
    for p in nb.prange(n):
        a0, a1 = ptr[qi[p]], ptr[qi[p] + 1]
        c = 0
        for a in range(a0, a1):
            c += oov[tok[a]]
        out[p, 0] = c / (a1 - a0) if a1 > a0 else -1.0
        out[p, 1] = c
    return out


def legal_relation(lq, ls):
    """0 both none, 1 query none, 2 Source 1 none, 3 agree, 4 conflict."""
    return np.select([(lq == 0) & (ls == 0), lq == 0, ls == 0, lq == ls], [0, 1, 2, 3], 4).astype(np.float32)


# ------------------------------------------------------------ string feats

def _rf(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def features_pair_features(st: RecordStore, qi: np.ndarray, si: np.ndarray) -> pl.DataFrame:
    t0 = time.time()
    F = {}
    o = _tok_overlap(qi, si, st.nm_ptr, st.nm_tok, st.nm_idf)
    for k, name in enumerate(['nm_cov_q', 'nm_cov_s', 'nm_sh_idf', 'nm_nsh', 'nm_extra_idf', 'nm_miss_idf',
                              'nm_nq', 'nm_ns', 'nm_first_eq']):
        F[name] = o[:, k]
    o = _tok_overlap(qi, si, st.ad_ptr, st.ad_tok, st.ad_idf)
    for k, name in enumerate(['ad_cov_q', 'ad_cov_s', 'ad_sh_idf', 'ad_nsh', 'ad_extra_idf', 'ad_miss_idf',
                              'ad_nq', 'ad_ns', 'ad_first_eq']):
        F[name] = o[:, k]
    o = _num_feats(qi, si, st.num_ptr, st.num_val)
    for k, name in enumerate(['num_nq', 'num_ns', 'num_s0_in_q', 'num_first_eq', 'num_best_ed', 'num_best_rel',
                              'num_jacc', 'num_len_diff', 'num_best_abs', 'num_prefix', 'num_first_abs']):
        F[name] = o[:, k]
    o = _oov(qi, st.nm_ptr, st.nm_tok, st.nm_oov)
    F['q_oov_frac'] = o[:, 0]
    F['q_oov_n'] = o[:, 1]
    F['leg_rel'] = legal_relation(st.leg[qi], st.leg[si])
    sq, ss = st.state[qi], st.state[si]
    F['state_eq'] = np.where((sq == st.state_empty) | (ss == st.state_empty), -1, (sq == ss).astype(np.int8)).astype(np.float32)
    F['q_flags'] = st.flags[qi].astype(np.float32)
    F['q_src'] = st.src[qi].astype(np.float32)
    F['country'] = st.country[qi].astype(np.float32)
    F['s_name_dup'] = st.name_dup[si].astype(np.float32)
    t1 = time.time()
    g = RecordStore.gather
    core_q, core_s = g(st.s_core, qi), g(st.s_core, si)
    F['nm_ratio'] = _rf(fuzz.ratio, core_q, core_s)
    F['nm_tset'] = _rf(fuzz.token_set_ratio, core_q, core_s)
    F['nm_tsort'] = _rf(fuzz.token_sort_ratio, core_q, core_s)
    F['nm_partial'] = _rf(fuzz.partial_ratio, core_q, core_s)
    del core_q, core_s
    jq, js = g(st.s_join, qi), g(st.s_join, si)
    F['nm_join_jw'] = _rf(JaroWinkler.normalized_similarity, jq, js)
    F['nm_join_partial'] = _rf(fuzz.partial_ratio, jq, js)
    F['nm_join_lev'] = _rf(Levenshtein.normalized_similarity, jq, js)
    del jq, js
    fq, fs = g(st.s_full, qi), g(st.s_full, si)
    F['nm_full_ratio'] = _rf(fuzz.ratio, fq, fs)
    del fq, fs
    aq, as_ = g(st.s_addr, qi), g(st.s_addr, si)
    F['ad_tset'] = _rf(fuzz.token_set_ratio, aq, as_)
    F['ad_tsort'] = _rf(fuzz.token_sort_ratio, aq, as_)
    F['ad_partial'] = _rf(fuzz.partial_ratio, aq, as_)
    F['ad_len_q'] = np.array([len(x) for x in aq], dtype=np.float32)
    del aq, as_
    t2 = time.time()
    print(f'    pair_features n={len(qi)} numba {t1-t0:.1f}s rapidfuzz {t2-t1:.1f}s', flush=True)
    return pl.DataFrame(F)


# ========================================================================================
# stage0.py
#
# Stage-0 pruning: a small LightGBM on retrieval signals only decides which
# candidates get full features, instead of a fixed top-K by combined score.
#
# Signals per pair: the three channel cosines, the combined score and rank, and
# per-query context (best combined score, gap to it, gaps per channel, number of
# candidates). Cross-fitted by query halves on train: queries in half h are
# pruned by the model trained on the other half. Test uses the mean of both.
#
# On train (eval half): top-7+extras keeps 9.6 pairs/query at 98.15% recall of
# true links; stage-0 p >= 0.001 keeps 1.57 pairs/query at 98.81%.
# ========================================================================================

stage0_NAMES = ['sc_nw', 'sc_nc', 'sc_ad', 'comb', 'rk', 'q_best', 'gap_best', 'q_n',
         'gap_nw', 'gap_nc', 'gap_ad']
stage0_PARAMS = dict(objective='binary', learning_rate=0.1, num_leaves=127, min_data_in_leaf=500,
              verbose=-1, num_threads=8)


def stage0_signals(q, sc, comb, rk):
    """Per-pair signal matrix; pairs grouped by query in retrieval order."""
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    cnt = np.diff(np.r_[starts, len(q)])
    gid = np.repeat(np.arange(len(starts)), cnt)
    best = np.maximum.reduceat(comb, starts)[gid]
    cols = [sc[:, 0], sc[:, 1], sc[:, 2], comb, rk.astype(np.float32), best, comb - best, cnt[gid].astype(np.float32)]
    for c in range(3):
        cols.append(np.maximum.reduceat(sc[:, c], starts)[gid] - sc[:, c])
    return np.stack(cols, 1).astype(np.float32)


def stage0_half_of(q):
    return ((q.astype(np.int64) * 2654435761) >> 5) % 2


def stage0_train(cand_tag='_a', frac=0.3):
    """Two models, each trained on a sample of one query half."""
    t0 = time.time()
    z = np.load(f'{WORK}/cand_train{cand_tag}.npz')
    q = z['q']
    ts = true_s1()
    rng = np.random.default_rng(0)
    uq = np.unique(q)
    pick = np.zeros(q.max() + 1, bool)
    pick[rng.choice(uq, int(len(uq) * frac), replace=False)] = True
    m = pick[q]
    q = q[m]; s = z['s'][m]; sc = z['sc'][m]; comb = z['comb'][m]; rk = z['rk'][m]
    del z
    y = (ts[q] == s).astype(np.int8)
    X = stage0_signals(q, sc, comb, rk)
    h = stage0_half_of(q)
    for k in (0, 1):
        mk = h == k
        model = lgb.train(stage0_PARAMS, lgb.Dataset(X[mk], y[mk], feature_name=stage0_NAMES), 300)
        model.save_model(f'{WORK}/stage0{cand_tag}_h{k}.txt')
        print(f'[{time.time()-t0:.0f}s] stage-0 model for half {k} trained on {mk.sum()} pairs', flush=True)


def stage0_predict(q, sc, comb, rk, cand_tag='_a', split='train', chunk=20_000_000):
    """Stage-0 probability for every candidate pair, in chunks aligned to
    query boundaries. Train: half-h queries use the model trained on the other
    half. Test: mean of both models."""
    models = [lgb.Booster(model_file=f'{WORK}/stage0{cand_tag}_h{k}.txt') for k in (0, 1)]
    p = np.empty(len(q), np.float32)
    a = 0
    while a < len(q):
        b = min(len(q), a + chunk)
        while b < len(q) and q[b] == q[b - 1]:
            b += 1
        X = stage0_signals(q[a:b], sc[a:b], comb[a:b], rk[a:b])
        if split == 'train':
            h = stage0_half_of(q[a:b])
            for k in (0, 1):
                mk = h == k
                if mk.any():
                    p[a:b][mk] = models[1 - k].predict(X[mk], num_threads=8)
        else:
            p[a:b] = (models[0].predict(X, num_threads=8) + models[1].predict(X, num_threads=8)) / 2
        a = b
    return p


def stage0_cli():
    """`python stage0.py ...` of the original pipeline (reads sys.argv)."""
    stage0_train(sys.argv[1] if len(sys.argv) > 1 else '_a')


# ========================================================================================
# build_features.py
#
# Compute pair features for the pruned candidate set of a split.
#
# Writes WORK/feat_{split}/part_{i}.parquet, each holding whole queries
# (a query's candidates never straddle two files), with columns
# q, s, [y], channel scores/ranks, pair features, and per-query context.
# ========================================================================================

K0 = int(os.environ.get('ER_K0', '5'))       # keep a candidate if it is top-K0 in any channel
CHUNK = int(os.environ.get('ER_CHUNK', '4000000'))
# Stage-0 pruning (ER_STAGE0=<cand tag of the stage-0 models>): keep a pair if
# stage-0 p >= T0 or its combined rank is below KMIN.
STAGE0 = os.environ.get('ER_STAGE0', '')
T0 = float(os.environ.get('ER_T0', '0.001'))
KMIN = int(os.environ.get('ER_KMIN', '3'))


def query_context(df: pl.DataFrame) -> pl.DataFrame:
    """Per-query competition features: how this candidate compares with the
    query's other candidates on the main similarity scores."""
    cols = ['comb', 'sc_nw', 'sc_nc', 'sc_ad', 'nm_tset', 'nm_join_jw', 'ad_tset', 'nm_cov_s', 'ad_cov_s']
    exprs = [pl.len().over('q').alias('q_ncand')]
    for c in cols:
        exprs.append(pl.col(c).rank('min', descending=True).over('q').cast(pl.Float32).alias(f'{c}_qrank'))
    df = df.with_columns(exprs)
    # margin to the best *other* candidate (positive when this one is the unique best)
    ex2 = []
    for c in cols:
        first = pl.col(c).max().over('q')
        second = pl.col(c).sort(descending=True).slice(1, 1).first().over('q')
        other_best = pl.when(pl.col(c) >= first).then(second).otherwise(first)
        ex2.append((pl.col(c) - other_best.fill_null(0)).alias(f'{c}_margin'))
    return df.with_columns(ex2)


def build_features_main(split, cand_tag='', out_tag=''):
    t0 = time.time()
    z = np.load(f'{WORK}/cand_{split}{cand_tag}.npz')
    p0 = None
    if STAGE0:
        # stage-0 pruning: keep pairs the retrieval-signal model finds plausible
        q, s, sc, comb, rk = z['q'], z['s'], z['sc'], z['comb'], z['rk']
        p0 = stage0_predict(q, sc, comb, rk, cand_tag=STAGE0, split=split)
        keep = (p0 >= T0) | (rk < KMIN)
        # stage-0 learned US/India retrieval signals; for other countries also
        # keep the old top-K0 + channel extras so nothing is lost to the shift
        ctry = pl.read_parquet(f'{WORK}/norm_{split}.parquet', columns=['country'])['country']
        unseen = ~ctry.is_in(['US', 'India']).to_numpy()
        keep |= unseen[q] & ((rk < K0) | (rk == 99))
        q, s, sc, comb, rk, p0 = q[keep], s[keep], sc[keep], comb[keep], rk[keep], p0[keep]
        print(f'[{time.time()-t0:.0f}s] stage-0 kept {keep.mean():.4f} of pairs', flush=True)
    else:
        rk = z['rk']
        keep = (rk < K0) | (rk == 99)
        rk = rk[keep]
        q = z['q'][keep]
        s = z['s'][keep]
        comb = z['comb'][keep]
        sc = z['sc'][keep]
    del z, keep
    order = np.lexsort((s, q))
    q, s, sc, rk, comb = q[order], s[order], sc[order], rk[order], comb[order]
    if p0 is not None:
        p0 = p0[order]
    print(f'[{time.time()-t0:.0f}s] {split}: {len(q)} pairs after top-{K0} pruning', flush=True)
    y = None
    st = RecordStore(split)
    if split == 'train':
        ts = true_s1()
        y = (ts[q] == s).astype(np.int8)
        fo = fold_of(s)
        c0 = tokrisk_counts(st, q[fo == 0], s[fo == 0], y[fo == 0])
        c1 = tokrisk_counts(st, q[fo == 1], s[fo == 1], y[fo == 1])
        npos = [int(y[fo == k].sum()) for k in (0, 1)]
        nneg = [int((fo == k).sum()) - npos[k] for k in (0, 1)]
        l0 = leg_counts(st.leg[q[fo == 0]], st.leg[s[fo == 0]], y[fo == 0])
        l1 = leg_counts(st.leg[q[fo == 1]], st.leg[s[fo == 1]], y[fo == 1])
        risk_by_fold = {1: tokrisk_risk(c0, npos[0], nneg[0]) + (leg_risk(*l0),),
                        0: tokrisk_risk(c1, npos[1], nneg[1]) + (leg_risk(*l1),)}
        full = tokrisk_risk(c0 + c1, sum(npos), sum(nneg))
        to_table(st, *full).write_parquet(f'{WORK}/tokrisk{out_tag}.parquet')
        np.save(f'{WORK}/legrisk{out_tag}.npy', leg_risk(l0[0] + l1[0], l0[1] + l1[1]))
        del c0, c1
    else:
        table = pl.read_parquet(f'{WORK}/tokrisk{out_tag}.parquet')
        fo = np.zeros(len(q), dtype=np.int64)
        risk_by_fold = {0: from_table(st, table) + (np.load(f'{WORK}/legrisk{out_tag}.npy'),)}
    print(f'[{time.time()-t0:.0f}s] token risk ready', flush=True)
    outdir = f'{WORK}/feat_{split}{out_tag}'
    os.makedirs(outdir, exist_ok=True)
    # chunk boundaries aligned to query changes
    bounds = [0]
    while bounds[-1] < len(q):
        e = min(len(q), bounds[-1] + CHUNK)
        while e < len(q) and q[e] == q[e - 1]:
            e += 1
        bounds.append(e)
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        F = features_pair_features(st, q[a:b], s[a:b])
        base = {'q': q[a:b], 's': s[a:b]}
        if y is not None:
            base['y'] = y[a:b]
        for c, name in enumerate(['nw', 'nc', 'ad']):
            base[f'sc_{name}'] = sc[a:b, c]
        base['comb'] = comb[a:b]
        base['rk_comb'] = rk[a:b].astype(np.float32)
        if p0 is not None:
            base['p0'] = p0[a:b]
        R = None
        for k, (re_, rm_, rl_) in risk_by_fold.items():
            m = fo[a:b] == k
            if not m.any():
                continue
            qq, ss = q[a:b][m], s[a:b][m]
            part = tokrisk_pair_features(st, qq, ss, re_, rm_)
            part['leg_pair_risk'] = leg_pair(st.leg[qq], st.leg[ss], rl_)
            if R is None:
                R = {c: np.zeros(b - a, dtype=np.float32) for c in part}
            for c, v in part.items():
                R[c][m] = v
        base.update(R)
        df = pl.concat([pl.DataFrame(base), F], how='horizontal')
        df = query_context(df)
        df.write_parquet(f'{outdir}/part_{i:03d}.parquet')
        print(f'[{time.time()-t0:.0f}s]   part {i}: rows {a}-{b}', flush=True)
    print(f'[{time.time()-t0:.0f}s] done', flush=True)


def build_features_cli():
    """`python build_features.py ...` of the original pipeline (reads sys.argv)."""
    build_features_main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '', sys.argv[3] if len(sys.argv) > 3 else '')


# ========================================================================================
# unsup.py
#
# Label-free token statistics that transfer to a country without labels.
#
# For candidate pairs whose core names differ by exactly one token w (the query
# has one extra token, or lacks one Source 1 token), we count how often the
# Source 1 house number still appears in the query address. Generator noise
# words ("services", "center") leave the address alone, while sibling markers
# ("holding", "groupe", "(france)") come with a different house number. The
# rate is computed per (country, token) from the split's own candidates, so
# France gets real statistics even though it has no labels.
# ========================================================================================

@nb.njit(cache=True)
def _unsup_count(qi, si, nptr, ntok, mptr, mval, agree_e, n_e, agree_m, n_m):
    for p in range(qi.shape[0]):
        a0, a1 = nptr[qi[p]], nptr[qi[p] + 1]
        b0, b1 = nptr[si[p]], nptr[si[p] + 1]
        # house-number agreement: Source 1 first number present in query
        c0, c1 = mptr[si[p]], mptr[si[p] + 1]
        d0, d1 = mptr[qi[p]], mptr[qi[p] + 1]
        if c1 == c0 or d1 == d0:
            continue
        s0 = mval[c0]
        ag = 0
        for d in range(d0, d1):
            if mval[d] == s0:
                ag = 1
                break
        ne = 0
        te = -1
        for a in range(a0, a1):
            t = ntok[a]
            found = False
            for b in range(b0, b1):
                if ntok[b] == t:
                    found = True
                    break
            if not found:
                ne += 1
                te = t
        nm = 0
        tm = -1
        for b in range(b0, b1):
            t = ntok[b]
            found = False
            for a in range(a0, a1):
                if ntok[a] == t:
                    found = True
                    break
            if not found:
                nm += 1
                tm = t
        if ne == 1 and nm == 0:
            n_e[te] += 1
            agree_e[te] += ag
        if nm == 1 and ne == 0:
            n_m[tm] += 1
            agree_m[tm] += ag


def unsup_token_stats(st, q, s, prior=20.0):
    V = len(st.nm_idf)
    ae = np.zeros(V, np.int64); ne = np.zeros(V, np.int64)
    am = np.zeros(V, np.int64); nm = np.zeros(V, np.int64)
    _unsup_count(q, s, st.nm_ptr, st.nm_tok, st.num_ptr, st.num_val, ae, ne, am, nm)
    mu_e = ae.sum() / max(1, ne.sum())
    mu_m = am.sum() / max(1, nm.sum())
    re = ((ae + prior * mu_e) / (ne + prior)).astype(np.float32)
    rm = ((am + prior * mu_m) / (nm + prior)).astype(np.float32)
    return re, np.log1p(ne).astype(np.float32), rm, np.log1p(nm).astype(np.float32)


@nb.njit(parallel=True, cache=True)
def _unsup_pair(qi, si, ptr, tok, re, le, rm, lm):
    n = qi.shape[0]
    out = np.full((n, 4), -1.0, dtype=np.float32)
    for p in nb.prange(n):
        a0, a1 = ptr[qi[p]], ptr[qi[p] + 1]
        b0, b1 = ptr[si[p]], ptr[si[p] + 1]
        mn_e = 2.0
        sup_e = 0.0
        for a in range(a0, a1):
            t = tok[a]
            found = False
            for b in range(b0, b1):
                if tok[b] == t:
                    found = True
                    break
            if not found and re[t] < mn_e:
                mn_e = re[t]
                sup_e = le[t]
        mn_m = 2.0
        sup_m = 0.0
        for b in range(b0, b1):
            t = tok[b]
            found = False
            for a in range(a0, a1):
                if tok[a] == t:
                    found = True
                    break
            if not found and rm[t] < mn_m:
                mn_m = rm[t]
                sup_m = lm[t]
        if mn_e < 2.0:
            out[p, 0] = mn_e
            out[p, 1] = sup_e
        if mn_m < 2.0:
            out[p, 2] = mn_m
            out[p, 3] = sup_m
    return out


def unsup_pair_features(st, q, s, stats):
    o = _unsup_pair(q, s, st.nm_ptr, st.nm_tok, *stats)
    return {'u_extra_agree_min': o[:, 0], 'u_extra_support': o[:, 1],
            'u_miss_agree_min': o[:, 2], 'u_miss_support': o[:, 3]}


# ========================================================================================
# add_unsup.py
#
# Append the label-free token statistics (unsup.py) to existing feature
# parts of a split, writing a new feature directory.
# ========================================================================================

def add_unsup_main(split, src_tag, dst_tag):
    t0 = time.time()
    st = RecordStore(split)
    files = sorted(glob.glob(f'{WORK}/feat_{split}{src_tag}/part_*.parquet'))
    Q, S = [], []
    for f in files:
        d = pl.read_parquet(f, columns=['q', 's'])
        Q.append(d['q'].to_numpy()); S.append(d['s'].to_numpy())
    stats = unsup_token_stats(st, np.concatenate(Q), np.concatenate(S))
    del Q, S
    print(f'[{time.time()-t0:.0f}s] token stats ready', flush=True)
    out = f'{WORK}/feat_{split}{dst_tag}'
    os.makedirs(out, exist_ok=True)
    for f in files:
        df = pl.read_parquet(f)
        U = unsup_pair_features(st, df['q'].to_numpy(), df['s'].to_numpy(), stats)
        df.with_columns([pl.Series(k, v) for k, v in U.items()]).write_parquet(f'{out}/{os.path.basename(f)}')
    print(f'[{time.time()-t0:.0f}s] wrote {len(files)} parts to {out}', flush=True)


def add_unsup_cli():
    """`python add_unsup.py ...` of the original pipeline (reads sys.argv)."""
    add_unsup_main(sys.argv[1], sys.argv[2], sys.argv[3])


# ========================================================================================
# groupfeat.py
#
# Group-consistency features (label-free).
#
# A sibling business (second location, subsidiary) has several noisy copies of
# its own, so its deviations from the Source 1 record repeat across queries:
# the same different house number, the same inserted word. Random noise on a
# true copy produces a deviation no other query shares. For every candidate
# pair (q, s) we look at all queries that have s as a candidate and count how
# many share q's house number, q's name, or q's extra-token signature, and how
# many carry s's own house number and name.
# ========================================================================================

@nb.njit(cache=True)
def _name_hash(ptr, tok, r):
    """Order-free hash of a record's core token set."""
    h = np.uint64(1469598103934665603)
    acc = np.uint64(0)
    for a in range(ptr[r], ptr[r + 1]):
        x = np.uint64(tok[a]) * np.uint64(11400714819323198485) + np.uint64(12345)
        x ^= x >> np.uint64(29)
        acc += x  # sum is order-free; duplicates count twice (rare)
    return acc ^ h


@nb.njit(cache=True)
def _extra_sig(ptr, tok, q, s):
    """Hash of the tokens in q's core name that s's core name lacks (0 if none)."""
    acc = np.uint64(0)
    a0, a1 = ptr[q], ptr[q + 1]
    b0, b1 = ptr[s], ptr[s + 1]
    for a in range(a0, a1):
        t = tok[a]
        found = False
        for b in range(b0, b1):
            if tok[b] == t:
                found = True
                break
        if not found:
            x = np.uint64(t) * np.uint64(11400714819323198485) + np.uint64(777)
            x ^= x >> np.uint64(31)
            acc += x
    return acc


@nb.njit(parallel=True, cache=True)
def _group(gptr, q, s, nptr, ntok, mptr, mval):
    """Pairs sorted by s; gptr delimits groups. Returns (n, 7) features."""
    n = q.shape[0]
    out = np.zeros((n, 7), dtype=np.int16)
    ng = gptr.shape[0] - 1
    for g in nb.prange(ng):
        a, b = gptr[g], gptr[g + 1]
        m = b - a
        sr = s[a]
        s_num = mval[mptr[sr]] if mptr[sr + 1] > mptr[sr] else -1
        s_name = _name_hash(nptr, ntok, sr)
        qnum = np.empty(m, dtype=np.int64)
        qname = np.empty(m, dtype=np.uint64)
        qsig = np.empty(m, dtype=np.uint64)
        for i in range(m):
            qr = q[a + i]
            qnum[i] = mval[mptr[qr]] if mptr[qr + 1] > mptr[qr] else -1
            qname[i] = _name_hash(nptr, ntok, qr)
            qsig[i] = _extra_sig(nptr, ntok, qr, sr)
        n_snum = 0
        n_sname = 0
        for i in range(m):
            if s_num >= 0 and qnum[i] == s_num:
                n_snum += 1
            if qname[i] == s_name:
                n_sname += 1
        for i in range(m):
            c_num = 0
            c_name = 0
            c_sig = 0
            for j in range(m):
                if j == i:
                    continue
                if qnum[i] >= 0 and qnum[j] == qnum[i]:
                    c_num += 1
                if qname[j] == qname[i]:
                    c_name += 1
                if qsig[i] != 0 and qsig[j] == qsig[i]:
                    c_sig += 1
            own_num = 1 if (s_num >= 0 and qnum[i] == s_num) else 0
            own_name = 1 if qname[i] == s_name else 0
            out[a + i, 0] = min(m, 32767)
            out[a + i, 1] = min(c_num, 32767)            # others sharing q's first house number
            out[a + i, 2] = min(n_snum - own_num, 32767)  # others carrying s's house number
            out[a + i, 3] = min(c_name, 32767)           # others with q's exact core name
            out[a + i, 4] = min(n_sname - own_name, 32767)  # others with s's exact core name
            out[a + i, 5] = min(c_sig, 32767)            # others with the same extra-token signature
            out[a + i, 6] = 1 if (qnum[i] >= 0 and s_num >= 0 and qnum[i] != s_num) else 0
    return out


def groupfeat_group_features(st: RecordStore, q, s):
    """q, s aligned arrays for all candidate pairs of a split. Returns dict of
    arrays aligned with the input order."""
    order = np.argsort(s, kind='stable')
    qs, ss = q[order], s[order]
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    gptr = np.r_[starts, len(ss)].astype(np.int64)
    o = _group(gptr, qs, ss, st.nm_ptr, st.nm_tok, st.num_ptr, st.num_val)
    del qs, ss
    names = ['g_size', 'g_qnum_shared', 'g_snum_support', 'g_qname_shared', 'g_sname_support',
             'g_extra_sig_shared', 'g_num_deviates']
    out = {}
    for i, k in enumerate(names):
        col = np.empty(len(q), dtype=np.int16)
        col[order] = o[:, i]
        out[k] = col
    return out


# ========================================================================================
# add_group.py
#
# Append group-consistency features (groupfeat.py) to existing feature parts
# of a split, writing a new feature directory.
# ========================================================================================

def add_group_main(split, src_tag, dst_tag):
    t0 = time.time()
    st = RecordStore(split)
    files = sorted(glob.glob(f'{WORK}/feat_{split}{src_tag}/part_*.parquet'))
    Q, S, N = [], [], []
    for f in files:
        d = pl.read_parquet(f, columns=['q', 's'])
        Q.append(d['q'].to_numpy()); S.append(d['s'].to_numpy()); N.append(d.height)
    G = groupfeat_group_features(st, np.concatenate(Q), np.concatenate(S))
    del Q, S
    print(f'[{time.time()-t0:.0f}s] group features ready', flush=True)
    out = f'{WORK}/feat_{split}{dst_tag}'
    os.makedirs(out, exist_ok=True)
    off = 0
    for f, n in zip(files, N):
        df = pl.read_parquet(f)
        df.with_columns([pl.Series(k, v[off:off + n]) for k, v in G.items()]).write_parquet(f'{out}/{os.path.basename(f)}')
        off += n
    print(f'[{time.time()-t0:.0f}s] wrote {len(files)} parts to {out}', flush=True)


def add_group_cli():
    """`python add_group.py ...` of the original pipeline (reads sys.argv)."""
    add_group_main(sys.argv[1], sys.argv[2], sys.argv[3])


# ========================================================================================
# build_loco.py
#
# Leave-one-country-out risk features for training the unseen-country
# (France) model.
#
# France has no labels, so at test time its token and legal-form risks come
# from other countries' statistics plus the fallback in tokrisk.from_table.
# To train a model that expects exactly that, each train country's pairs get
# risks computed only from the *other* train country, with the same fallback.
# In the US-train / India-test proxy this lifted the unseen-country macro
# F0.5 from 0.927 to 0.946.
# ========================================================================================

def country_stats(st, files, mask_rows):
    cnt = np.zeros((len(st.nm_idf), 4), dtype=np.int64)
    lp = ln = 0
    npos = nneg = 0
    for f in files:
        df = pl.read_parquet(f, columns=['q', 's', 'y'])
        q, s, y = df['q'].to_numpy(), df['s'].to_numpy(), df['y'].to_numpy()
        m = mask_rows[q]
        cnt += tokrisk_counts(st, q[m], s[m], y[m])
        a, b = leg_counts(st.leg[q[m]], st.leg[s[m]], y[m])
        lp = lp + a; ln = ln + b
        npos += int(y[m].sum()); nneg += int(m.sum() - y[m].sum())
    return cnt, npos, nneg, lp, ln


def build_loco_main(src_tag, dst_tag):
    t0 = time.time()
    st = RecordStore('train')
    files = sorted(glob.glob(f'{WORK}/feat_train{src_tag}/part_*.parquet'))
    names = {0: 'US', 1: 'India'}
    risks = {}
    for c in (0, 1):
        cnt, npos, nneg, lp, ln = country_stats(st, files, st.country == c)
        re_, rm_ = tokrisk_risk(cnt, npos, nneg)
        table = to_table(st, re_, rm_).filter(pl.col('country') == names[c])
        # statistics of country c, applied to the *other* country with fallback
        risks[1 - c] = from_table(st, table) + (leg_risk(lp, ln),)
    print(f'[{time.time()-t0:.0f}s] cross-country risks ready', flush=True)
    out = f'{WORK}/feat_train{dst_tag}'
    os.makedirs(out, exist_ok=True)
    for f in files:
        df = pl.read_parquet(f)
        q, s = df['q'].to_numpy(), df['s'].to_numpy()
        cq = st.country[q]
        cols = {}
        for c in (0, 1):
            m = cq == c
            if not m.any():
                continue
            re_, rm_, rl_ = risks[c]
            R = tokrisk_pair_features(st, q[m], s[m], re_, rm_)
            R['leg_pair_risk'] = leg_pair(st.leg[q[m]], st.leg[s[m]], rl_)
            for k, v in R.items():
                if k not in cols:
                    cols[k] = df[k].to_numpy().copy()
                cols[k][m] = v
        df.with_columns([pl.Series(k, v) for k, v in cols.items()]).write_parquet(f'{out}/{os.path.basename(f)}')
    print(f'[{time.time()-t0:.0f}s] wrote {len(files)} parts to {out}', flush=True)


def build_loco_cli():
    """`python build_loco.py ...` of the original pipeline (reads sys.argv)."""
    build_loco_main(sys.argv[1], sys.argv[2])


# ========================================================================================
# stage2.py
#
# Stage-2 context features from stage-1 probabilities.
#
# For each pair (q, s) with stage-1 probability p:
#   query side : best competing p among q's other candidates, p rank, sum of p
#   entity side: how many other queries already pick s confidently, the best
#                and total p of s's other queries
# Train uses out-of-fold stage-1 p; test uses the averaged fold models.
# Rows stay aligned with the input order. Implemented with numpy/numba
# aggregates so memory stays at a few arrays of the pair count.
# ========================================================================================

@nb.njit(cache=True)
def _query_side(q, p, prank, pmax_other, psum):
    """Pairs grouped contiguously by q (feature parts are query-aligned)."""
    n = q.shape[0]
    a = 0
    while a < n:
        b = a
        while b < n and q[b] == q[a]:
            b += 1
        m1 = -1.0
        m2 = -1.0
        i1 = -1
        tot = 0.0
        for i in range(a, b):
            tot += p[i]
            if p[i] > m1:
                m2 = m1
                m1 = p[i]
                i1 = i
            elif p[i] > m2:
                m2 = p[i]
        for i in range(a, b):
            r = 1
            for j in range(a, b):
                if p[j] > p[i] or (p[j] == p[i] and j < i):
                    r += 1
            prank[i] = r
            pmax_other[i] = (m2 if m2 >= 0 else 0.0) if i == i1 else m1
            psum[i] = tot
        a = b


@nb.njit(cache=True)
def _entity_aggregates(s, p, best, n_s):
    ncand = np.zeros(n_s, np.float32)
    nstrong = np.zeros(n_s, np.float32)
    bestpsum = np.zeros(n_s, np.float32)
    psum = np.zeros(n_s, np.float32)
    top1 = np.zeros(n_s, np.float32)
    top2 = np.zeros(n_s, np.float32)
    for i in range(s.shape[0]):
        k = s[i]
        ncand[k] += 1
        psum[k] += p[i]
        if best[i]:
            bestpsum[k] += p[i]
            if p[i] >= 0.5:
                nstrong[k] += 1
        if p[i] > top1[k]:
            top2[k] = top1[k]
            top1[k] = p[i]
        elif p[i] > top2[k]:
            top2[k] = p[i]
    return ncand, nstrong, bestpsum, psum, top1, top2


def stage2_context(q, s, p):
    q = np.asarray(q); s = np.asarray(s); p = np.asarray(p, dtype=np.float32)
    n = len(q)
    if n > 1 and not (np.diff(q) != 0).sum() + 1 == len(np.unique(q)):
        raise ValueError('pairs must be grouped by query')
    prank = np.zeros(n, np.float32); pmax_other = np.zeros(n, np.float32); psum_q = np.zeros(n, np.float32)
    _query_side(q, p, prank, pmax_other, psum_q)
    best = prank == 1
    n_s = int(s.max()) + 1
    ncand, nstrong, bestpsum, psum_s, top1, top2 = _entity_aggregates(s, p, best, n_s)
    strong_i = (best & (p >= 0.5)).astype(np.float32)
    bestp_i = np.where(best, p, 0).astype(np.float32)
    out = {
        'q_prank': prank,
        'q_psum': psum_q,
        'q_pmax_other': pmax_other,
        's_ncand': ncand[s],
        's_nstrong_other': nstrong[s] - strong_i,
        's_bestpsum_other': bestpsum[s] - bestp_i,
        's_psum_other': psum_s[s] - p,
        's_pmax_other': np.where(p >= top1[s], top2[s], top1[s]).astype(np.float32),
    }
    out['p1'] = p
    out['p1_margin'] = (p - pmax_other).astype(np.float32)
    return out


# ========================================================================================
# train_model.py
#
# 2-fold cross-fitted LightGBM on train pair features, OOF macro F0.5.
#
# Folds are by Source 1 entity: pairs whose Source 1 record is in fold f are
# predicted by the model trained on the other fold. Assignment is many-to-one
# (each query keeps its best candidate) and a pair is kept when p >= thr.
#
# Stage 2 (ER_STAGE2=<stage-1 tag>) adds context features computed from the
# stage-1 out-of-fold probabilities (see stage2.py).
# ========================================================================================

DROP = {'q', 's', 'y', 'country'}
# extra features to leave out: exact names, or prefixes ending in '*' (e.g. g_*)
_EXTRA_DROP = [x for x in os.environ.get('ER_DROP_FEATS', '').split(',') if x]


def dropped(c):
    return c in DROP or any(c == x or (x.endswith('*') and c.startswith(x[:-1])) for x in _EXTRA_DROP)
MAX_TRAIN = int(os.environ.get('ER_MAX_TRAIN', '8000000'))
ROUNDS = int(os.environ.get('ER_ROUNDS', '800'))
# ER_SEED > 0 gives an independent copy (row sample, bagging, feature sampling) for seed averaging
SEED = int(os.environ.get('ER_SEED', '0'))
train_model_PARAMS = dict(objective='binary', learning_rate=float(os.environ.get('ER_LR', '0.08')),
              num_leaves=int(os.environ.get('ER_LEAVES', '255')),
              min_data_in_leaf=int(os.environ.get('ER_MIN_LEAF', '200')),
              feature_fraction=float(os.environ.get('ER_FF', '0.7')), bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
              max_bin=255, verbose=-1, num_threads=8)
if SEED:
    train_model_PARAMS.update(seed=SEED, bagging_seed=SEED + 1, feature_fraction_seed=SEED + 2, data_random_seed=SEED + 3)


def load_parts(split):
    return sorted(glob.glob(f'{WORK}/feat_{split}{os.environ.get("ER_FEAT", "")}/part_*.parquet'))


def part_frames(files, cols, extra):
    """Yield (df, extra_slice) per part; extra arrays are aligned with the
    concatenation of all parts."""
    off = 0
    for f in files:
        df = pl.read_parquet(f, columns=cols)
        n = df.height
        if extra:
            df = df.with_columns([pl.Series(k, v[off:off + n]) for k, v in extra.items()])
        off += n
        yield df


def train_model_main(tag, extra=None):
    t0 = time.time()
    files = load_parts('train')
    schema = pl.read_parquet_schema(files[0])
    base = [c for c in schema if not dropped(c)]
    if extra:
        extra = {k: v for k, v in extra.items() if not dropped(k)}
    feats = base + (list(extra) if extra else [])
    print(f'{len(files)} parts, {len(feats)} features', flush=True)
    n_total = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in files)
    frac = min(1.0, 2 * MAX_TRAIN / n_total)
    print(f'{n_total} pairs, sampling {frac:.3f} for training', flush=True)
    models = {}
    for k in (0, 1):
        rng = np.random.default_rng(k + 1000 * SEED)
        Xl, yl = [], []
        for df in part_frames(files, base + ['s', 'y'], extra):
            m = (fold_of(df['s'].to_numpy()) == k) & (rng.random(df.height) < frac)
            Xl.append(df.filter(pl.Series(m)).select(feats).to_numpy().astype(np.float32))
            yl.append(df['y'].to_numpy()[m])
            del df
        X = np.concatenate(Xl); y = np.concatenate(yl)
        del Xl, yl
        perm = rng.permutation(len(y))
        X, y = X[perm], y[perm]
        nval = min(500000, len(y) // 10)
        dtr = lgb.Dataset(X[nval:], y[nval:], feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(X[:nval], y[:nval], reference=dtr)
        print(f'[{time.time()-t0:.0f}s] fold-model {k}: train {len(y)-nval} pos {y.mean():.3f}', flush=True)
        m = lgb.train(train_model_PARAMS, dtr, ROUNDS, valid_sets=[dva], callbacks=[lgb.log_evaluation(100), lgb.early_stopping(50)])
        models[k] = m
        del X, y, dtr, dva
        m.save_model(f'{WORK}/lgb_{tag}_fold{k}.txt')
    Q, S, P, Y = [], [], [], []
    for df in part_frames(files, base + ['q', 's', 'y'], extra):
        X = df.select(feats).to_numpy().astype(np.float32)
        fo = fold_of(df['s'].to_numpy())
        p = np.empty(df.height, dtype=np.float32)
        for k in (0, 1):
            mk = fo == k
            if mk.any():
                p[mk] = models[1 - k].predict(X[mk], num_threads=8)
        Q.append(df['q'].to_numpy()); S.append(df['s'].to_numpy()); P.append(p); Y.append(df['y'].to_numpy())
    q = np.concatenate(Q); s = np.concatenate(S); p = np.concatenate(P); y = np.concatenate(Y)
    np.savez(f'{WORK}/oof_{tag}.npz', q=q, s=s, p=p, y=y)
    print(f'[{time.time()-t0:.0f}s] OOF done: {len(q)} pairs', flush=True)
    report(q, s, p, tag)
    imp = sorted(zip(feats, models[0].feature_importance('gain')), key=lambda x: -x[1])
    print('top features:', [(a, int(b)) for a, b in imp[:30]])


def report(q, s, p, tag, thrs=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    ts = true_s1()
    norm = pl.read_parquet(f'{WORK}/norm_train.parquet', columns=['src', 'country'])
    src = norm['src'].to_numpy(); country = norm['country'].to_numpy()
    cs = np.unique(country[q])
    s1_mask = (src == 1) & np.isin(country, cs)
    uq, bs, bp = assign(q, s, p)
    res = {}
    for thr in thrs:
        m = bp >= thr
        f, _ = macro_f05(uq[m], bs[m], ts, s1_mask)
        res[thr] = f
        per = []
        for cc in cs:
            fc, _ = macro_f05(uq[m], bs[m], ts, s1_mask & (country == cc))
            per.append(f'{cc}={fc:.4f}')
        print(f'  thr {thr:.2f}: macro F0.5 {f:.4f}  ' + ' '.join(per), flush=True)
    json.dump({str(k): v for k, v in res.items()}, open(f'{WORK}/oof_{tag}_scores.json', 'w'))


def train_model_cli():
    """`python train_model.py ...` of the original pipeline (reads sys.argv)."""
    tag = sys.argv[1] if len(sys.argv) > 1 else 'v1'
    extra = None
    st1 = os.environ.get('ER_STAGE2')
    if st1:
        z = np.load(f'{WORK}/oof_{st1}.npz')
        extra = stage2_context(z['q'], z['s'], z['p'])
        print('stage-2 context from', st1, list(extra), flush=True)
    train_model_main(tag, extra)


# ========================================================================================
# decide.py
#
# Per-entity expected-F0.5 decision.
#
# After many-to-one assignment each Source 1 entity s holds the queries whose
# best candidate is s, with probabilities p (sorted descending). Keeping the
# top k gives, with X = true among kept and Y = true among dropped,
#     F0.5 = 1.25 X / (0.25 (X + Y) + k)        (k >= 1)
#     F0.5 = 1 if X + Y == 0 else 0             (k == 0)
# Treating the p as independent, X and Y are Poisson-binomial; we pick the k
# with the highest expectation.
# ========================================================================================

@nb.njit(cache=True)
def _pb(p):
    """Poisson-binomial pmf of the number of successes."""
    n = p.shape[0]
    f = np.zeros(n + 1)
    f[0] = 1.0
    for i in range(n):
        for j in range(i + 1, 0, -1):
            f[j] = f[j] * (1 - p[i]) + f[j - 1] * p[i]
        f[0] *= (1 - p[i])
    return f


@nb.njit(cache=True)
def best_k(p, extra_true=0.0):
    """p sorted descending. extra_true: expected true links outside the
    list (added to Y as a constant). Returns (k, expected F)."""
    n = p.shape[0]
    best = 0
    # k = 0: F = 1 only if nothing is true
    prod = 1.0
    for i in range(n):
        prod *= (1 - p[i])
    best_v = prod if extra_true == 0.0 else prod * np.exp(-extra_true)
    for k in range(1, n + 1):
        fx = _pb(p[:k])
        fy = _pb(p[k:])
        v = 0.0
        for x in range(1, k + 1):
            if fx[x] == 0.0:
                continue
            for y in range(0, n - k + 1):
                if fy[y] == 0.0:
                    continue
                v += fx[x] * fy[y] * 1.25 * x / (0.25 * (x + y + extra_true) + k)
        if v > best_v:
            best_v = v
            best = k
    return best, best_v


@nb.njit(parallel=True, cache=True)
def _decide(ptr, p_sorted, extra_true, max_n):
    ng = ptr.shape[0] - 1
    keep_k = np.zeros(ng, dtype=np.int32)
    for g in nb.prange(ng):
        a, b = ptr[g], ptr[g + 1]
        n = b - a
        if n > max_n:  # huge groups: fall back to p >= 0.5
            k = 0
            for i in range(a, b):
                if p_sorted[i] >= 0.5:
                    k += 1
            keep_k[g] = k
            continue
        k, _ = best_k(p_sorted[a:b], extra_true)
        keep_k[g] = k
    return keep_k


def decide_decide(uq, bs, bp, floor=0.05, extra_true=0.0, max_n=40):
    """uq, bs, bp: assigned (query, Source 1, p). Returns a boolean keep mask."""
    m = bp >= floor
    idx = np.where(m)[0]
    order = idx[np.lexsort((-bp[idx], bs[idx]))]
    s_sorted = bs[order]
    p_sorted = bp[order].astype(np.float64)
    starts = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
    ptr = np.r_[starts, len(order)].astype(np.int64)
    kk = _decide(ptr, p_sorted, extra_true, max_n)
    rank = np.arange(len(order)) - np.repeat(ptr[:-1], np.diff(ptr))
    keep_sorted = rank < np.repeat(kk, np.diff(ptr))
    keep = np.zeros(len(uq), dtype=bool)
    keep[order[keep_sorted]] = True
    return keep


# ========================================================================================
# predict.py
#
# Score test candidates, assign, and write the two submission files.
#
# p = mean of the two cross-fitted fold models. Each Source 2/3 record keeps
# its best-scoring Source 1 candidate when p >= thr (many-to-one). Every test
# Source 1 entity gets exactly one row.
#
# candidate_pairs.tsv lists exactly the pairs the model scored (the pruned
# candidate set), grouped by Source 1 entity.
# ========================================================================================

def predict_score(tag, split='test', feat_tag='', extra=None):
    files = sorted(glob.glob(f'{WORK}/feat_{split}{feat_tag}/part_*.parquet'))
    models = [lgb.Booster(model_file=f'{WORK}/lgb_{tag}_fold{k}.txt') for k in (0, 1)]
    feats = models[0].feature_name()
    base = [c for c in feats if not (extra and c in extra)]
    Q, S, P = [], [], []
    t0 = time.time()
    off = 0
    for f in files:
        df = pl.read_parquet(f, columns=base + ['q', 's'])
        if extra:
            df = df.with_columns([pl.Series(k, v[off:off + df.height]) for k, v in extra.items()])
        off += df.height
        X = df.select(feats).to_numpy().astype(np.float32)
        p = np.mean([m.predict(X, num_threads=8) for m in models], axis=0).astype(np.float32)
        Q.append(df['q'].to_numpy()); S.append(df['s'].to_numpy()); P.append(p)
        print(f'  scored {os.path.basename(f)} ({time.time()-t0:.0f}s)', flush=True)
    q, s, p = np.concatenate(Q), np.concatenate(S), np.concatenate(P)
    np.savez(f'{WORK}/pred_{split}_{tag}.npz', q=q, s=s, p=p)
    return q, s, p


def write_outputs(q, s, p, thr, split='test', out_dir=OUT, thr_by_country=None, expf_countries=None):
    """thr_by_country: optional {country: threshold} overriding thr.
    expf_countries: countries decided by the per-entity expected-F0.5 rule
    (decide.py) instead of a threshold."""
    os.makedirs(out_dir, exist_ok=True)
    norm = pl.read_parquet(f'{WORK}/norm_{split}.parquet', columns=['id', 'src', 'country'])
    ids = norm['id']
    s1 = norm.with_row_index('row').filter(pl.col('src') == 1).select(pl.col('row').cast(pl.Int32), pl.col('id').alias('source1_entity_id'))
    uq, bs, bp = assign(q, s, p)
    t = np.full(len(uq), thr, dtype=np.float32)
    cq = norm['country'].to_numpy()[uq]
    if thr_by_country:
        for c, v in thr_by_country.items():
            t[cq == c] = v
    keep = bp >= t
    if expf_countries:
        m = np.isin(cq, expf_countries)
        ke = decide_decide(uq, bs, bp)
        keep = np.where(m, ke, keep)
    m = (pl.DataFrame({'row': bs[keep], 'qid': ids.gather(uq[keep])})
         .group_by('row').agg(pl.col('qid').sort().str.join(',').alias('matched_entity_ids')))
    res = s1.join(m, on='row', how='left').with_columns(pl.col('matched_entity_ids').fill_null(''))
    res.select('source1_entity_id', 'matched_entity_ids').write_csv(f'{out_dir}/matching_results.tsv', separator='\t', quote_style='never')
    c = (pl.DataFrame({'row': s, 'qid': ids.gather(q)}).unique()
         .group_by('row').agg(pl.col('qid').sort().str.join(',').alias('candidate_entity_ids')))
    cres = s1.join(c, on='row', how='left').with_columns(pl.col('candidate_entity_ids').fill_null(''))
    cres.select('source1_entity_id', 'candidate_entity_ids').write_csv(f'{out_dir}/candidate_pairs.tsv', separator='\t', quote_style='never')
    n_match = (res['matched_entity_ids'] != '').sum()
    print(f'wrote {res.height} rows; {n_match} with >=1 match ({res.height - n_match} empty); '
          f'{keep.sum()} matched ids; thr={thr}', flush=True)


def predict_cli():
    """`python predict.py ...` of the original pipeline (reads sys.argv)."""
    tag = sys.argv[1]
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
    by_country = None
    if len(sys.argv) > 3 and sys.argv[3]:  # e.g. France=0.5,US=0.7
        by_country = {kv.split('=')[0]: float(kv.split('=')[1]) for kv in sys.argv[3].split(',')}
    expf = os.environ.get('ER_EXPF', '').split(',') if os.environ.get('ER_EXPF') else None
    cand_keep = os.environ.get('ER_CAND_KEEP')  # boolean mask from cand_filter.py
    if os.path.exists(f'{WORK}/pred_test_{tag}.npz') and os.environ.get('ER_RESCORE') != '1':
        z = np.load(f'{WORK}/pred_test_{tag}.npz'); q, s, p = z['q'], z['s'], z['p']
    else:
        extra = None
        st1 = os.environ.get('ER_STAGE2')
        if st1:
            z = np.load(f'{WORK}/pred_test_{st1}.npz')
            extra = stage2_context(z['q'], z['s'], z['p'])
        q, s, p = predict_score(tag, extra=extra)
    if cand_keep:
        k = np.load(cand_keep)
        assert len(k) == len(q)
        q, s, p = q[k], s[k], p[k]
    write_outputs(q, s, p, thr, thr_by_country=by_country, expf_countries=expf)


# ========================================================================================
# merge_preds.py
#
# Combine test predictions: rows whose query is in `country` take p from the
# country-specific model, all others from the base model. Both prediction
# files must come from the same candidate order (same feature-part lineage).
# ========================================================================================

def merge_preds_main(base_tag, special_tag, country, out_tag):
    a = np.load(f'{WORK}/pred_test_{base_tag}.npz')
    b = np.load(f'{WORK}/pred_test_{special_tag}.npz')
    assert (a['q'] == b['q']).all() and (a['s'] == b['s']).all()
    c = pl.read_parquet(f'{WORK}/norm_test.parquet', columns=['country'])['country'].to_numpy()
    m = c[a['q']] == country
    p = np.where(m, b['p'], a['p']).astype(np.float32)
    np.savez(f'{WORK}/pred_test_{out_tag}.npz', q=a['q'], s=a['s'], p=p)
    print(f'{m.sum()} {country} pairs from {special_tag}, {(~m).sum()} from {base_tag}')


def merge_preds_cli():
    """`python merge_preds.py ...` of the original pipeline (reads sys.argv)."""
    merge_preds_main(*sys.argv[1:5])


# ========================================================================================
# score_v7.py
#
# Score v7 test features: stage 1 (t1) and stage 2 (t2) for all rows, the
# France models (tfr, then tfr2) for France rows; merge into pred_test_v7.
# ========================================================================================


def score_v7_main():
    FT = '_sug'
    q, s, p = predict_score('t1', feat_tag=FT)
    predict_score('t2', feat_tag=FT, extra=stage2_context(q, s, p))
    q, s, p = predict_score('tfr', feat_tag=FT)
    predict_score('tfr2', feat_tag=FT, extra=stage2_context(q, s, p))
    merge_preds_main('t2', 'tfr2', 'France', 'v7')


# ========================================================================================
# france_noise_fix.py
#
# France noise-word correction.
#
# Measured on test France candidate pairs on the same street (same address
# token set) whose names differ by one added query word:
#   - Participations / Holding / Distribution / International: ~16k pairs each,
#     house number agrees in 0.2-0.3% -> sibling businesses at a nearby number.
#   - Fils / Associes / Compagnie / Services: 0.5-1.9k pairs each, house number
#     agrees in 86-89% -> the true-copy noise words (the French counterpart of
#     the US/India Center/Services/Service/Partners list, which is ~100% true at
#     the same address in train).
#   - Groupe / Developpement / France: ~16k sibling pairs (0.3% same number)
#     plus ~850 extra same-number pairs each -> on both lists.
# The France model accepts "services" at the same number (p 0.998) but rejects
# the other noise words (p ~0.001), because they never occur in US/India train.
#
# Rule: a France pair whose address token set and first house number agree,
# whose query name adds only words from the noise list (at least one French
# one) and drops at most one word, and whose legal forms do not conflict, gets
# p = max(p, P_FIX).
#
#     python france_noise_fix.py <in_tag> <out_tag>
# ========================================================================================

NOISE_FR = ['fils', 'associes', 'compagnie', 'groupe', 'developpement', 'france']
NOISE_OK = NOISE_FR + ['services', 'service']
NOISE_FR_SWAP = [w for w in NOISE_FR if w != 'compagnie']
NOISE_SWAP_OK = NOISE_FR_SWAP + ['services', 'service']
P_FIX = float(os.environ.get('ER_P_FIX', '0.97'))
RELAX = os.environ.get('ER_FIX_RELAX', '1') == '1'
GENERIC_DF = 1500
france_noise_fix_COUNTRY = 'France'


def fix_mask(q, s, split='test'):
    """Boolean mask over the (q, s) pairs that the rule applies to."""
    norm = pl.read_parquet(f'{WORK}/norm_{split}.parquet', columns=['src', 'country', 'a_toks', 'a_nums', 'n_core', 'n_toks'])
    # address tokens frequent among the country's Source 1 addresses (street types, cities,
    # regions) do not identify a street on their own
    generic = set(norm.filter((pl.col('src') == 1) & (pl.col('country') == france_noise_fix_COUNTRY))
                  .select(pl.col('a_toks').list.unique()).explode('a_toks', empty_as_null=True)
                  .group_by('a_toks').len().filter(pl.col('len') > GENERIC_DF)['a_toks'].to_list())
    leg = {t: i + 1 for i, t in enumerate(LEGAL_ORDER)}
    rec = norm.select(
        pl.col('country'),
        pl.col('a_toks').list.sort().list.join(' ').hash(1).alias('ah'),
        (pl.col('a_toks').list.len() > 0).alias('has'),
        pl.col('a_nums').list.first().fill_null('').alias('n0'),
        pl.col('n_toks').list.eval(pl.element().replace_strict(leg, default=99, return_dtype=pl.Int8)).list.min().fill_null(99).alias('leg'))
    core = norm['n_core']
    atoks = norm['a_toks']
    del norm
    ctry = rec['country'].to_numpy(); ah = rec['ah'].to_numpy(); has = rec['has'].to_numpy()
    n0 = rec['n0'].to_numpy(); lg = rec['leg'].to_numpy(); lg[lg == 99] = 0
    m = (ctry[q] == france_noise_fix_COUNTRY) & has[q] & has[s] & (n0[q] != '') & (n0[q] == n0[s])
    if not RELAX:
        m &= ah[q] == ah[s]
    m &= (lg[q] == 0) | (lg[s] == 0) | (lg[q] == lg[s])
    idx = np.flatnonzero(m)
    d = pl.DataFrame({'i': idx, 'qc': core.gather(q[idx]), 'sc': core.gather(s[idx])})
    if RELAX:
        d = d.with_columns(atoks.gather(q[idx]).alias('qa'), atoks.gather(s[idx]).alias('sa'))
    d = d.with_columns(pl.col('qc').list.set_difference('sc').alias('ex'), pl.col('sc').list.set_difference('qc').alias('mi'))
    # added words only from the noise list, at least one French one; when a word is also
    # dropped (swap), "compagnie" is excluded: as a swap it is a category word (house number
    # agrees in only 42% of same-street pairs, like club/ecole/amicale)
    sup = pl.col('mi').list.len() == 0
    swap = pl.col('mi').list.len() == 1
    d = d.filter((pl.col('ex').list.len() > 0)
                 & ((sup & pl.col('ex').list.eval(pl.element().is_in(NOISE_OK)).list.all()
                     & pl.col('ex').list.eval(pl.element().is_in(NOISE_FR)).list.any())
                    | (swap & pl.col('ex').list.eval(pl.element().is_in(NOISE_SWAP_OK)).list.all()
                       & pl.col('ex').list.eval(pl.element().is_in(NOISE_FR_SWAP)).list.any())))
    if RELAX:
        # same address up to an omitted component (one token set contains the other)
        # or one misspelt token (all others shared, the odd one close to an unmatched token)
        # identical addresses need nothing more; partial or misspelt ones must share a
        # distinctive token
        keep = [_addr_close(a, b) and (_canon(a) == _canon(b) or bool((set(a) & set(b)) - generic))
                for a, b in zip(d['qa'].to_list(), d['sa'].to_list())]
        d = d.filter(pl.Series(keep))
    out = np.zeros(len(q), bool)
    out[d['i'].to_numpy()] = True
    # a query that fits the rule with two or more Source 1 records is ambiguous: leave it
    qq = q[out]
    uq_, cnt = np.unique(qq, return_counts=True)
    amb = np.isin(q, uq_[cnt > 1]) & out
    out &= ~amb
    print(f'rule: {out.sum()} pairs, {len(np.unique(q[amb]))} ambiguous queries skipped', flush=True)
    return out


ADDR_EQUIV = {'b': 'bis'}  # "9 B Rue Neuve" = "9 bis Rue Neuve"


def _canon(a):
    return {ADDR_EQUIV.get(t, t) for t in a}


def _addr_close(a, b):
    a, b = _canon(a), _canon(b)
    if a == b or a <= b or b <= a:
        return True
    xa, xb = a - b, b - a
    if len(xa) > len(xb):
        xa, xb = xb, xa
    if len(xa) != 1 or len(xb) > 3:
        return False
    u = next(iter(xa))
    return len(u) >= 4 and any(len(v) >= 4 and Levenshtein.normalized_similarity(u, v) >= 0.7 for v in xb)


def france_noise_fix_main(in_tag, out_tag):
    z = np.load(f'{WORK}/pred_test_{in_tag}.npz')
    q, s, p = z['q'], z['s'], z['p']
    m = fix_mask(q, s)
    # ties between two rule pairs of one query go to the higher model score
    p2 = np.where(m, np.maximum(p, P_FIX + 0.02 * p), p).astype(np.float32)
    print(f'{m.sum()} France pairs matched the rule; {(p[m] < 0.5).sum()} of them had p < 0.5')
    np.savez(f'{WORK}/pred_test_{out_tag}.npz', q=q, s=s, p=p2)


def france_noise_fix_cli():
    """`python france_noise_fix.py ...` of the original pipeline (reads sys.argv)."""
    france_noise_fix_main(sys.argv[1], sys.argv[2])


# ========================================================================================
# france_selftrain.py
#
# Self-training for France (no labels).
#
# Round 0 = the leave-one-country-out France model's test predictions.
# Confident France decisions become pseudo-labels (assigned best pair with
# p >= HI -> 1, p <= LO -> 0). France token and legal-form risks are then
# learned from those pseudo-labels (cross-fitted by Source 1 fold; words with
# no pseudo-label evidence keep their round-0 fallback value), and a model
# trained on US/India labels (leave-one-country-out features) plus France
# pseudo-labels rescores the France pairs out-of-fold.
#
# In the US-train / India-as-unlabeled proxy this lifted India macro F0.5 from
# 0.9419 to 0.9473 (thr 0.5) and 0.9368 to 0.9460 (thr 0.7).
# ========================================================================================

HI = float(os.environ.get('ER_HI', '0.95'))
LO = float(os.environ.get('ER_LO', '0.05'))
N_TRAIN = int(os.environ.get('ER_N_TRAIN', '5000000'))
N_FR = int(os.environ.get('ER_N_FR', '4000000'))
france_selftrain_COUNTRY = 'France'


def france_selftrain_fit(X, y, feats):
    nval = min(300000, len(y) // 10)
    return lgb.train(train_model_PARAMS, lgb.Dataset(X[nval:], y[nval:], feature_name=feats), 800,
                     valid_sets=[lgb.Dataset(X[:nval], y[:nval])],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def france_selftrain_main(round0_tag, test_feat, train_loco_feat, base_tag, out_tag):
    t0 = time.time()
    te_files = sorted(glob.glob(f'{WORK}/feat_test{test_feat}/part_*.parquet'))
    tr_files = sorted(glob.glob(f'{WORK}/feat_train{train_loco_feat}/part_*.parquet'))
    feats = [c for c in pl.read_parquet_schema(tr_files[0]) if not dropped(c)]
    st = RecordStore('test')
    ctry = pl.read_parquet(f'{WORK}/norm_test.parquet', columns=['country'])['country'].to_numpy()
    is_fr = ctry == france_selftrain_COUNTRY
    # round-0 predictions (all test pairs, feature-part order)
    z = np.load(f'{WORK}/pred_test_{round0_tag}.npz')
    q_all, s_all, p_all = z['q'], z['s'], z['p']
    frm = is_fr[q_all]
    q, s, p0 = q_all[frm], s_all[frm], p_all[frm]
    uq, bs, bp = assign(q, s, p0)
    best_s = np.full(len(ctry), -1, np.int64); best_s[uq] = bs
    best_p = np.zeros(len(ctry), np.float32); best_p[uq] = bp
    pos = (s == best_s[q]) & (best_p[q] >= HI)
    neg = (p0 <= LO) & ~pos
    lab = np.full(len(q), -1, np.int8); lab[neg] = 0; lab[pos] = 1
    keep = lab >= 0
    print(f'[{time.time()-t0:.0f}s] France pairs {len(q)}: pseudo pos {pos.sum()} neg {neg.sum()}', flush=True)
    # round-0 risks for France (fallback from train tables)
    table = pl.read_parquet(f'{WORK}/tokrisk_s.parquet')
    re0, rm0 = from_table(st, table)
    fo = fold_of(s)
    risk_k = {}
    for k in (0, 1):
        src = keep & (fo == 1 - k)
        c = tokrisk_counts(st, q[src], s[src], lab[src].astype(np.int64))
        re_, rm_ = tokrisk_risk(c, int(lab[src].sum()), int(src.sum() - lab[src].sum()))
        re_ = np.where((c[:, 0] + c[:, 1]) > 0, re_, re0).astype(np.float32)
        rm_ = np.where((c[:, 2] + c[:, 3]) > 0, rm_, rm0).astype(np.float32)
        a, b = leg_counts(st.leg[q[src]], st.leg[s[src]], lab[src].astype(np.int64))
        risk_k[k] = (re_, rm_, leg_risk(a, b))
    # US/India training rows (true labels, leave-one-country-out features)
    rng = np.random.default_rng(0)
    n_tr = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in tr_files)
    frac = min(1.0, N_TRAIN / n_tr)
    Xt, yt = [], []
    for f in tr_files:
        d = pl.read_parquet(f, columns=feats + ['y'])
        m = rng.random(d.height) < frac
        Xt.append(d.filter(pl.Series(m)).select(feats).to_numpy().astype(np.float32)); yt.append(d['y'].to_numpy()[m])
    Xt = np.concatenate(Xt); yt = np.concatenate(yt)
    print(f'[{time.time()-t0:.0f}s] train rows {len(yt)}', flush=True)

    def patched(d, k):
        qq, ss = d['q'].to_numpy(), d['s'].to_numpy()
        R = tokrisk_pair_features(st, qq, ss, risk_k[k][0], risk_k[k][1])
        R['leg_pair_risk'] = leg_pair(st.leg[qq], st.leg[ss], risk_k[k][2])
        return d.with_columns([pl.Series(c, v) for c, v in R.items() if c in feats])

    # France pseudo-labeled rows per fold, in feature-part order
    fr_frac = min(1.0, N_FR / max(1, keep.sum()))
    Xf = {0: [], 1: []}; yf = {0: [], 1: []}
    off = 0
    for f in te_files:
        d = pl.read_parquet(f, columns=feats + ['q', 's'])
        n = d.height
        m_fr = is_fr[d['q'].to_numpy()]
        idx = np.arange(off, off + n)[m_fr]
        off += n
        if not m_fr.any():
            continue
        d = d.filter(pl.Series(m_fr))
        # position of these rows inside the France arrays
        pos_fr = np.searchsorted(np.flatnonzero(frm), idx)
        lk, fk = lab[pos_fr], fo[pos_fr]
        for k in (0, 1):
            m = (fk == k) & (lk >= 0) & (rng.random(d.height) < fr_frac)
            if m.any():
                dd = patched(d.filter(pl.Series(m)), k)
                Xf[k].append(dd.select(feats).to_numpy().astype(np.float32)); yf[k].append(lk[m])
    models = {}
    for k in (0, 1):
        X = np.concatenate([Xt] + Xf[k]); y = np.concatenate([yt] + yf[k]).astype(np.int8)
        perm = rng.permutation(len(y))
        models[k] = france_selftrain_fit(X[perm], y[perm], feats)
        print(f'[{time.time()-t0:.0f}s] model {k} trained on {len(y)} rows', flush=True)
        del X, y
    # rescore France rows out-of-fold
    p1 = p_all.copy()
    off = 0
    fr_idx = np.flatnonzero(frm)
    for f in te_files:
        d = pl.read_parquet(f, columns=feats + ['q', 's'])
        n = d.height
        m_fr = is_fr[d['q'].to_numpy()]
        rows = np.arange(off, off + n)[m_fr]
        off += n
        if not m_fr.any():
            continue
        d = d.filter(pl.Series(m_fr))
        fk = fold_of(d['s'].to_numpy())
        pp = np.zeros(d.height, np.float32)
        for k in (0, 1):
            m = fk == k
            if m.any():
                dd = patched(d.filter(pl.Series(m)), k)
                pp[m] = models[1 - k].predict(dd.select(feats).to_numpy().astype(np.float32), num_threads=8)
        p1[rows] = pp
    base = np.load(f'{WORK}/pred_test_{base_tag}.npz')
    assert (base['q'] == q_all).all()
    p_out = np.where(is_fr[q_all], p1, base['p']).astype(np.float32)
    np.savez(f'{WORK}/pred_test_{out_tag}.npz', q=q_all, s=s_all, p=p_out)
    print(f'[{time.time()-t0:.0f}s] saved pred_test_{out_tag}', flush=True)


def france_selftrain_cli():
    """`python france_selftrain.py ...` of the original pipeline (reads sys.argv)."""
    france_selftrain_main(*sys.argv[1:6])


# ========================================================================================
# france_combine.py
#
# Combine the two France prediction rounds (both after the noise-word rule).
#
# round 1 = leave-one-country-out model (+ rule); round 2 = self-trained on round-1 +
# rule pseudo-labels (+ rule). Round 2 learns the French noise words beyond the rule's
# exact conditions (missing house number, '& Cie' parsed as a legal form, frequent street
# names), but drifts down on mid-confidence cases (blank-address exact names). Round 1
# accepts swaps to 'centre'/'service', which are true-copy noise words in US/India but
# category words in France (same-number share 0.43 like club/ecole, vs 0.87 for the
# French noise words). So: p = max(round1, round2), except pairs where the query swaps
# in 'center'/'service' for another word, which take round 2.
#
#     python france_combine.py <round1_tag> <round2_tag> <out_tag>
# ========================================================================================

CAT_EN = ['center', 'service']


def france_combine_main(t1, t2, out):
    a = np.load(f'{WORK}/pred_test_{t1}.npz'); b = np.load(f'{WORK}/pred_test_{t2}.npz')
    q, s = a['q'], a['s']
    assert (b['q'] == q).all() and (b['s'] == s).all()
    p1, p2 = a['p'], b['p']
    ctry = pl.read_parquet(f'{WORK}/norm_test.parquet', columns=['country'])['country'].to_numpy()
    fr = ctry[q] == 'France'
    p = np.where(fr, np.maximum(p1, p2), p1).astype(np.float32)
    idx = np.flatnonzero(fr & (p1 >= 0.5) & (p2 < 0.5))
    core = pl.read_parquet(f'{WORK}/norm_test.parquet', columns=['n_core'])['n_core']
    d = pl.DataFrame({'i': idx, 'qc': core.gather(q[idx]), 'sc': core.gather(s[idx])})
    d = d.with_columns(pl.col('qc').list.set_difference('sc').alias('ex'), pl.col('sc').list.set_difference('qc').alias('mi'))
    d = d.filter((pl.col('mi').list.len() > 0) & pl.col('ex').list.eval(pl.element().is_in(CAT_EN)).list.any())
    j = d['i'].to_numpy()
    p[j] = p2[j]
    print(f'France pairs {fr.sum()}; round-1-only accepts {len(idx)}, of which centre/service swaps {len(j)} set to round 2')
    np.savez(f'{WORK}/pred_test_{out}.npz', q=q, s=s, p=p)


def france_combine_cli():
    """`python france_combine.py ...` of the original pipeline (reads sys.argv)."""
    france_combine_main(*sys.argv[1:4])


# ========================================================================================
# france_mean.py
#
# France decision variant: mean of the two self-training rounds (both after the
# noise-word rule) instead of the max, plus a French legal-form conflict cap.
#
# "& Cie" is normalised to the legal token "co", which outranks the French forms, so
# "X & Cie SAS" vs "X & Cie SARL" looked like agreeing legal forms. Explicit French
# legal conflicts are kept 0.1% of the time; these hidden ones 7.3% (mostly nearby house
# numbers, i.e. siblings). Pairs whose French legal forms (first of sasu/sas/sarl/eurl/
# sci/sa/snc/ei) differ get p = min(p, 0.01).
#
#     python france_mean.py <round1_tag> <round2_tag> <base_all_tag> <out_tag> [max]
# With "max", France keeps the max of the two rounds (the v11/v12a combination, e.g.
# round2_tag = fr_final) and only the legal-conflict cap is added.
# ========================================================================================

FR_LEGAL = ['sasu', 'sas', 'sarl', 'eurl', 'sci', 'sa', 'snc', 'ei']


def france_mean_main(t1, t2, base, out, mode='mean'):
    a = np.load(f'{WORK}/pred_test_{t1}.npz'); b = np.load(f'{WORK}/pred_test_{t2}.npz'); c = np.load(f'{WORK}/pred_test_{base}.npz')
    q, s = c['q'], c['s']
    assert (a['q'] == q).all() and (b['q'] == q).all() and (a['s'] == s).all() and (b['s'] == s).all()
    norm = pl.read_parquet(f'{WORK}/norm_test.parquet', columns=['country', 'n_toks'])
    fr = norm['country'].to_numpy()[q] == 'France'
    fl = norm.select(pl.col('n_toks').list.eval(pl.element().filter(pl.element().is_in(FR_LEGAL))).list.first().fill_null(''))['n_toks'].to_numpy()
    p = c['p'].copy()
    p[fr] = np.maximum(a['p'][fr], b['p'][fr]) if mode == 'max' else (a['p'][fr] + b['p'][fr]) / 2
    conflict = fr & (fl[q] != '') & (fl[s] != '') & (fl[q] != fl[s])
    print(f'France pairs {fr.sum()}; French legal conflicts capped: {conflict.sum()} (of which p >= 0.6 before: {(p[conflict] >= 0.6).sum()})')
    p[conflict] = np.minimum(p[conflict], 0.01)
    np.savez(f'{WORK}/pred_test_{out}.npz', q=q, s=s, p=p.astype(np.float32))


def france_mean_cli():
    """`python france_mean.py ...` of the original pipeline (reads sys.argv)."""
    france_mean_main(*sys.argv[1:6])


# ========================================================================================
# cand_filter.py
#
# Last filtering stage before the final matchers.
#
# Stage 1 (US/India: t1, France: tfr, the leave-one-country-out model) scores every
# stage-0 candidate; a pair goes on to the final matchers (stage 2 / France models)
# only if its stage-1 probability is >= EPS or it fits the France noise-word rule.
# The stage-2 context features are aggregates of the stage-1 scores. On test this keeps
# ~4.6 (US/India) and ~5.6 (France) pairs per Source 1 entity instead of 19-54, and no
# final match falls outside it.
#
#     python cand_filter.py <any prediction tag with the test pair order> [eps]
# writes WORK/cand_keep.npy (boolean over the test pair order)
# ========================================================================================

def cand_filter_main(tag, eps=0.0005):
    t1 = np.load(f'{WORK}/pred_test_t1.npz'); tfr = np.load(f'{WORK}/pred_test_tfr.npz')
    z = np.load(f'{WORK}/pred_test_{tag}.npz')
    q, s = z['q'], z['s']
    assert (t1['q'] == q).all() and (tfr['q'] == q).all()
    ctry = pl.read_parquet(f'{WORK}/norm_test.parquet', columns=['country'])['country'].to_numpy()
    p1 = np.where(ctry[q] == 'France', tfr['p'], t1['p'])
    keep = (p1 >= eps) | fix_mask(q, s)
    np.save(f'{WORK}/cand_keep.npy', keep)
    print(f'kept {keep.sum()} of {len(keep)} pairs (eps {eps})')


def cand_filter_cli():
    """`python cand_filter.py ...` of the original pipeline (reads sys.argv)."""
    cand_filter_main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.0005)


# ========================================================================================
# avg_preds.py
#
# Average saved predictions of several model tags (same candidate order).
#
#     python avg_preds.py <split: test|oof> <tag1,tag2,...> <out_tag> [mean|logit]
#
# "logit" averages log-odds (p clipped to [1e-6, 1 - 1e-6]) instead of probabilities.
# ========================================================================================

def logit(p):
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def avg_preds_main(split, tags, out, mode='mean'):
    pre = 'pred_test' if split == 'test' else 'oof'
    zs = [np.load(f'{WORK}/{pre}_{t}.npz') for t in tags.split(',')]
    q, s = zs[0]['q'], zs[0]['s']
    for z in zs[1:]:
        assert (z['q'] == q).all() and (z['s'] == s).all()
    if mode == 'logit':
        p = 1 / (1 + np.exp(-np.mean([logit(z['p']) for z in zs], axis=0)))
    else:
        p = np.mean([z['p'] for z in zs], axis=0)
    p = p.astype(np.float32)
    extra = {'y': zs[0]['y']} if 'y' in zs[0] else {}
    np.savez(f'{WORK}/{pre}_{out}.npz', q=q, s=s, p=p, **extra)
    print(f'{pre}_{out}: {mode} of {tags}, {len(p)} pairs')


def avg_preds_cli():
    """`python avg_preds.py ...` of the original pipeline (reads sys.argv)."""
    avg_preds_main(*sys.argv[1:5])


# ========================================================================================
# orchestration
#
# The steps and settings of the original run scripts (run_v4/run_v6/run_v7 up to the test
# features and models, the round-1 France self-training, run_seed x4, the France rounds of
# run_final, and build_final), restricted to what the final submission v14l depends on.
# Every step runs as `python er_pipeline.py <step> <args>` in a fresh process, so the
# module-level ER_* settings are read per step exactly as before.
# ========================================================================================

def score_seed_cli():
    """Test scoring of an extra-seed US/India chain (the inline step of run_seed.sh)."""
    n = sys.argv[1]
    q, s, p = predict_score(f't1s{n}', feat_tag='_sug')
    predict_score(f't2s{n}', feat_tag='_sug', extra=stage2_context(q, s, p))


STEPS = {
    'to_parquet': to_parquet_cli, 'translit': translit_cli, 'prep': prep_cli,
    'run_retrieve': run_retrieve_cli, 'stage0': stage0_cli, 'build_features': build_features_cli,
    'add_unsup': add_unsup_cli, 'add_group': add_group_cli, 'build_loco': build_loco_cli,
    'train_model': train_model_cli, 'score_v7': score_v7_main, 'score_seed': score_seed_cli,
    'france_selftrain': france_selftrain_cli, 'france_noise_fix': france_noise_fix_cli,
    'merge_preds': merge_preds_cli, 'france_combine': france_combine_cli, 'france_mean': france_mean_cli,
    'cand_filter': cand_filter_cli, 'avg_preds': avg_preds_cli, 'predict': predict_cli,
}
# group-consistency and candidate-count features shift between train and test: left out
DROP_FEATS = 'g_*,q_ncand,s_ncand'


def step(name, *args, **env):
    e = dict(os.environ)
    e.update({k: str(v) for k, v in env.items()})
    sets = ' '.join(f'{k}={v}' for k, v in env.items())
    print(f'[{time.strftime("%H:%M:%S")}] {name} {" ".join(repr(a) if a == "" else a for a in args)}'
          + (f'   ({sets})' if sets else ''), flush=True)
    subprocess.run([sys.executable, os.path.abspath(__file__), name, *args], env=e, check=True)


def seed_chain(n, **env):
    """Extra copy n of the US/India stage-1/2 chain (run_seed.sh): new row sample and
    LightGBM seeds, 12M training rows per fold."""
    env = dict(ER_MAX_TRAIN=12000000, ER_DROP_FEATS=DROP_FEATS, ER_SEED=n, **env)
    step('train_model', f't1s{n}', ER_FEAT='_sug', **env)
    step('train_model', f't2s{n}', ER_FEAT='_sug', ER_STAGE2=f't1s{n}', **env)
    step('score_seed', str(n), **env)


def build_final(name='v14l'):
    """Final stage from saved predictions: 5-model US/India mean, France = max of the two
    self-training rounds (fr_final) with the French legal-form conflict cap, last
    filtering stage, then assignment + expected-F0.5 (US/India) / threshold 0.6 (France)."""
    step('avg_preds', 'test', 't2,t2s1,t2s2,t2s3,t2s4', 't2avg5')
    step('france_mean', 'fr_final', 'fr_final', 't2avg5', name, 'max')
    step('cand_filter', name, '0.0005')
    step('predict', name, '0.7', 'France=0.6', ER_CAND_KEEP=f'{WORK}/cand_keep.npy', ER_EXPF='US,India')


def run_all():
    """Data -> blocking -> matching -> output, in the order the final submission was built."""
    # 0-2: parquet caches, Indic dictionary, normalization
    step('to_parquet')
    step('translit')
    step('prep', 'train', 'test')
    # 3: blocking (joint inverted index, adaptive second pass)
    step('run_retrieve', 'train', '', '_a')
    step('run_retrieve', 'test', '', '_a')
    # 4: stage-0 pruning, pair features, label-free statistics, leave-one-country-out risks
    v6 = dict(ER_CHUNK=3000000, ER_MAX_TRAIN=10000000, ER_STAGE0='_a', ER_T0=0.001, ER_KMIN=3, ER_K0=7)
    step('stage0', '_a', **v6)
    step('build_features', 'train', '_a', '_s', **v6)
    step('add_unsup', 'train', '_s', '_su', **v6)
    step('add_group', 'train', '_su', '_sug', **v6)
    step('build_loco', '_sug', '_sgloco', **v6)
    step('build_features', 'test', '_a', '_s', **v6)
    step('add_unsup', 'test', '_s', '_su', **v6)
    step('add_group', 'test', '_su', '_sug', **v6)
    # 5: stage-1/2 US+India models and leave-one-country-out France models, test scoring
    v7 = dict(ER_MAX_TRAIN=10000000, ER_DROP_FEATS=DROP_FEATS)
    step('train_model', 't1', ER_FEAT='_sug', **v7)
    step('train_model', 't2', ER_FEAT='_sug', ER_STAGE2='t1', **v7)
    step('train_model', 'tfr', ER_FEAT='_sgloco', **v7)
    step('train_model', 'tfr2', ER_FEAT='_sgloco', ER_STAGE2='tfr', **v7)
    step('score_v7', **v7)
    # 6: France self-training round 1
    step('france_selftrain', 'tfr2', '_sug', '_sgloco', 't2', 'v8fr', ER_DROP_FEATS=DROP_FEATS)
    # 7: two extra US/India seeds; France noise-word rule, self-training round 2, combination.
    #    (Round 2 was run with the 3-model US/India mean as its base; only non-France rows
    #    come from the base, and those are replaced later, so it does not change the output.)
    seed_chain(1)
    seed_chain(2)
    step('avg_preds', 'test', 't2,t2s1,t2s2', 't2avg')
    fr = dict(ER_DROP_FEATS=DROP_FEATS)
    step('france_noise_fix', 'v8fr', 'fr_r1', **fr)
    step('merge_preds', 't2avg', 'fr_r1', 'France', 'r1all', **fr)
    step('france_selftrain', 'r1all', '_sug', '_sgloco', 't2avg', 'fr_r2_raw', **fr)
    step('france_noise_fix', 'fr_r2_raw', 'fr_r2', **fr)
    step('france_combine', 'fr_r1', 'fr_r2', 'fr_final', **fr)
    # 8: two more US/India copies at learning rate 0.04 / 1600 rounds
    seed_chain(3, ER_LR=0.04, ER_ROUNDS=1600)
    seed_chain(4, ER_LR=0.04, ER_ROUNDS=1600)
    # 9: final stage -> matching_results.tsv, candidate_pairs.tsv
    build_final()


PIPELINES = {'run_all': run_all, 'build_final': build_final}


def cli():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        print('steps:', ', '.join(STEPS))
        return
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd in PIPELINES:
        PIPELINES[cmd](*args)
    elif cmd in STEPS:
        sys.argv = [cmd] + args
        STEPS[cmd]()
    else:
        raise SystemExit(f'unknown command {cmd!r}; commands: run_all, build_final, ' + ', '.join(STEPS))


if __name__ == '__main__':
    cli()
