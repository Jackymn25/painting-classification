
"""
pred.py

Numpy/Pandas-only implementation of the uploaded Logistic+NB pipeline.

Allowed imports only:
- csv
- random
- re
- numpy
- pandas

This file exposes:
    predict_all(filename)

It will:
1) load the training CSV (prefers cleaned 1610x17 CSV if present),
2) train the full model with the tuned hyperparameters,
3) predict painting titles for all rows in `filename`.

It also includes an optional K-fold validation in __main__ to verify
that the numpy version matches the original sklearn pipeline closely.
"""

import csv
import random
import re

import numpy as np
import pandas as pd

# =========================
# Paths / Columns
# =========================
TRAIN_CSV_CANDIDATES = [
    "cleaned_data_final.csv",
]

LABEL_COL = "label"
UNIQUE_ID_COL = "unique_id"

FOOD_COL = "If this painting was a food, what would be?"
FEEL_COL = "Describe how this painting makes you feel."
SOUND_COL = "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting."

ROOM_COL = "If you could purchase this painting, which room would you put that painting in?"
WHO_COL = "If you could view this art in person, who would you want to view it with?"
SEASON_COL = "What season does this art piece remind you of?"

MULTIHOT_COLS = [ROOM_COL, WHO_COL, SEASON_COL]

INTENSITY_COL = "On a scale of 1–10, how intense is the emotion conveyed by the artwork?"
PROMINENT_COLOURS_COL = "How many prominent colours do you notice in this painting?"
OBJECTS_COL = "How many objects caught your eye in the painting?"
PAYMENT_COL = "How much (in Canadian dollars) would you be willing to pay for this painting?"

LIKERT_COLS = [
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy.",
]

LIKERT_MAP = {
    "1 - Strongly disagree": 1,
    "2 - Disagree": 2,
    "3 - Neutral": 3,
    "4 - Agree": 4,
    "5 - Strongly agree": 5,
}

# =========================
# Best params
# =========================
FOOD_K = 42
FEELING_K = 38
SOUNDTRACK_K = 62
NB_ALPHA = 1.0

# Original tuning log on the cleaned 1610x17 CSV:
# C=0.1 -> mean_val_acc=0.9124
# C=0.5 -> mean_val_acc=0.9124
# We use the slightly more regularized of the tied best values.
LOGREG_C = 0.1

MIN_TOKEN_LEN = 2
EXCLUDE_MISSING_FROM_VOCAB = True
ZERO_HIT_RETURNS_PRIOR = True
SORT_TIES_ALPHABETICALLY = True

# Small numerical damping for the intercept block in the Newton solve.
# This makes the full-K softmax Hessian non-singular while keeping behaviour
# essentially identical to sklearn multinomial logistic regression.
INTERCEPT_DAMPING = 1e-6
NEWTON_MAX_ITER = 25
NEWTON_TOL = 1e-8

# Validation config
RUN_KFOLD_VALIDATION_IN_MAIN = True
KFOLD_N_SPLITS = 5
KFOLD_SHUFFLE = True
KFOLD_RANDOM_STATE = 42

LABEL_TO_NAME = {
    0: "The Persistence of Memory",
    1: "The Starry Night",
    2: "The Water Lily Pond",
}

ALIASES = {
    "blueberries": "blueberry",
    "noodles": "noodle",
    "violins": "violin",
    "slowly": "slow",
    "icecream": "ice",
}

# =========================
# Cleaning helpers
# =========================
def normalize_text(text):
    if pd.isna(text):
        return ""
    s = str(text).strip().lower()
    s = s.replace("\xa0", " ")
    s = s.replace("–", "-").replace("—", "-")
    s = " ".join(s.split())
    return s

def collapse_spaced_thousands(text):
    if not text:
        return text
    pattern = re.compile(r"(?<!\d)(\d{1,3}(?: \d{3})+)(?!\d)")
    prev = None
    s = text
    while prev != s:
        prev = s
        s = pattern.sub(lambda m: m.group(1).replace(" ", ""), s)
    return s

def contains_any(text, patterns):
    for pat in patterns:
        if re.search(pat, text):
            return True
    return False

def looks_like_uncertain_text(s):
    uncertain_patterns = [
        r"\bnot sure\b",
        r"\bunsure\b",
        r"\bidk\b",
        r"\bi don't know\b",
        r"\bcannot decide\b",
        r"\bcan't decide\b",
        r"\bno idea\b",
    ]
    return contains_any(s, uncertain_patterns)

def looks_like_zero_text(s):
    zero_patterns = [
        r"^0+(\.0+)?$",
        r"\bfree\b",
        r"\bnothing\b",
        r"\bnone\b",
        r"\bno money\b",
        r"\bzero\b",
    ]
    return contains_any(s, zero_patterns)

def scale_multiplier(scale):
    if not scale:
        return 1.0
    scale = scale.lower().strip()
    if scale in {"k", "thousand"}:
        return 1_000.0
    if scale in {"mil", "million"}:
        return 1_000_000.0
    if scale in {"b", "billion"}:
        return 1_000_000_000.0
    return 1.0

def parse_money_value(text):
    if pd.isna(text):
        return np.nan
    s = normalize_text(text)
    if s in {"", "nan", "n/a", "na", "none"}:
        return np.nan
    s = collapse_spaced_thousands(s)
    if looks_like_zero_text(s):
        return 0.0
    num_pattern = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    scale_pattern = r"(?:k|thousand|mil|million|b|billion)?"
    money_pattern = re.compile(
        rf"""(?: 
            (?:cad\s*\$?|\$|usd\s*\$?)\s*(?P<num1>{num_pattern})\s*(?P<scale1>{scale_pattern})
        )|(
            (?P<num2>{num_pattern})\s*(?P<scale2>{scale_pattern})\s*(?:cad|usd|dollars?|bucks?)\b
        )|(
            (?P<num3>{num_pattern})\s+(?P<scale3>k|thousand|mil|million|b|billion)\b
        )|(
            (?P<num4>{num_pattern})(?P<scale4>k|mil|b)\b
        )|(
            \b(?P<num5>{num_pattern})\b
        )""".format(num_pattern=num_pattern, scale_pattern=scale_pattern),
        re.VERBOSE | re.IGNORECASE,
    )
    candidates = []
    for match in money_pattern.finditer(s):
        num = None
        scale = ""
        for num_key, scale_key in [
            ("num1", "scale1"),
            ("num2", "scale2"),
            ("num3", "scale3"),
            ("num4", "scale4"),
            ("num5", None),
        ]:
            if match.group(num_key) is not None:
                num = match.group(num_key)
                scale = match.group(scale_key) if scale_key else ""
                scale = scale or ""
                break
        if num is None:
            continue
        try:
            value = float(num.replace(",", ""))
        except Exception:
            continue
        value *= scale_multiplier(scale)
        candidates.append(value)
    if candidates:
        return max(candidates)
    if looks_like_uncertain_text(s):
        return np.nan
    return np.nan

def parse_payment_column(series):
    out = series.apply(parse_money_value)
    out = pd.to_numeric(out, errors="coerce")
    neg_count = int((out < 0).fillna(False).sum())
    out.loc[out < 0] = 0
    cap_count = int((out > 324_000_000).fillna(False).sum())
    out = out.clip(0, 324_000_000)
    missing_count = int(out.isna().sum())
    return out, missing_count, neg_count, cap_count

def clean_bounded_scale(series, lower, upper):
    x = pd.to_numeric(series, errors="coerce")
    invalid_mask = (x < lower) | (x > upper)
    invalid_count = int(invalid_mask.fillna(False).sum())
    x[invalid_mask] = np.nan
    return x, invalid_count

def clean_count_column(series):
    x = pd.to_numeric(series, errors="coerce")
    neg_count = int((x < 0).fillna(False).sum())
    x.loc[x < 0] = 0
    non_missing = x.dropna()
    if len(non_missing) == 0:
        return x, neg_count, np.nan, 0
    q99 = float(non_missing.quantile(0.99))
    upper = int(np.ceil(q99))
    cap_count = int((x > upper).fillna(False).sum())
    x = x.clip(lower=0, upper=upper)
    x = np.round(x)
    return x, neg_count, upper, cap_count

def clean_raw_dataframe(df_raw):
    df = df_raw.copy()
    if "Painting" in df.columns and LABEL_COL not in df.columns:
        label_map = {
            "The Persistence of Memory": 0,
            "The Starry Night": 1,
            "The Water Lily Pond": 2,
        }
        df["Painting"] = df["Painting"].astype(str).str.strip()
        df["Painting"] = df["Painting"].replace("nan", np.nan)
        df[LABEL_COL] = df["Painting"].map(label_map)
    for col in LIKERT_COLS:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col], _ = clean_bounded_scale(df[col], 1, 5)
            else:
                mapped = df[col].astype(str).str.strip().replace("nan", np.nan).map(LIKERT_MAP)
                df[col] = pd.to_numeric(mapped, errors="coerce")
                df[col], _ = clean_bounded_scale(df[col], 1, 5)
    if INTENSITY_COL in df.columns:
        df[INTENSITY_COL], _ = clean_bounded_scale(df[INTENSITY_COL], 1, 10)
    if PROMINENT_COLOURS_COL in df.columns:
        df[PROMINENT_COLOURS_COL], _, _, _ = clean_count_column(df[PROMINENT_COLOURS_COL])
    if OBJECTS_COL in df.columns:
        df[OBJECTS_COL], _, _, _ = clean_count_column(df[OBJECTS_COL])
    if PAYMENT_COL in df.columns:
        if pd.api.types.is_numeric_dtype(df[PAYMENT_COL]):
            x = pd.to_numeric(df[PAYMENT_COL], errors="coerce")
            x.loc[x < 0] = 0
            df[PAYMENT_COL] = x.clip(0, 324_000_000)
        else:
            df[PAYMENT_COL], _, _, _ = parse_payment_column(df[PAYMENT_COL])
    if LABEL_COL in df.columns:
        df = df[df[LABEL_COL].notna()].copy()
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]
    text_cols = [c for c in df.columns if c not in numeric_cols]
    for col in numeric_cols:
        if df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)
    for col in text_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna("Missing")
    return df.reset_index(drop=True)

def resolve_training_dataframe():
    # Prefer an already-cleaned CSV because that is what the original reported
    # 0.9124/0.9112 validation results were run on.
    last_exc = None
    for path in TRAIN_CSV_CANDIDATES:
        try:
            df = pd.read_csv(path)
            # If cleaned (has label already), use directly.
            if LABEL_COL in df.columns:
                return df, path, True
            # Otherwise it is raw; clean it.
            return clean_raw_dataframe(df), path, False
        except Exception as e:
            last_exc = e
    raise FileNotFoundError("Could not load any training CSV candidate.") from last_exc

stop_words = {
    "'d",
    "'ll",
    "'m",
    "'re",
    "'s",
    "'ve",
    "a",
    "about",
    "above",
    "across",
    "after",
    "afterwards",
    "again",
    "against",
    "all",
    "almost",
    "alone",
    "along",
    "already",
    "also",
    "although",
    "always",
    "am",
    "among",
    "amongst",
    "amoungst",
    "amount",
    "an",
    "and",
    "another",
    "any",
    "anyhow",
    "anyone",
    "anything",
    "anyway",
    "anywhere",
    "are",
    "around",
    "as",
    "at",
    "back",
    "be",
    "became",
    "because",
    "become",
    "becomes",
    "becoming",
    "been",
    "before",
    "beforehand",
    "behind",
    "being",
    "below",
    "beside",
    "besides",
    "between",
    "beyond",
    "bill",
    "both",
    "bottom",
    "but",
    "by",
    "ca",
    "call",
    "can",
    "cannot",
    "cant",
    "co",
    "con",
    "could",
    "couldnt",
    "cry",
    "de",
    "describe",
    "detail",
    "did",
    "do",
    "does",
    "doing",
    "done",
    "down",
    "due",
    "during",
    "each",
    "eg",
    "eight",
    "either",
    "eleven",
    "else",
    "elsewhere",
    "empty",
    "enough",
    "etc",
    "even",
    "ever",
    "every",
    "everyone",
    "everything",
    "everywhere",
    "except",
    "few",
    "fifteen",
    "fifty",
    "fill",
    "find",
    "fire",
    "first",
    "five",
    "for",
    "former",
    "formerly",
    "forty",
    "found",
    "four",
    "from",
    "front",
    "full",
    "further",
    "get",
    "give",
    "go",
    "had",
    "has",
    "hasnt",
    "have",
    "he",
    "hence",
    "her",
    "here",
    "hereafter",
    "hereby",
    "herein",
    "hereupon",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "however",
    "hundred",
    "i",
    "ie",
    "if",
    "in",
    "inc",
    "indeed",
    "interest",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "keep",
    "last",
    "latter",
    "latterly",
    "least",
    "less",
    "ltd",
    "made",
    "make",
    "many",
    "may",
    "me",
    "meanwhile",
    "might",
    "mill",
    "mine",
    "more",
    "moreover",
    "most",
    "mostly",
    "move",
    "much",
    "must",
    "my",
    "myself",
    "n't",
    "name",
    "namely",
    "neither",
    "never",
    "nevertheless",
    "next",
    "nine",
    "no",
    "nobody",
    "none",
    "noone",
    "nor",
    "not",
    "nothing",
    "now",
    "nowhere",
    "n‘t",
    "n’t",
    "of",
    "off",
    "often",
    "on",
    "once",
    "one",
    "only",
    "onto",
    "or",
    "other",
    "others",
    "otherwise",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "part",
    "per",
    "perhaps",
    "please",
    "put",
    "quite",
    "rather",
    "re",
    "really",
    "regarding",
    "same",
    "say",
    "see",
    "seem",
    "seemed",
    "seeming",
    "seems",
    "serious",
    "several",
    "she",
    "should",
    "show",
    "side",
    "since",
    "sincere",
    "six",
    "sixty",
    "so",
    "some",
    "somehow",
    "someone",
    "something",
    "sometime",
    "sometimes",
    "somewhere",
    "still",
    "such",
    "system",
    "take",
    "ten",
    "than",
    "that",
    "the",
    "their",
    "them",
    "themselves",
    "then",
    "thence",
    "there",
    "thereafter",
    "thereby",
    "therefore",
    "therein",
    "thereupon",
    "these",
    "they",
    "thick",
    "thin",
    "third",
    "this",
    "those",
    "though",
    "three",
    "through",
    "throughout",
    "thru",
    "thus",
    "to",
    "together",
    "too",
    "top",
    "toward",
    "towards",
    "twelve",
    "twenty",
    "two",
    "un",
    "under",
    "unless",
    "until",
    "up",
    "upon",
    "us",
    "used",
    "using",
    "various",
    "very",
    "via",
    "was",
    "we",
    "well",
    "were",
    "what",
    "whatever",
    "when",
    "whence",
    "whenever",
    "where",
    "whereafter",
    "whereas",
    "whereby",
    "wherein",
    "whereupon",
    "wherever",
    "whether",
    "which",
    "while",
    "whither",
    "who",
    "whoever",
    "whole",
    "whom",
    "whose",
    "why",
    "will",
    "with",
    "within",
    "without",
    "would",
    "yet",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "‘d",
    "‘ll",
    "‘m",
    "‘re",
    "‘s",
    "‘ve",
    "’d",
    "’ll",
    "’m",
    "’re",
    "’s",
    "’ve"
}

# =========================
# Text preprocessing / NB
# =========================

def simple_singularize(token):
    if token in ALIASES:
        return ALIASES[token]
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 4 and not token.endswith("ses"):
        return token[:-2]
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token

def tokenize_text(text):
    if pd.isna(text):
        text = "missing"
    text = str(text).lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z'\s]", " ", text)
    tokens = []
    for raw_tok in text.split():
        tok = raw_tok.strip("'")
        if not tok:
            continue
        tok = simple_singularize(tok)
        if len(tok) < MIN_TOKEN_LEN:
            continue
        if tok in stop_words:
            continue
        tokens.append(tok)
    return tokens

def select_top_k_vocab_from_training(texts, k):
    if k <= 0:
        raise ValueError("k must be > 0.")
    doc_counter = {}
    for text in pd.Series(texts).fillna("missing"):
        uniq_tokens = set(tokenize_text(text))
        if EXCLUDE_MISSING_FROM_VOCAB and "missing" in uniq_tokens:
            uniq_tokens.remove("missing")
        for tok in uniq_tokens:
            doc_counter[tok] = doc_counter.get(tok, 0) + 1
    if not doc_counter:
        raise ValueError("No usable tokens found in the training data after preprocessing.")
    items = list(doc_counter.items())
    if SORT_TIES_ALPHABETICALLY:
        items.sort(key=lambda x: (-x[1], x[0]))
    else:
        items.sort(key=lambda x: -x[1])
    vocab = [tok for tok, _ in items[:k]]
    return vocab, doc_counter

class TopKBernoulliNB:
    def __init__(self, k, alpha=NB_ALPHA):
        self.k = int(k)
        self.alpha = float(alpha)
        self.vocab_ = None
        self.vocab_index_ = None
        self.doc_freq_ = None
        self.class_count_ = None
        self.feature_prob_ = None

    def _vectorize_one(self, text):
        row = np.zeros(len(self.vocab_), dtype=np.int8)
        uniq_tokens = set(tokenize_text(text))
        for tok in uniq_tokens:
            j = self.vocab_index_.get(tok)
            if j is not None:
                row[j] = 1
        return row

    def _vectorize(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = pd.Series(texts).fillna("missing")
        X = np.zeros((len(texts), len(self.vocab_)), dtype=np.int8)
        for i, text in enumerate(texts):
            X[i] = self._vectorize_one(text)
        return X

    def fit(self, texts, y):
        self.vocab_, self.doc_freq_ = select_top_k_vocab_from_training(texts, self.k)
        self.vocab_index_ = {tok: i for i, tok in enumerate(self.vocab_)}
        X = self._vectorize(texts)
        y = np.asarray(y, dtype=int)
        K = int(np.max(y)) + 1
        self.class_count_ = np.bincount(y, minlength=K).astype(np.float64)
        V = X.shape[1]
        self.feature_prob_ = np.zeros((K, V), dtype=np.float64)
        for c in range(K):
            mask = (y == c)
            n_c = float(mask.sum())
            count = X[mask].sum(axis=0).astype(np.float64)
            self.feature_prob_[c] = (count + self.alpha) / (n_c + 2.0 * self.alpha)
        self.feature_prob_ = np.clip(self.feature_prob_, 1e-12, 1.0 - 1e-12)
        return self

    def hit_count(self, texts):
        X = self._vectorize(texts)
        return X.sum(axis=1)

    def predict_proba(self, texts):
        X = self._vectorize(texts).astype(np.float64)
        K = len(self.class_count_)
        log_prior = np.log(self.class_count_ / self.class_count_.sum())
        log_p = np.log(self.feature_prob_)
        log_not_p = np.log(1.0 - self.feature_prob_)
        log_scores = X @ log_p.T + (1.0 - X) @ log_not_p.T + log_prior.reshape(1, K)
        if ZERO_HIT_RETURNS_PRIOR:
            hits = X.sum(axis=1)
            zero_mask = hits == 0
            if np.any(zero_mask):
                prior = self.class_count_ / self.class_count_.sum()
                log_scores[zero_mask] = np.log(prior)
        log_scores = log_scores - log_scores.max(axis=1, keepdims=True)
        probs = np.exp(log_scores)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, texts):
        probs = self.predict_proba(texts)
        return np.argmax(probs, axis=1)

    def predict_one(self, text):
        return int(self.predict([text])[0])

def train_text_nb_models(df_train, food_k=FOOD_K, feeling_k=FEELING_K, soundtrack_k=SOUNDTRACK_K, alpha=NB_ALPHA):
    food_model = TopKBernoulliNB(k=food_k, alpha=alpha).fit(df_train[FOOD_COL], df_train[LABEL_COL])
    feeling_model = TopKBernoulliNB(k=feeling_k, alpha=alpha).fit(df_train[FEEL_COL], df_train[LABEL_COL])
    soundtrack_model = TopKBernoulliNB(k=soundtrack_k, alpha=alpha).fit(df_train[SOUND_COL], df_train[LABEL_COL])
    return food_model, feeling_model, soundtrack_model

def build_nb_feature_frame(df_any, food_model, feeling_model, soundtrack_model):
    food_probs = food_model.predict_proba(df_any[FOOD_COL].fillna("missing"))
    feeling_probs = feeling_model.predict_proba(df_any[FEEL_COL].fillna("missing"))
    soundtrack_probs = soundtrack_model.predict_proba(df_any[SOUND_COL].fillna("missing"))

    food_hits = food_model.hit_count(df_any[FOOD_COL].fillna("missing"))
    feeling_hits = feeling_model.hit_count(df_any[FEEL_COL].fillna("missing"))
    soundtrack_hits = soundtrack_model.hit_count(df_any[SOUND_COL].fillna("missing"))

    return pd.DataFrame(
        {
            "food_nb_p0": food_probs[:, 0],
            "food_nb_p1": food_probs[:, 1],
            "food_nb_p2": food_probs[:, 2],
            "food_nb_pred": np.argmax(food_probs, axis=1),
            "food_nb_hit_count": food_hits,
            "food_nb_zero_hit": (food_hits == 0).astype(int),
            "feeling_nb_p0": feeling_probs[:, 0],
            "feeling_nb_p1": feeling_probs[:, 1],
            "feeling_nb_p2": feeling_probs[:, 2],
            "feeling_nb_pred": np.argmax(feeling_probs, axis=1),
            "feeling_nb_hit_count": feeling_hits,
            "feeling_nb_zero_hit": (feeling_hits == 0).astype(int),
            "soundtrack_nb_p0": soundtrack_probs[:, 0],
            "soundtrack_nb_p1": soundtrack_probs[:, 1],
            "soundtrack_nb_p2": soundtrack_probs[:, 2],
            "soundtrack_nb_pred": np.argmax(soundtrack_probs, axis=1),
            "soundtrack_nb_hit_count": soundtrack_hits,
            "soundtrack_nb_zero_hit": (soundtrack_hits == 0).astype(int),
        },
        index=df_any.index,
    )

# =========================
# multihot for the 3 categorical variables
# =========================
def split_multiselect_cell(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if (not s) or s.lower() == "missing":
        return []
    return [part.strip() for part in s.split(",") if part.strip() and part.strip().lower() != "missing"]

class MultiHotCategoryBundle:
    def __init__(self, columns):
        self.columns = list(columns)
        self.encoders = {}

    def fit(self, df_train):
        for col in self.columns:
            classes = sorted(set(sum(df_train[col].apply(split_multiselect_cell).tolist(), [])))
            self.encoders[col] = classes
        return self

    def transform(self, df_any):
        frames = []
        for col in self.columns:
            classes = self.encoders[col]
            index_map = {cls: i for i, cls in enumerate(classes)}
            arr = np.zeros((len(df_any), len(classes)), dtype=np.int8)
            rows = df_any[col].apply(split_multiselect_cell)
            for i, items in enumerate(rows):
                for item in items:
                    j = index_map.get(item)
                    if j is not None:
                        arr[i, j] = 1
            safe_col_prefix = re.sub(r"[^0-9a-zA-Z]+", "_", col).strip("_").lower()[:24]
            col_names = [f"{safe_col_prefix}__{re.sub(r'[^0-9a-zA-Z]+', '_', cls).strip('_').lower()}" for cls in classes]
            frames.append(pd.DataFrame(arr, columns=col_names, index=df_any.index))
        return pd.concat(frames, axis=1)

def get_base_feature_cols(df_any):
    exclude_cols = {
        LABEL_COL,
        UNIQUE_ID_COL,
        "Painting",
        FOOD_COL,
        FEEL_COL,
        SOUND_COL,
        ROOM_COL,
        WHO_COL,
        SEASON_COL,
    }
    candidate_cols = [col for col in df_any.columns if col not in exclude_cols]
    numeric_cols = []
    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(df_any[col]) or pd.api.types.is_bool_dtype(df_any[col]):
            numeric_cols.append(col)
    return numeric_cols

def build_logistic_feature_matrix(df_any, food_model, feeling_model, soundtrack_model, multihot_bundle):
    base_cols = get_base_feature_cols(df_any)
    base_X = df_any[base_cols].copy()
    multihot_X = multihot_bundle.transform(df_any)
    nb_X = build_nb_feature_frame(df_any, food_model, feeling_model, soundtrack_model)
    X = pd.concat([
        base_X.reset_index(drop=True),
        multihot_X.reset_index(drop=True),
        nb_X.reset_index(drop=True),
    ], axis=1)
    return X

def fit_standardizer(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma[sigma < 1e-12] = 1.0
    return mu, sigma

def apply_standardizer(X, mu, sigma):
    return (X - mu) / sigma

def softmax_full(scores):
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def fit_multinomial_logistic_newton(X, y, C=LOGREG_C, max_iter=NEWTON_MAX_ITER, tol=NEWTON_TOL, intercept_damping=INTERCEPT_DAMPING):
    y = np.asarray(y, dtype=int)
    N, D = X.shape
    K = int(np.max(y)) + 1

    Xb = np.concatenate([np.ones((N, 1), dtype=np.float64), X], axis=1)
    D1 = D + 1

    W = np.zeros((D1, K), dtype=np.float64)
    # Exact sklearn lbfgs scaling for multinomial logistic:
    # l2_reg_strength = 1.0 / (C * sw_sum), with sw_sum = N here.
    lam = 1.0 / (float(C) * float(N))

    reg_diag = np.ones(D1, dtype=np.float64)
    reg_diag[0] = intercept_damping

    Y = np.zeros((N, K), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    def objective(Wcur):
        P = softmax_full(Xb @ Wcur)
        loss = -np.mean(np.log(np.clip(P[np.arange(N), y], 1e-15, 1.0)))
        reg = 0.5 * lam * (np.sum(Wcur[1:, :] ** 2) + intercept_damping * np.sum(Wcur[0:1, :] ** 2))
        return loss + reg

    for _ in range(max_iter):
        P = softmax_full(Xb @ W)

        G = np.zeros((D1, K), dtype=np.float64)
        H = np.zeros((D1 * K, D1 * K), dtype=np.float64)
        Xt = Xb.T

        for a in range(K):
            G[:, a] = (Xt @ (P[:, a] - Y[:, a])) / N + lam * reg_diag * W[:, a]

        for a in range(K):
            pa = P[:, a]
            for b in range(K):
                if a == b:
                    w_ab = pa * (1.0 - pa)
                else:
                    w_ab = -pa * P[:, b]
                H_block = (Xt @ (Xb * w_ab[:, None])) / N
                if a == b:
                    H_block = H_block + lam * np.diag(reg_diag)
                r0 = a * D1
                c0 = b * D1
                H[r0:r0 + D1, c0:c0 + D1] = H_block

        H = H + 1e-8 * np.eye(H.shape[0], dtype=np.float64)
        g = G.reshape(-1, order="F")

        try:
            delta = np.linalg.solve(H, g)
        except Exception:
            delta = np.linalg.lstsq(H, g, rcond=None)[0]

        step = 1.0
        W_new = W - delta.reshape(D1, K, order="F")
        old_obj = objective(W)
        new_obj = objective(W_new)
        while new_obj > old_obj and step > 1e-8:
            step *= 0.5
            W_new = W - step * delta.reshape(D1, K, order="F")
            new_obj = objective(W_new)

        max_change = np.max(np.abs(W_new - W))
        W = W_new
        if max_change < tol:
            break

    return W

def predict_multinomial_logistic(W, X):
    Xb = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), X], axis=1)
    probs = softmax_full(Xb @ W)
    pred = np.argmax(probs, axis=1)
    return pred, probs

def train_logistic_with_nb_features(
    df_train,
    food_k=FOOD_K,
    feeling_k=FEELING_K,
    soundtrack_k=SOUNDTRACK_K,
    nb_alpha=NB_ALPHA,
    C=LOGREG_C,
):
    food_model, feeling_model, soundtrack_model = train_text_nb_models(
        df_train,
        food_k=food_k,
        feeling_k=feeling_k,
        soundtrack_k=soundtrack_k,
        alpha=nb_alpha,
    )
    multihot_bundle = MultiHotCategoryBundle(MULTIHOT_COLS).fit(df_train)

    X_train = build_logistic_feature_matrix(df_train, food_model, feeling_model, soundtrack_model, multihot_bundle)
    y_train = df_train[LABEL_COL].to_numpy(dtype=int)

    mu, sigma = fit_standardizer(X_train.to_numpy(dtype=np.float64))
    X_train_scaled = apply_standardizer(X_train.to_numpy(dtype=np.float64), mu, sigma)

    W = fit_multinomial_logistic_newton(X_train_scaled, y_train, C=C)

    return {
        "W": W,
        "mu": mu,
        "sigma": sigma,
        "food_model": food_model,
        "feeling_model": feeling_model,
        "soundtrack_model": soundtrack_model,
        "multihot_bundle": multihot_bundle,
    }

def predict_logistic_from_models(df_any, trained):
    X_any = build_logistic_feature_matrix(
        df_any,
        trained["food_model"],
        trained["feeling_model"],
        trained["soundtrack_model"],
        trained["multihot_bundle"],
    )
    X_scaled = apply_standardizer(X_any.to_numpy(dtype=np.float64), trained["mu"], trained["sigma"])
    pred, probs = predict_multinomial_logistic(trained["W"], X_scaled)
    out = df_any.copy()
    out["pred"] = pred
    return out, probs

def stratified_kfold_indices(y, n_splits=5, shuffle=True, random_state=42):
    y = np.asarray(y)
    # y_encoded: class labels encoded by order of appearance, just like sklearn
    _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
    _, class_perm = np.unique(y_idx, return_inverse=True)
    y_encoded = class_perm[y_inv]

    n_classes = len(y_idx)
    y_order = np.sort(y_encoded)
    allocation = np.asarray([
        np.bincount(y_order[i::n_splits], minlength=n_classes)
        for i in range(n_splits)
    ])

    rng = np.random.RandomState(random_state)
    test_folds = np.empty(len(y), dtype=int)
    for k in range(n_classes):
        folds_for_class = np.arange(n_splits).repeat(allocation[:, k])
        if shuffle:
            rng.shuffle(folds_for_class)
        test_folds[y_encoded == k] = folds_for_class

    splits = []
    all_idx = np.arange(len(y))
    for fold_id in range(n_splits):
        val_idx = np.where(test_folds == fold_id)[0]
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[val_idx] = False
        train_idx = all_idx[train_mask]
        splits.append((train_idx, val_idx))
    return splits

def evaluate_one_param_combo_kfold(
    df,
    food_k=FOOD_K,
    feeling_k=FEELING_K,
    soundtrack_k=SOUNDTRACK_K,
    nb_alpha=NB_ALPHA,
    C=LOGREG_C,
    n_splits=5,
    random_state=42,
):
    y = df[LABEL_COL].to_numpy(dtype=int)
    splits = stratified_kfold_indices(y, n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores = []
    for train_idx, val_idx in splits:
        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()

        trained = train_logistic_with_nb_features(
            df_train=df_train,
            food_k=food_k,
            feeling_k=feeling_k,
            soundtrack_k=soundtrack_k,
            nb_alpha=nb_alpha,
            C=C,
        )
        val_pred_df, _ = predict_logistic_from_models(df_val, trained)
        acc = float((val_pred_df["pred"].to_numpy(dtype=int) == df_val[LABEL_COL].to_numpy(dtype=int)).mean())
        fold_scores.append(acc)

    return {
        "food_k": food_k,
        "feeling_k": feeling_k,
        "soundtrack_k": soundtrack_k,
        "nb_alpha": nb_alpha,
        "C": C,
        "mean_val_acc": float(np.mean(fold_scores)),
        "std_val_acc": float(np.std(fold_scores)),
        "fold_scores": fold_scores,
    }



def train_val_split_indices(y, train_size=0.8, random_state=42, stratify=True):
    y = np.asarray(y)
    n = len(y)
    rng = np.random.RandomState(random_state)

    if not stratify:
        indices = np.arange(n)
        rng.shuffle(indices)
        n_train = int(round(n * train_size))
        train_idx = np.sort(indices[:n_train])
        val_idx = np.sort(indices[n_train:])
        return train_idx, val_idx

    classes, y_encoded = np.unique(y, return_inverse=True)
    train_parts = []
    val_parts = []
    for class_id in range(len(classes)):
        class_idx = np.where(y_encoded == class_id)[0]
        rng.shuffle(class_idx)
        n_train_class = int(round(len(class_idx) * train_size))
        if len(class_idx) >= 2:
            n_train_class = max(1, min(len(class_idx) - 1, n_train_class))
        train_parts.append(class_idx[:n_train_class])
        val_parts.append(class_idx[n_train_class:])

    train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=int)
    val_idx = np.sort(np.concatenate(val_parts)) if val_parts else np.array([], dtype=int)
    return train_idx, val_idx

def evaluate_one_param_combo_holdout(
    df,
    food_k=FOOD_K,
    feeling_k=FEELING_K,
    soundtrack_k=SOUNDTRACK_K,
    nb_alpha=NB_ALPHA,
    C=LOGREG_C,
    train_size=0.8,
    random_state=42,
    stratify=True,
):
    y = df[LABEL_COL].to_numpy(dtype=int)
    train_idx, val_idx = train_val_split_indices(
        y,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify,
    )
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()

    trained = train_logistic_with_nb_features(
        df_train=df_train,
        food_k=food_k,
        feeling_k=feeling_k,
        soundtrack_k=soundtrack_k,
        nb_alpha=nb_alpha,
        C=C,
    )
    val_pred_df, _ = predict_logistic_from_models(df_val, trained)
    acc = float((val_pred_df["pred"].to_numpy(dtype=int) == df_val[LABEL_COL].to_numpy(dtype=int)).mean())
    return {
        "food_k": food_k,
        "feeling_k": feeling_k,
        "soundtrack_k": soundtrack_k,
        "nb_alpha": nb_alpha,
        "C": C,
        "train_size": train_size,
        "random_state": random_state,
        "stratify": stratify,
        "train_rows": int(len(df_train)),
        "val_rows": int(len(df_val)),
        "val_acc": acc,
    }


# Public API
_MODEL_CACHE = None
_TRAIN_INFO_CACHE = None

def _get_trained_model():
    global _MODEL_CACHE, _TRAIN_INFO_CACHE
    if _MODEL_CACHE is None:
        df_train, chosen_path, used_clean_directly = resolve_training_dataframe()
        _TRAIN_INFO_CACHE = (chosen_path, used_clean_directly, len(df_train))
        _MODEL_CACHE = train_logistic_with_nb_features(df_train, C=LOGREG_C)
    return _MODEL_CACHE

def predict(x):
    model = _get_trained_model()
    df_one = pd.DataFrame([x])
    if LABEL_COL not in df_one.columns:
        df_one[LABEL_COL] = 0
    if UNIQUE_ID_COL not in df_one.columns:
        df_one[UNIQUE_ID_COL] = -1
    # Minimal NA fill so transform functions work.
    for col in MULTIHOT_COLS + [FOOD_COL, FEEL_COL, SOUND_COL]:
        if col not in df_one.columns:
            df_one[col] = "Missing"
    for col in [INTENSITY_COL, PROMINENT_COLOURS_COL, OBJECTS_COL, PAYMENT_COL] + LIKERT_COLS:
        if col not in df_one.columns:
            df_one[col] = np.nan
    df_one = clean_raw_dataframe(df_one) if LABEL_COL not in x else df_one
    pred_df, _ = predict_logistic_from_models(df_one, model)
    return LABEL_TO_NAME[int(pred_df["pred"].iloc[0])]

def predict_all(filename):
    data = csv.DictReader(open(filename, newline="", encoding="utf-8"))
    predictions = []
    for test_example in data:
        pred = predict(test_example)
        predictions.append(pred)
    return predictions

if __name__ == "__main__":
    df_train, chosen_path, used_clean_directly = resolve_training_dataframe()
    print("Training CSV:", chosen_path)
    print("Used cleaned CSV directly:", used_clean_directly)
    print("Rows:", len(df_train))

    if RUN_KFOLD_VALIDATION_IN_MAIN:
        result = evaluate_one_param_combo_kfold(
            df_train,
            food_k=FOOD_K,
            feeling_k=FEELING_K,
            soundtrack_k=SOUNDTRACK_K,
            nb_alpha=NB_ALPHA,
            C=LOGREG_C,
            n_splits=KFOLD_N_SPLITS,
            random_state=KFOLD_RANDOM_STATE,
        )
        print("KFold accuracies:", [round(x, 4) for x in result["fold_scores"]])
        print("KFold mean:", round(result["mean_val_acc"], 4), "Std:", round(result["std_val_acc"], 4))

    holdout_runs = []
    holdout_scores = []
    base_random_state = 42
    n_holdout_repeats = 50

    for repeat_i in range(n_holdout_repeats):
        holdout_result = evaluate_one_param_combo_holdout(
            df_train,
            food_k=FOOD_K,
            feeling_k=FEELING_K,
            soundtrack_k=SOUNDTRACK_K,
            nb_alpha=NB_ALPHA,
            C=LOGREG_C,
            train_size=0.8,
            random_state=base_random_state + repeat_i,
            stratify=True,
        )
        holdout_runs.append(holdout_result)
        holdout_scores.append(float(holdout_result["val_acc"]))

    print("Random 80/20 validation accuracies (50 runs):", [round(x, 4) for x in holdout_scores])
    print(
        "Random 80/20 mean:",
        round(float(np.mean(holdout_scores)), 4),
        "Std:",
        round(float(np.std(holdout_scores)), 4),
    )
    print(
        "Last split sizes:",
        f"train={holdout_runs[-1]['train_rows']},",
        f"val={holdout_runs[-1]['val_rows']}"
    )
