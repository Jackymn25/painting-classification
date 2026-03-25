import re
import time
import warnings
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# =========================
# Paths / Columns
# =========================
CSV_PATH = "cleaned_data_final.csv"

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
# Best params placeholder
# Keep all hyperparameters at the top, same style as original
# =========================
FOOD_K = 42
FEELING_K = 38
SOUNDTRACK_K = 62

NB_ALPHA = 1.0

KNN_N_NEIGHBORS = 11
KNN_WEIGHTS = "distance"
KNN_METRIC = "minkowski"
KNN_P = 2

MIN_TOKEN_LEN = 2
EXCLUDE_MISSING_FROM_VOCAB = True
ZERO_HIT_RETURNS_PRIOR = True
SORT_TIES_ALPHABETICALLY = True

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
# kept in this file so the whole pipeline is self-contained
# =========================
def normalize_text(text):
    """Normalize whitespace / dashes for payment parsing."""
    if pd.isna(text):
        return ""
    s = str(text).strip().lower()
    s = s.replace("\xa0", " ")
    s = s.replace("–", "-").replace("—", "-")
    s = " ".join(s.split())
    return s



def collapse_spaced_thousands(text):
    """
    Convert spaced thousand groups like:
    5 000 -> 5000
    10 000 000 -> 10000000
    """
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
    return any(re.search(pat, text) for pat in patterns)



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
        r"\b0+\b",
        r"\bzero\b",
        r"\bnothing\b",
        r"\bno money\b",
        r"\bfree\b",
    ]
    return contains_any(s, zero_patterns)



def scale_multiplier(scale):
    scale = (scale or "").lower().strip()
    if scale in {"k", "thousand"}:
        return 1_000
    if scale in {"mil", "million"}:
        return 1_000_000
    if scale in {"b", "billion"}:
        return 1_000_000_000
    return 1



def score_candidate(context):
    score = 0
    realistic_patterns = [
        r"\bmax\b",
        r"\bat most\b",
        r"\bno more than\b",
        r"\bwould pay\b",
        r"\bi'd pay\b",
        r"\bwilling to pay\b",
        r"\bbecause i'm not\b",
        r"\bbecause i am not\b",
        r"\brealistically\b",
    ]
    hypothetical_patterns = [
        r"\bif i were\b",
        r"\bif i was\b",
        r"\bif i had\b",
        r"\bbillionaire\b",
        r"\bin a perfect world\b",
    ]
    if contains_any(context, realistic_patterns):
        score += 5
    if contains_any(context, hypothetical_patterns):
        score -= 5
    return score



def choose_best_payment_value(candidates, full_text):
    if not candidates:
        return np.nan

    scored = []
    for idx, item in enumerate(candidates):
        scored.append((score_candidate(item["context"]), idx, item["value"]))

    best_score = max(x[0] for x in scored)

    if best_score > 0:
        best_group = [x for x in scored if x[0] == best_score]
        best_group.sort(key=lambda x: x[1])
        return best_group[-1][2]

    if re.search(r"\bdepends\b", full_text) or re.search(r"\bif i were\b", full_text):
        return candidates[-1]["value"]

    return max(item["value"] for item in candidates)



def parse_money_value(text):
    """
    Extract a payment value from free text.
    """
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
        rf"""
        (?:
            (?:cad\s*\$?|\$|usd\s*\$?)\s*
            (?P<num1>{num_pattern})
            \s*(?P<scale1>{scale_pattern})
        )
        |
        (?:
            (?P<num2>{num_pattern})
            \s*(?P<scale2>{scale_pattern})
            \s*(?:cad|usd|dollars?|bucks?)\b
        )
        |
        (?:
            (?P<num3>{num_pattern})
            \s+(?P<scale3>k|thousand|mil|million|b|billion)\b
        )
        |
        (?:
            (?P<num4>{num_pattern})(?P<scale4>k|mil|b)\b
        )
        |
        (?:
            \b(?P<num5>{num_pattern})\b
        )
        """,
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
        except ValueError:
            continue

        value *= scale_multiplier(scale)
        start = max(0, match.start() - 35)
        end = min(len(s), match.end() + 35)
        context = s[start:end]
        candidates.append({"value": value, "context": context})

    if candidates:
        return choose_best_payment_value(candidates, s)

    if looks_like_uncertain_text(s):
        return np.nan

    return np.nan



def parse_payment_column(series):
    x = series.apply(parse_money_value)
    negative_count = int((x < 0).sum())
    x.loc[x < 0] = 0
    cap_value = 324_000_000
    capped_count = int((x > cap_value).sum())
    x = x.clip(lower=0, upper=cap_value)
    return x, negative_count, cap_value, capped_count



def clean_count_column(series, upper_q=0.99):
    x = pd.to_numeric(series, errors="coerce")
    negative_count = int((x < 0).sum())
    x.loc[x < 0] = 0

    non_na = x.dropna()
    if len(non_na) == 0:
        return x, negative_count, None, 0

    upper = int(np.ceil(non_na.quantile(upper_q)))
    before = x.copy()
    x = x.clip(lower=0, upper=upper).round()
    capped_count = int((before > x).sum())
    return x, negative_count, upper, capped_count



def clean_bounded_scale(series, lower, upper):
    x = pd.to_numeric(series, errors="coerce")
    invalid_count = int(((x < lower) | (x > upper)).sum())
    x = x.where((x >= lower) & (x <= upper), np.nan)
    return x, invalid_count



def clean_raw_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct a cleaned dataframe in the same style as the uploaded cleaned CSV.
    This is included so the whole workflow lives in one file.
    """
    df = df_raw.copy()

    if "Painting" in df.columns:
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



def clean_raw_csv_to_cleaned_csv(input_csv_path: str, output_csv_path: str = "cleaned_data_final.csv") -> pd.DataFrame:
    df_raw = pd.read_csv(input_csv_path)
    df_clean = clean_raw_dataframe(df_raw)
    df_clean.to_csv(output_csv_path, index=False)
    return df_clean


# =========================
# Stop words
# embedded from the original uploaded code so this file is standalone
# =========================
stop_words = {
    "'d", "'ll", "'m", "'re", "'s", "'ve", "a", "about", "above", "across", "after",
    "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also",
    "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and",
    "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around",
    "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been",
    "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond",
    "bill", "both", "bottom", "but", "by", "ca", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "did", "do", "does", "doing", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere",
    "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere",
    "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for", "former",
    "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein",
    "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "ie",
    "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "just", "keep",
    "last", "latter", "latterly", "least", "less", "ltd", "made", "make", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much",
    "must", "my", "myself", "n't", "name", "namely", "neither", "never", "nevertheless", "next",
    "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "n‘t",
    "n’t", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others",
    "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps", "please",
    "put", "quite", "rather", "re", "really", "regarding", "same", "say", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere",
    "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes",
    "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon",
    "these", "they", "thick", "thin", "third", "this", "those", "though", "three", "through",
    "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve",
    "twenty", "two", "un", "under", "unless", "until", "up", "upon", "us", "used", "using", "various",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever",
    "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which",
    "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within",
    "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "‘d", "‘ll", "‘m",
    "‘re", "‘s", "‘ve", "’d", "’ll", "’m", "’re", "’s", "’ve"
}

# =========================
# Text preprocessing / NB
# =========================
def simple_singularize(token: str) -> str:
    """Keep the original light singularization logic."""
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



def tokenize_text(text) -> list[str]:
    """
    Original logic:
    1) lowercase
    2) remove punctuation
    3) light singularization
    4) remove stopwords
    """
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



def select_top_k_vocab_from_training(texts, k: int) -> tuple[list[str], dict]:
    """
    Use document frequency on training texts only.
    """
    if k <= 0:
        raise ValueError("k must be > 0.")

    doc_counter = Counter()

    for text in pd.Series(texts).fillna("missing"):
        uniq_tokens = set(tokenize_text(text))

        if EXCLUDE_MISSING_FROM_VOCAB:
            uniq_tokens.discard("missing")

        for tok in uniq_tokens:
            doc_counter[tok] += 1

    if not doc_counter:
        raise ValueError("No usable tokens found in the training data after preprocessing.")

    items = list(doc_counter.items())

    if SORT_TIES_ALPHABETICALLY:
        items.sort(key=lambda x: (-x[1], x[0]))
    else:
        items.sort(key=lambda x: -x[1])

    vocab = [tok for tok, _ in items[:k]]
    return vocab, dict(doc_counter)


class TopKBernoulliNB:
    """
    Same idea as the original code:
    - choose top-k vocabulary from training set only
    - binary multi-hot vector
    - BernoulliNB
    - zero-hit -> class prior
    """

    def __init__(self, k: int, alpha: float = NB_ALPHA):
        self.k = k
        self.alpha = alpha

        self.vocab_ = None
        self.vocab_index_ = None
        self.doc_freq_ = None
        self.model_ = None

    def fit(self, texts, y):
        self.vocab_, self.doc_freq_ = select_top_k_vocab_from_training(texts, self.k)
        self.vocab_index_ = {tok: i for i, tok in enumerate(self.vocab_)}

        X = self._vectorize(texts)

        if X.shape[1] == 0:
            raise ValueError("Vocabulary is empty after top-k selection.")

        self.model_ = BernoulliNB(alpha=self.alpha)
        self.model_.fit(X, np.asarray(y, dtype=int))
        return self

    def _vectorize_one(self, text) -> np.ndarray:
        row = np.zeros(len(self.vocab_), dtype=np.int8)

        uniq_tokens = set(tokenize_text(text))
        for tok in uniq_tokens:
            j = self.vocab_index_.get(tok)
            if j is not None:
                row[j] = 1

        return row

    def _vectorize(self, texts) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        texts = pd.Series(texts).fillna("missing")
        X = np.zeros((len(texts), len(self.vocab_)), dtype=np.int8)

        for i, text in enumerate(texts):
            X[i] = self._vectorize_one(text)

        return X

    def hit_count(self, texts) -> np.ndarray:
        X = self._vectorize(texts)
        return X.sum(axis=1)

    def predict_proba(self, texts) -> np.ndarray:
        X = self._vectorize(texts)
        probs = self.model_.predict_proba(X)

        if ZERO_HIT_RETURNS_PRIOR:
            hits = X.sum(axis=1)
            zero_mask = hits == 0
            if np.any(zero_mask):
                prior = self.model_.class_count_ / self.model_.class_count_.sum()
                probs[zero_mask] = prior

        return probs

    def predict(self, texts) -> np.ndarray:
        probs = self.predict_proba(texts)
        return np.argmax(probs, axis=1)

    def predict_one(self, text) -> int:
        return int(self.predict([text])[0])



def load_cleaned_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    return pd.read_csv(csv_path)



def train_food_nb(df_train: pd.DataFrame, k: int = FOOD_K, alpha: float = NB_ALPHA) -> TopKBernoulliNB:
    model = TopKBernoulliNB(k=k, alpha=alpha)
    model.fit(df_train[FOOD_COL], df_train[LABEL_COL])
    return model



def train_feeling_nb(df_train: pd.DataFrame, k: int = FEELING_K, alpha: float = NB_ALPHA) -> TopKBernoulliNB:
    model = TopKBernoulliNB(k=k, alpha=alpha)
    model.fit(df_train[FEEL_COL], df_train[LABEL_COL])
    return model



def train_soundtrack_nb(df_train: pd.DataFrame, k: int = SOUNDTRACK_K, alpha: float = NB_ALPHA) -> TopKBernoulliNB:
    model = TopKBernoulliNB(k=k, alpha=alpha)
    model.fit(df_train[SOUND_COL], df_train[LABEL_COL])
    return model



def predict_food(text: str, food_model: TopKBernoulliNB) -> int:
    return int(food_model.predict_one(text))



def predict_proba_food(text: str, food_model: TopKBernoulliNB) -> np.ndarray:
    return food_model.predict_proba([text])[0]



def predict_feeling(text: str, feeling_model: TopKBernoulliNB) -> int:
    return int(feeling_model.predict_one(text))



def predict_proba_feeling(text: str, feeling_model: TopKBernoulliNB) -> np.ndarray:
    return feeling_model.predict_proba([text])[0]



def predict_soundtrack(text: str, soundtrack_model: TopKBernoulliNB) -> int:
    return int(soundtrack_model.predict_one(text))



def predict_proba_soundtrack(text: str, soundtrack_model: TopKBernoulliNB) -> np.ndarray:
    return soundtrack_model.predict_proba([text])[0]



def train_text_nb_models(
    df_train: pd.DataFrame,
    food_k: int = FOOD_K,
    feeling_k: int = FEELING_K,
    soundtrack_k: int = SOUNDTRACK_K,
    alpha: float = NB_ALPHA,
):
    food_model = TopKBernoulliNB(k=food_k, alpha=alpha)
    feeling_model = TopKBernoulliNB(k=feeling_k, alpha=alpha)
    soundtrack_model = TopKBernoulliNB(k=soundtrack_k, alpha=alpha)

    food_model.fit(df_train[FOOD_COL], df_train[LABEL_COL])
    feeling_model.fit(df_train[FEEL_COL], df_train[LABEL_COL])
    soundtrack_model.fit(df_train[SOUND_COL], df_train[LABEL_COL])

    return food_model, feeling_model, soundtrack_model



def build_nb_feature_frame(
    df_any: pd.DataFrame,
    food_model: TopKBernoulliNB,
    feeling_model: TopKBernoulliNB,
    soundtrack_model: TopKBernoulliNB,
) -> pd.DataFrame:
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
def split_multiselect_cell(x) -> list[str]:
    """
    Turn:
    "Bedroom,Living room"
    into:
    ["Bedroom", "Living room"]
    """
    if pd.isna(x):
        return []

    s = str(x).strip()
    if not s or s.lower() == "missing":
        return []

    return [
        part.strip()
        for part in s.split(",")
        if part.strip() and part.strip().lower() != "missing"
    ]


class MultiHotCategoryBundle:
    """
    Fit MultiLabelBinarizer on train only for each multi-select categorical column.
    """

    def __init__(self, columns: list[str]):
        self.columns = list(columns)
        self.encoders = {}

    def fit(self, df_train: pd.DataFrame):
        for col in self.columns:
            mlb = MultiLabelBinarizer()
            mlb.fit(df_train[col].apply(split_multiselect_cell))
            self.encoders[col] = mlb
        return self

    def transform(self, df_any: pd.DataFrame) -> pd.DataFrame:
        frames = []

        for col in self.columns:
            mlb = self.encoders[col]
            rows = df_any[col].apply(split_multiselect_cell)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                arr = mlb.transform(rows)

            safe_col_prefix = re.sub(r"[^0-9a-zA-Z]+", "_", col).strip("_").lower()[:24]
            col_names = [
                f"{safe_col_prefix}__{re.sub(r'[^0-9a-zA-Z]+', '_', cls).strip('_').lower()}"
                for cls in mlb.classes_
            ]

            frames.append(pd.DataFrame(arr, columns=col_names, index=df_any.index))

        return pd.concat(frames, axis=1)


# =========================
# KNN features
# =========================
def get_base_feature_cols(df_any: pd.DataFrame) -> list[str]:
    """
    Keep original numeric features,
    but explicitly:
    - remove unique_id
    - remove raw text columns
    - remove raw multihot categorical columns
    - remove Painting
    """
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



def build_knn_feature_matrix(
    df_any: pd.DataFrame,
    food_model: TopKBernoulliNB,
    feeling_model: TopKBernoulliNB,
    soundtrack_model: TopKBernoulliNB,
    multihot_bundle: MultiHotCategoryBundle,
) -> pd.DataFrame:
    base_cols = get_base_feature_cols(df_any)
    base_X = df_any[base_cols].copy()

    multihot_X = multihot_bundle.transform(df_any)

    nb_X = build_nb_feature_frame(
        df_any,
        food_model,
        feeling_model,
        soundtrack_model,
    )

    X = pd.concat(
        [
            base_X.reset_index(drop=True),
            multihot_X.reset_index(drop=True),
            nb_X.reset_index(drop=True),
        ],
        axis=1,
    )
    return X



def train_knn_with_nb_features(
    df_train: pd.DataFrame,
    food_k: int,
    feeling_k: int,
    soundtrack_k: int,
    nb_alpha: float,
    n_neighbors: int,
    weights: str,
    metric: str,
    p: int,
):
    food_model, feeling_model, soundtrack_model = train_text_nb_models(
        df_train,
        food_k=food_k,
        feeling_k=feeling_k,
        soundtrack_k=soundtrack_k,
        alpha=nb_alpha,
    )

    multihot_bundle = MultiHotCategoryBundle(MULTIHOT_COLS).fit(df_train)

    X_train = build_knn_feature_matrix(
        df_train,
        food_model,
        feeling_model,
        soundtrack_model,
        multihot_bundle,
    )
    y_train = df_train[LABEL_COL].to_numpy()

    knn_model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        p=p,
        n_jobs=-1,
    )
    knn_model.fit(X_train, y_train)

    return knn_model, food_model, feeling_model, soundtrack_model, multihot_bundle



def predict_knn_from_models(
    df_any: pd.DataFrame,
    knn_model: KNeighborsClassifier,
    food_model: TopKBernoulliNB,
    feeling_model: TopKBernoulliNB,
    soundtrack_model: TopKBernoulliNB,
    multihot_bundle: MultiHotCategoryBundle,
) -> pd.DataFrame:
    X_any = build_knn_feature_matrix(
        df_any,
        food_model,
        feeling_model,
        soundtrack_model,
        multihot_bundle,
    )

    pred = knn_model.predict(X_any)

    out = df_any.copy()
    out["pred"] = pred
    return out


# =========================
# CV / tuning
# =========================
def evaluate_one_param_combo_kfold(
    df: pd.DataFrame,
    food_k: int,
    feeling_k: int,
    soundtrack_k: int,
    nb_alpha: float,
    n_neighbors: int,
    weights: str,
    metric: str,
    p: int,
    n_splits: int = 3,
    random_state: int = 42,
) -> dict:
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_scores = []
    X_dummy = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL].to_numpy()

    for train_idx, val_idx in skf.split(X_dummy, y):
        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()

        (
            knn_model,
            food_model,
            feeling_model,
            soundtrack_model,
            multihot_bundle,
        ) = train_knn_with_nb_features(
            df_train=df_train,
            food_k=food_k,
            feeling_k=feeling_k,
            soundtrack_k=soundtrack_k,
            nb_alpha=nb_alpha,
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p,
        )

        val_pred_df = predict_knn_from_models(
            df_val,
            knn_model,
            food_model,
            feeling_model,
            soundtrack_model,
            multihot_bundle,
        )

        val_acc = accuracy_score(df_val[LABEL_COL], val_pred_df["pred"])
        fold_scores.append(val_acc)

    return {
        "food_k": food_k,
        "feeling_k": feeling_k,
        "soundtrack_k": soundtrack_k,
        "nb_alpha": nb_alpha,
        "n_neighbors": n_neighbors,
        "weights": weights,
        "metric": metric,
        "p": p,
        "fold_scores": fold_scores,
        "mean_val_acc": float(np.mean(fold_scores)),
        "std_val_acc": float(np.std(fold_scores)),
    }



def tune_knn_nb_hyperparameters_kfold(
    csv_path: str = CSV_PATH,
    n_splits: int = 3,
    random_state: int = 42,
    food_k_grid=None,
    feeling_k_grid=None,
    soundtrack_k_grid=None,
    nb_alpha_grid=None,
    n_neighbors_grid=None,
    weights_grid=None,
    metric_grid=None,
    p_grid=None,
):
    df = load_cleaned_data(csv_path)

    if food_k_grid is None:
        food_k_grid = [42]
    if feeling_k_grid is None:
        feeling_k_grid = [38]
    if soundtrack_k_grid is None:
        soundtrack_k_grid = [62]
    if nb_alpha_grid is None:
        nb_alpha_grid = [1.0]
    if n_neighbors_grid is None:
        n_neighbors_grid = [5, 7, 9, 11, 13]
    if weights_grid is None:
        weights_grid = ["uniform", "distance"]
    if metric_grid is None:
        metric_grid = ["minkowski"]
    if p_grid is None:
        p_grid = [1, 2]

    param_grid = list(
        product(
            food_k_grid,
            feeling_k_grid,
            soundtrack_k_grid,
            nb_alpha_grid,
            n_neighbors_grid,
            weights_grid,
            metric_grid,
            p_grid,
        )
    )

    print("total combinations:", len(param_grid))
    print("total model fits:", len(param_grid) * n_splits)

    all_results = []
    start_time = time.time()

    for combo_id, params in enumerate(param_grid, start=1):
        (
            food_k,
            feeling_k,
            soundtrack_k,
            nb_alpha,
            n_neighbors,
            weights,
            metric,
            p,
        ) = params

        result = evaluate_one_param_combo_kfold(
            df=df,
            food_k=food_k,
            feeling_k=feeling_k,
            soundtrack_k=soundtrack_k,
            nb_alpha=nb_alpha,
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p,
            n_splits=n_splits,
            random_state=random_state,
        )

        all_results.append(result)

        elapsed = time.time() - start_time
        avg_per_combo = elapsed / combo_id
        remain = avg_per_combo * (len(param_grid) - combo_id)

        print(
            f"[{combo_id}/{len(param_grid)}] "
            f"food_k={food_k}, feeling_k={feeling_k}, soundtrack_k={soundtrack_k}, "
            f"n_neighbors={n_neighbors}, weights={weights}, metric={metric}, p={p} "
            f"=> mean_val_acc={result['mean_val_acc']:.4f}, std={result['std_val_acc']:.4f} "
            f"| elapsed={elapsed/60:.1f} min, remain≈{remain/60:.1f} min"
        )

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(
        by=["mean_val_acc", "std_val_acc"],
        ascending=[False, True],
    ).reset_index(drop=True)

    best_row = results_df.iloc[0]

    print("\n=========================")
    print("Best Hyperparameter Combination")
    print("=========================")
    print("food_k:", best_row["food_k"])
    print("feeling_k:", best_row["feeling_k"])
    print("soundtrack_k:", best_row["soundtrack_k"])
    print("nb_alpha:", best_row["nb_alpha"])
    print("n_neighbors:", best_row["n_neighbors"])
    print("weights:", best_row["weights"])
    print("metric:", best_row["metric"])
    print("p:", best_row["p"])
    print("best mean validation accuracy:", round(best_row["mean_val_acc"], 4))
    print("best std:", round(best_row["std_val_acc"], 4))
    print("fold scores:", best_row["fold_scores"])

    return {
        "best_params": {
            "food_k": best_row["food_k"],
            "feeling_k": best_row["feeling_k"],
            "soundtrack_k": best_row["soundtrack_k"],
            "nb_alpha": best_row["nb_alpha"],
            "n_neighbors": best_row["n_neighbors"],
            "weights": best_row["weights"],
            "metric": best_row["metric"],
            "p": best_row["p"],
        },
        "best_mean_val_acc": best_row["mean_val_acc"],
        "best_std_val_acc": best_row["std_val_acc"],
        "results_df": results_df,
    }


# =========================
# Fit one final model with the retuned params
# =========================
def train_final_model(
    csv_path: str = CSV_PATH,
):
    df = load_cleaned_data(csv_path)

    return train_knn_with_nb_features(
        df_train=df,
        food_k=FOOD_K,
        feeling_k=FEELING_K,
        soundtrack_k=SOUNDTRACK_K,
        nb_alpha=NB_ALPHA,
        n_neighbors=KNN_N_NEIGHBORS,
        weights=KNN_WEIGHTS,
        metric=KNN_METRIC,
        p=KNN_P,
    )



def train_from_raw_or_clean_csv(
    csv_path: str,
    is_cleaned: bool = True,
    cleaned_output_path: str = "cleaned_data_final.csv",
):
    """
    Convenience entry:
    - if is_cleaned=True, read csv_path directly
    - if is_cleaned=False, clean it first and then train
    """
    if is_cleaned:
        df = load_cleaned_data(csv_path)
    else:
        df = clean_raw_csv_to_cleaned_csv(csv_path, cleaned_output_path)

    return train_knn_with_nb_features(
        df_train=df,
        food_k=FOOD_K,
        feeling_k=FEELING_K,
        soundtrack_k=SOUNDTRACK_K,
        nb_alpha=NB_ALPHA,
        n_neighbors=KNN_N_NEIGHBORS,
        weights=KNN_WEIGHTS,
        metric=KNN_METRIC,
        p=KNN_P,
    )


if __name__ == "__main__":
    print("Updated pipeline summary:")
    print("- remove unique_id from KNN input")
    print("- add room / who / season multihot features")
    print("- keep original 3-text-NB + KNN overall logic")
    print()
    print("Current params:")
    print(
        {
            "food_k": FOOD_K,
            "feeling_k": FEELING_K,
            "soundtrack_k": SOUNDTRACK_K,
            "nb_alpha": NB_ALPHA,
            "n_neighbors": KNN_N_NEIGHBORS,
            "weights": KNN_WEIGHTS,
            "metric": KNN_METRIC,
            "p": KNN_P,
        }
    )

    # Example search:
    # search_result = tune_knn_nb_hyperparameters_kfold(
    #     csv_path=CSV_PATH,
    #     n_splits=5,
    #     random_state=42,
    #     food_k_grid=[42],
    #     feeling_k_grid=[38],
    #     soundtrack_k_grid=[62],
    #     nb_alpha_grid=[1.0],
    #     n_neighbors_grid=[5, 7, 9, 11, 13, 15],
    #     weights_grid=["uniform", "distance"],
    #     metric_grid=["minkowski"],
    #     p_grid=[1, 2],
    # )

    search_result = tune_knn_nb_hyperparameters_kfold(
        csv_path=CSV_PATH,
        n_splits=5,
        random_state=42,
        food_k_grid=[42],
        feeling_k_grid=[38],
        soundtrack_k_grid=[62],
        nb_alpha_grid=[1.0],
        n_neighbors_grid=[5, 7, 9, 11, 13, 15, 17, 18, 19, 20, 22, 24, 25, 27, 29, 28],
        weights_grid=["uniform", "distance"],
        metric_grid=["minkowski"],
        p_grid=[1, 2],
    )
    results_df = search_result["results_df"]
    results_df.to_csv("knn_nb_tuning_results.csv", index=False)
    print("\nSaved tuning results to knn_nb_tuning_results.csv")
