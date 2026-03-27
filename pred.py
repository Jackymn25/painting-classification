"""
Random-Forest + Naive-Bayes predictor for the painting challenge.


Pipeline:
1. load the cleaned training CSV,
2. train 3 Bernoulli Naive Bayes text models on the three free-response columns,
3. turn room / who / season into multi-hot features,
4. combine numeric features + 9 NB probabilities + multihot features,
5. train a custom random forest and predict painting titles.
"""

import numpy
import pandas


LABEL_COL = "label"
UNIQUE_ID_COL = "unique_id"

FOOD_COL = "If this painting was a food, what would be?"
FEEL_COL = "Describe how this painting makes you feel."
SOUND_COL = "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting."

ROOM_COL = "If you could purchase this painting, which room would you put that painting in?"
WHO_COL = "If you could view this art in person, who would you want to view it with?"
SEASON_COL = "What season does this art piece remind you of?"

INTENSITY_COL = "On a scale of 1â€“10, how intense is the emotion conveyed by the artwork?"
PROMINENT_COLOURS_COL = "How many prominent colours do you notice in this painting?"
OBJECTS_COL = "How many objects caught your eye in the painting?"
PAYMENT_COL = "How much (in Canadian dollars) would you be willing to pay for this painting?"

LIKERT_COLS = [
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy.",
]

MULTIHOT_COLS = [ROOM_COL, WHO_COL, SEASON_COL]

TRAIN_CSV = __file__.replace("\\", "/").rsplit("/", 1)[0] + "/painting-classification/cleaned_data_final.csv"

TOPK1 = 42
TOPK2 = 38
TOPK3 = 62
NB_ALPHA = 1.0

RF_N_ESTIMATORS = 175
RF_MAX_DEPTH = 12
RF_MIN_SAMPLES_LEAF = 1
RF_MAX_FEATURES = "log2"
RF_BOOTSTRAP = True
RF_RANDOM_STATE = 42
RF_MAX_THRESHOLDS_PER_FEATURE = 8

MIN_TOKEN_LEN = 2

LABEL_TO_NAME = {
    0: "The Persistence of Memory",
    1: "The Starry Night",
    2: "The Water Lily Pond",
}

LIKERT_MAP = {
    "1 - strongly disagree": 1,
    "2 - disagree": 2,
    "3 - neutral": 3,
    "4 - agree": 4,
    "5 - strongly agree": 5,
}

ALIASES = {
    "blueberries": "blueberry",
    "berries": "berry",
    "noodles": "noodle",
    "violins": "violin",
    "drums": "drum",
    "clocks": "clock",
    "memories": "memory",
    "feelings": "feeling",
    "slowly": "slow",
    "icecream": "ice",
}

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "almost", "also", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "cannot", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "itself", "just", "like", "me", "more", "most", "my", "myself",
    "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with",
    "would", "you", "your", "yours", "yourself", "yourselves"
}

_MODEL_CACHE = None

_COLUMN_HINTS = {
    FOOD_COL: ["painting was a food", "what would be"],
    FEEL_COL: ["describe how this painting makes you feel"],
    SOUND_COL: ["imagine a soundtrack for this painting", "without naming any objects"],
    ROOM_COL: ["which room would you put"],
    WHO_COL: ["who would you want to view it with"],
    SEASON_COL: ["what season does this art piece remind you of"],
    INTENSITY_COL: ["how intense is the emotion conveyed by the artwork"],
    PROMINENT_COLOURS_COL: ["how many prominent colours"],
    OBJECTS_COL: ["how many objects caught your eye"],
    PAYMENT_COL: ["how much", "canadian dollars", "willing to pay"],
}


def _normalize_text(text):
    if pandas.isna(text):
        return ""
    return " ".join(str(text).replace("\xa0", " ").strip().lower().split())


def _rename_similar_columns(df):
    rename_map = {}
    columns = list(df.columns)

    for target, hints in _COLUMN_HINTS.items():
        if target in columns:
            continue
        for col in columns:
            lowered = _normalize_text(col)
            matched = True
            for hint in hints:
                if hint not in lowered:
                    matched = False
                    break
            if matched:
                rename_map[col] = target
                break

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _parse_numeric_string(text):
    if pandas.isna(text):
        return numpy.nan

    s = _normalize_text(text)
    if s in {"", "nan", "na", "n/a", "none", "missing"}:
        return numpy.nan

    chars = []
    for ch in s:
        if ("0" <= ch <= "9") or ch in {".", "-", " "}:
            chars.append(ch)
        elif ch in {",", "$"}:
            continue
        else:
            chars.append(" ")
    pieces = [piece for piece in "".join(chars).split() if piece not in {"-", ".", "-."}]
    if not pieces:
        return numpy.nan

    value = numpy.nan
    for piece in pieces:
        try:
            value = float(piece)
            break
        except Exception:
            continue
    if pandas.isna(value):
        return numpy.nan

    if " billion" in s or s.endswith("b") or " b " in (" " + s + " "):
        value *= 1_000_000_000.0
    elif " million" in s or " mil" in (" " + s + " "):
        value *= 1_000_000.0
    elif " thousand" in s or s.endswith("k") or " k " in (" " + s + " "):
        value *= 1_000.0

    return value


def _clean_numeric_column(series, lower=None, upper=None):
    x = pandas.to_numeric(series, errors="coerce")
    if lower is not None:
        x = x.where(x >= lower, numpy.nan)
    if upper is not None:
        x = x.where(x <= upper, numpy.nan)
    return x


def _clean_payment_column(series):
    parsed = series.apply(_parse_numeric_string)
    parsed = pandas.to_numeric(parsed, errors="coerce")
    parsed = parsed.where(parsed >= 0, 0.0)
    parsed = parsed.clip(0, 324_000_000)
    return parsed


def _clean_raw_dataframe(df_raw, reference_medians=None):
    df = _rename_similar_columns(df_raw.copy())

    if "Painting" in df.columns and LABEL_COL not in df.columns:
        label_map = {
            "The Persistence of Memory": 0,
            "The Starry Night": 1,
            "The Water Lily Pond": 2,
        }
        df[LABEL_COL] = df["Painting"].astype(str).str.strip().map(label_map)

    for col in LIKERT_COLS:
        if col not in df.columns:
            df[col] = numpy.nan
        if not pandas.api.types.is_numeric_dtype(df[col]):
            mapped = df[col].astype(str).str.strip().str.lower().map(LIKERT_MAP)
            df[col] = pandas.to_numeric(mapped, errors="coerce")
        df[col] = _clean_numeric_column(df[col], 1, 5)

    if INTENSITY_COL not in df.columns:
        df[INTENSITY_COL] = numpy.nan
    df[INTENSITY_COL] = _clean_numeric_column(df[INTENSITY_COL], 1, 10)

    if PROMINENT_COLOURS_COL not in df.columns:
        df[PROMINENT_COLOURS_COL] = numpy.nan
    df[PROMINENT_COLOURS_COL] = _clean_numeric_column(df[PROMINENT_COLOURS_COL], 0, None)

    if OBJECTS_COL not in df.columns:
        df[OBJECTS_COL] = numpy.nan
    df[OBJECTS_COL] = _clean_numeric_column(df[OBJECTS_COL], 0, None)

    if PAYMENT_COL not in df.columns:
        df[PAYMENT_COL] = numpy.nan
    if pandas.api.types.is_numeric_dtype(df[PAYMENT_COL]):
        df[PAYMENT_COL] = pandas.to_numeric(df[PAYMENT_COL], errors="coerce").clip(0, 324_000_000)
    else:
        df[PAYMENT_COL] = _clean_payment_column(df[PAYMENT_COL])

    text_cols = [FOOD_COL, FEEL_COL, SOUND_COL, ROOM_COL, WHO_COL, SEASON_COL]
    for col in text_cols:
        if col not in df.columns:
            df[col] = "Missing"
        df[col] = df[col].fillna("Missing").astype(str)

    numeric_cols = [
        INTENSITY_COL,
        PROMINENT_COLOURS_COL,
        OBJECTS_COL,
        PAYMENT_COL,
    ] + LIKERT_COLS

    medians = {}
    for col in numeric_cols:
        if reference_medians is not None and col in reference_medians:
            med = reference_medians[col]
        else:
            med = df[col].median()
        if pandas.isna(med):
            med = 0.0
        medians[col] = float(med)
        df[col] = df[col].fillna(med)

    return df.reset_index(drop=True), medians


def _simple_singularize(token):
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


def _tokenize_text(text):
    text = _normalize_text(text)
    chars = []
    for ch in text:
        if ("a" <= ch <= "z") or ch == "'" or ch == " ":
            chars.append(ch)
        else:
            chars.append(" ")

    tokens = []
    for raw_token in "".join(chars).split():
        token = raw_token.strip("'")
        if not token:
            continue
        token = _simple_singularize(token)
        if len(token) < MIN_TOKEN_LEN:
            continue
        if token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _top_k_vocab(texts, k):
    doc_counts = {}
    for text in pandas.Series(texts).fillna("missing"):
        token_set = set(_tokenize_text(text))
        if "missing" in token_set:
            token_set.remove("missing")
        for token in token_set:
            doc_counts[token] = doc_counts.get(token, 0) + 1
    items = sorted(doc_counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in items[:k]]


class _BernoulliNBTextModel:
    def __init__(self, k, alpha):
        self.k = int(k)
        self.alpha = float(alpha)
        self.vocab = []
        self.vocab_index = {}
        self.class_count = None
        self.feature_prob = None

    def _vectorize_one(self, text):
        row = numpy.zeros(len(self.vocab), dtype=numpy.float64)
        for token in set(_tokenize_text(text)):
            j = self.vocab_index.get(token)
            if j is not None:
                row[j] = 1.0
        return row

    def _vectorize(self, texts):
        texts = pandas.Series(texts).fillna("missing")
        X = numpy.zeros((len(texts), len(self.vocab)), dtype=numpy.float64)
        for i, text in enumerate(texts):
            X[i] = self._vectorize_one(text)
        return X

    def fit(self, texts, labels):
        self.vocab = _top_k_vocab(texts, self.k)
        self.vocab_index = {token: i for i, token in enumerate(self.vocab)}
        X = self._vectorize(texts)
        y = numpy.asarray(labels, dtype=int)
        n_classes = int(y.max()) + 1

        self.class_count = numpy.bincount(y, minlength=n_classes).astype(numpy.float64)
        self.feature_prob = numpy.zeros((n_classes, X.shape[1]), dtype=numpy.float64)

        for cls in range(n_classes):
            mask = (y == cls)
            class_total = float(mask.sum())
            hit_counts = X[mask].sum(axis=0)
            self.feature_prob[cls] = (hit_counts + self.alpha) / (class_total + 2.0 * self.alpha)

        self.feature_prob = numpy.clip(self.feature_prob, 1e-12, 1.0 - 1e-12)
        return self

    def predict_proba(self, texts):
        X = self._vectorize(texts)
        log_prior = numpy.log(self.class_count / self.class_count.sum())
        log_p = numpy.log(self.feature_prob)
        log_not_p = numpy.log(1.0 - self.feature_prob)
        log_scores = X @ log_p.T + (1.0 - X) @ log_not_p.T + log_prior.reshape(1, -1)

        zero_hit_mask = (X.sum(axis=1) == 0)
        if zero_hit_mask.any():
            log_scores[zero_hit_mask] = log_prior

        log_scores = log_scores - log_scores.max(axis=1, keepdims=True)
        probs = numpy.exp(log_scores)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs


class _MultiHotEncoder:
    def __init__(self, columns):
        self.columns = list(columns)
        self.classes = {}

    def _split_cell(self, value):
        if pandas.isna(value):
            return []
        text = str(value).strip()
        if text == "" or text.lower() == "missing":
            return []
        return [part.strip() for part in text.split(",") if part.strip() and part.strip().lower() != "missing"]

    def fit(self, df):
        for col in self.columns:
            seen = set()
            for items in df[col].apply(self._split_cell):
                for item in items:
                    seen.add(item)
            self.classes[col] = sorted(seen)
        return self

    def transform(self, df):
        frames = []
        for col in self.columns:
            values = self.classes[col]
            value_to_index = {value: idx for idx, value in enumerate(values)}
            arr = numpy.zeros((len(df), len(values)), dtype=numpy.float64)
            for i, items in enumerate(df[col].apply(self._split_cell)):
                for item in items:
                    j = value_to_index.get(item)
                    if j is not None:
                        arr[i, j] = 1.0
            col_names = [col + "__" + value for value in values]
            frames.append(pandas.DataFrame(arr, columns=col_names, index=df.index))
        if not frames:
            return pandas.DataFrame(index=df.index)
        return pandas.concat(frames, axis=1)


def _build_nb_feature_frame(df, food_model, feel_model, sound_model):
    food_probs = food_model.predict_proba(df[FOOD_COL].fillna("missing"))
    feel_probs = feel_model.predict_proba(df[FEEL_COL].fillna("missing"))
    sound_probs = sound_model.predict_proba(df[SOUND_COL].fillna("missing"))

    return pandas.DataFrame(
        {
            "food_nb_p0": food_probs[:, 0],
            "food_nb_p1": food_probs[:, 1],
            "food_nb_p2": food_probs[:, 2],
            "feel_nb_p0": feel_probs[:, 0],
            "feel_nb_p1": feel_probs[:, 1],
            "feel_nb_p2": feel_probs[:, 2],
            "sound_nb_p0": sound_probs[:, 0],
            "sound_nb_p1": sound_probs[:, 1],
            "sound_nb_p2": sound_probs[:, 2],
        },
        index=df.index,
    )


def _get_base_feature_cols(df):
    excluded = {
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
    cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pandas.api.types.is_numeric_dtype(df[col]) or pandas.api.types.is_bool_dtype(df[col]):
            cols.append(col)
    return cols


def _build_feature_matrix(df, base_cols, food_model, feel_model, sound_model, multihot_bundle):
    base_X = df[base_cols].copy()
    multihot_X = multihot_bundle.transform(df)
    nb_X = _build_nb_feature_frame(df, food_model, feel_model, sound_model)
    return pandas.concat(
        [
            base_X.reset_index(drop=True),
            multihot_X.reset_index(drop=True),
            nb_X.reset_index(drop=True),
        ],
        axis=1,
    )


def _gini(y, n_classes):
    if len(y) == 0:
        return 0.0
    counts = numpy.bincount(y, minlength=n_classes).astype(numpy.float64)
    probs = counts / counts.sum()
    return 1.0 - numpy.sum(probs * probs)


def _leaf_probs(y, n_classes):
    counts = numpy.bincount(y, minlength=n_classes).astype(numpy.float64)
    total = counts.sum()
    if total == 0:
        return numpy.ones(n_classes, dtype=numpy.float64) / float(n_classes)
    return counts / total


class _RandomForestTree:
    def __init__(self, n_classes, max_depth, min_samples_leaf, max_features, random_state):
        self.n_classes = int(n_classes)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.rng = numpy.random.RandomState(random_state)
        self.root = None

    def _feature_count(self, n_features):
        if isinstance(self.max_features, str):
            if self.max_features == "log2":
                return max(1, int(numpy.log2(max(n_features, 2))))
            if self.max_features == "sqrt":
                return max(1, int(numpy.sqrt(n_features)))
        if isinstance(self.max_features, (int, numpy.integer)):
            return max(1, min(int(self.max_features), n_features))
        return n_features

    def _candidate_thresholds(self, values):
        unique_values = numpy.unique(values)
        if len(unique_values) <= 1:
            return numpy.array([], dtype=numpy.float64)
        if len(unique_values) <= RF_MAX_THRESHOLDS_PER_FEATURE + 1:
            return (unique_values[:-1] + unique_values[1:]) / 2.0

        quantiles = numpy.linspace(0.1, 0.9, RF_MAX_THRESHOLDS_PER_FEATURE)
        thresholds = numpy.quantile(unique_values, quantiles)
        thresholds = numpy.unique(thresholds)
        return thresholds.astype(numpy.float64)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        parent_gini = _gini(y, self.n_classes)
        best = None

        feature_count = self._feature_count(n_features)
        chosen_features = self.rng.choice(n_features, size=feature_count, replace=False)

        for feature_idx in chosen_features:
            column = X[:, feature_idx]
            thresholds = self._candidate_thresholds(column)
            for threshold in thresholds:
                left_mask = column <= threshold
                right_mask = column > threshold
                left_n = int(left_mask.sum())
                right_n = int(right_mask.sum())

                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue

                left_gini = _gini(y[left_mask], self.n_classes)
                right_gini = _gini(y[right_mask], self.n_classes)
                weighted = (left_n * left_gini + right_n * right_gini) / float(n_samples)
                gain = parent_gini - weighted

                if best is None or gain > best["gain"]:
                    best = {
                        "feature_idx": feature_idx,
                        "threshold": float(threshold),
                        "gain": float(gain),
                        "left_mask": left_mask,
                        "right_mask": right_mask,
                    }
        return best

    def _build(self, X, y, depth):
        node = {
            "is_leaf": False,
            "probs": _leaf_probs(y, self.n_classes),
            "feature_idx": None,
            "threshold": None,
            "left": None,
            "right": None,
        }

        if depth >= self.max_depth:
            node["is_leaf"] = True
            return node
        if len(y) <= 2 * self.min_samples_leaf:
            node["is_leaf"] = True
            return node
        if numpy.unique(y).size == 1:
            node["is_leaf"] = True
            return node

        best = self._best_split(X, y)
        if best is None or best["gain"] <= 1e-12:
            node["is_leaf"] = True
            return node

        node["feature_idx"] = best["feature_idx"]
        node["threshold"] = best["threshold"]
        node["left"] = self._build(X[best["left_mask"]], y[best["left_mask"]], depth + 1)
        node["right"] = self._build(X[best["right_mask"]], y[best["right_mask"]], depth + 1)
        return node

    def fit(self, X, y):
        self.root = self._build(X, y, 0)
        return self

    def _predict_one_proba(self, row, node):
        current = node
        while not current["is_leaf"]:
            if row[current["feature_idx"]] <= current["threshold"]:
                current = current["left"]
            else:
                current = current["right"]
        return current["probs"]

    def predict_proba(self, X):
        probs = numpy.zeros((X.shape[0], self.n_classes), dtype=numpy.float64)
        for i in range(X.shape[0]):
            probs[i] = self._predict_one_proba(X[i], self.root)
        return probs


class _RandomForestClassifier:
    def __init__(self, n_estimators, max_depth, min_samples_leaf, max_features, bootstrap, random_state):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.random_state = int(random_state)
        self.trees = []
        self.n_classes = None

    def fit(self, X, y):
        X = numpy.asarray(X, dtype=numpy.float64)
        y = numpy.asarray(y, dtype=int)
        self.n_classes = int(y.max()) + 1
        rng = numpy.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        self.trees = []

        for tree_idx in range(self.n_estimators):
            if self.bootstrap:
                indices = rng.randint(0, n_samples, size=n_samples)
            else:
                indices = numpy.arange(n_samples)
            tree = _RandomForestTree(
                n_classes=self.n_classes,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=int(rng.randint(0, 1_000_000_000)),
            )
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        X = numpy.asarray(X, dtype=numpy.float64)
        probs = numpy.zeros((X.shape[0], self.n_classes), dtype=numpy.float64)
        for tree in self.trees:
            probs += tree.predict_proba(X)
        probs /= float(len(self.trees))
        return probs

    def predict(self, X):
        return numpy.argmax(self.predict_proba(X), axis=1)


def _fit_models():
    train_df = pandas.read_csv(TRAIN_CSV)
    train_df, numeric_medians = _clean_raw_dataframe(train_df)

    food_model = _BernoulliNBTextModel(TOPK1, NB_ALPHA).fit(train_df[FOOD_COL], train_df[LABEL_COL])
    feel_model = _BernoulliNBTextModel(TOPK2, NB_ALPHA).fit(train_df[FEEL_COL], train_df[LABEL_COL])
    sound_model = _BernoulliNBTextModel(TOPK3, NB_ALPHA).fit(train_df[SOUND_COL], train_df[LABEL_COL])

    multihot_bundle = _MultiHotEncoder(MULTIHOT_COLS).fit(train_df)
    base_cols = _get_base_feature_cols(train_df)
    X_train = _build_feature_matrix(train_df, base_cols, food_model, feel_model, sound_model, multihot_bundle)
    y_train = train_df[LABEL_COL].to_numpy(dtype=int)

    forest = _RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features=RF_MAX_FEATURES,
        bootstrap=RF_BOOTSTRAP,
        random_state=RF_RANDOM_STATE,
    ).fit(X_train.to_numpy(dtype=numpy.float64), y_train)

    return {
        "food_model": food_model,
        "feel_model": feel_model,
        "sound_model": sound_model,
        "multihot_bundle": multihot_bundle,
        "base_cols": base_cols,
        "numeric_medians": numeric_medians,
        "forest": forest,
    }


def _get_models():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = _fit_models()
    return _MODEL_CACHE


def predict_all(filename):
    """
    Return one predicted painting title per row in filename.
    """
    models = _get_models()
    test_df = pandas.read_csv(filename)
    test_df, _ = _clean_raw_dataframe(test_df, reference_medians=models["numeric_medians"])

    X_test = _build_feature_matrix(
        test_df,
        models["base_cols"],
        models["food_model"],
        models["feel_model"],
        models["sound_model"],
        models["multihot_bundle"],
    )

    pred_labels = models["forest"].predict(X_test.to_numpy(dtype=numpy.float64))
    return [LABEL_TO_NAME[int(label)] for label in pred_labels]
