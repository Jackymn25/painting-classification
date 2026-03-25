import pandas as pd
import numpy as np
import re


# =========================================================
# 辅助函数
# =========================================================
def normalize_text(text):
    """
    统一清理文本中的格式问题：
    1. 缺失值转为空字符串
    2. 去掉首尾空格
    3. 转小写
    4. 统一特殊空格和破折号
    5. 把连续空白压成单个空格
    """
    if pd.isna(text):
        return ""

    s = str(text).strip().lower()
    s = s.replace("\xa0", " ")
    s = s.replace("–", "-").replace("—", "-")
    s = " ".join(s.split())
    return s


def collapse_spaced_thousands(text):
    """
    把带空格分组的大数字合并，例如：
    - 5 000 -> 5000
    - 5 000 000 -> 5000000
    - $10 000 -> $10000

    这里只合并“每组正好 3 位”的千分位写法，
    避免误改普通句子中的数字。
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
    """
    只要 text 能匹配任意一个正则 pattern，就返回 True。
    """
    for pat in patterns:
        if re.search(pat, text):
            return True
    return False


def looks_like_uncertain_text(s):
    """
    判断一句话是否看起来像“不确定回答”。
    例如：not sure / idk / no idea ...
    """
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
    """
    判断一句话是否明显表示“0 金额”。
    例如：0 / zero / nothing / free ...
    """
    zero_patterns = [
        r"\b0+\b",
        r"\bzero\b",
        r"\bnothing\b",
        r"\bno money\b",
        r"\bfree\b",
    ]
    return contains_any(s, zero_patterns)


def scale_multiplier(scale):
    """
    将金额单位换算成倍数。

    注意：
    这里故意不支持单独 'm'，
    避免 '$5000 max' 被误识别成 '$5000 m'。
    """
    scale = (scale or "").lower().strip()

    if scale in {"k", "thousand"}:
        return 1_000
    elif scale in {"mil", "million"}:
        return 1_000_000
    elif scale in {"b", "billion"}:
        return 1_000_000_000
    else:
        return 1


def score_candidate(context):
    """
    给候选金额的上下文打分。

    分数高：更像真实愿付金额
    分数低：更像假设金额 / 夸张金额
    """
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
    """
    当一句话里提到多个金额时，不直接取最大值。

    规则：
    1. 先看上下文打分，优先更像“真实愿付金额”的候选
    2. 如果同分，取最后出现的那个
    3. 如果整句明显是 depends / if i were 这种假设句，
       也优先最后一个候选
    4. 最后才 fallback 到 max
    """
    if not candidates:
        return np.nan

    scored = []
    for idx, item in enumerate(candidates):
        scored.append((score_candidate(item["context"]), idx, item["value"]))

    best_score = max(x[0] for x in scored)

    # 如果有明显“更现实”的金额，优先选这些
    if best_score > 0:
        best_group = [x for x in scored if x[0] == best_score]
        best_group.sort(key=lambda x: x[1])   # 按出现顺序排序
        return best_group[-1][2]              # 同分时取最后一个

    # 常见模式：前面说“如果我是富豪会付多少”，后面才说现实愿付金额
    if re.search(r"\bdepends\b", full_text) or re.search(r"\bif i were\b", full_text):
        return candidates[-1]["value"]

    # 默认 fallback：取最大值
    return max(item["value"] for item in candidates)


def parse_money_value(text):
    """
    尽量从自由文本中提取金额。

    支持的形式包括：
    - $5000
    - 5000 dollars
    - 5k / 5 k
    - 5mil / 5 million
    - 2b / 2 billion
    - 80 000

    改进点：
    1. 不接受单独 'm'，避免 '$5000 max' 被误读
    2. 多金额句子不再直接取最大值，而是根据上下文选更合理的值
    """
    if pd.isna(text):
        return np.nan

    s = normalize_text(text)

    if s in {"", "nan", "n/a", "na", "none"}:
        return np.nan

    s = collapse_spaced_thousands(s)

    # 先处理明显表示“0”的文本
    if looks_like_zero_text(s):
        return 0.0

    num_pattern = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

    # 故意去掉单独的 m
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
        re.VERBOSE | re.IGNORECASE
    )

    candidates = []

    for match in money_pattern.finditer(s):
        num = None
        scale = ""

        if match.group("num1") is not None:
            num = match.group("num1")
            scale = match.group("scale1") or ""
        elif match.group("num2") is not None:
            num = match.group("num2")
            scale = match.group("scale2") or ""
        elif match.group("num3") is not None:
            num = match.group("num3")
            scale = match.group("scale3") or ""
        elif match.group("num4") is not None:
            num = match.group("num4")
            scale = match.group("scale4") or ""
        elif match.group("num5") is not None:
            num = match.group("num5")
            scale = ""

        if num is None:
            continue

        num_clean = num.replace(",", "")

        try:
            value = float(num_clean)
        except ValueError:
            continue

        # 按单位放大
        value *= scale_multiplier(scale)

        # 记录上下文，用于后续判断哪个金额更合理
        start = max(0, match.start() - 35)
        end = min(len(s), match.end() + 35)
        context = s[start:end]

        candidates.append({
            "value": value,
            "context": context
        })

    if candidates:
        return choose_best_payment_value(candidates, s)

    # 没抓到金额，再看是否属于“不确定回答”
    if looks_like_uncertain_text(s):
        return np.nan

    return np.nan


def parse_payment_column(series):
    """
    清洗 payment 金额列：
    1. 从自由文本中提取金额
    2. 负数改成 0
    3. 超过上限的值 cap 到 324000000
    """
    x = series.apply(parse_money_value)

    negative_count = (x < 0).sum()
    x.loc[x < 0] = 0

    cap_value = 324_000_000
    capped_count = (x > cap_value).sum()
    x = x.clip(lower=0, upper=cap_value)

    return x, negative_count, cap_value, capped_count


def clean_count_column(series, upper_q=0.99):
    """
    清洗 count 类型列：
    1. 转成数值
    2. 负数改成 0
    3. 用全局高分位数做上界
    4. clip 后 round，避免出现小数
    """
    x = pd.to_numeric(series, errors="coerce")

    negative_count = (x < 0).sum()
    x.loc[x < 0] = 0

    non_na = x.dropna()
    if len(non_na) == 0:
        return x, negative_count, None, 0

    upper = int(np.ceil(non_na.quantile(upper_q)))
    before = x.copy()
    x = x.clip(lower=0, upper=upper)
    x = x.round()

    capped_count = (before > x).sum()
    return x, negative_count, upper, capped_count


def clean_bounded_scale(series, lower, upper):
    """
    清洗固定范围的量表列：
    - 超出范围的值直接设为 NaN
    - 不进行 quantile cap
    """
    x = pd.to_numeric(series, errors="coerce")
    invalid_count = ((x < lower) | (x > upper)).sum()
    x = x.where((x >= lower) & (x <= upper), np.nan)
    return x, invalid_count


# =========================================================
# 0. 读取数据
# =========================================================
df = pd.read_csv("data1.csv")

print("=" * 60)
print("Step 0: Read CSV")
print("=" * 60)
print("Original shape:", df.shape)


# =========================================================
# 1. 新增 label 列
# =========================================================
print("\n" + "=" * 60)
print("Step 1: Add label column")
print("=" * 60)

label_map = {
    "The Persistence of Memory": 0,
    "The Starry Night": 1,
    "The Water Lily Pond": 2
}

# 清理 Painting 列并映射成标签
df["Painting"] = df["Painting"].astype(str).str.strip()
df["Painting"] = df["Painting"].replace("nan", np.nan)
df["label"] = df["Painting"].map(label_map)


# =========================================================
# 2. Likert 列先映射成 1~5
# =========================================================
print("\n" + "=" * 60)
print("Step 2: Convert Likert columns to 1~5")
print("=" * 60)

likert_cols = [
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy."
]

likert_map = {
    "1 - Strongly disagree": 1,
    "2 - Disagree": 2,
    "3 - Neutral/Unsure": 3,
    "4 - Agree": 4,
    "5 - Strongly agree": 5
}

for col in likert_cols:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace("nan", np.nan)
    df[col] = df[col].map(likert_map)


# =========================================================
# 3. 清洗金额列
# =========================================================
print("\n" + "=" * 60)
print("Step 3: Parse and cap payment column")
print("=" * 60)

pay_col = "How much (in Canadian dollars) would you be willing to pay for this painting?"

df[pay_col], neg_count_pay, pay_cap_value, pay_capped_count = parse_payment_column(df[pay_col])

print("Negative values replaced with 0:", neg_count_pay)
print("Cap value used:", pay_cap_value)
print("Values capped to 324000000:", pay_capped_count)
print("Max payment after cap:", df[pay_col].max())


# =========================================================
# 4. 定义数值列 / 文本列
# =========================================================
print("\n" + "=" * 60)
print("Step 4: Define numeric and text/categorical columns")
print("=" * 60)

intensity_col = "On a scale of 1–10, how intense is the emotion conveyed by the artwork?"

count_cols = [
    "How many prominent colours do you notice in this painting?",
    "How many objects caught your eye in the painting?"
]

numeric_cols = [
    intensity_col,
    count_cols[0],
    count_cols[1],
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy."
]

text_or_categorical_cols = [
    "Painting",
    "Describe how this painting makes you feel.",
    "If you could purchase this painting, which room would you put that painting in?",
    "If you could view this art in person, who would you want to view it with?",
    "What season does this art piece remind you of?",
    "If this painting was a food, what would be?",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting."
]


# =========================================================
# 5. 按列处理异常值 / 非法值
# =========================================================
print("\n" + "=" * 60)
print("Step 5: Column-specific cleaning")
print("=" * 60)

# 先处理两个 count 列
for col in count_cols:
    df[col], negative_count, upper, capped_count = clean_count_column(df[col], upper_q=0.99)
    print(f"\nColumn: {col}")
    print("Negative values replaced with 0:", negative_count)
    print("Integer upper bound from global P99:", upper)
    print("Values clipped above upper bound:", capped_count)

# intensity 列只保留 1~10
print(f"\nColumn: {intensity_col}")
df[intensity_col], invalid_count = clean_bounded_scale(df[intensity_col], 1, 10)
print("Out-of-range values set to NaN:", invalid_count)

# Likert 列只保留 1~5
for col in likert_cols:
    df[col], invalid_count = clean_bounded_scale(df[col], 1, 5)
    print(f"\nColumn: {col}")
    print("Out-of-range values set to NaN:", invalid_count)


# =========================================================
# 6. 统计每一行缺失值个数
# =========================================================
print("\n" + "=" * 60)
print("Step 6: Count missing values")
print("=" * 60)

missing_count = df.isna().sum(axis=1)


# =========================================================
# 7. 删除缺失值 >= 4 的行
# =========================================================
print("\n" + "=" * 60)
print("Step 7: Drop rows with >= 4 missing values")
print("=" * 60)

df_clean = df[missing_count < 4].copy()
print("Shape after dropping:", df_clean.shape)


# =========================================================
# 8. payment 缺失值用全局 median 填补
# =========================================================
print("\n" + "=" * 60)
print("Step 8: Fill payment missing values with global median")
print("=" * 60)

payment_median = df_clean[pay_col].median()
payment_missing_before = df_clean[pay_col].isna().sum()

df_clean[pay_col] = df_clean[pay_col].fillna(payment_median)

payment_missing_after = df_clean[pay_col].isna().sum()

print("Payment global median:", payment_median)
print("Payment missing before fill:", payment_missing_before)
print("Payment missing after fill:", payment_missing_after)


# =========================================================
# 9. 其他数值列用全局 median 填补
# =========================================================
print("\n" + "=" * 60)
print("Step 9: Fill other numeric columns with global median")
print("=" * 60)

for col in numeric_cols:
    global_median = df_clean[col].median()
    missing_before = df_clean[col].isna().sum()

    df_clean[col] = df_clean[col].fillna(global_median)

    missing_after = df_clean[col].isna().sum()

    print(f"\nColumn: {col}")
    print("Global median:", global_median)
    print("Missing before fill:", missing_before)
    print("Missing after fill:", missing_after)

# 把这些本应为整数的列再 round 并转成 Int64
discrete_int_cols = [intensity_col] + count_cols + likert_cols

for col in discrete_int_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").round().astype("Int64")


# =========================================================
# 10. 文本 / 类别列用 "Missing" 填补
# =========================================================
print("\n" + "=" * 60)
print('Step 10: Fill text/categorical columns with "Missing"')
print("=" * 60)

for col in text_or_categorical_cols:
    missing_before = df_clean[col].isna().sum()

    df_clean[col] = df_clean[col].fillna("Missing")

    missing_after = df_clean[col].isna().sum()

    print(f"\nColumn: {col}")
    print("Missing before fill:", missing_before)
    print("Missing after fill:", missing_after)


# =========================================================
# 11. 最终检查
# =========================================================
print("\n" + "=" * 60)
print("Step 11: Final check")
print("=" * 60)

print("Final shape:", df_clean.shape)
print("\nRemaining missing values per column:")
print(df_clean.isna().sum())


# =========================================================
# 12. 保存清洗后的文件
# =========================================================
output_file = "cleaned_data_final.csv"
df_clean.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("Step 12: Save cleaned file")
print("=" * 60)
print(f"Saved successfully to: {output_file}")


# =========================================================
# X. 再次把 4 个情绪态度列转成 1~5
# 注意：这里保留原始逻辑，不做改动
# =========================================================
print("\n" + "=" * 60)
print("Step X: Convert Likert-scale text to numeric 1~5")
print("=" * 60)

likert_cols = [
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy."
]

likert_map = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly agree": 5
}

for col in likert_cols:
    print(f"\nColumn: {col}")
    print("Unique values before mapping:")
    print(df[col].value_counts(dropna=False))

    # 去掉首尾空格
    df[col] = df[col].astype(str).str.strip()

    # 把字符串 "nan" 还原成真正的缺失值
    df[col] = df[col].replace("nan", np.nan)

    # 映射成数字
    df[col] = df[col].map(likert_map)

    print("Unique values after mapping:")
    print(df[col].value_counts(dropna=False))

print(df.head(12))

# 输出 "cleaned_data_final.csv"
