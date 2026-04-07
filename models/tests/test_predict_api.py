import sys
import unittest
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import predict_api as pred


VALID_LABELS = {
    "The Persistence of Memory",
    "The Starry Night",
    "The Water Lily Pond",
}


def build_extremely_low_quality_rows():
    long_noise = ("??? ### null none idk ??? " * 200).strip()

    return [
        # 1. 几乎全空：只给一个完全空字典
        {},

        # 2. 全是错误类型 / 乱填
        {
            "unique_id": "abc-not-a-number",
            "Painting": "Not A Real Painting Name",
            "On a scale of 1–10, how intense is the emotion conveyed by the artwork?": "very very intense",
            "Describe how this painting makes you feel.": 123456789,
            "This art piece makes me feel sombre.": "999",
            "This art piece makes me feel content.": "banana",
            "This art piece makes me feel calm.": "",
            "This art piece makes me feel uneasy.": None,
            "How many prominent colours do you notice in this painting?": "a lot",
            "How many objects caught your eye in the painting?": "many many many",
            "How much (in Canadian dollars) would you be willing to pay for this painting?": "depends maybe idk",
            "If this painting was a food, what would be?": 404,
            "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.": True,
            "If you could purchase this painting, which room would you put that painting in?": "Mars,Underwater,UnknownRoom",
            "If you could view this art in person, who would you want to view it with?": "Aliens,Robots",
            "What season does this art piece remind you of?": "Monsoon",
        },

        # 3. 极端数值、溢出风格、非法范围
        {
            "unique_id": -999999999999999999999,
            "Painting": "The Starry Night",
            "On a scale of 1–10, how intense is the emotion conveyed by the artwork?": 10**18,
            "Describe how this painting makes you feel.": "!!!",
            "This art piece makes me feel sombre.": -1000,
            "This art piece makes me feel content.": 999999,
            "This art piece makes me feel calm.": -3.14159,
            "This art piece makes me feel uneasy.": 88888,
            "How many prominent colours do you notice in this painting?": 10**30,
            "How many objects caught your eye in the painting?": -10**12,
            "How much (in Canadian dollars) would you be willing to pay for this painting?": "$999999999999999999999999999999999999",
            "If this painting was a food, what would be?": "icecream icecream icecream",
            "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.": "slow violin piano calm calm calm",
            "If you could purchase this painting, which room would you put that painting in?": "Bedroom,Living room,Office,Bathroom,Dining room,Unknown",
            "If you could view this art in person, who would you want to view it with?": "Friends,Strangers,Family members,Coworkers/Classmates,By yourself",
            "What season does this art piece remind you of?": "Spring,Summer,Winter,Fall",
        },

        # 4. 超长噪声文本 + 奇怪 unicode
        {
            "unique_id": 2,
            "Painting": "???",
            "On a scale of 1–10, how intense is the emotion conveyed by the artwork?": None,
            "Describe how this painting makes you feel.": long_noise + " 雨夜🌧️🔥🌀",
            "This art piece makes me feel sombre.": "3 - Neutral/Unsure",
            "This art piece makes me feel content.": "4 - Agree",
            "This art piece makes me feel calm.": "5 - Strongly agree",
            "This art piece makes me feel uneasy.": "1 - Strongly disagree",
            "How many prominent colours do you notice in this painting?": None,
            "How many objects caught your eye in the painting?": None,
            "How much (in Canadian dollars) would you be willing to pay for this painting?": "if i were a billionaire maybe 7b but realistically 0",
            "If this painting was a food, what would be?": "%%%%%%%%%%%% spaghetti blueberry soup ??????",
            "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.": long_noise,
            "If you could purchase this painting, which room would you put that painting in?": "",
            "If you could view this art in person, who would you want to view it with?": "",
            "What season does this art piece remind you of?": "",
        },

        # 5. 只给极少数字段
        {
            "Describe how this painting makes you feel.": "calm",
        },

        # 6. payment / category / likert 都很怪
        {
            "unique_id": "nan",
            "Painting": None,
            "On a scale of 1–10, how intense is the emotion conveyed by the artwork?": "NaN",
            "Describe how this painting makes you feel.": None,
            "This art piece makes me feel sombre.": "Strongly yes",
            "This art piece makes me feel content.": "Disagree a lot",
            "This art piece makes me feel calm.": "7",
            "This art piece makes me feel uneasy.": "-1",
            "How many prominent colours do you notice in this painting?": "inf",
            "How many objects caught your eye in the painting?": "-inf",
            "How much (in Canadian dollars) would you be willing to pay for this painting?": "free maybe maybe no idea",
            "If this painting was a food, what would be?": None,
            "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.": None,
            "If you could purchase this painting, which room would you put that painting in?": None,
            "If you could view this art in person, who would you want to view it with?": None,
            "What season does this art piece remind you of?": None,
        },

        # 7. 混合正常和极端，确保批量预测时不会因为单行坏数据崩掉
        {
            "unique_id": 777,
            "Painting": "The Persistence of Memory",
            "On a scale of 1–10, how intense is the emotion conveyed by the artwork?": 7,
            "Describe how this painting makes you feel.": "dreamy but also strange",
            "This art piece makes me feel sombre.": "2 - Disagree",
            "This art piece makes me feel content.": "4 - Agree",
            "This art piece makes me feel calm.": "5 - Strongly agree",
            "This art piece makes me feel uneasy.": "2 - Disagree",
            "How many prominent colours do you notice in this painting?": 4,
            "How many objects caught your eye in the painting?": 2,
            "How much (in Canadian dollars) would you be willing to pay for this painting?": "200",
            "If this painting was a food, what would be?": "blueberry cheesecake",
            "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.": "soft calm piano with a slow melody",
            "If you could purchase this painting, which room would you put that painting in?": "Bedroom,Living room",
            "If you could view this art in person, who would you want to view it with?": "Friends",
            "What season does this art piece remind you of?": "Spring",
        },
    ]

class TestPredictAPIStress(unittest.TestCase):
    def test_predict_handles_extremely_low_quality_single_rows(self):
        rows = build_extremely_low_quality_rows()

        for i, row in enumerate(rows):
            try:
                pred_name = pred.predict(row)
            except Exception as e:
                self.fail(f"predict(row #{i}) crashed with exception: {repr(e)}")

            self.assertIsInstance(pred_name, str, f"predict(row #{i}) did not return a string")
            self.assertIn(
                pred_name,
                VALID_LABELS,
                f"predict(row #{i}) returned unexpected label: {pred_name}",
            )

    def test_predict_all_handles_extremely_low_quality_csv_without_crashing(self):
        rows = build_extremely_low_quality_rows()
        df = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "extremely_bad_inputs.csv"
            df.to_csv(csv_path, index=False)

            try:
                preds = pred.predict_all(str(csv_path))
            except Exception as e:
                self.fail(f"predict_all(...) crashed with exception: {repr(e)}")

        self.assertIsInstance(preds, list, "predict_all did not return a list")
        self.assertEqual(
            len(preds),
            len(rows),
            "predict_all should return exactly one prediction per input row",
        )

        for i, pred_name in enumerate(preds):
            self.assertIsInstance(pred_name, str, f"predict_all result #{i} is not a string")
            self.assertIn(
                pred_name,
                VALID_LABELS,
                f"predict_all result #{i} returned unexpected label: {pred_name}",
            )

    def test_predict_all_handles_missing_columns_csv(self):
        # 故意只保留极少列，测试 ensure_raw_schema / cleaning 补齐缺失列
        df = pd.DataFrame(
            [
                {"Describe how this painting makes you feel.": "sad and quiet"},
                {"How much (in Canadian dollars) would you be willing to pay for this painting?": "idk"},
                {},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "missing_columns.csv"
            df.to_csv(csv_path, index=False)

            try:
                preds = pred.predict_all(str(csv_path))
            except Exception as e:
                self.fail(f"predict_all(missing_columns.csv) crashed with exception: {repr(e)}")

        self.assertEqual(len(preds), len(df))
        for pred_name in preds:
            self.assertIn(pred_name, VALID_LABELS)


if __name__ == "__main__":
    unittest.main()
