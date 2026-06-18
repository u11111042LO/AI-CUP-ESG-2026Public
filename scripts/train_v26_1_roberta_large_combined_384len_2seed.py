#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI CUP VeriPromiseESG - v26_1 roberta combined 512len holdout 20260618 3seed

用途：
1. 訓練 v20-1：新主線，不再只是 v19-2 的 local search。
2. 在輸入文字前加入 company / ticker / page_number metadata prefix，讓模型學公司與報告位置脈絡。
3. 延續 v19-2/v19-3 證明有效的 head-tail truncation + quality expert，但重新訓練模型。
4. 儲存 base_seed_*.pt、quality_seed_*.pt、26-1SumForGpt.txt/json，並同步保留 run_summary_for_chatgpt.txt/json。
5. 之後可用同一支腳本載入模型產生 Sub_v26_1.csv；submission 預設輸出 N/A，避免 AIdea 格式錯誤。

建議放置：
/content/gdrive/MyDrive/aicup_esg_2026/Scripts/train_v26_1_roberta_large_combined_384len_2seed.py

Colab 執行範例：
!python /content/gdrive/MyDrive/aicup_esg_2026/Scripts/train_v26_1_roberta_large_combined_384len_2seed.py

只用已訓練模型產生 submission：
!python /content/gdrive/MyDrive/aicup_esg_2026/Scripts/train_v26_1_roberta_large_combined_384len_2seed.py \
  --mode predict \
  --test_path /content/gdrive/MyDrive/aicup_esg_2026/data/vpesg4k_test_2000.json
"""

import argparse
import csv
import datetime as _dt
import gc
import json
import math
import os
import random
import re
import time
import shutil
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

try:
    from sklearn.metrics import f1_score
except Exception as e:
    raise RuntimeError("請先安裝 scikit-learn：!pip install scikit-learn") from e

try:
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
except Exception as e:
    raise RuntimeError("請先安裝 transformers：!pip install transformers") from e


# ============================================================
# v19-3 主要設定：v19-2 model reuse + quality_refine local search
# ============================================================

RUN_NAME = "v26_1_roberta_large_combined_384len_2seed"
MODEL_NAME = "hfl/chinese-roberta-wwm-ext-large"

DEFAULT_TRAIN_PATH = "/content/gdrive/MyDrive/aicup_esg_2026/data/vpesg_4k_train_1000.json"
DEFAULT_VAL_PATH = "/content/gdrive/MyDrive/aicup_esg_2026/data/vpesg4k_val_1000.json"
DEFAULT_OUTPUT_DIR = "/content/gdrive/MyDrive/aicup_esg_2026/outputs/outputs_esg_v26_1_roberta_large_combined_384len_2seed"
# v16_2 是超細搜版本，預設沿用 v16-1 輸出中的已訓練權重，避免重訓。
DEFAULT_SOURCE_MODEL_DIR = ""  # v26-1 預設不複製舊模型，改用另一個 holdout split 從頭訓練，增加模型多樣性

# v26-1：把 train + official val 合併後，改用另一個 holdout seed 留 10% 當內部 holdout。
# 注意：這個 holdout 分數不能和 v20 official-val 分數直接比較；此版重點是拿更多標註資料訓練，再用 public 測泛化。
COMBINE_TRAIN_AND_VAL = True
HOLDOUT_RATIO = 0.10
HOLDOUT_SEED = 20260619

BASE_SEEDS = [42, 2026]
QUALITY_SEEDS = [42, 2026]

MAX_LEN = 384
QUALITY_MAX_LEN = 384

# v19-3 延續 v19-2：head-tail truncation。
# 長文本超過長度時，保留前段與後段，避免只保留開頭或只保留結尾。
# 0.60 代表可用 token 預算中約 60% 給 head、40% 給 tail。
HEAD_TAIL_HEAD_RATIO = 0.60

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM_STEPS

EPOCHS = 8
QUALITY_EPOCHS = 6          # v19-3: 預設重用 v19-2 模型；若重訓則沿用 v19-2 設定
LR = 1e-5
QUALITY_LR = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
DROPOUT = 0.2
LABEL_SMOOTHING = 0.03
QUALITY_LABEL_SMOOTHING = 0.02  # v16: 回到 v14-2
PATIENCE = 3

QUALITY_NOT_CLEAR_OVERSAMPLE = 4   # v19-3: 沿用 v19-2
QUALITY_NOT_CLEAR_WEIGHT_BOOST = 2.3  # v19-3: 沿用 v19-2

SELECTED_BASE_PARAMS = {
    "promise_no_threshold": 0.65,
    "evidence_no_threshold": 0.55,
    "not_clear_ratio_threshold": 0.25,
    "timeline_mode": "year_from_2024",
}

# v16_2: 在 v16-1 最佳點附近做超細搜。
# v16-1 最佳：promise_no_threshold≈0.59、evidence_no_threshold≈0.535、first3、misleading_to_clear。
# 這版不重訓，主要在 0.001 粒度微調 threshold，希望在 official validation 上再撿一點分數。
# v22-4：不是細掃 threshold；只保留 v22-4 穩定點與 v20/v22 系列常見的 0.4875 做小範圍比較。
# 主要改動是補足 5 seeds，讓 combined-data 訓練更穩。
BASE_PARAM_CANDIDATES = [
    {
        "promise_no_threshold": 0.65,
        "evidence_no_threshold": 0.50,
        "not_clear_ratio_threshold": 0.25,
        "timeline_mode": "year_from_2024",
    },
    {
        "promise_no_threshold": 0.65,
        "evidence_no_threshold": 0.4875,
        "not_clear_ratio_threshold": 0.25,
        "timeline_mode": "year_from_2024",
    },
]

# v22-4：固定 v20/v22 成功的 quality_refine，不做大範圍 sweep。
QUALITY_THRESHOLD_CANDIDATES = [0.300]
QUALITY_GUARD_MODES = ["misleading_to_clear"]
QUALITY_REFINE_NC_THRESHOLDS = [0.300]
QUALITY_REFINE_CLEAR_THRESHOLDS = [0.525]
QUALITY_REFINE_MARGINS = [0.075]

# 保持與訓練資料相容：No 承諾時，官方說明寫 N/A，但資料常見為空字串。
# 這裡訓練/驗證用空字串；產生 submission 時預設也輸出空字串。
EMPTY_LABEL = ""
PROMISE_LABELS = ["No", "Yes"]
TIMELINE_LABELS = [EMPTY_LABEL, "already", "within_2_years", "between_2_and_5_years", "more_than_5_years"]
EVIDENCE_LABELS = [EMPTY_LABEL, "No", "Yes"]
QUALITY_LABELS = [EMPTY_LABEL, "Clear", "Not Clear", "Misleading"]

TASK_WEIGHTS = {
    "promise_status": 0.20,
    "verification_timeline": 0.15,
    "evidence_status": 0.30,
    "evidence_quality": 0.35,
}


# ============================================================
# Utilities
# ============================================================

def now_str() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def normalize_label(x) -> str:
    if x is None:
        return EMPTY_LABEL
    x = str(x).strip()
    if x in ["N/A", "NA", "nan", "None", "null"]:
        return EMPTY_LABEL
    return x


def read_records(path: str) -> List[dict]:
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到資料檔：{path}")
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # 兼容 {data:[...]} 形式
            for key in ["data", "records", "items"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            raise ValueError("JSON 是 dict，但找不到 data/records/items list")
        if not isinstance(data, list):
            raise ValueError("JSON 格式應為 list[dict]")
        return data
    if path.lower().endswith(".csv"):
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError("只支援 .json 或 .csv")


def get_text(r: dict) -> str:
    """
    v20-1 metadata-prefix input.

    測試集與訓練集都含有 company / ticker / page_number 這些公開欄位。
    前幾版只吃 data，本版把這些欄位加到文本前面，讓模型能學到：
    - 不同公司永續報告的寫法習慣
    - 不同頁碼區段常對應的章節脈絡

    注意：這不是人工修正測試答案，只是使用資料集中原本提供的輸入欄位做自動特徵。
    """
    text = str(r.get("data", "") or "")
    company = str(r.get("company", "") or "").strip()
    ticker = str(r.get("ticker", "") or "").strip()
    page = str(r.get("page_number", "") or "").strip()

    # 避免過長 metadata 影響主文；只放短欄位。
    parts = []
    if company:
        parts.append(f"公司:{company}")
    if ticker:
        parts.append(f"股票:{ticker}")
    if page:
        parts.append(f"頁碼:{page}")

    if parts:
        return "【" + "｜".join(parts) + "】" + text
    return text


def get_id(r: dict, idx: int) -> str:
    return str(r.get("id", idx))


def label_maps(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    l2i = {x: i for i, x in enumerate(labels)}
    i2l = {i: x for x, i in l2i.items()}
    return l2i, i2l


PROMISE_L2I, PROMISE_I2L = label_maps(PROMISE_LABELS)
TIMELINE_L2I, TIMELINE_I2L = label_maps(TIMELINE_LABELS)
EVIDENCE_L2I, EVIDENCE_I2L = label_maps(EVIDENCE_LABELS)
QUALITY_L2I, QUALITY_I2L = label_maps(QUALITY_LABELS)


def encode_label(value: str, mapping: Dict[str, int], task_name: str) -> int:
    value = normalize_label(value)
    if value not in mapping:
        # 避免官方資料中出現未預期空白或 N/A
        if value == "":
            return mapping[EMPTY_LABEL]
        raise ValueError(f"{task_name} 出現未知標籤：{value!r}; 可用標籤={list(mapping)}")
    return mapping[value]


def macro_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> float:
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def micro_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> float:
    return float(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0))


def weighted_score(task_scores: Dict[str, float]) -> float:
    return sum(task_scores[k] * TASK_WEIGHTS[k] for k in TASK_WEIGHTS)


def dist_str(values: List[str]) -> str:
    c = Counter(values)
    def fmt(k):
        return "(empty)" if k == "" else k
    return ", ".join(f"{fmt(k)}:{v}" for k, v in sorted(c.items(), key=lambda kv: str(kv[0])))


def year_timeline_from_text(text: str) -> Optional[str]:
    """以 2024 為基準，從文本明確年份推估 timeline。沒有明確年份就回傳 None。"""
    years = []
    for m in re.finditer(r"20\d{2}", text):
        y = int(m.group(0))
        if 2024 <= y <= 2060:
            years.append(y)
    if not years:
        return None
    y = max(years)
    if y <= 2024:
        return "already"
    if y <= 2026:
        return "within_2_years"
    if y <= 2029:
        return "between_2_and_5_years"
    return "more_than_5_years"


def make_class_weights(y: List[int], n_classes: int, boost_idx: Optional[int] = None, boost: float = 1.0) -> torch.Tensor:
    c = Counter(y)
    weights = []
    total = len(y)
    for i in range(n_classes):
        # sqrt inverse frequency，避免少數類權重爆炸
        freq = c.get(i, 0)
        if freq == 0:
            w = 0.0
        else:
            w = math.sqrt(total / (n_classes * freq))
        if boost_idx is not None and i == boost_idx:
            w *= boost
        weights.append(w)
    # 避免全 0
    arr = np.array(weights, dtype=np.float32)
    if arr.sum() > 0:
        arr = arr / arr.mean()
    return torch.tensor(arr, dtype=torch.float32)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)



# ============================================================
# Head-tail tokenization
# ============================================================

def encode_head_tail(tokenizer, text: str, max_len: int, head_ratio: float = HEAD_TAIL_HEAD_RATIO):
    """Encode one text with head-tail truncation.

    修正版：不使用 tokenizer.prepare_for_model，避免部分 transformers / BertTokenizer
    沒有該方法時報錯。手動組合 [CLS] + head/tail tokens + [SEP]，再 padding 到 max_len。
    """
    text = str(text or "")
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    if cls_id is None:
        cls_id = tokenizer.convert_tokens_to_ids("[CLS]")
    if sep_id is None:
        sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
    if pad_id is None:
        pad_id = 0

    body_max = max(1, max_len - 2)
    if len(token_ids) > body_max:
        head_n = int(round(body_max * head_ratio))
        head_n = min(max(1, head_n), body_max - 1)
        tail_n = body_max - head_n
        token_ids = token_ids[:head_n] + token_ids[-tail_n:]

    input_ids = [cls_id] + token_ids + [sep_id]
    if len(input_ids) > max_len:
        input_ids = input_ids[: max_len - 1] + [sep_id]

    attention_mask = [1] * len(input_ids)
    token_type_ids = [0] * len(input_ids)
    pad_len = max_len - len(input_ids)
    if pad_len > 0:
        input_ids += [pad_id] * pad_len
        attention_mask += [0] * pad_len
        token_type_ids += [0] * pad_len

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        "token_type_ids": torch.tensor([token_type_ids], dtype=torch.long),
    }

# ============================================================
# Datasets
# ============================================================

class BaseDataset(Dataset):
    def __init__(self, records: List[dict], tokenizer, max_len: int, has_labels: bool = True):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.has_labels = has_labels

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        enc = encode_head_tail(self.tokenizer, get_text(r), self.max_len)
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["idx"] = torch.tensor(idx, dtype=torch.long)
        if self.has_labels:
            item["promise_label"] = torch.tensor(encode_label(r.get("promise_status", ""), PROMISE_L2I, "promise_status"), dtype=torch.long)
            item["timeline_label"] = torch.tensor(encode_label(r.get("verification_timeline", ""), TIMELINE_L2I, "verification_timeline"), dtype=torch.long)
            item["evidence_label"] = torch.tensor(encode_label(r.get("evidence_status", ""), EVIDENCE_L2I, "evidence_status"), dtype=torch.long)
            item["quality_label"] = torch.tensor(encode_label(r.get("evidence_quality", ""), QUALITY_L2I, "evidence_quality"), dtype=torch.long)
        return item


class QualityDataset(Dataset):
    def __init__(self, records: List[dict], tokenizer, max_len: int, has_labels: bool = True, oversample_not_clear: int = 1):
        self.records = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.has_labels = has_labels
        if has_labels:
            for r in records:
                label = normalize_label(r.get("evidence_quality", ""))
                self.records.append(r)
                if label == "Not Clear":
                    for _ in range(max(0, oversample_not_clear - 1)):
                        self.records.append(r)
        else:
            self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        enc = encode_head_tail(self.tokenizer, get_text(r), self.max_len)
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["idx"] = torch.tensor(idx, dtype=torch.long)
        if self.has_labels:
            item["label"] = torch.tensor(encode_label(r.get("evidence_quality", ""), QUALITY_L2I, "evidence_quality"), dtype=torch.long)
        return item


# ============================================================
# Models
# ============================================================

class RobertaMultiTask(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.promise_head = nn.Linear(hidden, len(PROMISE_LABELS))
        self.timeline_head = nn.Linear(hidden, len(TIMELINE_LABELS))
        self.evidence_head = nn.Linear(hidden, len(EVIDENCE_LABELS))
        self.quality_head = nn.Linear(hidden, len(QUALITY_LABELS))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        cls = out.last_hidden_state[:, 0, :]
        x = self.dropout(cls)
        return {
            "promise": self.promise_head(x),
            "timeline": self.timeline_head(x),
            "evidence": self.evidence_head(x),
            "quality": self.quality_head(x),
        }


class RobertaQuality(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, len(QUALITY_LABELS))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        cls = out.last_hidden_state[:, 0, :]
        x = self.dropout(cls)
        return self.head(x)


# ============================================================
# Loss / Prediction
# ============================================================

@dataclass
class TrainResult:
    seed: int
    best_epoch: int
    best_score_saved: float
    time_min: float
    path: str


def base_loss_fn(outputs, batch, weights, label_smoothing: float, device):
    ce_p = nn.CrossEntropyLoss(weight=weights["promise"].to(device), label_smoothing=label_smoothing)
    ce_t = nn.CrossEntropyLoss(weight=weights["timeline"].to(device), label_smoothing=label_smoothing)
    ce_e = nn.CrossEntropyLoss(weight=weights["evidence"].to(device), label_smoothing=label_smoothing)
    ce_q = nn.CrossEntropyLoss(weight=weights["quality"].to(device), label_smoothing=label_smoothing)
    return (
        0.20 * ce_p(outputs["promise"], batch["promise_label"].to(device))
        + 0.15 * ce_t(outputs["timeline"], batch["timeline_label"].to(device))
        + 0.30 * ce_e(outputs["evidence"], batch["evidence_label"].to(device))
        + 0.35 * ce_q(outputs["quality"], batch["quality_label"].to(device))
    )


def quality_loss_fn(logits, labels, weights, label_smoothing: float, device):
    ce = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=label_smoothing)
    return ce(logits, labels.to(device))


@torch.no_grad()
def predict_base_logits(model: nn.Module, records: List[dict], tokenizer, max_len: int, device, batch_size: int = BATCH_SIZE):
    ds = BaseDataset(records, tokenizer, max_len=max_len, has_labels=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    all_logits = {"promise": [], "timeline": [], "evidence": [], "quality": []}
    for batch in dl:
        inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
        outputs = model(**inputs)
        for k in all_logits:
            all_logits[k].append(outputs[k].detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in all_logits.items()}


@torch.no_grad()
def predict_quality_logits(model: nn.Module, records: List[dict], tokenizer, max_len: int, device, batch_size: int = BATCH_SIZE):
    ds = QualityDataset(records, tokenizer, max_len=max_len, has_labels=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    logits_all = []
    for batch in dl:
        inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
        logits = model(**inputs)
        logits_all.append(logits.detach().cpu().numpy())
    return np.concatenate(logits_all, axis=0)


def apply_base_predictions(base_logits: Dict[str, np.ndarray], records: List[dict], params: dict):
    p_prob = softmax_np(base_logits["promise"])
    t_prob = softmax_np(base_logits["timeline"])
    e_prob = softmax_np(base_logits["evidence"])
    q_prob = softmax_np(base_logits["quality"])

    no_p_idx = PROMISE_L2I["No"]
    yes_p_idx = PROMISE_L2I["Yes"]
    empty_t_idx = TIMELINE_L2I[EMPTY_LABEL]
    empty_e_idx = EVIDENCE_L2I[EMPTY_LABEL]
    no_e_idx = EVIDENCE_L2I["No"]
    yes_e_idx = EVIDENCE_L2I["Yes"]
    empty_q_idx = QUALITY_L2I[EMPTY_LABEL]

    out = []
    for i, r in enumerate(records):
        # promise threshold
        if p_prob[i, no_p_idx] >= params.get("promise_no_threshold", 0.5):
            promise = "No"
        else:
            promise = "Yes"

        if promise == "No":
            timeline = EMPTY_LABEL
            evidence = EMPTY_LABEL
            quality = EMPTY_LABEL
        else:
            # timeline: year rule 優先，沒有明確年份才用模型
            timeline = None
            if params.get("timeline_mode") == "year_from_2024":
                timeline = year_timeline_from_text(get_text(r))
            if timeline is None:
                # 不允許非 No promise 預測 empty timeline，若模型選到 empty，就取非 empty 最大
                order = np.argsort(-t_prob[i])
                timeline_idx = next((j for j in order if j != empty_t_idx), int(order[0]))
                timeline = TIMELINE_I2L[int(timeline_idx)]

            # evidence threshold
            if e_prob[i, no_e_idx] >= params.get("evidence_no_threshold", 0.5):
                evidence = "No"
            else:
                evidence = "Yes"

            if evidence != "Yes":
                quality = EMPTY_LABEL
            else:
                # base quality 粗估
                order = np.argsort(-q_prob[i])
                q_idx = next((j for j in order if j != empty_q_idx), int(order[0]))
                quality = QUALITY_I2L[int(q_idx)]
        out.append({
            "promise_status": promise,
            "verification_timeline": timeline,
            "evidence_status": evidence,
            "evidence_quality": quality,
        })
    return out


def apply_quality_override(base_preds: List[dict], quality_logits: np.ndarray, q_threshold: float):
    q_prob = softmax_np(quality_logits)
    clear_idx = QUALITY_L2I["Clear"]
    nc_idx = QUALITY_L2I["Not Clear"]
    misleading_idx = QUALITY_L2I["Misleading"]
    out = []
    for i, pred in enumerate(base_preds):
        p = dict(pred)
        if p["promise_status"] == "Yes" and p["evidence_status"] == "Yes":
            # v16：quality 模型仍只負責 Clear / Not Clear；Misleading 需要極高信心才輸出。
            if q_prob[i, nc_idx] >= q_threshold:
                p["evidence_quality"] = "Not Clear"
            else:
                # Misleading 只有極少樣本，除非壓倒性最高，不然避免亂猜。
                if q_prob[i, misleading_idx] >= 0.80 and q_prob[i, misleading_idx] == q_prob[i].max():
                    p["evidence_quality"] = "Misleading"
                else:
                    p["evidence_quality"] = "Clear"
        else:
            p["evidence_quality"] = EMPTY_LABEL
        out.append(p)
    return out


def apply_quality_refine(base_preds: List[dict], quality_logits: np.ndarray, nc_threshold: float, clear_threshold: float, margin: float):
    """保守的 quality-only refine。

    只在 promise_status=Yes 且 evidence_status=Yes 時處理 evidence_quality。
    不改 promise_status / verification_timeline / evidence_status。
    只允許 Clear <-> Not Clear 的小範圍修正；Misleading 留給 guard 處理。
    """
    q_prob = softmax_np(quality_logits)
    clear_idx = QUALITY_L2I["Clear"]
    nc_idx = QUALITY_L2I["Not Clear"]
    out = []
    for i, pred in enumerate(base_preds):
        p = dict(pred)
        if p.get("promise_status") == "Yes" and p.get("evidence_status") == "Yes":
            clear_p = float(q_prob[i, clear_idx])
            nc_p = float(q_prob[i, nc_idx])
            # 只有 quality 專家明顯偏向 Not Clear 時，才把 Clear/Misleading 改成 Not Clear。
            if nc_p >= nc_threshold and (nc_p - clear_p) >= margin:
                p["evidence_quality"] = "Not Clear"
            # 只有 quality 專家明顯偏向 Clear 時，才把 Not Clear/Misleading 改成 Clear。
            elif clear_p >= clear_threshold and (clear_p - nc_p) >= margin:
                p["evidence_quality"] = "Clear"
            # 否則保留 base 模型原判斷。
        else:
            p["evidence_quality"] = EMPTY_LABEL
        out.append(p)
    return out


def apply_quality_guard(preds: List[dict], guard_mode: str = "none") -> List[dict]:
    """
    v16 後處理：避免 evidence_quality 亂猜 Misleading。
    訓練集中 Misleading 只有極少數，v14-4 曾把 Misleading 預測過多而拖累分數。
    這裡不固定使用哪個 guard，而是在 validation 自動選 none / 轉 Clear / 轉 Not Clear。
    """
    if guard_mode == "none":
        return [dict(p) for p in preds]
    out = []
    for pred in preds:
        p = dict(pred)
        if p.get("promise_status") == "Yes" and p.get("evidence_status") == "Yes" and p.get("evidence_quality") == "Misleading":
            if guard_mode == "misleading_to_clear":
                p["evidence_quality"] = "Clear"
            elif guard_mode == "misleading_to_not_clear":
                p["evidence_quality"] = "Not Clear"
        out.append(p)
    return out


def eval_preds(records: List[dict], preds: List[dict]) -> Dict[str, object]:
    true = {
        "promise_status": [normalize_label(r.get("promise_status", "")) for r in records],
        "verification_timeline": [normalize_label(r.get("verification_timeline", "")) for r in records],
        "evidence_status": [normalize_label(r.get("evidence_status", "")) for r in records],
        "evidence_quality": [normalize_label(r.get("evidence_quality", "")) for r in records],
    }
    pred = {
        "promise_status": [p["promise_status"] for p in preds],
        "verification_timeline": [p["verification_timeline"] for p in preds],
        "evidence_status": [p["evidence_status"] for p in preds],
        "evidence_quality": [p["evidence_quality"] for p in preds],
    }
    labels = {
        "promise_status": PROMISE_LABELS,
        "verification_timeline": TIMELINE_LABELS,
        "evidence_status": EVIDENCE_LABELS,
        "evidence_quality": QUALITY_LABELS,
    }
    task_scores = {}
    task_micro = {}
    per_class_f1 = {}
    for task in TASK_WEIGHTS:
        task_scores[task] = macro_f1(true[task], pred[task], labels[task])
        task_micro[task] = micro_f1(true[task], pred[task], labels[task])
        f1s = f1_score(true[task], pred[task], labels=labels[task], average=None, zero_division=0)
        per_class_f1[task] = {lab: float(score) for lab, score in zip(labels[task], f1s)}
    return {
        "score": weighted_score(task_scores),
        "task_macro_f1": task_scores,
        "task_micro_f1": task_micro,
        "per_class_f1": per_class_f1,
        "prediction_distribution": {task: Counter(pred[task]) for task in TASK_WEIGHTS},
        "true_distribution": {task: Counter(true[task]) for task in TASK_WEIGHTS},
    }


# ============================================================
# Train / load
# ============================================================

def compute_base_weights(records: List[dict]) -> Dict[str, torch.Tensor]:
    return {
        "promise": make_class_weights([encode_label(r.get("promise_status", ""), PROMISE_L2I, "promise_status") for r in records], len(PROMISE_LABELS)),
        "timeline": make_class_weights([encode_label(r.get("verification_timeline", ""), TIMELINE_L2I, "verification_timeline") for r in records], len(TIMELINE_LABELS)),
        "evidence": make_class_weights([encode_label(r.get("evidence_status", ""), EVIDENCE_L2I, "evidence_status") for r in records], len(EVIDENCE_LABELS)),
        "quality": make_class_weights([encode_label(r.get("evidence_quality", ""), QUALITY_L2I, "evidence_quality") for r in records], len(QUALITY_LABELS), boost_idx=QUALITY_L2I["Not Clear"], boost=1.2),
    }


def compute_quality_weights(records: List[dict]) -> torch.Tensor:
    ys = [encode_label(r.get("evidence_quality", ""), QUALITY_L2I, "evidence_quality") for r in records]
    return make_class_weights(ys, len(QUALITY_LABELS), boost_idx=QUALITY_L2I["Not Clear"], boost=QUALITY_NOT_CLEAR_WEIGHT_BOOST)


def train_base_seed(seed: int, train_records, val_records, tokenizer, output_dir: str, device, reuse_existing: bool) -> TrainResult:
    save_path = os.path.join(output_dir, f"base_seed_{seed}.pt")
    if reuse_existing and os.path.exists(save_path) and os.path.getsize(save_path) > 10_000_000:
        print(f"[base seed={seed}] 已存在，跳過訓練：{save_path}")
        return TrainResult(seed=seed, best_epoch=-1, best_score_saved=-1.0, time_min=0.0, path=save_path)

    set_seed(seed)
    t0 = time.time()
    model = RobertaMultiTask(MODEL_NAME, dropout=DROPOUT).to(device)
    ds = BaseDataset(train_records, tokenizer, max_len=MAX_LEN, has_labels=True)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(dl) / GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    weights = compute_base_weights(train_records)

    best_score = -1.0
    best_epoch = -1
    bad = 0
    print(f"[base seed={seed}] train_size={len(train_records)}, val_size={len(val_records)}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(dl, start=1):
            inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(**inputs)
                loss = base_loss_fn(outputs, batch, weights, LABEL_SMOOTHING, device) / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
            total_loss += float(loss.detach().cpu()) * GRAD_ACCUM_STEPS
            if step % GRAD_ACCUM_STEPS == 0 or step == len(dl):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        logits = predict_base_logits(model, val_records, tokenizer, MAX_LEN, device)
        preds = apply_base_predictions(logits, val_records, SELECTED_BASE_PARAMS)
        metrics = eval_preds(val_records, preds)
        score = metrics["score"]
        print(f"[base seed={seed}] epoch={epoch:02d} loss={total_loss/len(dl):.4f} val_score={score:.5f}")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            bad = 0
            torch.save({
                "state_dict": model.state_dict(),
                "model_name": MODEL_NAME,
                "run_name": RUN_NAME,
                "seed": seed,
                "task": "base_multitask",
                "params": {
                    "MAX_LEN": MAX_LEN,
                    "DROPOUT": DROPOUT,
                    "LABEL_SMOOTHING": LABEL_SMOOTHING,
                    "base_params": SELECTED_BASE_PARAMS,
                },
                "labels": {
                    "promise_status": PROMISE_LABELS,
                    "verification_timeline": TIMELINE_LABELS,
                    "evidence_status": EVIDENCE_LABELS,
                    "evidence_quality": QUALITY_LABELS,
                },
                "best_epoch": best_epoch,
                "best_score": best_score,
            }, save_path)
            print(f"[base seed={seed}] saved score={best_score:.5f} -> {save_path}")
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[base seed={seed}] early stop at epoch={epoch}")
                break

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return TrainResult(seed=seed, best_epoch=best_epoch, best_score_saved=best_score, time_min=(time.time()-t0)/60, path=save_path)


def train_quality_seed(seed: int, train_records, val_records, tokenizer, output_dir: str, device, reuse_existing: bool) -> TrainResult:
    save_path = os.path.join(output_dir, f"quality_seed_{seed}.pt")
    if reuse_existing and os.path.exists(save_path) and os.path.getsize(save_path) > 10_000_000:
        print(f"[quality seed={seed}] 已存在，跳過訓練：{save_path}")
        return TrainResult(seed=seed, best_epoch=-1, best_score_saved=-1.0, time_min=0.0, path=save_path)

    set_seed(seed)
    t0 = time.time()
    model = RobertaQuality(MODEL_NAME, dropout=DROPOUT).to(device)
    ds = QualityDataset(train_records, tokenizer, max_len=QUALITY_MAX_LEN, has_labels=True, oversample_not_clear=QUALITY_NOT_CLEAR_OVERSAMPLE)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    optimizer = AdamW(model.parameters(), lr=QUALITY_LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(dl) / GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * QUALITY_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    weights = compute_quality_weights(train_records)

    best_score = -1.0
    best_epoch = -1
    bad = 0
    print(f"[quality seed={seed}] train_size_after_oversample={len(ds)}, val_size={len(val_records)}")
    for epoch in range(1, QUALITY_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(dl, start=1):
            inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
            labels = batch["label"].to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(**inputs)
                loss = quality_loss_fn(logits, labels, weights, QUALITY_LABEL_SMOOTHING, device) / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
            total_loss += float(loss.detach().cpu()) * GRAD_ACCUM_STEPS
            if step % GRAD_ACCUM_STEPS == 0 or step == len(dl):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        q_logits = predict_quality_logits(model, val_records, tokenizer, QUALITY_MAX_LEN, device)
        q_probs = softmax_np(q_logits)
        y_true = [normalize_label(r.get("evidence_quality", "")) for r in val_records]
        y_pred = []
        for i in range(len(val_records)):
            idx = int(np.argmax(q_probs[i]))
            y_pred.append(QUALITY_I2L[idx])
        score = macro_f1(y_true, y_pred, QUALITY_LABELS)
        print(f"[quality seed={seed}] epoch={epoch:02d} loss={total_loss/len(dl):.4f} quality_macro_f1={score:.5f}")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            bad = 0
            torch.save({
                "state_dict": model.state_dict(),
                "model_name": MODEL_NAME,
                "run_name": RUN_NAME,
                "seed": seed,
                "task": "quality",
                "params": {
                    "QUALITY_MAX_LEN": QUALITY_MAX_LEN,
                    "DROPOUT": DROPOUT,
                    "QUALITY_LABEL_SMOOTHING": QUALITY_LABEL_SMOOTHING,
                    "QUALITY_NOT_CLEAR_OVERSAMPLE": QUALITY_NOT_CLEAR_OVERSAMPLE,
                    "QUALITY_NOT_CLEAR_WEIGHT_BOOST": QUALITY_NOT_CLEAR_WEIGHT_BOOST,
                },
                "labels": {"evidence_quality": QUALITY_LABELS},
                "best_epoch": best_epoch,
                "best_score": best_score,
            }, save_path)
            print(f"[quality seed={seed}] saved score={best_score:.5f} -> {save_path}")
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[quality seed={seed}] early stop at epoch={epoch}")
                break

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return TrainResult(seed=seed, best_epoch=best_epoch, best_score_saved=best_score, time_min=(time.time()-t0)/60, path=save_path)


def load_base_model(path: str, device) -> nn.Module:
    ckpt = torch.load(path, map_location=device)
    model = RobertaMultiTask(ckpt.get("model_name", MODEL_NAME), dropout=DROPOUT).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def load_quality_model(path: str, device) -> nn.Module:
    ckpt = torch.load(path, map_location=device)
    model = RobertaQuality(ckpt.get("model_name", MODEL_NAME), dropout=DROPOUT).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def ensemble_base_logits(paths: List[str], records: List[dict], tokenizer, device) -> Dict[str, np.ndarray]:
    acc = None
    used = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[warn] base model not found: {path}")
            continue
        model = load_base_model(path, device)
        logits = predict_base_logits(model, records, tokenizer, MAX_LEN, device)
        if acc is None:
            acc = {k: v.astype(np.float64) for k, v in logits.items()}
        else:
            for k in acc:
                acc[k] += logits[k]
        used.append(path)
        del model
        torch.cuda.empty_cache(); gc.collect()
    if not used:
        raise RuntimeError("沒有可用的 base 模型")
    return {k: (v / len(used)).astype(np.float32) for k, v in acc.items()}


def ensemble_quality_logits(paths: List[str], records: List[dict], tokenizer, device) -> np.ndarray:
    acc = None
    used = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[warn] quality model not found: {path}")
            continue
        model = load_quality_model(path, device)
        logits = predict_quality_logits(model, records, tokenizer, QUALITY_MAX_LEN, device)
        if acc is None:
            acc = logits.astype(np.float64)
        else:
            acc += logits
        used.append(path)
        del model
        torch.cuda.empty_cache(); gc.collect()
    if not used:
        raise RuntimeError("沒有可用的 quality 模型")
    return (acc / len(used)).astype(np.float32)



def import_existing_models_if_needed(output_dir: str, source_dir: Optional[str]):
    """
    v16 主要是 threshold grid / 後處理搜尋，不需要重訓時，可沿用 v16 權重。
    若 output_dir 尚未有模型，且 source_dir 有 v16 模型，就自動複製過來。
    """
    if not source_dir:
        return
    if not os.path.isdir(source_dir):
        print(f"[model import] source_model_dir 不存在，略過：{source_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    copied = 0
    for name in [f"base_seed_{s}.pt" for s in BASE_SEEDS] + [f"quality_seed_{s}.pt" for s in QUALITY_SEEDS]:
        src = os.path.join(source_dir, name)
        dst = os.path.join(output_dir, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 10_000_000:
            continue
        if os.path.exists(src) and os.path.getsize(src) > 10_000_000:
            shutil.copy2(src, dst)
            copied += 1
    if copied:
        print(f"[model import] 已從 {source_dir} 複製 {copied} 個模型到 {output_dir}")
    else:
        print("[model import] 沒有需要複製的模型，或目標資料夾已存在模型。")

# ============================================================
# Main workflow
# ============================================================

def find_model_paths(output_dir: str) -> Tuple[List[str], List[str]]:
    base_paths = [os.path.join(output_dir, f"base_seed_{s}.pt") for s in BASE_SEEDS]
    quality_paths = [os.path.join(output_dir, f"quality_seed_{s}.pt") for s in QUALITY_SEEDS]
    return base_paths, quality_paths


def evaluate_candidates(output_dir: str, val_records: List[dict], tokenizer, device):
    base_paths, quality_paths = find_model_paths(output_dir)

    print("\n[eval] ensemble base first3")
    base_logits_first3 = ensemble_base_logits(base_paths[:3], val_records, tokenizer, device)
    print("\n[eval] ensemble base all")
    base_logits_all = ensemble_base_logits(base_paths, val_records, tokenizer, device)

    base_sources = {
        "first3": base_logits_first3,
        "all": base_logits_all,
    }

    print("\n[eval] ensemble quality")
    q_logits = ensemble_quality_logits(quality_paths, val_records, tokenizer, device)

    base_candidate_scores = []
    final_candidates = []

    # v16: 同時評估 first3 / all、threshold grid、Misleading guard。
    for source_name, base_logits in base_sources.items():
        for params in BASE_PARAM_CANDIDATES:
            base_preds = apply_base_predictions(base_logits, val_records, params)
            base_metrics = eval_preds(val_records, base_preds)
            base_candidate_scores.append({
                "name": f"base_{source_name}_p{params['promise_no_threshold']}_e{params['evidence_no_threshold']}",
                "score": base_metrics["score"],
                "params": params,
                "base_source": source_name,
            })

            # base only + guard 候選
            for guard_mode in QUALITY_GUARD_MODES:
                guarded = apply_quality_guard(base_preds, guard_mode=guard_mode)
                m = eval_preds(val_records, guarded)
                final_candidates.append({
                    "mode": "base_only",
                    "score": m["score"],
                    "q_th": None,
                    "guard_mode": guard_mode,
                    "base_source": source_name,
                    "base_params": params,
                    "metrics": m,
                    "preds": guarded,
                })

            # quality override + guard 候選
            for q_th in QUALITY_THRESHOLD_CANDIDATES:
                preds = apply_quality_override(base_preds, q_logits, q_threshold=q_th)
                for guard_mode in QUALITY_GUARD_MODES:
                    guarded = apply_quality_guard(preds, guard_mode=guard_mode)
                    m = eval_preds(val_records, guarded)
                    final_candidates.append({
                        "mode": "base_quality",
                        "score": m["score"],
                        "q_th": q_th,
                        "guard_mode": guard_mode,
                        "base_source": source_name,
                        "base_params": params,
                        "metrics": m,
                        "preds": guarded,
                    })


            # v19-2：quality-only refine 候選。
            # 不改 evidence_status，只用 quality 專家在 Clear / Not Clear 之間做保守修正。
            for nc_th in QUALITY_REFINE_NC_THRESHOLDS:
                for clear_th in QUALITY_REFINE_CLEAR_THRESHOLDS:
                    for margin in QUALITY_REFINE_MARGINS:
                        preds = apply_quality_refine(base_preds, q_logits, nc_threshold=nc_th, clear_threshold=clear_th, margin=margin)
                        for guard_mode in QUALITY_GUARD_MODES:
                            guarded = apply_quality_guard(preds, guard_mode=guard_mode)
                            m = eval_preds(val_records, guarded)
                            final_candidates.append({
                                "mode": "quality_refine",
                                "score": m["score"],
                                "q_th": None,
                                "quality_refine_params": {
                                    "nc_threshold": nc_th,
                                    "clear_threshold": clear_th,
                                    "margin": margin,
                                },
                                "guard_mode": guard_mode,
                                "base_source": source_name,
                                "base_params": params,
                                "metrics": m,
                                "preds": guarded,
                            })

    base_candidate_scores.sort(key=lambda x: x["score"], reverse=True)
    final_candidates.sort(key=lambda x: x["score"], reverse=True)
    selected = final_candidates[0]
    return base_candidate_scores, final_candidates, selected

def summary_to_text(summary: dict) -> str:
    lines = []
    lines.append("[ESG_RUN_SUMMARY]")
    for key in [
        "run_started_at", "approx_train_runtime_min", "device", "gpu", "model_name", "training_mode", "quick_test",
        "base_seeds", "quality_seeds", "run_seed", "train_path", "official_val_path", "output_dir",
        "train_size", "val_size", "MAX_LEN", "QUALITY_MAX_LEN", "BATCH_SIZE", "effective_batch_size",
        "EPOCHS", "QUALITY_EPOCHS", "LR", "QUALITY_LR", "WEIGHT_DECAY", "WARMUP_RATIO", "DROPOUT",
        "LABEL_SMOOTHING", "QUALITY_LABEL_SMOOTHING", "GRAD_ACCUM_STEPS", "PATIENCE",
        "QUALITY_NOT_CLEAR_OVERSAMPLE", "QUALITY_NOT_CLEAR_WEIGHT_BOOST",
    ]:
        lines.append(f"{key} = {summary.get(key)}")

    lines.append("\nbase_single_seed_results:")
    for r in summary.get("base_single_seed_results", []):
        lines.append(f"  seed={r['seed']} best_epoch={r['best_epoch']} best_score_saved={r['best_score_saved']:.5f} time_min={r['time_min']:.2f}")

    lines.append("\nquality_single_seed_results:")
    for r in summary.get("quality_single_seed_results", []):
        lines.append(f"  seed={r['seed']} best_epoch={r['best_epoch']} best_score_saved={r['best_score_saved']:.5f} time_min={r['time_min']:.2f}")

    lines.append("\nbase_candidate_scores:")
    for r in summary.get("base_candidate_scores", []):
        lines.append(f"  {r['name']} score={r['score']:.5f} params={r['params']}")

    lines.append("\nfinal_candidate_scores_top:")
    for r in summary.get("final_candidate_scores_top", [])[:20]:
        lines.append(f"  mode={r['mode']} source={r.get('base_source')} guard={r.get('guard_mode')} score={r['score']:.5f} q_th={r['q_th']}")

    lines.append(f"\nselected_mode = {summary.get('selected_mode')}")
    lines.append(f"selected_validation_score = {summary.get('selected_validation_score'):.5f}")
    lines.append(f"selected_base_source = {summary.get('selected_base_source')}")
    lines.append(f"selected_guard_mode = {summary.get('selected_guard_mode')}")
    lines.append(f"selected_base_params = {summary.get('selected_base_params')}")
    lines.append(f"selected_quality_threshold = {summary.get('selected_quality_threshold')}")
    lines.append(f"selected_quality_refine_params = {summary.get('selected_quality_refine_params')}")

    lines.append("\ntask_scores:")
    for task, score in summary.get("task_scores", {}).items():
        micro = summary.get("task_micro_scores", {}).get(task, None)
        lines.append(f"  {task}: macro_f1={score:.5f}, micro_f1={micro:.5f}, weight={TASK_WEIGHTS[task]}")
    lines.append(f"lowest_field = {summary.get('lowest_field')}")

    lines.append("\ntrain_label_distribution:")
    for task, s in summary.get("train_label_distribution", {}).items():
        lines.append(f"  {task}: {s}")

    lines.append("\nval_true_distribution:")
    for task, s in summary.get("val_true_distribution", {}).items():
        lines.append(f"  {task}: {s}")

    lines.append("\nper_class_f1:")
    for task, d in summary.get("per_class_f1", {}).items():
        parts = []
        for k, v in d.items():
            kk = "" if k == EMPTY_LABEL else k
            parts.append(f"{kk}:{v:.4f}")
        lines.append(f"  {task}: " + ", ".join(parts))

    lines.append("\nval_prediction_distribution:")
    for task, s in summary.get("val_prediction_distribution", {}).items():
        lines.append(f"  {task}: {s}")

    lines.append(f"\ntest_path = {summary.get('test_path')}")
    lines.append(f"submission_paths = {summary.get('submission_paths')}")
    lines.append(f"summary_json_path = {summary.get('summary_json_path')}")
    lines.append(f"summary_txt_path = {summary.get('summary_txt_path')}")
    lines.append("[/ESG_RUN_SUMMARY]")
    return "\n".join(lines)


def write_submission(records: List[dict], preds: List[dict], out_path: str, empty_as_na: bool = False):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "promise_status", "verification_timeline", "evidence_status", "evidence_quality"])
        writer.writeheader()
        for idx, (r, p) in enumerate(zip(records, preds)):
            row = {
                "id": get_id(r, idx),
                "promise_status": p["promise_status"],
                "verification_timeline": p["verification_timeline"],
                "evidence_status": p["evidence_status"],
                "evidence_quality": p["evidence_quality"],
            }
            if empty_as_na:
                for k in ["verification_timeline", "evidence_status", "evidence_quality"]:
                    if row[k] == "":
                        row[k] = "N/A"
            writer.writerow(row)
    return out_path


def run_predict(args, tokenizer, device):
    if not args.test_path:
        raise ValueError("predict 模式需要 --test_path")
    records = read_records(args.test_path)
    output_dir = args.output_dir
    base_paths, quality_paths = find_model_paths(output_dir)

    # 讀取 validation 自動選出的 v16 設定。
    summary_path = os.path.join(output_dir, "26-1SumForGpt.json")
    if not os.path.exists(summary_path):
        summary_path = os.path.join(output_dir, "run_summary_for_chatgpt.json")
    selected_mode = "base_quality"
    selected_base_source = "first3"
    selected_guard_mode = "none"
    selected_base_params = SELECTED_BASE_PARAMS
    selected_quality_refine_params = None
    q_th = 0.725
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        selected_mode = s.get("selected_mode", selected_mode)
        selected_base_source = s.get("selected_base_source", selected_base_source)
        selected_guard_mode = s.get("selected_guard_mode", selected_guard_mode)
        selected_base_params = s.get("selected_base_params", selected_base_params) or selected_base_params
        selected_quality_refine_params = s.get("selected_quality_refine_params", None)
        q_th = s.get("selected_quality_threshold", q_th) or q_th

    if selected_base_source == "all":
        use_base_paths = base_paths
    else:
        use_base_paths = base_paths[:3]

    print(f"[predict] selected_mode={selected_mode}, base_source={selected_base_source}, guard={selected_guard_mode}, q_th={q_th}")
    print(f"[predict] base_params={selected_base_params}")
    print(f"[predict] quality_refine_params={selected_quality_refine_params}")

    base_logits = ensemble_base_logits(use_base_paths, records, tokenizer, device)
    base_preds = apply_base_predictions(base_logits, records, selected_base_params)

    if selected_mode == "base_quality":
        q_logits = ensemble_quality_logits(quality_paths, records, tokenizer, device)
        preds = apply_quality_override(base_preds, q_logits, q_threshold=float(q_th))
    elif selected_mode == "quality_refine":
        q_logits = ensemble_quality_logits(quality_paths, records, tokenizer, device)
        rp = selected_quality_refine_params or {"nc_threshold": 0.35, "clear_threshold": 0.55, "margin": 0.05}
        preds = apply_quality_refine(
            base_preds, q_logits,
            nc_threshold=float(rp.get("nc_threshold", 0.35)),
            clear_threshold=float(rp.get("clear_threshold", 0.55)),
            margin=float(rp.get("margin", 0.05)),
        )
    else:
        preds = base_preds

    preds = apply_quality_guard(preds, guard_mode=selected_guard_mode)

    out_path = args.submission_path or os.path.join(output_dir, "Sub_v26_1.csv")
    write_submission(records, preds, out_path, empty_as_na=args.empty_as_na)
    print(f"[submission] saved: {out_path}")
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "predict", "train_eval_predict"], default="train")
    parser.add_argument("--train_path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val_path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--test_path", default=None)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--submission_path", default=None)
    parser.add_argument("--source_model_dir", default=DEFAULT_SOURCE_MODEL_DIR, help="v19_3 預設從 v19-2 output 複製既有模型，只做 local search；若要從頭訓練才改 source。")
    parser.add_argument("--no_import_existing_models", action="store_true", help="不要從 source_model_dir 複製既有模型。")
    parser.add_argument("--quick_test", action="store_true", help="快速測試流程，只取少量資料與少量 seed/epoch。")
    parser.add_argument("--reuse_existing", action="store_true", default=True)
    parser.add_argument("--no_reuse_existing", dest="reuse_existing", action="store_false")
    parser.add_argument("--empty_as_na", action="store_true", default=True, help="submission 預設把空字串輸出成 N/A，避免 AIdea 格式不合規。")
    args = parser.parse_args()

    run_started = now_str()
    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.mode in ["train", "eval", "predict", "train_eval_predict"] and not args.no_import_existing_models:
        import_existing_models_if_needed(args.output_dir, args.source_model_dir)

    device = get_device()
    print(f"[info] run={RUN_NAME}")
    print(f"[info] device={device}, gpu={gpu_name()}")
    print(f"[info] output_dir={args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    print(f"[tokenizer] mode=head_tail, head_ratio={HEAD_TAIL_HEAD_RATIO}, max_len={MAX_LEN}, quality_max_len={QUALITY_MAX_LEN}")

    base_seeds = BASE_SEEDS[:]
    quality_seeds = QUALITY_SEEDS[:]
    epochs_note = "normal"
    global EPOCHS, QUALITY_EPOCHS
    if args.quick_test:
        print("[quick_test] 啟用：只跑少量資料、少量 epoch、少量 seed。")
        base_seeds = [42]
        quality_seeds = [42]
        EPOCHS = 1
        QUALITY_EPOCHS = 1
        epochs_note = "quick_test"

    train_records = []
    val_records = []
    if args.mode in ["train", "eval", "train_eval_predict"]:
        raw_train_records = read_records(args.train_path)
        raw_val_records = read_records(args.val_path)
        if COMBINE_TRAIN_AND_VAL:
            combined_records = list(raw_train_records) + list(raw_val_records)
            rng = random.Random(HOLDOUT_SEED)
            rng.shuffle(combined_records)
            holdout_n = max(1, int(round(len(combined_records) * HOLDOUT_RATIO)))
            val_records = combined_records[:holdout_n]
            train_records = combined_records[holdout_n:]
            print(f"[data] COMBINED train+official_val={len(combined_records)} -> train={len(train_records)}, holdout={len(val_records)}")
            print("[data] 注意：這版 selected_validation_score 是內部 holdout，不可直接和 v20 official-val score 比。")
        else:
            train_records = raw_train_records
            val_records = raw_val_records
        if args.quick_test:
            train_records = train_records[:64]
            val_records = val_records[:64]
        print(f"[data] train={len(train_records)}, val={len(val_records)}")
        print("[data] train evidence_quality:", dist_str([normalize_label(r.get("evidence_quality", "")) for r in train_records]))
        print("[data] val evidence_quality:", dist_str([normalize_label(r.get("evidence_quality", "")) for r in val_records]))

    base_results: List[TrainResult] = []
    quality_results: List[TrainResult] = []

    if args.mode in ["train", "train_eval_predict"]:
        for seed in base_seeds:
            base_results.append(train_base_seed(seed, train_records, val_records, tokenizer, args.output_dir, device, args.reuse_existing))
        for seed in quality_seeds:
            quality_results.append(train_quality_seed(seed, train_records, val_records, tokenizer, args.output_dir, device, args.reuse_existing))

    if args.mode in ["train", "eval", "train_eval_predict"]:
        base_candidate_scores, final_candidates, selected = evaluate_candidates(args.output_dir, val_records, tokenizer, device)
        selected_metrics = selected["metrics"]
        task_scores = selected_metrics["task_macro_f1"]
        lowest_field = min(task_scores.items(), key=lambda kv: kv[1])[0]

        def counter_to_dist(counter):
            # Counter 可能來自 dict json 不可直接存，所以轉文字
            return dist_str(list(counter.elements()))

        summary = {
            "run_started_at": run_started,
            "approx_train_runtime_min": round((time.time() - start_time) / 60, 2),
            "device": str(device),
            "gpu": gpu_name(),
            "model_name": MODEL_NAME,
            "training_mode": RUN_NAME,
            "quick_test": args.quick_test,
            "epochs_note": epochs_note,
            "base_seeds": base_seeds,
            "quality_seeds": quality_seeds,
            "run_seed": base_seeds[0] if base_seeds else None,
            "train_path": args.train_path,
            "official_val_path": args.val_path,
            "output_dir": args.output_dir,
            "train_size": len(train_records),
            "val_size": len(val_records),
            "MAX_LEN": MAX_LEN,
            "QUALITY_MAX_LEN": QUALITY_MAX_LEN,
            "TRUNCATION_MODE": "head_tail",
            "METADATA_PREFIX": True,
            "HEAD_TAIL_HEAD_RATIO": HEAD_TAIL_HEAD_RATIO,
            "COMBINE_TRAIN_AND_VAL": COMBINE_TRAIN_AND_VAL,
            "HOLDOUT_RATIO": HOLDOUT_RATIO,
            "HOLDOUT_SEED": HOLDOUT_SEED,
            "score_note": "selected_validation_score is internal holdout after combining train+official_val; not directly comparable with official-val runs. v26-1 uses another holdout seed and trains 3 base/quality seeds from scratch for model diversity.",
            "BATCH_SIZE": BATCH_SIZE,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "EPOCHS": EPOCHS,
            "QUALITY_EPOCHS": QUALITY_EPOCHS,
            "LR": LR,
            "QUALITY_LR": QUALITY_LR,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "WARMUP_RATIO": WARMUP_RATIO,
            "DROPOUT": DROPOUT,
            "LABEL_SMOOTHING": LABEL_SMOOTHING,
            "QUALITY_LABEL_SMOOTHING": QUALITY_LABEL_SMOOTHING,
            "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
            "PATIENCE": PATIENCE,
            "QUALITY_NOT_CLEAR_OVERSAMPLE": QUALITY_NOT_CLEAR_OVERSAMPLE,
            "QUALITY_NOT_CLEAR_WEIGHT_BOOST": QUALITY_NOT_CLEAR_WEIGHT_BOOST,
            "base_single_seed_results": [asdict(r) for r in base_results],
            "quality_single_seed_results": [asdict(r) for r in quality_results],
            "base_candidate_scores": base_candidate_scores,
            "final_candidate_scores_top": [
                {
                    "mode": c["mode"],
                    "score": float(c["score"]),
                    "q_th": c["q_th"],
                    "base_source": c.get("base_source"),
                    "guard_mode": c.get("guard_mode"),
                    "base_params": c.get("base_params"),
                    "quality_refine_params": c.get("quality_refine_params"),
                }
                for c in final_candidates[:20]
            ],
            "selected_mode": selected["mode"],
            "selected_validation_score": float(selected["score"]),
            "selected_base_source": selected.get("base_source"),
            "selected_guard_mode": selected.get("guard_mode"),
            "selected_base_params": selected.get("base_params", SELECTED_BASE_PARAMS),
            "selected_quality_threshold": selected["q_th"],
            "selected_quality_refine_params": selected.get("quality_refine_params"),
            "task_scores": {k: float(v) for k, v in selected_metrics["task_macro_f1"].items()},
            "task_micro_scores": {k: float(v) for k, v in selected_metrics["task_micro_f1"].items()},
            "lowest_field": lowest_field,
            "train_label_distribution": {
                "promise_status": dist_str([normalize_label(r.get("promise_status", "")) for r in train_records]),
                "verification_timeline": dist_str([normalize_label(r.get("verification_timeline", "")) for r in train_records]),
                "evidence_status": dist_str([normalize_label(r.get("evidence_status", "")) for r in train_records]),
                "evidence_quality": dist_str([normalize_label(r.get("evidence_quality", "")) for r in train_records]),
            },
            "val_true_distribution": {
                k: counter_to_dist(v) for k, v in selected_metrics["true_distribution"].items()
            },
            "per_class_f1": {
                task: {label: float(score) for label, score in scores.items()}
                for task, scores in selected_metrics["per_class_f1"].items()
            },
            "val_prediction_distribution": {
                k: counter_to_dist(v) for k, v in selected_metrics["prediction_distribution"].items()
            },
            "test_path": args.test_path,
            "submission_paths": {},
        }

        if args.mode == "train_eval_predict" and args.test_path:
            sub_path = run_predict(args, tokenizer, device)
            summary["submission_paths"] = {"v19_3": sub_path}

        summary_json_path = os.path.join(args.output_dir, "26-1SumForGpt.json")
        summary_txt_path = os.path.join(args.output_dir, "26-1SumForGpt.txt")
        generic_json_path = os.path.join(args.output_dir, "run_summary_for_chatgpt.json")
        generic_txt_path = os.path.join(args.output_dir, "run_summary_for_chatgpt.txt")
        summary["summary_json_path"] = summary_json_path
        summary["summary_txt_path"] = summary_txt_path
        summary["generic_summary_json_path"] = generic_json_path
        summary["generic_summary_txt_path"] = generic_txt_path

        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        text = summary_to_text(summary)
        with open(summary_txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        shutil.copy2(summary_json_path, generic_json_path)
        shutil.copy2(summary_txt_path, generic_txt_path)
        print("\n" + text)
        print("\n摘要檔已儲存：")
        print(summary_json_path)
        print(summary_txt_path)
        print(generic_json_path)
        print(generic_txt_path)

    elif args.mode == "predict":
        run_predict(args, tokenizer, device)


if __name__ == "__main__":
    main()
