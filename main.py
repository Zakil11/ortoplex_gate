#!/usr/bin/env python3
"""
Copyright (c) 2026 Sawelew Tech / Ortoplex Research Division

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

NOTE: Portions of this file implement the cross7 / G2 / QMRS algorithms
covered by patent application PATENT_QMRS.md (Zastrzeżenie 1).
See NOTICE for licensing restrictions on cryptographic derivative use.

CANOPY L1 — AI-NATIVE APPCHAIN LOGIC (orthoplex-gate + G2+Zero classifier)
========================================================================
Kompletna logika L1 do szablonu Canopy (Python appchain).

WARSTWY:
  A) ORTHOPLEX GATE (unsupervised, 0 KB wag, czysta geometria):
       L1, L2, sphere_radius, orthoplex_fuel, SVD-projection → NORMAL/ANOMALY
  B) PERCEPTRON G2+Zero (supervised, wagi z perceptron_genlayer_weights.json):
       extended 42D fusion (raw7 + fused14 + inter7 + meta7 + cross7_self)
         → ZeroBlock → 16 klas → decyzja

UŻYCIE (Canopy):
  from canopy_l1_ai import AiContract
  c = AiContract()
  res = c.handle_event({"text":"raport kwartalny zysk",
                        "numbers":[200.0,300.0,400.0],
                        "timestamp":1757200000.0})
"""

import hashlib
import json
import math
import os
from collections import deque

PHI = (1 + 5 ** 0.5) / 2
GA = 2 * math.pi * (1 - 1 / PHI)

WEIGHTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "perceptron_genlayer_weights.json")

# ============================================================
# 1. ORTHOPLEX GATE — unsupervised (zero wagi)
# ============================================================
class OrthoplexGate:
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self._recent = deque(maxlen=64)
        self._mean = [0.0] * 7
        self._n = 0

    def _l2(self, v):
        return math.sqrt(sum(x * x for x in v))

    def _l1(self, v):
        return sum(abs(x) for x in v)

    def _fuel(self, v):
        mx = max(abs(x) for x in v)
        return mx / (self._l2(v) + 1e-12)

    def _svd(self, v):
        if self._n < 2:
            return 0.0
        dot = sum(a * b for a, b in zip(v, self._mean))
        nv = self._l2(v) + 1e-12
        nm = self._l2(self._mean) + 1e-12
        return abs(dot) / (nv * nm)

    def update(self, x):
        for i in range(7):
            self._mean[i] = (self._mean[i] * self._n + x[i]) / (self._n + 1)
        self._n += 1
        self._recent.append(x)

    def score(self, x):
        l2 = self._l2(x)
        l1 = self._l1(x)
        fuel = self._fuel(x)
        sv = self._svd(x)
        return l2 + 0.35 * l1 + 1.5 * fuel + (1.0 - sv)

    def is_anomaly(self, x):
        s = self.score(x)
        return s > self.threshold, s


# ============================================================
# 2. G2 — cross7 (identyczna z treningiem)
# ============================================================
def cross7(a, b):
    return [
        a[1]*b[3] - a[3]*b[1] + a[2]*b[6] - a[6]*b[2] + a[4]*b[5] - a[5]*b[4],
        a[2]*b[4] - a[4]*b[2] + a[3]*b[0] - a[0]*b[3] + a[5]*b[6] - a[6]*b[5],
        a[3]*b[5] - a[5]*b[3] + a[4]*b[1] - a[1]*b[4] + a[6]*b[0] - a[0]*b[6],
        a[4]*b[6] - a[6]*b[4] + a[5]*b[2] - a[2]*b[5] + a[0]*b[1] - a[1]*b[0],
        a[5]*b[0] - a[0]*b[5] + a[6]*b[3] - a[3]*b[6] + a[1]*b[2] - a[2]*b[1],
        a[6]*b[1] - a[1]*b[6] + a[0]*b[4] - a[4]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[2]*b[0] + a[1]*b[5] - a[5]*b[1] + a[3]*b[4] - a[4]*b[3],
    ]


# ============================================================
# 3. ENCODERS 7D (deterministic, czysty Python)
# ============================================================
def text_encode(text: str) -> list:
    v = [0.0] * 7
    if not text:
        return v
    grams = [text[i:i + 3] for i in range(max(1, len(text) - 2))]
    for g in grams:
        h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
        for k in range(4):
            v[k] += math.sin((h >> (k * 8)) % 256 * GA + k * PHI)
    denom = len(grams) + 1e-8
    for k in range(4):
        v[k] /= denom
    counts = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    probs = [c / len(text) for c in counts.values()]
    v[4] = -sum(p * math.log(p + 1e-12) for p in probs) / 4.0
    v[5] = sum(c.isdigit() for c in text) / len(text) if text else 0.0
    v[6] = math.log1p(len(text)) / 8.0
    return [max(-1.0, min(1.0, x)) for x in v]


def num_encode(numbers: list) -> list:
    v = [0.0] * 7
    xs = sorted([float(n) for n in numbers if n is not None and n > 0])
    if not xs:
        return v
    mn, mx = min(xs), max(xs)
    for k in range(min(3, len(xs))):
        v[k] = (xs[k] - mn) / (mx - mn + 1e-8)
    if len(xs) >= 2:
        v[3] = math.log(xs[0] / xs[1] + 1e-8) if xs[1] > 0 else 0.0
    mean = sum(xs) / len(xs)
    v[4] = math.log1p(mean) / 8.0
    var = sum((x - mean) ** 2 for x in xs)
    v[5] = math.sqrt(var / len(xs)) / (mean + 1e-8)
    v[6] = sum(1.0 / x for x in xs)
    return [max(-1.0, min(1.0, x)) for x in v]


def time_encode(ts: float) -> list:
    import datetime as _dt
    dtm = _dt.datetime.fromtimestamp(ts)
    v = [0.0] * 7
    ang_h = 2 * math.pi * (dtm.hour + dtm.minute / 60.0) / 24.0
    v[0], v[1] = math.sin(ang_h), math.cos(ang_h)
    ang_d = 2 * math.pi * dtm.weekday() / 7.0
    v[2], v[3] = math.sin(ang_d), math.cos(ang_d)
    ang_m = 2 * math.pi * (dtm.day - 1) / 30.0
    v[4], v[5] = math.sin(ang_m), math.cos(ang_m)
    v[6] = math.sin(GA * (ts / 86400.0))
    return v


# ============================================================
# 4. FUZJA EXTENDED42 — 42D (identyczna z ai_model.py / treningiem kucoin_futures)
# ============================================================
def _raw_avg(modalities, include):
    acc = [0.0] * 7
    n = 0
    for name in include:
        v = modalities.get(name)
        if v is not None and len(v) == 7:
            acc = [acc[i] + v[i] for i in range(7)]
            n += 1
    if n:
        acc = [x / n for x in acc]
    return acc


def _octo_fusion(modalities):
    vecs = [v for v in modalities.values() if v is not None and len(v) == 7]
    if not vecs:
        return [0.0] * 14
    if len(vecs) == 1:
        return vecs[0] + [0.0] * 7
    fused = [0.0] * 7
    norms = [0.0] * 7
    n_pairs = 0
    for i in range(len(vecs)):
        for k in range(i + 1, len(vecs)):
            c = cross7(vecs[i], vecs[k])
            fused = [fused[j] + c[j] for j in range(7)]
            norms = [norms[j] + abs(c[j]) for j in range(7)]
            n_pairs += 1
    fused = [x / max(1, n_pairs) for x in fused]
    norms = [x / max(1, n_pairs) for x in norms]
    scale = max(abs(x) for x in fused) + 1e-8
    fused = [x / scale for x in fused]
    return fused + norms


def encode_event(text: str, numbers: list, timestamp: float = 0.0,
                 recent: deque = None) -> list:
    if recent is None:
        recent = deque(maxlen=30)
    m_t = text_encode(text)
    m_n = num_encode(numbers)
    m_ti = time_encode(timestamp) if timestamp > 0 else None
    modalities = {"text": m_t, "numbers": m_n}
    present = ["text", "numbers"]
    if m_ti is not None:
        modalities["time"] = m_ti
        present.append("time")

    fused14 = _octo_fusion(modalities)
    ctx = sum(recent) / len(recent) if len(recent) >= 5 else 0.5
    fused14 = list(fused14)
    fused14[13] = 0.5 * fused14[13] + 0.5 * ctx
    recent.append(fused14[13])

    raw7 = _raw_avg(modalities, present)
    inter7 = [raw7[i] * fused14[i] for i in range(7)]

    # meta7 — średnia kwadratów modalności (energia)
    meta7 = [0.0] * 7
    n_present = 0
    for name in present:
        v = modalities.get(name)
        if v is not None and len(v) == 7:
            meta7 = [meta7[i] + v[i] * v[i] for i in range(7)]
            n_present += 1
    if n_present > 0:
        meta7 = [x / n_present for x in meta7]
    meta7 = [max(-1.0, min(1.0, x)) for x in meta7]

    # cross7_self — druga runda oktonionowa raw7 × fused7
    cross7_self = cross7(raw7, fused14[:7])
    scale = max(abs(x) for x in cross7_self) + 1e-8
    cross7_self = [x / scale for x in cross7_self]

    return raw7 + fused14 + inter7 + meta7 + cross7_self


# ============================================================
# 5. BACKBONE — czysty forward (eval)
# ============================================================
def matmul_vec(w, x, b):
    return [sum(w[i][j] * x[j] for j in range(len(x))) + b[i]
            for i in range(len(w))]


def zero_act(s):
    return [(1.0/(1.0+abs(v))) * (v/(1.0+abs(v))) +
            (1.0/(1.0+abs(v))) * math.tanh(v)
            for v in s]


def bn_eval(x, w, b, rm, rv, eps=1e-5):
    return [((xi - rm[i]) / math.sqrt(rv[i] + eps)) * w[i] + b[i]
            for i, xi in enumerate(x)]


def softmax(x, temperature: float = 1.0):
    if temperature <= 0:
        temperature = 1.0
    scaled = [v / temperature for v in x]
    mx = max(scaled)
    ex = [math.exp(v - mx) for v in scaled]
    s = sum(ex)
    return [v / s for v in ex]


# ============================================================
# 6. AI CONTRACT — entrypoint dla Canopy L1
# ============================================================
try:
    from ensemble import (
        ALL_CLASSES,
        action_class,
        action_name,
        get_ensemble,
        oracle_signature,
    )
except ImportError:  # pakiet vs skrypt
    from .ensemble import (
        ALL_CLASSES,
        action_class,
        action_name,
        get_ensemble,
        oracle_signature,
    )


class AiContract:
    def __init__(self, weights_path: str = WEIGHTS_FILE):
        self.gate = OrthoplexGate()
        with open(weights_path) as f:
            data = json.load(f)
        self.meta = data.get("meta", {})
        self.w = data["weights"]
        self.recent = deque(maxlen=30)
        self.ensemble = get_ensemble()

    def _weights_count(self):
        total = 0
        for k, v in self.w.items():
            if k.endswith("num_batches_tracked"):
                continue
            if isinstance(v, list) and v and isinstance(v[0], list):
                total += sum(len(row) for row in v)
            elif isinstance(v, list):
                total += len(v)
        return total

    def forward(self, x42):
        w = self.w
        s = matmul_vec(w["backbone.fc00.weight"], x42, w["backbone.fc00.bias"])
        h = zero_act(s)
        s = matmul_vec(w["backbone.block1.fc.weight"], h,
                       w["backbone.block1.fc.bias"])
        h = zero_act(s)
        h = bn_eval(h, w["backbone.block1.bn.weight"],
                    w["backbone.block1.bn.bias"],
                    w["backbone.block1.bn.running_mean"],
                    w["backbone.block1.bn.running_var"])
        s = matmul_vec(w["backbone.block2.fc.weight"], h,
                       w["backbone.block2.fc.bias"])
        h = zero_act(s)
        h = bn_eval(h, w["backbone.block2.bn.weight"],
                    w["backbone.block2.bn.bias"],
                    w["backbone.block2.bn.running_mean"],
                    w["backbone.block2.bn.running_var"])
        return matmul_vec(w["backbone.head.weight"], h,
                          w["backbone.head.bias"])

    def handle_event(self, event: dict) -> dict:
        text = event.get("text", "")
        nums = event.get("numbers", [])
        ts = event.get("timestamp", 0.0)

        # --- GATE (unsupervised, zero wagi) ---
        raw7 = _raw_avg({"text": text_encode(text),
                         "numbers": num_encode(nums)},
                        ["text", "numbers"])
        self.gate.update(raw7)
        is_anom, score = self.gate.is_anomaly(raw7)

        # --- KLASYFIKATOR (extended42) ---
        x42 = encode_event(text, nums, ts, self.recent)
        logits = self.forward(x42)
        y_base = logits.index(max(logits))
        probs = softmax(logits, temperature=1.5)

        # --- ENSEMBLE: G2 + Spiral + Resonance (weighted vote refine) ---
        refined = self.ensemble.refine(probs, x42, timestamp=float(ts))
        probs_ref = refined["probs"]
        y = probs_ref.index(max(probs_ref))

        # --- ACTION HEAD: klasa akcji 16-31 (deterministyczna) ---
        entropy7 = -sum(
            (p + 1e-12) * math.log(p + 1e-12) for p in probs[:7]
        ) if len(probs) >= 7 else 1.0
        a_idx = action_class(y_base, probs[y_base], is_anom, entropy7,
                             numbers=nums)

        # --- CROSS-CHAIN ORACLE: sygnatura sha256 predykcji ---
        sig = oracle_signature(x42, y, a_idx, probs_ref[y], float(ts),
                               1264119, 0)

        # top-3 klasy (po ensemble refine)
        ranked = sorted(zip(ALL_CLASSES, probs_ref), key=lambda kv: kv[1],
                        reverse=True)
        top3 = [{"class": name, "prob": round(float(p), 6)}
                for name, p in ranked[:3]]

        # feature importance — numeryczny gradient logitu klasy zwycięskiej
        importance = self._feature_importance(x42, logits, y_base)

        return {
            "gate": "ANOMALY" if is_anom else "NORMAL",
            "anomaly_score": round(score, 4),
            "y": int(y),
            "y_base": int(y_base),
            "decision": ALL_CLASSES[y] if 0 <= y < len(ALL_CLASSES) else "CLASS_{}".format(y),
            "top3": top3,
            "feature_importance": [round(float(v), 6) for v in importance],
            "temperature": 1.5,
            "confidence": round(probs_ref[y], 4),
            "ensemble": {
                "spiral": refined["spiral"],
                "resonance": refined["resonance"],
                "changed": bool(y != y_base),
            },
            "action": {
                "y": int(a_idx),
                "name": action_name(a_idx),
            },
            "oracle": {
                "signature": sig,
                "chain_id": 1264119,
                "n_classes": 32,
                "reproducible": True,
            },
            "storage_weights": self._weights_count(),
            "meta": self.meta,
        }

    def _feature_importance(self, x42, logits, y, eps: float = 1e-3):
        """Numeryczny gradient logitu klasy y względem każdego z 42 wymiarów."""
        base = logits[y]
        imp = []
        for i in range(len(x42)):
            x_plus = list(x42)
            x_plus[i] += eps
            logits_plus = self.forward(x_plus)
            imp.append((logits_plus[y] - base) / eps)
        mx = max(abs(v) for v in imp) if imp else 1.0
        if mx > 1e-12:
            imp = [v / mx for v in imp]
        return imp


_CLASS_NAMES = [
    "BENIGN",       # 0
    "LOW_RISK",     # 1
    "INFO",         # 2
    "RAISE_ALERT",  # 3
    "MEME",         # 4
    "SENTIMENT_UP", # 5
    "SENTIMENT_DN", # 6
    "VOLATILITY",   # 7
    "LIQUIDITY",    # 8
    "RESONANCE_UP", # 9
    "RESONANCE_DN", # 10
    "RECURRENCE",   # 11
    "FRACTAL_SELL", # 12
    "FRACTAL_BUY",  # 13
    "AUTO_ONLY",    # 14
    "ANOMALY",      # 15
]


if __name__ == "__main__":
    c = AiContract()
    res = c.handle_event({
        "text": "raport kwartalny zysk",
        "numbers": [200.0, 300.0, 400.0],
        "timestamp": 1757200000.0,
    })
    print(json.dumps({"gate": res["gate"], "y": res["y"],
                      "decision": res["decision"],
                      "top3": res["top3"],
                      "confidence": res["confidence"],
                      "ensemble": res["ensemble"],
                      "action": res["action"],
                      "oracle_sig": res["oracle"]["signature"][:16] + "...",
                      "storage_weights": res["storage_weights"]}, indent=2))
