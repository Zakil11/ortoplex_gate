# 🛡️ OrtoplexGate — AI-native L1 (Canopy)

**Ortoplex gate (unsupervised) → G₂+Zero classifier (supervised) — czysty Python, zero numpy/torch.**

## Architektura

```
CANOPY APPCHAIN L1 (Python)
├─▶ ORTHOPLEX GATE (0 KB wag, unsupervised)
│     L1, L2, sphere_radius, fuel, SVD → NORMAL/ANOMALY
└─▶ G₂+ZERO KLASYFIKATOR (wagi: data/perceptron_genlayer_weights.json)
      extended 28D fusion → 256→512→256→16 klas → decyzja
```

Model wytrenowany: **13 400 epok na RTX 4060 Laptop GPU, acc=1.0** (28D extended, hidden=256).

## Użycie

```bash
python3 main.py
```

Lub jako moduł:

```python
from main import AiContract
c = AiContract()
res = c.handle_event({
    "text": "mecz live bramka",
    "numbers": [1.0, 2.0, 3.0],
    "timestamp": 1757200000.0,
})
# → {"gate": "NORMAL"|"ANOMALY", "y": 1, "confidence": 0.99, ...}
```

## Struktura

```
ortoplex_gate/
├─ main.py                 # kompletna logika AI (gate + classifier)
├─ data/
│  └─ perceptron_genlayer_weights.json   # wagi 6.3MB (277 520 floats)
└─ README.md
```

## Zgodność

- **Czysty Python** (math/hashlib/json/deque) — działa tam, gdzie nie ma numpy/torch (Canopy, sandboxy).
- Matmul + ZeroBlock exact forward zgodny z treningiem torch (zweryfikowano: identyczne logity).
- L1-ready: zero Solidity, zero Rust — cała logika w standardowym Pythonie dla Canopy Appchain.

## Wyniki testu

```
ev0: ANOMALY score=3.62 y=0 conf=1.00 (raport)
ev1: ANOMALY score=3.05 y=1 conf=1.00 (mecz)
ev2: NORMAL  score=2.43 y=1 conf=0.57 (pogoda — po fit)
ev3: NORMAL  score=2.70 y=0 conf=1.00 (cena)
ev4: ANOMALY score=3.09 y=3 conf=0.79 (faktura — odchyl)
ev5: NORMAL  score=2.69 y=0 conf=1.00 (raport powtórzony)
```

---

# 🧠 G2 ZeroPerceptron na Canopy — Roadmapa AI

## 🏗️ Fundament (v1 — GOTOWE)

**G2 ZeroPerceptron wdrożony jako transakcje na Canopy L1** (chain `M12890`, token QARD).

- **Architektura**: cross7 (oktoniony G2 7D) → fuzja multimodalna 28D (raw7 + fused14 + inter7)
- **16 klas**: BENIGN, LOW_RISK, INFO, RAISE_ALERT, MEME, SENTIMENT_UP/DN, VOLATILITY, LIQUIDITY, RESONANCE_UP/DN, RECURRENCE, FRACTAL_SELL/BUY, AUTO_ONLY, ANOMALY
- **Wagi**: 277 520 floatów (fc00[256,28] → block1[512,256] → block2[256,512] → head[16,256])
- **Inferencja**: czysty Python (bit-identical z torch), lazy loading, singleton
- **Transakcje**: `MessagePredict` (0x03) + fee
- **Repozytoria**:
  - `Zakil11/ortoplex_gate` — brama/inferencja (Apache-2.0 + NOTICE patentowe)
  - `Zakil11/canopy` — fork Canopy L1 z plugin/python/contract (Apache-2.0 + NOTICE)

## 📜 Licencja — Apache-2.0 + NOTICE patentowe

| | MIT (Canopy) | Apache-2.0 (nasz) |
|---|---|---|
| Użycie komercyjne | Tak | Tak |
| Ochrona patentowa | ❌ Brak | ✅ TAK — sekcja 3 |
| Obligatoryjna atribucja | ✅ | ✅ (+NOTICE) |
| Można pozwac za patent | ⚠️ Ryzyko | 🛡️ Klauzula ogranicza |

- **NOTICE** — klauzula PATENT_QMRS: cross7 / G2 / QMRS objęte zgłoszeniem patentowym (Zastrzeżenie 1)
- Użytek kryptograficzny (faktoryzacja RSA, post-quantum) wymaga kontaktu przed dystrybucją

---

## 🚀 FAZA 1 — Quick wins

### 1. ✅ Explainability / Feature Importance — **GOTOWE** (commit `60a4278`)
- `top3_classes` + `feature_importance` (28 wymiarów) w odpowiedzi modelu
- Numeryczny gradient logitu klasy zwycięskiej względem każdego wymiaru
- Znormalizowany do [-1, 1]

### 2. ✅ Temperature Scaling — **GOTOWE** (commit `60a4278`)
- Softmax dzielony przez T=1.5 — realne confidence, nie zawyżone
- `probs` sumują się do 1.0 (zweryfikowano)

### 3. ⏳ Dynamic fee wg złożoności — **DO ZROBIENIA**
- Fee zależne od: długości tekstu, liczby features, timestamp (peak hours)

## 🧠 FAZA 2 — Rozbudowa modelu

### 4. 🔄 Nowe modalności (28D → 42D) — **W TRAKCIE**
- **Encoder cenowy** (OHLCV: open/high/low/close/volume → 7D)
- **Encoder blockchain** (gas price, tx count, block time → 7D)
- **Encoder sentymentu** (emoji, znaki interpunkcyjne, CAPS ratio → 7D)
- Pipeline: `colab_perceptron_multimodal.py` (trening GPU) → `export_genlayer_weights.py` (eksport wag) → aktualizacja `ai_model.py` + `contract.py`

### 5. ⏳ Ensemble G2 + Spiral + Resonance — **DO ZROBIENIA**
- 3 modele: obecny G2, `g2_spiral_simulator.py`, `g2_med_resonance.py`
- Voting (majority) lub weighted average

### 6. ⏳ Rozszerzenie klas: 16 → 32 — **DO ZROBIENIA**
- Dodanie klas akcji: `BUY_SIGNAL`, `SELL_SIGNAL`, `HOLD`, `WHALE_ALERT`, `PUMP_DUMP`, `SCAM`, `LEGIT`

## 🔄 FAZA 3 — On-chain learning

### 7. ✅ Feedback Loop (uczenie z użycia) — **GOTOWE** (commit `f9b2218`)
- Nowy typ transakcji `MessageFeedback` (0x04) — użytkownik zgłasza trafność predykcji
- Zapis do state: `predict_seq`, `correct`, `actual_class`, `height`
- Off-chain: retraining na GPU z nowymi danymi, nowe wagi przez governance
- **To czyni QARD pierwszym chainem z prawdziwym "AI, które uczy się z użycia"**

### 8. ✅ Prediction Marketplace / Staking — **GOTOWE**
- Nowe transakcje: `MessageStake` (0x06), `MessageCreateMarket` (0x05), `MessageResolveMarket`, `MessageClaimReward`
- Staking QARD na predykcje — kto trafnie przewidzi, dostaje nagrodę proporcjonalną do stawki
- Rynek: pytanie + resolution_height + pula nagród (outcome_pools)
- Tylko twórca rynku może go rozwiązać (actual_class 0-15)
- Nagroda = (stake × total_pool) / winning_pool — wypłacana po rozwiązaniu
- Testy: **44 passed** (34 check_tx + 10 deliver_tx — pełny cykl end-to-end: create_market → stake → resolve → claim + model registry)
- Testy deliver_tx: MockPlugin (in-memory state), weryfikacja sald, błędów (brak rynku, nie-kreator, zły outcome, podwójny claim, brak funduszy)

### 9. ⏳ Cross-chain Oracle — **DO ZROBIENIA**
- G2 jako oracle dla innych chainów (predykcje rynku, anomalie)

## 📊 FAZA 4 — Infrastruktura

### 10. ⏳ Dashboard on-chain — **DO ZROBIENIA**
- Agregacje w state: liczba predykcji, accuracy, top klasy, revenue z fee

### 11. ✅ Model versioning — **GOTOWE** (commit `4e824e5`)
- Nowy typ transakcji `MessageRegisterModel` (0x09 registry + 0x0a counter)
- Rejestr: wersja, hash wag, accuracy, in_dim (28/42), n_classes, opis
- Active model pointer (0x09/active/) — wskazuje najnowszą wersję
- Governance: aktualizacja wag bez hard-forku (rejestr on-chain)
- Testy: **44 passed** (34 + 10 nowych: 7 check + 3 deliver)

---

## 🎯 Postęp wdrożenia

| # | Element | Status | Commit |
|---|---------|--------|--------|
| 1 | Explainability + Temperature | ✅ GOTOWE | `60a4278` |
| 2 | Feedback Loop | ✅ GOTOWE | `f9b2218` |
| 3 | Nowe modalności 42D | ✅ GOTOWE | `83f9345` |
| 4 | Prediction Marketplace | ✅ GOTOWE | `cdfa718` |
| 5 | Ensemble G2 + Spiral + Resonance | ⏳ DO ZROBIENIA | — |
| 6 | Rozszerzenie klas 16→32 | ⏳ DO ZROBIENIA | — |
| 7 | Dynamic fee | ⏳ DO ZROBIENIA | — |
| 8 | Dashboard on-chain | ⏳ DO ZROBIENIA | — |
| 9 | Model versioning | ✅ GOTOWE | `4e824e5` |
| 10 | Cross-chain Oracle | ⏳ DO ZROBIENIA | — |

## 🔄 Pełna pętla uczenia (Feedback Loop)

```
1. Użytkownik wysyła MessagePredict → model daje klasę (zapis 0x03)
2. Czas mija — znany jest prawdziwy wynik
3. Użytkownik wysyła MessageFeedback (predict_seq, correct, actual_class) → zapis 0x04
4. Off-chain: eksport feedback logów → retraining na GPU → nowe wagi
5. Governance: nowe wagi przez aktualizację modelu (MessageRegisterModel 0x09)
```

## License

**Apache-2.0** — (c) 2026 Sawelew Tech / Ortoplex Research Division

- Kod AI (cross7 / G2 / ZeroPerceptron) chroniony zgłoszeniem patentowym QMRS — patrz `NOTICE`.
- Wykorzystanie komercyjne / kryptograficzne (faktoryzacja, RSA, post-quantum) wymaga zgody — patrz `PATENT_QMRS.md`.