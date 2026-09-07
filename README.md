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

## License

**Apache-2.0** — (c) 2026 Sawelew Tech / Ortoplex Research Division

- Kod AI (cross7 / G2 / ZeroPerceptron) chroniony zgłoszeniem patentowym QMRS — patrz `NOTICE`.
- Wykorzystanie komercyjne / kryptograficzne (faktoryzacja, RSA, post-quantum) wymaga zgody — patrz `PATENT_QMRS.md`.
