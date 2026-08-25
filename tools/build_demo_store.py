#!/usr/bin/env python
"""Build the demo-mode results store (demo_results/results.json).

Demo mode serves pre-computed results instantly instead of running the
(slow) CM/ACM checks, MR/AMR repair and baselines live. This script assembles
that store.

Store hierarchy (schema v3):

    datasets[<dataset>]                         # e.g. malware
        variants[<variant>]                     # missingness level: "1", "20", "natural"
            stats                               # samples/features/missing + feature_missing
            models[<model>]                     # svm / logistic / linear / mlp / ft
                check                           # CM/ACM verdict (convex models only)
                by_method[<method>]             # mr / amr / full_impute / activeclean
                drop_incomplete                 # method-independent baseline

Datasets carry *injected* missingness at several levels (Malware, Tuandromd,
Parkinsons, Heart, Fraud, Higgs, Gas, Superconductivity, Concrete) or a single
*natural/observed* missing rate (Breast Cancer, Water-Potability, Online-Ed,
Bankruptcy, Air Quality, Cancer Rate).

Numbers are deterministic placeholders (seeded) so the store is stable and can
be designed/reviewed before the real paper numbers are dropped in. Per-feature
missingness is synthesized (real column names used when a CSV is available) so
the profile page renders for every dataset even without a data file.

Regenerate after editing the matrix or dropping in real numbers:

    python tools/build_demo_store.py
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "Sample-Datasets"
OUT_PATH = REPO_ROOT / "demo_results" / "results.json"

SCHEMA_VERSION = 3

# Imputation methods offered in Step 5: (internal_key, display_label).
METHODS = [
    ("manual", "Manual"),
    ("knn", "KNN"),
    ("mice", "MICE"),
    ("tcsdi", "TCSDI (Diffusion-based)"),
    ("missforest", "MissForest (Ensemble-based)"),
    ("llm", "LLM-based"),
]
METHOD_QUALITY = {
    "manual": 0.4, "knn": 0.5, "mice": 0.7,
    "tcsdi": 1.0, "missforest": 0.85, "llm": 0.9,
}

# Only convex models (SVM, logistic, linear regression) support CM/ACM
# checking; MLP and FT-Transformer skip the check and go straight to minimal
# repair. ActiveClean is only a valid baseline for the convex models.
MODEL_ORDER = ["svm", "logistic", "linear", "mlp", "ft"]
MODELS = {
    "svm": dict(display="Linear SVM", tasks={"classification"},
                check=True, ac=True, acc_off=-0.010, mse_off=0.00),
    "logistic": dict(display="Logistic Regression", tasks={"classification"},
                     check=True, ac=True, acc_off=0.000, mse_off=0.00),
    "linear": dict(display="Linear Regression", tasks={"regression"},
                   check=True, ac=True, acc_off=0.000, mse_off=0.00),
    "mlp": dict(display="MLP", tasks={"classification", "regression"},
                check=False, ac=False, acc_off=0.015, mse_off=-0.06),
    "ft": dict(display="FT-Transformer", tasks={"classification", "regression"},
               check=False, ac=False, acc_off=0.022, mse_off=-0.10),
}

# A threshold above the 0-10 slider max means "ACM never exists".
ACM_NEVER = 999

# ── Runtime model (seconds) ───────────────────────────────────────────────────
# Repair time ordering (the demo's core story): Drop < MinPrep < ActiveClean <
# Full Repair. Times scale with training-set size (n) and the imputation
# method's per-cell cost. A baseline whose time exceeds the 30-minute budget
# "did not finish" (Scenario 3, scalability).
BUDGET_S = 1800.0            # 30-minute time budget
C_FULL = 0.05               # Full Repair seconds per (row · method-cost)
C_AC = 0.02                 # ActiveClean seconds per (row · method-cost)
OOM_CELLS = 5e7             # n_rows * n_features beyond which ActiveClean OOMs
# Per-method relative cost (diffusion/LLM are heavy; KNN/mean are cheap).
METHOD_TIME = {
    "manual": 1.0, "knn": 0.8, "mice": 2.0,
    "tcsdi": 6.0, "missforest": 3.0, "llm": 10.0,
}

# The demo roster (from the paper's dataset tables). `base` is the clean-data
# optimal score at ~0 missingness (accuracy for classification, MSE for
# regression); it is degraded as the missingness level rises.
#
# CM/ACM existence (convex models):
#   cm_levels   : set of levels at which a Certain Model exists
#   acm_levels  : {level: slider_threshold} at which an ACM appears (no CM)
# Everything not listed has neither CM nor ACM at any threshold.
DATASETS = {
    # ── Classification ────────────────────────────────────────────────────
    "malware": dict(
        display="Malware", task="classification", file=None, target=None,
        n_features=6823, train_rows=1596, base=0.94,
        missingness="injected", levels=[1, 5, 20, 40, 60],
        kinds="MCAR, MAR, MNAR",
        cm_levels={1, 5}, acm_levels={},
    ),
    "tuandromd": dict(
        display="Tuandromd", task="classification", file=None, target=None,
        n_features=242, train_rows=3571, base=0.97,
        missingness="injected", levels=[1, 5, 20, 40, 60],
        kinds="MCAR, MAR, MNAR",
        cm_levels={1}, acm_levels={5: 2},
    ),
    "parkinsons": dict(
        display="Parkinsons", task="classification", file=None, target=None,
        n_features=22, train_rows=197, base=0.88,
        missingness="injected", levels=[20, 40, 60],
        kinds="MCAR, MAR, MNAR",
        cm_levels=set(), acm_levels={},
    ),
    "heart": dict(
        display="Heart Disease", task="classification",
        file="heart.csv", target="label",
        n_features=13, train_rows=303, base=0.85,
        missingness="injected", levels=[20, 40, 60],
        kinds="MCAR, MAR, MNAR",
        cm_levels=set(), acm_levels={},
    ),
    "fraud": dict(
        display="Fraud", task="classification", file=None, target=None,
        n_features=11, train_rows=6353307, base=0.99,
        missingness="injected", levels=[20, 40, 60],
        kinds="MCAR, MAR, MNAR",
        cm_levels=set(), acm_levels={},
    ),
    "higgs": dict(
        display="Higgs", task="classification", file=None, target=None,
        n_features=21, train_rows=11000000, base=0.74,
        missingness="injected", levels=[20, 40, 60],
        kinds="MCAR, MAR, MNAR",
        cm_levels=set(), acm_levels={},
    ),
    "breast_cancer": dict(
        display="Breast Cancer", task="classification", file=None, target=None,
        n_features=10, train_rows=559, base=0.96,
        missingness="natural", observed_rate=1.97, kinds="MCAR",
        cm_levels=set(), acm_levels={},
    ),
    "water_potability": dict(
        display="Water Potability", task="classification",
        file="water_potability.csv", target="Potability",
        n_features=9, train_rows=2620, base=0.69,
        missingness="natural", observed_rate=39.0, kinds="MCAR",
        cm_levels=set(), acm_levels={},
    ),
    "online_ed": dict(
        display="Online-Ed", task="classification", file=None, target=None,
        n_features=36, train_rows=7026, base=0.65,
        missingness="natural", observed_rate=35.48, kinds="MNAR, MCAR",
        cm_levels=set(), acm_levels={},
    ),
    "bankruptcy": dict(
        display="Bankruptcy", task="classification", file=None, target=None,
        n_features=64, train_rows=8402, base=0.97,
        missingness="natural", observed_rate=54.0, kinds="MNAR",
        cm_levels=set(), acm_levels={},
    ),
    # ── Regression (score = MSE) ──────────────────────────────────────────
    "gas": dict(
        display="Gas", task="regression", file=None, target=None,
        n_features=129, train_rows=2566, base=12.0,
        missingness="injected", levels=[20, 40, 60], kinds="MCAR",
        cm_levels=set(), acm_levels={},
    ),
    "superconductivity": dict(
        display="Superconductivity", task="regression", file=None, target=None,
        n_features=82, train_rows=21262, base=310.0,
        missingness="injected", levels=[20, 40, 60], kinds="MCAR",
        cm_levels=set(), acm_levels={},
    ),
    "concrete": dict(
        display="Concrete Strength", task="regression",
        file="concrete.csv", target="Concrete compressive strength",
        n_features=8, train_rows=1030, base=55.0,
        missingness="injected", levels=[20, 40, 60], kinds="MCAR",
        cm_levels=set(), acm_levels={},
    ),
    "air_quality": dict(
        display="Air Quality", task="regression", file=None, target=None,
        n_features=12, train_rows=7344, base=45.0,
        missingness="natural", observed_rate=90.8, kinds="MNAR",
        cm_levels=set(), acm_levels={},
    ),
    "cancer_rate": dict(
        display="Cancer Rate", task="regression", file=None, target=None,
        n_features=32, train_rows=3048, base=180.0,
        missingness="natural", observed_rate=81.0, kinds="MCAR",
        cm_levels=set(), acm_levels={},
    ),
}

FEATURES_SHOWN = 12  # per-feature missingness bars shown on the profile page


def _applicable_models(task):
    return [mk for mk in MODEL_ORDER if task in MODELS[mk]["tasks"]]


def _feature_names(cfg):
    """Real column names when a CSV is present, else generic names."""
    n = cfg["n_features"]
    if cfg.get("file"):
        path = DATASETS_DIR / cfg["file"]
        if path.exists():
            cols = list(pd.read_csv(path, nrows=1).columns)
            if cfg.get("target") in cols:
                cols = [c for c in cols if c != cfg["target"]]
            if cols:
                return cols[:n] if len(cols) >= n else cols
    return [f"feature {i}" for i in range(1, n + 1)]


def _synth_feature_missing(rng, feature_names, pct_missing, top_n=FEATURES_SHOWN):
    """Synthesize plausible per-feature missing%.

    The level (pct_missing) is the share of *rows* that are incomplete, so an
    individual feature's missing rate should sit at or below it. We spread the
    features over [~0.25*level, level] and keep the highest ``top_n``.
    """
    lo = pct_missing * 0.25
    vals = []
    for name in feature_names:
        v = rng.uniform(lo, pct_missing)
        vals.append([name, round(v, 1)])
    vals.sort(key=lambda x: x[1], reverse=True)
    return vals[:top_n]


def _gen_model(rng, cfg, model_key, level, stats):
    """Placeholder results for one (dataset, variant, model)."""
    mspec = MODELS[model_key]
    classification = cfg["task"] == "classification"
    score_label = "Accuracy" if classification else "MSE"
    dirty = stats["rows_with_missing"]

    # Base score degrades as missingness rises.
    if classification:
        model_base = min(cfg["base"] + mspec["acc_off"] - 0.0015 * level, 0.985)
        model_base = max(model_base, 0.50)
    else:
        model_base = round(cfg["base"] * (1.0 + mspec["mse_off"]) * (1.0 + 0.004 * level), 3)

    def score_from(delta, method_key="knn"):
        q = METHOD_QUALITY[method_key]
        if classification:
            val = model_base - delta + 0.010 * q
            return round(min(val, model_base + 0.005), 4)
        val = model_base * (1.0 + delta) * (1.0 - 0.02 * q)
        return round(val, 3)

    def pct(lo, hi):
        return round(rng.uniform(lo, hi), 1)

    n = stats["train_rows"]
    n_features = stats["n_features"]
    mem_cells = n * n_features  # rough memory-pressure proxy for OOM

    def dnf_reason(kind):
        # Full repair blows the time budget; ActiveClean tends to blow memory
        # first on very wide/large data.
        if kind == "activeclean" and mem_cells > OOM_CELLS:
            return "oom"
        return "budget"

    # ── Check step (convex models only) ────────────────────────────────────
    check = None
    if mspec["check"]:
        cm_exists = level in cfg["cm_levels"]
        if cm_exists:
            acm_gap = 0
        else:
            acm_gap = cfg["acm_levels"].get(level, ACM_NEVER)
        # The CM/ACM check is efficient: scales sublinearly and is cheap.
        chk_time = round(0.05 + 0.00002 * n, 3)
        check = {
            "cm": {
                "exists": bool(cm_exists),
                "score": score_from(rng.uniform(-0.005, 0.005)),
                "time_s": chk_time,
            },
            "acm_gap": acm_gap,
            "acm": {
                "score": score_from(rng.uniform(-0.010, 0.005)),
                "time_s": round(chk_time * rng.uniform(1.0, 1.3), 3),
            },
        }

    # ── Per-method repair + baselines (imputation ratio = % of dirty rows) ──
    # Runtime ordering (the demo's core story): Drop < MinPrep < ActiveClean <
    # Full Repair. Times scale with training-set size and the imputation
    # method's cost; baselines beyond BUDGET_S "did not finish".
    by_method = {}
    for key, _label in METHODS:
        mf = METHOD_TIME[key]
        mr_pct = pct(6, 18)
        amr_pct = round(max(3.0, mr_pct * rng.uniform(0.7, 0.95)), 1)

        # MinPrep repairs only a minimal subset -> fast and always finishes.
        mr_time = round((2.0 + 0.00004 * n + mf) * rng.uniform(0.9, 1.1), 1)
        amr_time = round(max(1.5, mr_time * rng.uniform(0.75, 0.95)), 1)
        # Full imputes everything; ActiveClean cleans an iterative subset.
        full_time = round(C_FULL * n * mf * rng.uniform(0.85, 1.15), 1)
        ac_time = round(C_AC * n * mf * rng.uniform(0.85, 1.15), 1)
        full_fin = full_time <= BUDGET_S
        ac_fin = ac_time <= BUDGET_S

        entry = {
            "mr": {
                "score": score_from(rng.uniform(0.0, 0.020), key),
                "time_s": mr_time,
                "finished": True,
                "pct_imputed": mr_pct,
                "rows_imputed": int(round(dirty * mr_pct / 100.0)),
            },
            "amr": {
                "score": score_from(rng.uniform(0.005, 0.025), key),
                "time_s": amr_time,
                "finished": True,
                "pct_imputed": amr_pct,
                "rows_imputed": int(round(dirty * amr_pct / 100.0)),
            },
            "full_impute": {
                "score": score_from(rng.uniform(0.005, 0.030), key),
                "time_s": full_time,
                "finished": bool(full_fin),
                "dnf_reason": None if full_fin else dnf_reason("full_impute"),
                "pct_imputed": 100.0,
                "rows_imputed": dirty,
            },
        }
        if mspec["ac"]:
            ac_pct = pct(22, 45)
            entry["activeclean"] = {
                "score": score_from(rng.uniform(0.010, 0.040), key),
                "time_s": ac_time,
                "finished": bool(ac_fin),
                "dnf_reason": None if ac_fin else dnf_reason("activeclean"),
                "pct_imputed": ac_pct,
                "rows_imputed": int(round(dirty * ac_pct / 100.0)),
            }
        by_method[key] = entry

    # ── Drop-all-incomplete-samples baseline (method-independent) ──────────
    drop_incomplete = {
        "score": score_from(rng.uniform(0.020, 0.055)),
        "time_s": round(max(0.3, 0.5 + 0.0000008 * n) * rng.uniform(0.9, 1.1), 1),
        "finished": True,
        "rows_dropped": dirty,
        "pct_dropped": stats["pct_missing"],
    }

    return {
        "display": mspec["display"],
        "score_label": score_label,
        "supports_check": bool(mspec["check"]),
        "supports_activeclean": bool(mspec["ac"]),
        "check": check,
        "by_method": by_method,
        "drop_incomplete": drop_incomplete,
    }


def _gen_variant(dskey, cfg, variant_key, label, level, feature_names):
    train_rows = cfg["train_rows"]
    dirty = int(round(train_rows * level / 100.0))
    dirty = min(dirty, train_rows)
    frng = random.Random(f"{dskey}:{variant_key}:feat:v{SCHEMA_VERSION}")
    stats = {
        "train_rows": train_rows,
        "n_rows": train_rows,
        "n_features": cfg["n_features"],
        "rows_with_missing": dirty,
        "pct_missing": round(level, 2),
        "feature_missing": _synth_feature_missing(frng, feature_names, level),
        "features_total": cfg["n_features"],
    }
    models = {}
    for mk in _applicable_models(cfg["task"]):
        rng = random.Random(f"{dskey}:{variant_key}:{mk}:v{SCHEMA_VERSION}")
        models[mk] = _gen_model(rng, cfg, mk, level, stats)
    return {
        "label": label,
        "level": level,
        "kinds": cfg["kinds"],
        "stats": stats,
        "models": models,
    }


def _variants_for(cfg):
    """Return [(variant_key, label, level), ...] for a dataset."""
    if cfg["missingness"] == "injected":
        return [(str(l), f"{l}%", float(l)) for l in cfg["levels"]]
    rate = cfg["observed_rate"]
    return [("natural", f"{rate:g}% (observed)", float(rate))]


def build_placeholder_store():
    store = {
        "schema_version": SCHEMA_VERSION,
        "source": "placeholder",
        "methods": [{"key": k, "label": l} for k, l in METHODS],
        "datasets": {},
    }
    for dskey, cfg in DATASETS.items():
        feature_names = _feature_names(cfg)
        variants = {}
        for vkey, vlabel, level in _variants_for(cfg):
            variants[vkey] = _gen_variant(dskey, cfg, vkey, vlabel, level, feature_names)
        store["datasets"][dskey] = {
            "display": cfg["display"],
            "task": cfg["task"],
            "target": cfg.get("target"),
            "file": cfg.get("file"),
            "n_features": cfg["n_features"],
            "train_rows": cfg["train_rows"],
            "missingness": cfg["missingness"],
            "kinds": cfg["kinds"],
            "default_variant": next(iter(variants)),
            "variants": variants,
        }
    return store


def validate_store(store):
    """Fail loudly if the store is structurally wrong (cheap safety net)."""
    assert store.get("schema_version") == SCHEMA_VERSION, "schema mismatch"
    method_keys = [k for k, _ in METHODS]
    for name, ds in store["datasets"].items():
        for field in ("display", "task", "variants", "default_variant"):
            assert field in ds, f"{name} missing {field}"
        assert ds["variants"], f"{name} has no variants"
        assert ds["default_variant"] in ds["variants"], f"{name} bad default_variant"
        for vk, v in ds["variants"].items():
            assert "stats" in v and "models" in v, f"{name}/{vk} incomplete"
            assert v["models"], f"{name}/{vk} has no models"
            fm = v["stats"].get("feature_missing")
            assert isinstance(fm, list) and fm, f"{name}/{vk} missing feature_missing"
            for mk, m in v["models"].items():
                for field in ("supports_check", "supports_activeclean",
                              "by_method", "drop_incomplete", "score_label"):
                    assert field in m, f"{name}/{vk}/{mk} missing {field}"
                assert "finished" in m["drop_incomplete"], \
                    f"{name}/{vk}/{mk} drop_incomplete missing finished"
                if m["supports_check"]:
                    assert m["check"], f"{name}/{vk}/{mk} supports check but has none"
                for meth in method_keys:
                    bm = m["by_method"].get(meth)
                    assert bm, f"{name}/{vk}/{mk} missing method {meth}"
                    for appr in ("mr", "amr", "full_impute"):
                        assert appr in bm, f"{name}/{vk}/{mk}/{meth} missing {appr}"
                    assert "finished" in bm["full_impute"], \
                        f"{name}/{vk}/{mk}/{meth} full_impute missing finished"
                    if m["supports_activeclean"]:
                        assert "activeclean" in bm, \
                            f"{name}/{vk}/{mk}/{meth} missing activeclean"
                        assert "finished" in bm["activeclean"], \
                            f"{name}/{vk}/{mk}/{meth} activeclean missing finished"
    return True


def main():
    ap = argparse.ArgumentParser(description="Build demo results store")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    store = build_placeholder_store()
    validate_store(store)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(store, indent=2))
    n_ds = len(store["datasets"])
    n_var = sum(len(d["variants"]) for d in store["datasets"].values())
    n_models = sum(len(v["models"])
                   for d in store["datasets"].values()
                   for v in d["variants"].values())
    print(f"Wrote {out} ({n_ds} datasets, {n_var} variants, "
          f"{n_models} variant-models, {len(METHODS)} methods)")


if __name__ == "__main__":
    main()
