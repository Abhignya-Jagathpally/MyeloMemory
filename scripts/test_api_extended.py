#!/usr/bin/env python3
"""Extended test harness for MyeloMemory API — 4 additional test categories.

Extends scripts/test_api.py with:
    A. Dose-Response Gradients  (6 profiles, monotonicity check)
    B. Single-Protein Perturbation  (8 profiles, impact ranking)
    C. Drug-Mechanism Alignment  (4 profiles, combination superiority)
    D. Full-Proteome Profiles  (5 profiles from real test data)

Usage:
    python scripts/test_api_extended.py --url http://localhost:8001
"""

import argparse
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

ALL_PROTEINS = [
    "EZH2", "SUZ12", "EED", "DNMT1", "DNMT3A", "DNMT3B",
    "TET1", "TET2", "TET3", "KDM6A", "KDM6B", "SETD2",
    "KMT2A", "KMT2B", "KMT2C", "KMT2D", "KDM5A", "KDM5B",
    "HDAC1", "HDAC2", "HDAC3", "EP300", "CREBBP", "UHRF1",
    "SMARCB1", "SMARCA4",
]

BORTEZOMIB_RESISTANT = {
    "EZH2": 4.5, "SUZ12": 3.8, "EED": 3.5,
    "DNMT1": 4.0, "DNMT3A": 3.2, "DNMT3B": 2.5,
    "TET1": 0.2, "TET2": 0.3, "TET3": 0.1,
    "KDM6A": 0.3, "KDM6B": 0.2, "SETD2": 0.4,
    "KMT2A": 0.5, "KMT2B": 0.3, "KMT2C": 0.4, "KMT2D": 0.3,
    "KDM5A": 0.5, "KDM5B": 0.4,
    "HDAC1": 3.5, "HDAC2": 3.0, "HDAC3": 2.8,
    "EP300": 0.3, "CREBBP": 0.4, "UHRF1": 3.5,
    "SMARCB1": 0.5, "SMARCA4": 0.4,
}

TREATMENT_NAIVE = {
    "EZH2": 1.0, "SUZ12": 1.0, "EED": 1.0,
    "DNMT1": 1.2, "DNMT3A": 0.8, "DNMT3B": 0.5,
    "TET1": 1.0, "TET2": 1.1, "TET3": 0.7,
    "KDM6A": 1.0, "KDM6B": 0.9, "SETD2": 1.0,
    "KMT2A": 1.0, "KMT2B": 0.8, "KMT2C": 0.9, "KMT2D": 1.0,
    "KDM5A": 0.8, "KDM5B": 0.7,
    "HDAC1": 1.0, "HDAC2": 0.9, "HDAC3": 0.8,
    "EP300": 1.0, "CREBBP": 1.0, "UHRF1": 1.0,
    "SMARCB1": 1.0, "SMARCA4": 1.0,
}

_pass_count = 0
_fail_count = 0


def _record(passed: bool, description: str) -> bool:
    global _pass_count, _fail_count
    if passed:
        _pass_count += 1
        print(f"    [PASS] {description}")
    else:
        _fail_count += 1
        print(f"    [FAIL] {description}")
    return passed


def _predict(base_url: str, protein_abundances: dict, timeout: int = 60) -> dict | None:
    payload = {"protein_abundances": protein_abundances}
    try:
        r = requests.post(f"{base_url}/predict", json=payload, timeout=timeout)
        if r.status_code != 200:
            print(f"    ERROR (HTTP {r.status_code}): {r.text[:200]}")
            return None
        return r.json()
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def _interpolate(profile_a: dict, profile_b: dict, alpha: float) -> dict:
    return {p: alpha * profile_a[p] + (1.0 - alpha) * profile_b[p] for p in ALL_PROTEINS}


# ─── A. Dose-Response Gradients ──────────────────────────────

def test_dose_response(base_url: str) -> None:
    print(f"\n{'=' * 60}")
    print("  A. Dose-Response Gradients (6 profiles)")
    print("=" * 60)

    fractions = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    labels = ["100% resistant", "80% resistant", "60% resistant",
              "40% resistant", "20% resistant", "0% (naive)"]

    scores = []
    for frac, label in zip(fractions, labels):
        profile = _interpolate(BORTEZOMIB_RESISTANT, TREATMENT_NAIVE, frac)
        result = _predict(base_url, profile)
        if result is None:
            scores.append(None)
            continue
        stab = result["stability_score"]
        scores.append(stab)
        bar = "#" * int(stab * 40)
        print(f"    {label:25s}  stability={stab:.4f}  {bar}")

    valid = [(i, s) for i, s in enumerate(scores) if s is not None]
    if len(valid) < 2:
        _record(False, "Not enough valid scores")
        return

    print()
    tolerance = 0.02
    violations = sum(1 for k in range(len(valid) - 1) if valid[k + 1][1] > valid[k][1] + tolerance)
    _record(violations == 0, f"Monotonic decrease (tolerance={tolerance}, violations={violations})")

    spread = valid[0][1] - valid[-1][1]
    _record(spread > 0.01, f"Endpoint spread = {spread:.4f}")


# ─── B. Single-Protein Perturbation ─────────────────────────

PERTURBATIONS = {
    "flip_EZH2":    ("EZH2",    1.0),
    "flip_DNMT1":   ("DNMT1",   1.0),
    "flip_UHRF1":   ("UHRF1",   1.0),
    "flip_TET2":    ("TET2",    3.0),
    "flip_KDM6A":   ("KDM6A",   3.0),
    "flip_HDAC1":   ("HDAC1",   1.0),
    "flip_EP300":   ("EP300",   3.0),
    "flip_SMARCA4": ("SMARCA4", 2.5),
}


def test_single_protein_perturbation(base_url: str) -> None:
    print(f"\n{'=' * 60}")
    print("  B. Single-Protein Perturbation (8 profiles)")
    print("=" * 60)

    baseline = _predict(base_url, BORTEZOMIB_RESISTANT)
    if baseline is None:
        _record(False, "Baseline prediction failed")
        return
    base_stab = baseline["stability_score"]
    print(f"    Baseline stability: {base_stab:.4f}\n")

    print(f"    {'Name':20s}  {'Protein':10s}  {'To':>6s}  {'Stability':>10s}  {'Delta':>10s}")
    print(f"    {'-' * 65}")

    deltas = {}
    for name, (protein, to_val) in PERTURBATIONS.items():
        perturbed = dict(BORTEZOMIB_RESISTANT)
        perturbed[protein] = to_val
        result = _predict(base_url, perturbed)
        if result is None:
            continue
        stab = result["stability_score"]
        delta = stab - base_stab
        deltas[name] = abs(delta)
        print(f"    {name:20s}  {protein:10s}  {to_val:6.1f}  {stab:10.4f}  {delta:+10.4f}")

    if not deltas:
        _record(False, "No perturbation results")
        return

    ranked = sorted(deltas.items(), key=lambda x: x[1], reverse=True)
    print(f"\n    Impact ranking:")
    for rank, (name, d) in enumerate(ranked, 1):
        print(f"      {rank}. {name:20s}  |delta| = {d:.4f}")

    top4 = {n for n, _ in ranked[:4]}
    print()
    _record("flip_EZH2" in top4 and "flip_DNMT1" in top4,
            f"EZH2 and DNMT1 in top 4 (top4={top4})")


# ─── C. Drug-Mechanism Alignment ────────────────────────────

def test_drug_mechanism_alignment(base_url: str) -> None:
    print(f"\n{'=' * 60}")
    print("  C. Drug-Mechanism Alignment (4 profiles)")
    print("=" * 60)

    base = dict(BORTEZOMIB_RESISTANT)

    profiles = {
        "ezh2_inhibitor": {**base, "EZH2": 0.3},
        "dnmt_inhibitor": {**base, "DNMT1": 0.3, "DNMT3A": 0.3, "DNMT3B": 0.3},
        "hdac_inhibitor": {**base, "HDAC1": 0.3, "HDAC2": 0.3, "HDAC3": 0.3, "EP300": 3.0, "CREBBP": 3.0},
        "combination": {**base, "EZH2": 0.3, "DNMT1": 0.3, "DNMT3A": 0.3, "DNMT3B": 0.3,
                        "HDAC1": 0.3, "HDAC2": 0.3, "HDAC3": 0.3, "EP300": 3.0, "CREBBP": 3.0},
    }

    scores = {}
    for name, profile in profiles.items():
        result = _predict(base_url, profile)
        if result is None:
            continue
        stab = result["stability_score"]
        scores[name] = stab
        bar = "#" * int(stab * 40)
        print(f"    {name:25s}  stability={stab:.4f}  {bar}")

    if "combination" not in scores:
        _record(False, "Combination prediction failed")
        return

    combo = scores["combination"]
    indiv = {k: v for k, v in scores.items() if k != "combination"}

    print()
    if indiv:
        _record(all(combo <= v for v in indiv.values()),
                f"Combination ({combo:.4f}) <= all individual inhibitors")

    baseline = _predict(base_url, BORTEZOMIB_RESISTANT)
    if baseline:
        base_stab = baseline["stability_score"]
        for name, stab in indiv.items():
            _record(stab < base_stab, f"{name} ({stab:.4f}) < baseline ({base_stab:.4f})")


# ─── D. Full-Proteome Profiles ──────────────────────────────

def test_full_proteome_profiles(base_url: str) -> None:
    print(f"\n{'=' * 60}")
    print("  D. Full-Proteome Profiles (5 real test samples)")
    print("=" * 60)

    try:
        import torch
    except ImportError:
        _record(False, "PyTorch not installed")
        return

    ckpt_path = Path("checkpoints/data_ready.pt")
    if not ckpt_path.exists():
        ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints" / "data_ready.pt"
    if not ckpt_path.exists():
        _record(False, f"Checkpoint not found at {ckpt_path}")
        return

    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dataset = data["dataset"]
    test_indices = data["splits"]["test"]
    protein_names = dataset.protein_names
    n_samples = min(5, len(test_indices))

    print(f"    Proteins: {len(protein_names)}, test samples: {len(test_indices)}, using {n_samples}\n")

    scores = []
    for i in range(n_samples):
        idx = test_indices[i]
        abundances = {protein_names[j]: float(dataset.proteomics[idx, j]) for j in range(len(protein_names))}

        result = _predict(base_url, abundances, timeout=120)
        if result is None:
            scores.append(None)
            continue
        stab = result["stability_score"]
        scores.append(stab)
        coverage = result.get("coverage_pct", "N/A")
        print(f"    Sample {i+1} (idx={idx}): stability={stab:.4f}, coverage={coverage}%")

    valid = [s for s in scores if s is not None]
    print()
    _record(len(valid) == n_samples, f"All {n_samples} full-proteome samples predicted")
    _record(all(0.0 <= s <= 1.0 for s in valid), "All scores in [0, 1]")


# ─── Main ────────────────────────────────────────────────────

def main():
    global _pass_count, _fail_count
    _pass_count = _fail_count = 0

    parser = argparse.ArgumentParser(description="MyeloMemory extended test suite")
    parser.add_argument("--url", default="http://localhost:8001")
    args = parser.parse_args()

    print("=" * 60)
    print("  MyeloMemory API Extended Test Suite")
    print(f"  Target: {args.url}")
    print("=" * 60)

    try:
        r = requests.get(f"{args.url}/health", timeout=10)
        if not r.json().get("model_loaded"):
            print("  Model not loaded. Aborting.")
            return
    except Exception as e:
        print(f"  Cannot reach API: {e}")
        return

    test_dose_response(args.url)
    test_single_protein_perturbation(args.url)
    test_drug_mechanism_alignment(args.url)
    test_full_proteome_profiles(args.url)

    total = _pass_count + _fail_count
    print(f"\n{'=' * 60}")
    print("  EXTENDED TEST SUMMARY")
    print("=" * 60)
    print(f"  Passed: {_pass_count}/{total}  Failed: {_fail_count}/{total}")
    if total > 0:
        print(f"  Pass rate: {_pass_count / total * 100:.1f}%")
    print("=" * 60)

    sys.exit(1 if _fail_count > 0 else 0)


if __name__ == "__main__":
    main()
