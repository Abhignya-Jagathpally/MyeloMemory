#!/usr/bin/env python3
"""Test harness for MyeloMemory API — sends biologically meaningful requests.

Creates 5 distinct proteomic profiles representing real clinical scenarios:
    1. Treatment-naive MM (balanced chromatin)
    2. Bortezomib-resistant MM (high PRC2/DNMT, locked epigenetic memory)
    3. Lenalidomide-sensitive MM (high erasers, open chromatin)
    4. Post-washout MM (intermediate — partial memory recovery)
    5. Multi-drug resistant MM (extreme writer overexpression)

These profiles use the actual chromatin reader/writer proteins the model
was trained on, with abundance values that reflect published biology.

Usage:
    python scripts/test_api.py --url http://localhost:8001
    python scripts/test_api.py --url http://localhost:8001 --batch
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────
# Biologically motivated test profiles
# ─────────────────────────────────────────────────────────────

# Key proteins and their roles:
#   Writers (auto-catalysis, lock-in):  EZH2, SUZ12, EED, DNMT1, DNMT3A/B
#   Erasers (reverse, open chromatin):  TET1/2/3, KDM6A/B, KDM5A/B
#   Acetyltransferases (activation):    EP300, CREBBP
#   Deacetylases (repression):          HDAC1/2/3
#   Remodelers:                         SMARCB1, SMARCA4

PROFILES = {
    "treatment_naive": {
        "description": "Treatment-naive MM — balanced chromatin machinery",
        "expected_stability": "medium",
        "protein_abundances": {
            # Balanced writers
            "EZH2": 1.0, "SUZ12": 1.0, "EED": 1.0,
            "DNMT1": 1.2, "DNMT3A": 0.8, "DNMT3B": 0.5,
            # Active erasers
            "TET1": 1.0, "TET2": 1.1, "TET3": 0.7,
            "KDM6A": 1.0, "KDM6B": 0.9,
            # Normal H3K4me3 machinery
            "SETD2": 1.0,
            "KMT2A": 1.0, "KMT2B": 0.8, "KMT2C": 0.9, "KMT2D": 1.0,
            "KDM5A": 0.8, "KDM5B": 0.7,
            # Balanced ac/deac
            "HDAC1": 1.0, "HDAC2": 0.9, "HDAC3": 0.8,
            "EP300": 1.0, "CREBBP": 1.0,
            # Remodelers and recruiter
            "UHRF1": 1.0,
            "SMARCB1": 1.0, "SMARCA4": 1.0,
        },
    },
    "bortezomib_resistant": {
        "description": "Bortezomib-resistant MM — PRC2/DNMT overexpression locks in H3K27me3",
        "expected_stability": "high",
        "protein_abundances": {
            # HIGH writers → deep epigenetic lock
            "EZH2": 4.5, "SUZ12": 3.8, "EED": 3.5,
            "DNMT1": 4.0, "DNMT3A": 3.2, "DNMT3B": 2.5,
            # LOW erasers → can't reverse the marks
            "TET1": 0.2, "TET2": 0.3, "TET3": 0.1,
            "KDM6A": 0.3, "KDM6B": 0.2,
            # Suppressed activation
            "SETD2": 0.4,
            "KMT2A": 0.5, "KMT2B": 0.3, "KMT2C": 0.4, "KMT2D": 0.3,
            "KDM5A": 0.5, "KDM5B": 0.4,
            # High deacetylases, low acetyltransferases
            "HDAC1": 3.5, "HDAC2": 3.0, "HDAC3": 2.8,
            "EP300": 0.3, "CREBBP": 0.4,
            "UHRF1": 3.5,
            "SMARCB1": 0.5, "SMARCA4": 0.4,
        },
    },
    "lenalidomide_sensitive": {
        "description": "Lenalidomide-sensitive MM — high erasers, open/reversible chromatin",
        "expected_stability": "low",
        "protein_abundances": {
            # LOW writers → no locking
            "EZH2": 0.3, "SUZ12": 0.4, "EED": 0.3,
            "DNMT1": 0.5, "DNMT3A": 0.3, "DNMT3B": 0.2,
            # HIGH erasers → actively reversing marks
            "TET1": 3.5, "TET2": 4.0, "TET3": 2.5,
            "KDM6A": 3.0, "KDM6B": 3.2,
            # Active chromatin
            "SETD2": 2.5,
            "KMT2A": 2.5, "KMT2B": 2.0, "KMT2C": 2.3, "KMT2D": 2.5,
            "KDM5A": 1.0, "KDM5B": 0.8,
            # High acetyltransferases
            "HDAC1": 0.5, "HDAC2": 0.4, "HDAC3": 0.3,
            "EP300": 3.5, "CREBBP": 3.0,
            "UHRF1": 0.3,
            "SMARCB1": 2.5, "SMARCA4": 2.8,
        },
    },
    "post_washout": {
        "description": "Post-washout MM — partial recovery, intermediate state",
        "expected_stability": "medium",
        "protein_abundances": {
            # Partially elevated writers (recovering)
            "EZH2": 2.0, "SUZ12": 1.8, "EED": 1.7,
            "DNMT1": 2.0, "DNMT3A": 1.5, "DNMT3B": 1.0,
            # Partially recovering erasers
            "TET1": 0.8, "TET2": 1.0, "TET3": 0.5,
            "KDM6A": 0.9, "KDM6B": 0.8,
            # Mixed
            "SETD2": 1.2,
            "KMT2A": 1.3, "KMT2B": 1.0, "KMT2C": 1.1, "KMT2D": 1.2,
            "KDM5A": 0.9, "KDM5B": 0.8,
            "HDAC1": 1.8, "HDAC2": 1.5, "HDAC3": 1.3,
            "EP300": 0.8, "CREBBP": 0.7,
            "UHRF1": 1.8,
            "SMARCB1": 1.0, "SMARCA4": 0.9,
        },
    },
    "multi_drug_resistant": {
        "description": "Multi-drug resistant MM — extreme epigenetic silencing",
        "expected_stability": "high",
        "protein_abundances": {
            # Extreme writers
            "EZH2": 6.0, "SUZ12": 5.5, "EED": 5.0,
            "DNMT1": 5.5, "DNMT3A": 4.5, "DNMT3B": 4.0,
            # Near-zero erasers
            "TET1": 0.05, "TET2": 0.08, "TET3": 0.02,
            "KDM6A": 0.1, "KDM6B": 0.05,
            # Completely suppressed activation
            "SETD2": 0.1,
            "KMT2A": 0.15, "KMT2B": 0.1, "KMT2C": 0.12, "KMT2D": 0.1,
            "KDM5A": 0.3, "KDM5B": 0.2,
            # Maximal repression
            "HDAC1": 5.0, "HDAC2": 4.5, "HDAC3": 4.0,
            "EP300": 0.1, "CREBBP": 0.15,
            "UHRF1": 5.0,
            "SMARCB1": 0.2, "SMARCA4": 0.15,
        },
    },
}


def test_health(base_url: str) -> bool:
    """Test /health endpoint."""
    print("─" * 60)
    print("Testing /health")
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        data = r.json()
        print(f"  Status:       {data['status']}")
        print(f"  Model loaded: {data['model_loaded']}")
        print(f"  Device:       {data['device']}")
        print(f"  Version:      {data['version']}")
        return data["model_loaded"]
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_config(base_url: str) -> dict | None:
    """Test /config endpoint."""
    print("\n" + "─" * 60)
    print("Testing /config")
    try:
        r = requests.get(f"{base_url}/config", timeout=10)
        data = r.json()
        print(f"  Target drugs:  {data['target_drugs']}")
        print(f"  Latent dim:    {data['latent_dim']}")
        print(f"  GNN layers:    {data['num_gnn_layers']}")
        print(f"  Reader/writer proteins: {len(data['reader_writer_proteins'])} tracked")
        return data
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def test_single_prediction(base_url: str, name: str, profile: dict) -> dict | None:
    """Test /predict endpoint with a single profile."""
    print(f"\n{'─' * 60}")
    print(f"Testing /predict — {name}")
    print(f"  {profile['description']}")
    print(f"  Expected stability: {profile['expected_stability']}")

    payload = {"protein_abundances": profile["protein_abundances"]}

    try:
        start = time.time()
        r = requests.post(f"{base_url}/predict", json=payload, timeout=60)
        elapsed = time.time() - start

        if r.status_code != 200:
            print(f"  FAILED (HTTP {r.status_code}): {r.text[:200]}")
            return None

        data = r.json()
        print(f"  Latency:      {elapsed:.2f}s")
        print(f"  Stability:    {data['stability_score']:.4f}")

        # Memory state summary
        mem = data["memory_state"]
        mem_abs = [abs(x) for x in mem]
        print(f"  Memory state: dim={len(mem)}, "
              f"mean_abs={sum(mem_abs)/len(mem_abs):.4f}, "
              f"max_abs={max(mem_abs):.4f}")

        # Drug predictions
        print(f"  Drug predictions:")
        for dp in data["drug_predictions"]:
            print(f"    {dp['drug_name']:20s}  "
                  f"IC50={dp['predicted_ic50']:+8.4f}  "
                  f"P(resist)={dp['resistance_probability']:.3f}  "
                  f"P(revers)={dp['reversibility_probability']:.3f}")

        # Interpretation
        interp = data["interpretation"]
        if len(interp) > 120:
            interp = interp[:117] + "..."
        print(f"  Interpretation: {interp}")

        return data

    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def test_batch_prediction(base_url: str) -> list | None:
    """Test /predict/batch endpoint with all 5 profiles."""
    print(f"\n{'─' * 60}")
    print("Testing /predict/batch — all 5 profiles at once")

    samples = [
        {"protein_abundances": p["protein_abundances"]}
        for p in PROFILES.values()
    ]
    payload = {"samples": samples}

    try:
        start = time.time()
        r = requests.post(f"{base_url}/predict/batch", json=payload, timeout=120)
        elapsed = time.time() - start

        if r.status_code != 200:
            print(f"  FAILED (HTTP {r.status_code}): {r.text[:200]}")
            return None

        data = r.json()
        print(f"  Latency:  {elapsed:.2f}s for {len(data)} samples "
              f"({elapsed/len(data):.2f}s per sample)")

        names = list(PROFILES.keys())
        stabilities = [d["stability_score"] for d in data]
        print(f"\n  Stability scores across profiles:")
        for name, stab in zip(names, stabilities):
            expected = PROFILES[name]["expected_stability"]
            bar = "█" * int(stab * 30)
            match = "✓" if (
                (expected == "high" and stab > 0.6)
                or (expected == "medium" and 0.3 < stab < 0.7)
                or (expected == "low" and stab < 0.4)
            ) else "?"
            print(f"    {name:25s} {stab:.4f} {bar:30s} [{expected}] {match}")

        variance = sum((s - sum(stabilities)/len(stabilities))**2
                       for s in stabilities) / len(stabilities)
        print(f"\n  Score variance: {variance:.6f}")
        if variance < 0.001:
            print("  ⚠  Low variance — stability scorer may not be discriminating.")
        else:
            print("  ✓  Good variance — scores differentiate profiles.")

        return data

    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def run_all_tests(base_url: str, run_batch: bool = False) -> None:
    """Run the complete test suite."""
    print("=" * 60)
    print("  MyeloMemory API Test Suite")
    print(f"  Target: {base_url}")
    print("=" * 60)

    # Health check
    healthy = test_health(base_url)
    if not healthy:
        print("\n⚠  API is not healthy. Aborting.")
        return

    # Config
    test_config(base_url)

    # Individual predictions
    results = {}
    for name, profile in PROFILES.items():
        result = test_single_prediction(base_url, name, profile)
        if result:
            results[name] = result

    # Batch prediction
    if run_batch:
        test_batch_prediction(base_url)

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print("=" * 60)

    if results:
        stabilities = {name: r["stability_score"] for name, r in results.items()}
        print(f"\n  Stability scores:")
        for name, stab in sorted(stabilities.items(), key=lambda x: x[1], reverse=True):
            expected = PROFILES[name]["expected_stability"]
            print(f"    {name:25s} → {stab:.4f}  (expected: {expected})")

        print(f"\n  Score range: {min(stabilities.values()):.4f} — {max(stabilities.values()):.4f}")
        spread = max(stabilities.values()) - min(stabilities.values())
        print(f"  Score spread: {spread:.4f}")

        # Check biological ordering
        checks = []
        if "bortezomib_resistant" in stabilities and "lenalidomide_sensitive" in stabilities:
            ok = stabilities["bortezomib_resistant"] > stabilities["lenalidomide_sensitive"]
            checks.append(("Bortez-resistant > Lenal-sensitive", ok))
        if "multi_drug_resistant" in stabilities and "treatment_naive" in stabilities:
            ok = stabilities["multi_drug_resistant"] > stabilities["treatment_naive"]
            checks.append(("Multi-drug-resistant > Treatment-naive", ok))
        if "multi_drug_resistant" in stabilities and "lenalidomide_sensitive" in stabilities:
            ok = stabilities["multi_drug_resistant"] > stabilities["lenalidomide_sensitive"]
            checks.append(("Multi-drug-resistant > Lenal-sensitive", ok))

        if checks:
            print(f"\n  Biological ordering checks:")
            for desc, ok in checks:
                print(f"    {'✓' if ok else '✗'}  {desc}")

    print(f"\n  Tests completed: {len(results)}/{len(PROFILES)} profiles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MyeloMemory API test suite")
    parser.add_argument("--url", default="http://localhost:8001",
                        help="API base URL (default: http://localhost:8001)")
    parser.add_argument("--batch", action="store_true",
                        help="Also test batch endpoint")
    args = parser.parse_args()

    run_all_tests(args.url, run_batch=args.batch)