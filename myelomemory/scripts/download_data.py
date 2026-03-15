#!/usr/bin/env python3
"""download_data.py — Programmatically fetch all required public datasets.

Downloads real data from DepMap, STRING, and GDSC public APIs.
No registration required for programmatic access to these endpoints.

DepMap changed their API in 2025 — files are now accessed via a manifest
endpoint that returns signed Google Cloud Storage URLs.

Usage:
    python scripts/download_data.py [--data-dir data/raw]
"""

import argparse
import gzip
import io
import logging
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# DepMap manifest API (returns CSV with signed download URLs)
DEPMAP_MANIFEST_URL = "https://depmap.org/portal/api/download/files"

# Filenames to look for in the DepMap manifest
DEPMAP_WANTED = {
    "proteomics": {
        "filenames": ["harmonized_MS_CCLE_Gygi.csv"],
        "releases": ["Harmonized Public Proteomics 24Q4", None],
    },
    "sample_info": {
        "filenames": ["Model.csv"],
        "releases": ["DepMap Public 25Q3", "DepMap Public 25Q2", None],
    },
    "chromatin_profiling": {
        "filenames": ["CCLE_GlobalChromatinProfiling_20181130.csv"],
        "releases": ["CCLE 2019", None],
    },
    "gdsc_dose_response": {
        "filenames": ["sanger-dose-response.csv"],
        "releases": ["Sanger GDSC1 and GDSC2", None],
    },
}

# STRING PPI — fully public
STRING_PPI_URL = (
    "https://stringdb-downloads.org/download/"
    "protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
)
STRING_ALIASES_URL = (
    "https://stringdb-downloads.org/download/"
    "protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz"
)

# GDSC — public bulk download (IC50 fitted data, current_release path)
GDSC_URLS = [
    "https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/GDSC2_fitted_dose_response_24Jul22.csv",
    "https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/GDSC1_fitted_dose_response_24Jul22.csv",
]

# CTRPv2 — NCI CTD2 Data Portal
CTRPV2_URL = "https://ctd2-data.nci.nih.gov/Public/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset/CTRPv2.0_2015_ctd2_ExpandedDataset.zip"


def download_file(url: str, dest: Path, desc: str, timeout: int = 300) -> bool:
    """Download a file with progress logging.

    Args:
        url: Source URL.
        dest: Destination file path.
        desc: Human-readable description.
        timeout: Request timeout in seconds.

    Returns:
        True if downloaded successfully, False otherwise.
    """
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"  [SKIP] {desc} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    logger.info(f"  Downloading {desc}...")
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()

        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        size_mb = dest.stat().st_size / 1e6
        logger.info(f"  [OK] {desc} ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        logger.error(f"  [FAIL] {desc}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_and_gunzip(url: str, dest: Path, desc: str) -> bool:
    """Download a .gz file and extract it."""
    gz_dest = dest.with_suffix(dest.suffix + ".gz")

    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"  [SKIP] {desc} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    if not download_file(url, gz_dest, desc + " (compressed)"):
        return False

    logger.info(f"  Extracting {gz_dest.name}...")
    try:
        with gzip.open(gz_dest, "rb") as f_in, open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_dest.unlink()
        logger.info(f"  [OK] Extracted to {dest.name}")
        return True
    except Exception as e:
        logger.error(f"  [FAIL] Extraction failed: {e}")
        return False


# ---------------------------------------------------------------------------
# DepMap manifest-based downloads
# ---------------------------------------------------------------------------

_depmap_manifest: pd.DataFrame | None = None


def _get_depmap_manifest() -> pd.DataFrame | None:
    """Fetch and cache the DepMap file manifest."""
    global _depmap_manifest
    if _depmap_manifest is not None:
        return _depmap_manifest

    logger.info("  Fetching DepMap file manifest...")
    try:
        resp = requests.get(DEPMAP_MANIFEST_URL, timeout=30)
        resp.raise_for_status()
        _depmap_manifest = pd.read_csv(io.StringIO(resp.text))
        logger.info(f"  Manifest: {len(_depmap_manifest)} files across "
                     f"{_depmap_manifest.iloc[:, 0].nunique()} releases")
        return _depmap_manifest
    except Exception as e:
        logger.error(f"  [FAIL] Could not fetch DepMap manifest: {e}")
        return None


def _find_depmap_url(filenames: list[str], releases: list[str | None]) -> str | None:
    """Search the DepMap manifest for a file and return its signed download URL."""
    manifest = _get_depmap_manifest()
    if manifest is None:
        return None

    # Identify column names (they vary across API versions)
    cols = manifest.columns.tolist()
    name_col = next((c for c in cols if "file" in c.lower() and "name" in c.lower()), cols[-1])
    url_col = next((c for c in cols if "url" in c.lower() or "download" in c.lower()), None)
    release_col = next((c for c in cols if "release" in c.lower()), cols[0])

    if url_col is None:
        logger.error(f"  Could not find URL column in manifest. Columns: {cols}")
        return None

    for fname in filenames:
        matches = manifest[manifest[name_col].str.contains(fname, case=False, na=False)]
        if matches.empty:
            continue

        # Prefer specific release if requested
        for rel in releases:
            if rel is not None:
                rel_matches = matches[matches[release_col].str.contains(rel, case=False, na=False)]
                if not rel_matches.empty:
                    return rel_matches.iloc[0][url_col]
            else:
                # Return first match from any release
                return matches.iloc[0][url_col]

    return None


def download_from_depmap(data_dir: Path, key: str, dest_name: str, desc: str) -> bool:
    """Download a file from DepMap using the manifest API."""
    dest = data_dir / dest_name

    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"  [SKIP] {desc} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    wanted = DEPMAP_WANTED[key]
    url = _find_depmap_url(wanted["filenames"], wanted["releases"])

    if url is None:
        logger.error(
            f"  [FAIL] Could not find {desc} in DepMap manifest.\n"
            f"  Looked for: {wanted['filenames']}\n"
            f"  Manual download: https://depmap.org/portal/download/all/"
        )
        return False

    return download_file(url, dest, desc, timeout=300)


# ---------------------------------------------------------------------------
# Dataset-specific download and processing functions
# ---------------------------------------------------------------------------

def _validate_csv(dest: Path, desc: str, min_rows: int = 10) -> bool:
    """Check that a downloaded file is actually a CSV, not an HTML error page."""
    if not dest.exists() or dest.stat().st_size == 0:
        return False
    try:
        with open(dest, "r") as f:
            first_line = f.readline()
        if first_line.strip().startswith("<!DOCTYPE") or first_line.strip().startswith("<html"):
            logger.error(f"  [FAIL] {desc}: downloaded file is HTML, not CSV. Removing.")
            dest.unlink()
            return False
        # Quick row count check
        df = pd.read_csv(dest, nrows=min_rows)
        if len(df) < min_rows:
            logger.warning(f"  {desc}: only {len(df)} rows (expected >= {min_rows})")
        return True
    except Exception as e:
        logger.error(f"  [FAIL] {desc}: not a valid CSV: {e}")
        dest.unlink(missing_ok=True)
        return False


def download_ccle_proteomics(data_dir: Path) -> bool:
    """Download CCLE proteomics from DepMap manifest API."""
    logger.info("=== [1/6] CCLE Proteomics ===")
    dest = data_dir / "ccle_proteomics.csv"

    if dest.exists() and dest.stat().st_size > 0:
        if _validate_csv(dest, "CCLE proteomics"):
            logger.info(f"  [SKIP] CCLE proteomics already exists ({dest.stat().st_size / 1e6:.1f} MB)")
            return True
        # If validation failed, file was removed — re-download

    if download_from_depmap(data_dir, "proteomics", "ccle_proteomics.csv", "CCLE proteomics"):
        return _validate_csv(dest, "CCLE proteomics")
    return False


def download_sample_info(data_dir: Path) -> bool:
    """Download DepMap cell line metadata with lineage annotations."""
    logger.info("=== [2/6] CCLE Sample Metadata ===")
    return download_from_depmap(data_dir, "sample_info", "sample_info.csv", "Sample metadata (Model.csv)")


def download_epigenomics(data_dir: Path) -> bool:
    """Download CCLE chromatin profiling from DepMap manifest API.

    ATAC-seq is not available in the DepMap manifest (non-release dataset).
    Instead we use CCLE_GlobalChromatinProfiling which contains histone
    modification quantification across cell lines — suitable for training
    the proteome→epigenome VAE.
    """
    logger.info("=== [3/6] CCLE Epigenomics (Chromatin Profiling) ===")
    epi_dir = data_dir / "ccle_epigenomics"
    epi_dir.mkdir(parents=True, exist_ok=True)

    dest = epi_dir / "chromatin_profiling.csv"
    if dest.exists() and dest.stat().st_size > 0:
        if _validate_csv(dest, "Chromatin profiling"):
            logger.info(f"  [SKIP] Chromatin profiling already exists ({dest.stat().st_size / 1e6:.1f} MB)")
            return True

    wanted = DEPMAP_WANTED["chromatin_profiling"]
    url = _find_depmap_url(wanted["filenames"], wanted["releases"])

    if url is None:
        logger.error(
            "  [FAIL] Could not find chromatin profiling in DepMap manifest.\n"
            "  Manual download: https://depmap.org/portal/download/all/\n"
            "  Look for 'CCLE_GlobalChromatinProfiling_20181130.csv' and save as:\n"
            f"    {dest}"
        )
        return False

    if download_file(url, dest, "CCLE Global Chromatin Profiling", timeout=300):
        return _validate_csv(dest, "Chromatin profiling")
    return False


def download_string_ppi(data_dir: Path) -> bool:
    """Download STRING PPI and convert Ensembl protein IDs to gene symbols."""
    logger.info("=== [4/6] STRING PPI Network ===")

    ppi_dest = data_dir / "string_ppi_raw.txt"
    aliases_dest = data_dir / "string_aliases.txt"
    final_dest = data_dir / "string_ppi.txt"

    if final_dest.exists() and final_dest.stat().st_size > 0:
        logger.info(f"  [SKIP] STRING PPI already exists ({final_dest.stat().st_size / 1e6:.1f} MB)")
        return True

    # Download raw PPI links
    if not download_and_gunzip(STRING_PPI_URL, ppi_dest, "STRING protein links"):
        return False

    # Download aliases for ID mapping
    if not download_and_gunzip(STRING_ALIASES_URL, aliases_dest, "STRING protein aliases"):
        return False

    # Build Ensembl → gene symbol mapping
    logger.info("  Building Ensembl → gene symbol mapping...")
    try:
        alias_df = pd.read_csv(
            aliases_dest, sep="\t",
            names=["string_id", "alias", "source"],
            comment="#",
            low_memory=False,
        )

        # Prefer BioMart_HUGO and Ensembl_HGNC as gene symbol sources
        preferred_sources = ["BioMart_HUGO", "Ensembl_HGNC", "Ensembl_HGNC_symbol",
                             "Ensembl_gene_name", "BioMart_HUGO_Symbol"]
        best_map = {}
        for source in reversed(preferred_sources):
            subset = alias_df[alias_df["source"].str.contains(source, case=False, na=False)]
            for _, row in subset.iterrows():
                best_map[row["string_id"]] = row["alias"]

        # Also try any alias that looks like a gene symbol (all caps, short)
        if len(best_map) < 5000:
            for _, row in alias_df.iterrows():
                if row["string_id"] not in best_map:
                    alias = str(row["alias"])
                    if alias.isupper() and 2 <= len(alias) <= 15 and alias.isalpha():
                        best_map[row["string_id"]] = alias

        logger.info(f"  Mapped {len(best_map)} Ensembl IDs to gene symbols")

        # Load PPI, map IDs, filter, save
        logger.info("  Converting PPI to gene symbols...")
        ppi_df = pd.read_csv(ppi_dest, sep=" ")
        ppi_df["protein1"] = ppi_df["protein1"].map(best_map)
        ppi_df["protein2"] = ppi_df["protein2"].map(best_map)

        # Drop unmapped edges
        before = len(ppi_df)
        ppi_df = ppi_df.dropna(subset=["protein1", "protein2"])
        ppi_df = ppi_df[ppi_df["protein1"] != ppi_df["protein2"]]  # no self-loops
        logger.info(f"  Edges: {before} raw → {len(ppi_df)} after mapping & cleanup")

        # Save final gene-symbol PPI
        ppi_df.to_csv(final_dest, sep="\t", index=False)
        logger.info(f"  [OK] STRING PPI with gene symbols saved ({len(ppi_df)} edges)")

        # Cleanup intermediates
        ppi_dest.unlink(missing_ok=True)
        aliases_dest.unlink(missing_ok=True)
        return True

    except Exception as e:
        logger.error(f"  [FAIL] PPI processing: {e}")
        return False


def download_gdsc(data_dir: Path) -> bool:
    """Download GDSC drug sensitivity data.

    Tries DepMap manifest first (sanger-dose-response.csv), then Sanger FTP.
    """
    logger.info("=== [5/6] GDSC Drug Sensitivity ===")
    dest = data_dir / "gdsc_drug_sensitivity.csv"

    if dest.exists() and dest.stat().st_size > 0:
        if _validate_csv(dest, "GDSC"):
            logger.info(f"  [SKIP] GDSC already exists ({dest.stat().st_size / 1e6:.1f} MB)")
            return True

    # Try DepMap manifest first (cleaner, pre-processed)
    if download_from_depmap(data_dir, "gdsc_dose_response", "gdsc_drug_sensitivity.csv",
                            "GDSC dose-response (DepMap)"):
        if _validate_csv(dest, "GDSC"):
            return True

    # Fallback: Sanger FTP current_release
    for url in GDSC_URLS:
        if download_file(url, dest, f"GDSC dose-response ({Path(url).stem})"):
            if _validate_csv(dest, "GDSC"):
                return True

    logger.error(
        "  Could not download GDSC automatically.\n"
        "  Manual download: https://www.cancerrxgene.org/downloads/bulk_download\n"
        "  Save as: data/raw/gdsc_drug_sensitivity.csv\n"
        "  Expected: GDSC2_fitted_dose_response CSV with IC50 columns"
    )
    return False


def download_ctrpv2(data_dir: Path) -> bool:
    """Download CTRPv2 drug sensitivity from NCI CTD2 Data Portal."""
    logger.info("=== [6/6] CTRPv2 Drug Sensitivity ===")
    dest = data_dir / "ctrpv2_drug_sensitivity.csv"

    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"  [SKIP] CTRPv2 already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    # Download ZIP from NCI CTD2
    zip_dest = data_dir / "ctrpv2_raw.zip"
    if not download_file(CTRPV2_URL, zip_dest, "CTRPv2 expanded dataset (zip)", timeout=600):
        logger.error(
            "  Could not download CTRPv2 automatically.\n"
            "  Manual download: https://ctd2-data.nci.nih.gov/Public/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset/\n"
            "  Or: https://zenodo.org/records/3905470\n"
            "  Save as: data/raw/ctrpv2_drug_sensitivity.csv\n"
            "  Columns needed: cell_line_id, drug_name, ic50"
        )
        return False

    # Extract and process
    logger.info("  Extracting CTRPv2 zip...")
    try:
        extract_dir = data_dir / "ctrpv2_extracted"
        with zipfile.ZipFile(zip_dest, "r") as zf:
            zf.extractall(extract_dir)

        # Find the dose-response curve file
        csv_files = list(extract_dir.rglob("*.csv")) + list(extract_dir.rglob("*.txt"))
        logger.info(f"  Extracted {len(csv_files)} files")

        # Look for the AUC/EC50 file
        drc_file = None
        for f in csv_files:
            name_lower = f.name.lower()
            if "drc" in name_lower or "auc" in name_lower or "ec50" in name_lower or "cpd" in name_lower:
                drc_file = f
                break

        if drc_file is None and csv_files:
            # Use the largest CSV as fallback
            drc_file = max(csv_files, key=lambda f: f.stat().st_size)

        if drc_file is not None:
            shutil.copy2(drc_file, dest)
            logger.info(f"  [OK] CTRPv2: {drc_file.name} → {dest.name}")
        else:
            logger.error("  No suitable CSV found in CTRPv2 zip")
            return False

        # Cleanup
        shutil.rmtree(extract_dir, ignore_errors=True)
        zip_dest.unlink(missing_ok=True)
        return True

    except Exception as e:
        logger.error(f"  [FAIL] CTRPv2 extraction: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download MyeloMemory datasets")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  MyeloMemory — Automated Data Download")
    logger.info(f"  Target: {data_dir.resolve()}")
    logger.info("=" * 60)

    results = {
        "CCLE Proteomics": download_ccle_proteomics(data_dir),
        "Sample Metadata": download_sample_info(data_dir),
        "Epigenomics": download_epigenomics(data_dir),
        "STRING PPI": download_string_ppi(data_dir),
        "GDSC": download_gdsc(data_dir),
        "CTRPv2": download_ctrpv2(data_dir),
    }

    logger.info("")
    logger.info("=" * 60)
    logger.info("  DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    all_ok = True
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        if not ok:
            all_ok = False
        logger.info(f"  [{status:6s}] {name}")

    if all_ok:
        logger.info("")
        logger.info("  All datasets downloaded successfully.")
        logger.info("  Next: python main.py --config configs/default.yaml")
    else:
        logger.info("")
        logger.info("  Some downloads failed. See errors above for manual instructions.")
        logger.info("  The pipeline requires ALL datasets to run.")


if __name__ == "__main__":
    main()
