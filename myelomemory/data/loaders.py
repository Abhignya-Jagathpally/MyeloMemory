"""Dataset loaders for CCLE proteomics, epigenomics, STRING PPI, and drug sensitivity.

Each loader returns a standardized dictionary with tensors and metadata.
All loaders handle missing data, format conversion, and basic validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from myelomemory.config import DataConfig

logger = logging.getLogger(__name__)


class MultiOmicsDataset(Dataset):
    """PyTorch dataset holding matched proteomics + epigenomics for cell lines.

    Attributes:
        proteomics: (N, P) tensor of protein abundances.
        epigenomics: (N, E) tensor of ATAC-seq / histone mark signals.
        cell_line_ids: List of CCLE cell line identifiers.
        lineage: List of tissue lineage labels.
        drug_sensitivity: (N, D) tensor of IC50 values (NaN where missing).
        protein_names: List of protein/gene names for the P columns.
        epigenome_feature_names: List of epigenomic feature identifiers.
    """

    def __init__(
        self,
        proteomics: torch.Tensor,
        epigenomics: torch.Tensor,
        cell_line_ids: list[str],
        lineage: list[str],
        drug_sensitivity: torch.Tensor,
        protein_names: list[str],
        epigenome_feature_names: list[str],
        ppi_edges: list[tuple[str, str]] | None = None,
        ppi_scores: list[float] | None = None,
    ) -> None:
        assert proteomics.shape[0] == epigenomics.shape[0] == len(cell_line_ids)
        self.proteomics = proteomics
        self.epigenomics = epigenomics
        self.cell_line_ids = cell_line_ids
        self.lineage = lineage
        self.drug_sensitivity = drug_sensitivity
        self.protein_names = protein_names
        self.epigenome_feature_names = epigenome_feature_names
        self.ppi_edges = ppi_edges
        self.ppi_scores = ppi_scores

    def __len__(self) -> int:
        return self.proteomics.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "proteomics": self.proteomics[idx],
            "epigenomics": self.epigenomics[idx],
            "drug_sensitivity": self.drug_sensitivity[idx],
        }

    def subset_by_lineage(self, lineages: list[str]) -> "MultiOmicsDataset":
        """Return a new dataset filtered to specific tissue lineages."""
        mask = [lin in lineages for lin in self.lineage]
        indices = [i for i, m in enumerate(mask) if m]
        return MultiOmicsDataset(
            proteomics=self.proteomics[indices],
            epigenomics=self.epigenomics[indices],
            cell_line_ids=[self.cell_line_ids[i] for i in indices],
            lineage=[self.lineage[i] for i in indices],
            drug_sensitivity=self.drug_sensitivity[indices],
            protein_names=self.protein_names,
            epigenome_feature_names=self.epigenome_feature_names,
            ppi_edges=self.ppi_edges,
            ppi_scores=self.ppi_scores,
        )


def load_ccle_proteomics(config: DataConfig) -> dict[str, Any]:
    """Load CCLE proteomics data from DepMap.

    Expected format: CSV with cell lines as rows, proteins as columns.
    First column is cell line ID.

    Returns:
        Dict with keys: 'data' (DataFrame), 'cell_line_ids', 'protein_names'.
    """
    path = config.ccle_proteomics_path
    logger.info(f"Loading CCLE proteomics from {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"CCLE proteomics not found at {path}. "
            "Run scripts/download_data.sh first."
        )

    df = pd.read_csv(path, index_col=0)
    logger.info(f"Loaded proteomics: {df.shape[0]} cell lines, {df.shape[1]} proteins")

    # Drop proteins with too much missing data
    coverage = df.notna().mean(axis=0)
    kept = coverage >= config.min_protein_coverage
    df = df.loc[:, kept]
    logger.info(
        f"After coverage filter ({config.min_protein_coverage:.0%}): "
        f"{df.shape[1]} proteins retained"
    )

    # Map UniProt IDs to gene symbols if a mapping file is available
    mapping_path = path.parent / "uniprot_hugo_mapping.csv"
    if mapping_path.exists():
        mapping_df = pd.read_csv(mapping_path)
        uniprot_to_gene = dict(zip(mapping_df["UniprotID"], mapping_df["Symbol"]))
        old_cols = df.columns.tolist()
        new_cols = [uniprot_to_gene.get(c, c) for c in old_cols]
        n_mapped = sum(1 for o, n in zip(old_cols, new_cols) if o != n)
        df.columns = new_cols
        # Drop duplicate gene symbols (keep first occurrence)
        df = df.loc[:, ~df.columns.duplicated()]
        logger.info(f"Mapped {n_mapped}/{len(old_cols)} UniProt IDs to gene symbols")

    return {
        "data": df,
        "cell_line_ids": df.index.tolist(),
        "protein_names": df.columns.tolist(),
    }


def load_ccle_epigenomics(config: DataConfig) -> dict[str, Any]:
    """Load CCLE epigenomic data.

    Supports multiple data sources in order of preference:
        1. chromatin_profiling.csv — CCLE Global Chromatin Profiling (histone marks)
        2. atac_seq.csv, h3k4me3.csv, h3k27me3.csv — individual assay files

    Expected directory structure:
        ccle_epigenomics/
        ├── chromatin_profiling.csv  # Cell lines × histone modifications
        ├── atac_seq.csv             # Cell lines × peaks (optional)
        ├── h3k4me3.csv              # Cell lines × peaks (optional)
        └── h3k27me3.csv             # Cell lines × peaks (optional)

    Returns:
        Dict with keys for each loaded assay, plus 'cell_line_ids' and
        'feature_names'.
    """
    epi_dir = config.ccle_epigenomics_dir
    logger.info(f"Loading CCLE epigenomics from {epi_dir}")

    if not epi_dir.exists():
        raise FileNotFoundError(
            f"CCLE epigenomics directory not found at {epi_dir}. "
            "Run scripts/download_data.sh first."
        )

    result = {}
    all_feature_names = []

    # Try chromatin profiling first (CCLE Global Chromatin Profiling)
    chrom_path = epi_dir / "chromatin_profiling.csv"
    if chrom_path.exists():
        df = pd.read_csv(chrom_path)

        # CCLE Global Chromatin Profiling has CellLineName + BroadID columns
        # Use BroadID (ACH-XXXXXX) as index for consistency with proteomics
        if "BroadID" in df.columns:
            df = df.set_index("BroadID")
            # Drop non-numeric metadata columns
            df = df.select_dtypes(include=["number"])
            logger.info(f"  chromatin_profiling: indexed by BroadID (ACH format)")
        else:
            df = df.set_index(df.columns[0])
            df = df.select_dtypes(include=["number"])

        result["chromatin_profiling"] = df
        all_feature_names.extend([f"chromatin:{col}" for col in df.columns])
        logger.info(f"  chromatin_profiling: {df.shape[0]} lines, {df.shape[1]} features")

    # Also load individual assay files if available
    for assay in ("atac_seq", "h3k4me3", "h3k27me3"):
        csv_path = epi_dir / f"{assay}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0)
            result[assay] = df
            all_feature_names.extend([f"{assay}:{col}" for col in df.columns])
            logger.info(f"  {assay}: {df.shape[0]} lines, {df.shape[1]} features")

    # Determine cell line IDs from the first available source
    for key in ("chromatin_profiling", "atac_seq", "h3k4me3", "h3k27me3"):
        if key in result:
            result["cell_line_ids"] = result[key].index.tolist()
            break
    else:
        result["cell_line_ids"] = []

    if not all_feature_names:
        raise FileNotFoundError(
            f"No epigenomic data files found in {epi_dir}. "
            "Need at least one of: chromatin_profiling.csv, atac_seq.csv, "
            "h3k4me3.csv, h3k27me3.csv"
        )

    result["feature_names"] = all_feature_names
    return result


def load_string_ppi(config: DataConfig) -> dict[str, Any]:
    """Load STRING protein-protein interaction network.

    Expected format: TSV with columns [protein1, protein2, combined_score].
    Protein names should be gene symbols matching the proteomics columns.

    Returns:
        Dict with keys: 'edges' (list of tuples), 'scores' (list of floats),
        'proteins' (set of unique protein names).
    """
    path = config.string_ppi_path
    logger.info(f"Loading STRING PPI network from {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"STRING PPI not found at {path}. "
            "Run scripts/download_data.sh first."
        )

    df = pd.read_csv(path, sep="\t")
    expected_cols = {"protein1", "protein2", "combined_score"}
    if not expected_cols.issubset(df.columns):
        # Try alternative column names
        df.columns = ["protein1", "protein2", "combined_score"]

    # Normalize STRING scores to 0-1 if they're on the 0-1000 scale
    if df["combined_score"].max() > 1.0:
        logger.info("  STRING scores on 0-1000 scale — normalizing to 0-1")
        df["combined_score"] = df["combined_score"] / 1000.0

    # Filter by confidence threshold
    threshold = config.ppi_confidence_threshold
    df = df[df["combined_score"] >= threshold]
    logger.info(
        f"PPI edges after confidence filter (>={threshold}): "
        f"{len(df)}"
    )

    edges = list(zip(df["protein1"], df["protein2"]))
    scores = df["combined_score"].tolist()
    proteins = set(df["protein1"]) | set(df["protein2"])

    return {
        "edges": edges,
        "scores": scores,
        "proteins": proteins,
        "num_edges": len(edges),
        "num_proteins": len(proteins),
    }


def _standardize_drug_columns(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Remap drug sensitivity columns to standard names.

    Handles different column naming conventions across GDSC and CTRPv2.
    Standard output columns: cell_line_id, drug_name, ic50.
    """
    col_map = {}
    cols_lower = {c: c.lower().strip().replace(" ", "_") for c in df.columns}

    for orig, lower in cols_lower.items():
        if lower in ("arxspan_id", "depmap_id", "modelid"):
            col_map[orig] = "cell_line_id"
        elif lower in ("cell_line_name", "ccle_name", "cclename"):
            if "cell_line_id" not in col_map.values():
                col_map[orig] = "cell_line_id"
        elif lower == "drug_name":
            col_map[orig] = "drug_name"
        elif lower == "cpd_name" and "drug_name" not in col_map.values():
            col_map[orig] = "drug_name"
        elif lower in ("ic50_published", "ic50"):
            col_map[orig] = "ic50"
        elif lower == "log2.ic50" and "ic50" not in col_map.values():
            col_map[orig] = "log2_ic50"

    df = df.rename(columns=col_map)

    # Convert log2 IC50 to linear IC50 if needed
    if "ic50" not in df.columns and "log2_ic50" in df.columns:
        df["ic50"] = 2.0 ** df["log2_ic50"]

    logger.info(f"  {source_name}: mapped columns → {[c for c in ('cell_line_id', 'drug_name', 'ic50') if c in df.columns]}")
    return df


def load_drug_sensitivity(config: DataConfig) -> dict[str, Any]:
    """Load drug sensitivity data from GDSC and CTRPv2.

    Merges both sources, preferring GDSC when both are available for a
    cell line / drug pair. Handles column remapping for different formats.

    Returns:
        Dict with keys: 'data' (DataFrame, cell lines × drugs, IC50 values),
        'cell_line_ids', 'drug_names'.
    """
    logger.info("Loading drug sensitivity data")

    frames = []

    for name, path in [("GDSC", config.gdsc_path), ("CTRPv2", config.ctrpv2_path)]:
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            logger.info(f"  {name}: {len(df)} records, columns: {list(df.columns[:8])}")
            df = _standardize_drug_columns(df, name)
            frames.append(df)
        else:
            logger.warning(f"  {name}: not found at {path}, skipping")

    if not frames:
        raise FileNotFoundError("No drug sensitivity data found.")

    # Combine and pivot to cell_line × drug matrix
    combined = pd.concat(frames, ignore_index=True)

    # Filter to target drugs
    target = [d.lower() for d in config.target_drugs]
    if "drug_name" not in combined.columns:
        raise ValueError(
            "Drug sensitivity data has no 'drug_name' column after standardization. "
            f"Available columns: {list(combined.columns[:10])}"
        )
    combined["drug_lower"] = combined["drug_name"].str.lower()
    combined = combined[combined["drug_lower"].isin(target)]

    # Pivot: one row per cell line, one column per drug
    pivot = combined.pivot_table(
        index="cell_line_id",
        columns="drug_name",
        values="ic50",
        aggfunc="median",
    )

    logger.info(
        f"Drug sensitivity matrix: {pivot.shape[0]} cell lines × "
        f"{pivot.shape[1]} drugs"
    )

    return {
        "data": pivot,
        "cell_line_ids": pivot.index.tolist(),
        "drug_names": pivot.columns.tolist(),
    }