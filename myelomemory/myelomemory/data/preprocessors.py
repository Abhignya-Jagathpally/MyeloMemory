"""Data preprocessing: normalization, imputation, harmonization, and splitting.

Transforms raw DataFrames into matched PyTorch tensors ready for training.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.impute import KNNImputer
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from myelomemory.config import DataConfig
from myelomemory.data.loaders import MultiOmicsDataset, load_drug_sensitivity

logger = logging.getLogger(__name__)


def _load_lineage_labels(cell_line_ids: list[str], config: DataConfig) -> list[str]:
    """Load tissue lineage labels for cell lines from CCLE metadata.

    Expects a sample_info.csv (DepMap Model.csv) with ModelID and
    OncotreeLineage columns.

    Args:
        cell_line_ids: List of ModelID identifiers to look up.
        config: Data configuration with paths.

    Returns:
        List of lineage strings, one per cell line.

    Raises:
        FileNotFoundError: If CCLE metadata file is not found.
    """
    metadata_path = config.ccle_proteomics_path.parent / "sample_info.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"CCLE sample metadata not found at {metadata_path}.\n"
            "Download 'Model.csv' from DepMap (https://depmap.org/portal/download/all/) "
            "and save as sample_info.csv.\n"
            "Run: bash scripts/download_data.sh"
        )

    metadata = pd.read_csv(metadata_path)

    # Find the lineage column (DepMap uses OncotreeLineage)
    lineage_col = None
    for col in ("OncotreeLineage", "lineage", "Lineage", "primary_disease"):
        if col in metadata.columns:
            lineage_col = col
            break

    if lineage_col is None:
        logger.warning(f"No lineage column found in {metadata_path}. Using 'unknown'.")
        return ["unknown"] * len(cell_line_ids)

    # Build ModelID → lineage lookup
    id_col = "ModelID" if "ModelID" in metadata.columns else metadata.columns[0]
    lineage_map = dict(zip(metadata[id_col], metadata[lineage_col]))

    lineage = []
    for cl_id in cell_line_ids:
        label = lineage_map.get(cl_id)
        lineage.append(label if pd.notna(label) else "unknown")

    n_found = sum(1 for l in lineage if l != "unknown")
    logger.info(f"  Lineage labels: {n_found}/{len(cell_line_ids)} matched")

    return lineage


def _impute(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Impute missing values in a DataFrame.

    Args:
        df: Input DataFrame with potential NaN values.
        method: One of 'knn', 'median', 'zero'.

    Returns:
        Imputed DataFrame with same index and columns.
    """
    if method == "knn":
        imputer = KNNImputer(n_neighbors=5, weights="distance")
        values = imputer.fit_transform(df.values)
    elif method == "median":
        values = df.fillna(df.median()).values
    elif method == "zero":
        values = df.fillna(0.0).values
    else:
        raise ValueError(f"Unknown imputation method: {method}")

    return pd.DataFrame(values, index=df.index, columns=df.columns)


def _normalize(df: pd.DataFrame, method: str) -> tuple[pd.DataFrame, Any]:
    """Normalize feature values.

    Args:
        df: Input DataFrame.
        method: One of 'quantile', 'zscore', 'log2'.

    Returns:
        Tuple of (normalized DataFrame, fitted scaler for inverse transform).
    """
    if method == "quantile":
        scaler = QuantileTransformer(
            n_quantiles=min(1000, df.shape[0]),
            output_distribution="normal",
            random_state=42,
        )
        values = scaler.fit_transform(df.values)
    elif method == "zscore":
        scaler = StandardScaler()
        values = scaler.fit_transform(df.values)
    elif method == "log2":
        scaler = None
        values = np.log2(df.values + 1)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return pd.DataFrame(values, index=df.index, columns=df.columns), scaler


def _build_id_mapping(config: DataConfig) -> dict[str, str]:
    """Build a mapping from CCLEName → ModelID using sample_info.csv.

    This resolves the ID mismatch between datasets: proteomics uses ModelID
    (ACH-XXXXXX), while chromatin profiling uses CCLEName (NAME_TISSUE).

    Returns:
        Dict mapping CCLEName → ModelID.
    """
    metadata_path = config.ccle_proteomics_path.parent / "sample_info.csv"
    if not metadata_path.exists():
        return {}

    metadata = pd.read_csv(metadata_path)
    mapping = {}

    if "CCLEName" in metadata.columns and "ModelID" in metadata.columns:
        for _, row in metadata.iterrows():
            if pd.notna(row["CCLEName"]) and pd.notna(row["ModelID"]):
                mapping[row["CCLEName"]] = row["ModelID"]

    logger.info(f"  ID mapping: {len(mapping)} CCLEName → ModelID entries")
    return mapping


def harmonize_omics(
    proteomics: dict[str, Any],
    epigenomics: dict[str, Any],
    ppi_graph: dict[str, Any],
    config: DataConfig,
) -> MultiOmicsDataset:
    """Harmonize proteomics, epigenomics, and drug sensitivity into a single dataset.

    Steps:
        1. Build ID mapping between proteomics (ModelID) and epigenomics (CCLEName)
        2. Find cell lines present in BOTH modalities
        3. Impute missing values
        4. Normalize each modality
        5. Align protein names with PPI graph nodes
        6. Load and align drug sensitivity data
        7. Package into MultiOmicsDataset

    Args:
        proteomics: Output of load_ccle_proteomics().
        epigenomics: Output of load_ccle_epigenomics().
        ppi_graph: Output of load_string_ppi().
        config: Data configuration.

    Returns:
        MultiOmicsDataset with matched, preprocessed tensors.
    """
    logger.info("Harmonizing multi-omics data")

    prot_df = proteomics["data"]
    prot_ids = set(proteomics["cell_line_ids"])

    # Build ID mapping (CCLEName → ModelID) for cross-referencing
    ccle_to_model = _build_id_mapping(config)
    model_to_ccle = {v: k for k, v in ccle_to_model.items()}

    # Remap epigenomic cell line IDs to ModelID if needed
    epi_ids_raw = set(epigenomics.get("cell_line_ids", []))
    epi_uses_ccle_names = len(epi_ids_raw & prot_ids) == 0 and len(ccle_to_model) > 0

    if epi_uses_ccle_names:
        logger.info("  Epigenomic data uses CCLEName format — remapping to ModelID")
        epi_ids = {ccle_to_model.get(eid, eid) for eid in epi_ids_raw}
        # Remap DataFrames
        for key in ("chromatin_profiling", "atac_seq", "h3k4me3", "h3k27me3"):
            if key in epigenomics:
                df = epigenomics[key]
                new_index = [ccle_to_model.get(idx, idx) for idx in df.index]
                df.index = new_index
                epigenomics[key] = df
    else:
        epi_ids = epi_ids_raw

    # Intersect cell lines across modalities
    common_ids = sorted(prot_ids & epi_ids) if epi_ids else sorted(prot_ids)
    logger.info(f"Cell lines in common across modalities: {len(common_ids)}")

    if len(common_ids) < 10:
        logger.warning(
            f"Only {len(common_ids)} common cell lines found. "
            "Consider relaxing matching criteria."
        )

    # Subset and impute proteomics
    prot_df = prot_df.loc[common_ids]
    prot_df = _impute(prot_df, config.imputation_method)
    prot_df, prot_scaler = _normalize(prot_df, config.normalization)

    # Concatenate epigenomic assays and align to common cell lines
    epi_frames = []
    for assay in ("chromatin_profiling", "atac_seq", "h3k4me3", "h3k27me3"):
        if assay in epigenomics:
            df = epigenomics[assay]
            df = df.loc[df.index.intersection(common_ids)]
            epi_frames.append(df)

    if not epi_frames:
        raise FileNotFoundError(
            "No epigenomic data found. At least one of atac_seq.csv, "
            "h3k4me3.csv, or h3k27me3.csv is required in "
            f"{config.ccle_epigenomics_dir}.\n"
            "Download from ENCODE (https://www.encodeproject.org/) or DepMap "
            "(https://depmap.org/portal/download/all/).\n"
            "Run: bash scripts/download_data.sh"
        )

    epi_df = pd.concat(epi_frames, axis=1)
    epi_df = epi_df.loc[common_ids]
    epi_df = _impute(epi_df, config.imputation_method)
    epi_df, epi_scaler = _normalize(epi_df, "quantile")

    # Load and align drug sensitivity
    drug_data = load_drug_sensitivity(config)
    drug_df = drug_data["data"]
    drug_df = drug_df.reindex(common_ids)
    drug_tensor = torch.tensor(drug_df.values, dtype=torch.float32)
    drug_tensor = torch.where(
        torch.isnan(drug_tensor),
        torch.tensor(float("nan")),
        drug_tensor,
    )

    # Load lineage labels from CCLE metadata
    lineage = _load_lineage_labels(common_ids, config)

    # Validate PPI graph has edges matching our protein names
    ppi_proteins = ppi_graph["proteins"]
    matched_proteins = ppi_proteins & set(prot_df.columns)
    if len(matched_proteins) < 100:
        raise ValueError(
            f"Only {len(matched_proteins)} proteins in the PPI network match "
            f"the proteomics data (out of {len(prot_df.columns)} proteins and "
            f"{len(ppi_proteins)} PPI nodes). Check that protein names use the "
            "same convention (gene symbols) in both datasets.\n"
            "STRING uses Ensembl protein IDs by default — you may need the "
            "gene-symbol-mapped version from STRING or a mapping file."
        )
    logger.info(
        f"PPI-proteomics overlap: {len(matched_proteins)} proteins matched"
    )

    # Build dataset with PPI data attached
    dataset = MultiOmicsDataset(
        proteomics=torch.tensor(prot_df.values, dtype=torch.float32),
        epigenomics=torch.tensor(epi_df.values, dtype=torch.float32),
        cell_line_ids=common_ids,
        lineage=lineage,
        drug_sensitivity=drug_tensor,
        protein_names=prot_df.columns.tolist(),
        epigenome_feature_names=epi_df.columns.tolist(),
        ppi_edges=ppi_graph["edges"],
        ppi_scores=ppi_graph["scores"],
    )

    logger.info(
        f"MultiOmicsDataset: {len(dataset)} samples, "
        f"{dataset.proteomics.shape[1]} proteins, "
        f"{dataset.epigenomics.shape[1]} epigenomic features, "
        f"{ppi_graph['num_edges']} PPI edges"
    )

    return dataset


def build_train_val_test_splits(
    dataset: MultiOmicsDataset,
    config: DataConfig,
) -> dict[str, list[int]]:
    """Create stratified train/val/test splits.

    Stratifies by lineage to ensure hematological cell lines are
    represented in all splits.

    Args:
        dataset: The MultiOmicsDataset to split.
        config: Data configuration with split fractions and seed.

    Returns:
        Dict with keys 'train', 'val', 'test', each mapping to a list of indices.
    """
    n = len(dataset)
    rng = np.random.RandomState(config.random_seed)
    indices = rng.permutation(n)

    n_test = int(n * config.test_fraction)
    n_val = int(n * config.val_fraction)

    test_idx = indices[:n_test].tolist()
    val_idx = indices[n_test:n_test + n_val].tolist()
    train_idx = indices[n_test + n_val:].tolist()

    logger.info(
        f"Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )

    return {"train": train_idx, "val": val_idx, "test": test_idx}