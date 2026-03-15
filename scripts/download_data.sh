#!/usr/bin/env bash
# download_data.sh — Fetch all public datasets required by MyeloMemory
#
# IMPORTANT: Several datasets require free registration before download.
# This script will attempt automatic downloads where possible and give
# step-by-step manual instructions where registration is required.
#
# Usage:
#   bash scripts/download_data.sh [--data-dir data/raw]
#
# Required datasets (ALL are mandatory — the pipeline has no fallbacks):
#   1. CCLE proteomics          — DepMap (free registration)
#   2. CCLE sample metadata     — DepMap (same account)
#   3. CCLE ATAC-seq            — DepMap (same account)
#   4. Histone ChIP-seq         — ENCODE (no registration needed)
#   5. STRING PPI network       — STRING-db (no registration needed)
#   6. GDSC drug sensitivity    — Sanger (free registration)
#   7. CTRPv2 drug sensitivity  — Broad Institute (free registration)

set -euo pipefail

DATA_DIR="${1:-data/raw}"
mkdir -p "$DATA_DIR"
mkdir -p "$DATA_DIR/ccle_epigenomics"

MISSING=0

echo "================================================================"
echo "  MyeloMemory — Data Download & Validation"
echo "================================================================"
echo ""
echo "Target directory: $DATA_DIR"
echo "All datasets are REQUIRED. The pipeline will not run without them."
echo ""

# ---------------------------------------------------------------------------
# Helper: check if file exists and is non-empty
# ---------------------------------------------------------------------------
check_file() {
    local path="$1"
    local label="$2"
    if [ -f "$path" ] && [ -s "$path" ]; then
        local size
        size=$(du -h "$path" | cut -f1)
        echo "  [OK]     $label ($size)"
        return 0
    else
        echo "  [MISSING] $label"
        MISSING=$((MISSING + 1))
        return 1
    fi
}

# ---------------------------------------------------------------------------
# 1. CCLE Proteomics (DepMap)
# ---------------------------------------------------------------------------
echo "--- [1/7] CCLE Proteomics ---"
echo ""
if ! check_file "$DATA_DIR/ccle_proteomics.csv" "ccle_proteomics.csv"; then
    echo ""
    echo "  HOW TO GET THIS FILE:"
    echo "  1. Register at https://depmap.org/portal/ (free, institutional email)"
    echo "  2. Go to https://depmap.org/portal/download/all/"
    echo "  3. Search for 'proteomics' in the file list"
    echo "  4. Download 'Proteomics.csv' (or 'protein_quant_current_normalized.csv')"
    echo "  5. Save as: $DATA_DIR/ccle_proteomics.csv"
    echo "  Format: CSV with cell line IDs as rows, proteins as columns"
    echo ""
fi

# ---------------------------------------------------------------------------
# 2. CCLE Sample Metadata (DepMap)
# ---------------------------------------------------------------------------
echo "--- [2/7] CCLE Sample Metadata ---"
echo ""
if ! check_file "$DATA_DIR/sample_info.csv" "sample_info.csv"; then
    echo ""
    echo "  HOW TO GET THIS FILE:"
    echo "  1. Log in to https://depmap.org/portal/"
    echo "  2. Go to https://depmap.org/portal/download/all/"
    echo "  3. Search for 'Model.csv' (the cell line metadata file)"
    echo "  4. Download and save as: $DATA_DIR/sample_info.csv"
    echo "  REQUIRED COLUMNS: cell_line_id (or DepMap_ID as index), lineage"
    echo "  The 'lineage' column is used to identify hematological cell lines."
    echo ""
fi

# ---------------------------------------------------------------------------
# 3. CCLE ATAC-seq (DepMap)
# ---------------------------------------------------------------------------
echo "--- [3/7] CCLE ATAC-seq ---"
echo ""
if ! check_file "$DATA_DIR/ccle_epigenomics/atac_seq.csv" "ccle_epigenomics/atac_seq.csv"; then
    echo ""
    echo "  HOW TO GET THIS FILE:"
    echo "  1. Log in to https://depmap.org/portal/"
    echo "  2. Go to https://depmap.org/portal/download/all/"
    echo "  3. Search for 'ATAC' in the file list"
    echo "  4. Download CCLE ATAC-seq data"
    echo "  5. Save as: $DATA_DIR/ccle_epigenomics/atac_seq.csv"
    echo "  Format: CSV with cell line IDs as rows, ATAC-seq peaks as columns"
    echo ""
fi

# ---------------------------------------------------------------------------
# 4. Histone ChIP-seq (ENCODE)
# ---------------------------------------------------------------------------
echo "--- [4/7] Histone ChIP-seq (H3K4me3, H3K27me3) ---"
echo ""
H3K4_OK=0
H3K27_OK=0
check_file "$DATA_DIR/ccle_epigenomics/h3k4me3.csv" "ccle_epigenomics/h3k4me3.csv" && H3K4_OK=1
check_file "$DATA_DIR/ccle_epigenomics/h3k27me3.csv" "ccle_epigenomics/h3k27me3.csv" && H3K27_OK=1

if [ "$H3K4_OK" -eq 0 ] || [ "$H3K27_OK" -eq 0 ]; then
    echo ""
    echo "  HOW TO GET THESE FILES:"
    echo "  1. Go to https://www.encodeproject.org/"
    echo "  2. Search for: 'ChIP-seq H3K4me3 human cell line' (or H3K27me3)"
    echo "  3. Filter for cell lines matching CCLE (e.g., K562, MM.1S, U266)"
    echo "  4. Download peak signal files, process into cell-line × peak matrix"
    echo "  5. Save as: $DATA_DIR/ccle_epigenomics/h3k4me3.csv"
    echo "              $DATA_DIR/ccle_epigenomics/h3k27me3.csv"
    echo "  NOTE: At minimum ONE of atac_seq.csv, h3k4me3.csv, or h3k27me3.csv"
    echo "        must be present. All three are recommended for best results."
    echo ""
fi

# ---------------------------------------------------------------------------
# 5. STRING PPI Network (no registration required)
# ---------------------------------------------------------------------------
echo "--- [5/7] STRING PPI Network ---"
echo ""
STRING_URL="https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
if ! check_file "$DATA_DIR/string_ppi.txt" "string_ppi.txt"; then
    echo "  Attempting automatic download..."
    if curl -L -o "$DATA_DIR/string_ppi.txt.gz" "$STRING_URL" 2>/dev/null; then
        gunzip -f "$DATA_DIR/string_ppi.txt.gz"
        echo "  Downloaded and extracted successfully."
        echo ""
        echo "  IMPORTANT: STRING uses Ensembl protein IDs (e.g., 9606.ENSP00000269305)."
        echo "  You MUST convert to gene symbols to match the proteomics data."
        echo "  Download the mapping file from STRING:"
        echo "    https://string-db.org/mapping_files/STRING_display_names/"
        echo "  Or use the 'protein.aliases' file to create a lookup table."
        echo "  Expected final format: TSV with columns [protein1, protein2, combined_score]"
        echo "  where protein1/protein2 are GENE SYMBOLS (e.g., EZH2, DNMT1)."
    else
        echo ""
        echo "  Automatic download failed."
        echo "  HOW TO GET THIS FILE:"
        echo "  1. Go to https://string-db.org/cgi/download"
        echo "  2. Select organism: Homo sapiens (9606)"
        echo "  3. Download 'protein.links.v12.0.txt.gz'"
        echo "  4. Extract and convert Ensembl IDs to gene symbols"
        echo "  5. Save as: $DATA_DIR/string_ppi.txt"
        echo "  Format: TSV with columns [protein1, protein2, combined_score]"
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# 6. GDSC Drug Sensitivity (Sanger)
# ---------------------------------------------------------------------------
echo "--- [6/7] GDSC Drug Sensitivity ---"
echo ""
if ! check_file "$DATA_DIR/gdsc_drug_sensitivity.csv" "gdsc_drug_sensitivity.csv"; then
    echo ""
    echo "  HOW TO GET THIS FILE:"
    echo "  1. Register at https://www.cancerrxgene.org/ (free)"
    echo "  2. Go to https://www.cancerrxgene.org/downloads/bulk_download"
    echo "  3. Download 'Drug sensitivity data (IC50)' for GDSC1 and/or GDSC2"
    echo "  4. Reformat into a CSV with columns: cell_line_id, drug_name, ic50"
    echo "     - cell_line_id: must match the DepMap/CCLE identifiers"
    echo "     - drug_name: must include at least some of:"
    echo "       Bortezomib, Lenalidomide, Dexamethasone, Carfilzomib,"
    echo "       Pomalidomide, Daratumumab"
    echo "     - ic50: log-transformed IC50 values"
    echo "  5. Save as: $DATA_DIR/gdsc_drug_sensitivity.csv"
    echo ""
fi

# ---------------------------------------------------------------------------
# 7. CTRPv2 Drug Sensitivity (Broad)
# ---------------------------------------------------------------------------
echo "--- [7/7] CTRPv2 Drug Sensitivity ---"
echo ""
if ! check_file "$DATA_DIR/ctrpv2_drug_sensitivity.csv" "ctrpv2_drug_sensitivity.csv"; then
    echo ""
    echo "  HOW TO GET THIS FILE:"
    echo "  1. Go to https://portals.broadinstitute.org/ctrp.v2.1/"
    echo "     (or https://depmap.org/portal/ — CTRPv2 data is also on DepMap)"
    echo "  2. Download the compound sensitivity data"
    echo "  3. Reformat into a CSV with columns: cell_line_id, drug_name, ic50"
    echo "     - Same column format as GDSC above"
    echo "  4. Save as: $DATA_DIR/ctrpv2_drug_sensitivity.csv"
    echo ""
fi

# ---------------------------------------------------------------------------
# Final validation
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  VALIDATION SUMMARY"
echo "================================================================"
echo ""

REQUIRED_FILES=(
    "$DATA_DIR/ccle_proteomics.csv:CCLE Proteomics"
    "$DATA_DIR/sample_info.csv:CCLE Sample Metadata (lineage labels)"
    "$DATA_DIR/string_ppi.txt:STRING PPI Network"
    "$DATA_DIR/gdsc_drug_sensitivity.csv:GDSC Drug Sensitivity"
    "$DATA_DIR/ctrpv2_drug_sensitivity.csv:CTRPv2 Drug Sensitivity"
)

EPIGENOMIC_FILES=(
    "$DATA_DIR/ccle_epigenomics/atac_seq.csv:ATAC-seq"
    "$DATA_DIR/ccle_epigenomics/h3k4me3.csv:H3K4me3 ChIP-seq"
    "$DATA_DIR/ccle_epigenomics/h3k27me3.csv:H3K27me3 ChIP-seq"
)

TOTAL_MISSING=0

for entry in "${REQUIRED_FILES[@]}"; do
    path="${entry%%:*}"
    label="${entry##*:}"
    if [ -f "$path" ] && [ -s "$path" ]; then
        size=$(du -h "$path" | cut -f1)
        echo "  [OK]      $label ($size)"
    else
        echo "  [MISSING]  $label  ← REQUIRED"
        TOTAL_MISSING=$((TOTAL_MISSING + 1))
    fi
done

EPI_FOUND=0
for entry in "${EPIGENOMIC_FILES[@]}"; do
    path="${entry%%:*}"
    label="${entry##*:}"
    if [ -f "$path" ] && [ -s "$path" ]; then
        size=$(du -h "$path" | cut -f1)
        echo "  [OK]      $label ($size)"
        EPI_FOUND=$((EPI_FOUND + 1))
    else
        echo "  [MISSING]  $label"
    fi
done

if [ "$EPI_FOUND" -eq 0 ]; then
    echo ""
    echo "  ERROR: At least ONE epigenomic file is required (atac_seq, h3k4me3, or h3k27me3)."
    TOTAL_MISSING=$((TOTAL_MISSING + 1))
fi

echo ""
if [ "$TOTAL_MISSING" -gt 0 ]; then
    echo "  STATUS: $TOTAL_MISSING required file(s) missing."
    echo "  The pipeline WILL NOT RUN until all required files are present."
    echo "  Follow the instructions above for each missing file."
    echo ""
    echo "  After downloading, run this script again to validate:"
    echo "    bash scripts/download_data.sh"
    exit 1
else
    echo "  STATUS: All required files present."
    echo ""
    echo "  Next step: run data preprocessing:"
    echo "    python main.py --config configs/h100.yaml --stage data_prep"
    exit 0
fi