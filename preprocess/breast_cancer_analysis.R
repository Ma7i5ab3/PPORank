# breast_cancer_analysis.R
# Harmonize TCGA BRCA RNA-seq (Firehose 2016_01_28) with GDSC microarray
# using the pRRophetic pipeline (Geeleher et al.), as described in PPORank
# Section 3.4 (reference [63]).
#
# Input (download first — see JOURNAL.md TCGA section):
#   data/GDSC_ALL/gdac.broadinstitute.org_BRCA.mRNAseq_Preprocess.Level_3.2016012800.0.0/
#
# Output:
#   data/GDSC_ALL/pRRophetic_TCGA_BRCA_trainFrame.csv.gz
#   data/GDSC_ALL/pRRophetic_TCGA_BRCA_testFrame.csv.gz
#   data/GDSC_ALL/pRRophetic_TCGA_BRCA_preds.csv.gz
#
# One-time install (run in R):
#   install.packages("devtools")
#   devtools::install_github("paulgeeleher/pRRophetic", build_vignettes=FALSE)
#
# Run from the PPORank project root:
#   Rscript preprocess/breast_cancer_analysis.R

library(pRRophetic)

DATA_DIR    <- "data/GDSC_ALL"
FIREHOSE_DIR <- file.path(DATA_DIR,
    "gdac.broadinstitute.org_BRCA.mRNAseq_Preprocess.Level_3.2016012800.0.0")
OUT_PREFIX  <- file.path(DATA_DIR, "pRRophetic_TCGA_BRCA")

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load TCGA BRCA mRNAseq Level 3 expression from Firehose
# ─────────────────────────────────────────────────────────────────────────────
expr_files <- c(
    list.files(FIREHOSE_DIR,
               pattern = "mRNAseq_RSEM_normalized_log2\\.txt$",
               recursive = TRUE, full.names = TRUE),
    list.files(FIREHOSE_DIR,
               pattern = "RSEM_genes_normalized.*data\\.txt$",
               recursive = TRUE, full.names = TRUE),
    list.files(FIREHOSE_DIR,
               pattern = "mRNAseq_RPKM_log2\\.txt$",
               recursive = TRUE, full.names = TRUE)
)
if (length(expr_files) == 0) {
    stop("No expression file found. Expected RSEM_genes_normalized or RPKM_log2 in:\n  ", FIREHOSE_DIR)
}
expr_file <- expr_files[1]
message("[Step 1] Loading: ", expr_file)

raw <- read.table(expr_file, header = TRUE, sep = "\t",
                  row.names = 1, check.names = FALSE, comment.char = "")

# The first data row is a description row (gene_id / hybridization_ref) — remove it
first_val <- suppressWarnings(as.numeric(as.character(raw[1, 1])))
if (is.na(first_val)) raw <- raw[-1, ]

tcga_mat <- data.matrix(raw)

# Gene names are "SYMBOL|ENTREZ_ID" — keep symbol only
gene_symbols <- sapply(strsplit(rownames(tcga_mat), "\\|"), `[`, 1)
valid        <- !is.na(gene_symbols) & gene_symbols != "?"
tcga_mat     <- tcga_mat[valid, ]
rownames(tcga_mat) <- gene_symbols[valid]

# Collapse duplicate symbols by mean
dups <- duplicated(rownames(tcga_mat))
if (any(dups)) {
    message("  Collapsing ", sum(dups), " duplicate gene symbols")
    unique_genes <- unique(rownames(tcga_mat))
    collapsed <- do.call(rbind, lapply(unique_genes, function(g) {
        colMeans(tcga_mat[rownames(tcga_mat) == g, , drop = FALSE], na.rm = TRUE)
    }))
    rownames(collapsed) <- unique_genes
    tcga_mat <- collapsed
}

# Log2-transform if values are not already log-scaled (RPKM raw case)
if (max(tcga_mat, na.rm = TRUE) > 100) {
    message("  Values appear raw/RPKM — applying log2(x+1)")
    tcga_mat <- log2(tcga_mat + 1)
}

message("  TCGA expression: ", nrow(tcga_mat), " genes x ", ncol(tcga_mat), " samples")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Load pRRophetic's internal GDSC microarray expression
# ─────────────────────────────────────────────────────────────────────────────
message("[Step 2] Loading pRRophetic GDSC data...")

local_env <- new.env()

gdsc_expr <- tryCatch({
    data("cgp2016ExprRma", package = "pRRophetic", envir = local_env)
    local_env$cgp2016ExprRma
}, error = function(e) tryCatch({
    data("cgpExprRma", package = "pRRophetic", envir = local_env)
    local_env$cgpExprRma
}, error = function(e2) {
    avail <- data(package = "pRRophetic")$results[, "Item"]
    stop("Cannot load GDSC expression from pRRophetic.\nAvailable: ",
         paste(avail, collapse = ", "))
}))

gdsc_pheno <- tryCatch({
    data("cgp2016pheno", package = "pRRophetic", envir = local_env)
    local_env$cgp2016pheno
}, error = function(e) tryCatch({
    data("cgpPheno", package = "pRRophetic", envir = local_env)
    local_env$cgpPheno
}, error = function(e2) {
    message("  Warning: could not load GDSC phenotype data (Resp column will be NA)")
    NULL
}))

message("  GDSC expression: ", nrow(gdsc_expr), " genes x ", ncol(gdsc_expr), " cell lines")

# pRRophetic's internal GDSC data has NA-named and duplicate cell line columns;
# homogenizeData fails when these become row names after transposition.
na_cols  <- is.na(colnames(gdsc_expr))
dup_cols <- duplicated(colnames(gdsc_expr)) & !na_cols
bad_cols <- na_cols | dup_cols
if (any(bad_cols)) {
    message("  Removing ", sum(bad_cols), " bad cell line columns (",
            sum(na_cols), " NA-named, ", sum(dup_cols), " duplicates)")
    gdsc_expr <- gdsc_expr[, !bad_cols]
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Harmonize via ComBat (pRRophetic::homogenizeData)
# ─────────────────────────────────────────────────────────────────────────────
message("[Step 3] ComBat batch correction (may take several minutes)...")

harmonized <- homogenizeData(
    testExprMat  = tcga_mat,   # genes x TCGA patients
    trainExprMat = gdsc_expr,  # genes x GDSC cell lines
    batchCorrect = "eb",       # ComBat empirical Bayes (as in paper)
    selection    = 1,          # collapse duplicates by mean
    printOutput  = TRUE
)
# harmonized$test  — genes x TCGA patients  (harmonized)
# harmonized$train — genes x GDSC cell lines (harmonized)

n_common <- nrow(harmonized$train)
message("  Common genes after harmonization: ", n_common)

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Save trainFrame  [cell_line_id, Resp, gene1, gene2, ...]
# ─────────────────────────────────────────────────────────────────────────────
message("[Step 4] Building trainFrame...")

train_expr <- t(harmonized$train)  # cell lines x genes

# Build Resp column from lapatinib IC50 in pRRophetic's GDSC phenotype data
resp_vec <- rep(NA_real_, nrow(train_expr))
if (!is.null(gdsc_pheno)) {
    lap_col <- grep("lapatinib", colnames(gdsc_pheno), ignore.case = TRUE, value = TRUE)[1]
    if (!is.na(lap_col)) {
        lap_vals <- setNames(gdsc_pheno[, lap_col], rownames(gdsc_pheno))
        matched  <- lap_vals[rownames(train_expr)]
        resp_vec <- as.numeric(matched)
        na_count <- sum(is.na(resp_vec))
        if (na_count > 0) {
            resp_vec[is.na(resp_vec)] <- median(resp_vec, na.rm = TRUE)
            message("  Imputed ", na_count, " missing Resp values with median")
        }
    }
}

train_frame <- data.frame(
    cell_line_id     = rownames(train_expr),
    Resp             = resp_vec,
    train_expr,
    check.names      = FALSE,
    stringsAsFactors = FALSE
)

out_train <- paste0(OUT_PREFIX, "_trainFrame.csv.gz")
write.csv(train_frame, gzfile(out_train), row.names = FALSE)
message("  Saved: ", out_train)

# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Save testFrame  [patient_id, gene1, gene2, ...]
# ─────────────────────────────────────────────────────────────────────────────
message("[Step 5] Building testFrame...")

test_expr <- t(harmonized$test)  # patients x genes

test_frame <- data.frame(
    patient_id       = rownames(test_expr),
    test_expr,
    check.names      = FALSE,
    stringsAsFactors = FALSE
)

out_test <- paste0(OUT_PREFIX, "_testFrame.csv.gz")
write.csv(test_frame, gzfile(out_test), row.names = FALSE)
message("  Saved: ", out_test)

# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Save preds  (required by load_pRRophetic_data format check)
# create_TCGA_data.py does NOT use preds values (test_y_suffix=None),
# but load_pRRophetic_data asserts that preds[:,0] == testFrame[:,0].
# We save a placeholder with the same patient IDs and dummy IC50 = 0.
# ─────────────────────────────────────────────────────────────────────────────
message("[Step 6] Saving preds placeholder...")

preds_frame <- data.frame(
    patient_id       = test_frame$patient_id,
    Resp             = 0.0,
    stringsAsFactors = FALSE
)

out_preds <- paste0(OUT_PREFIX, "_preds.csv.gz")
write.csv(preds_frame, gzfile(out_preds), row.names = FALSE)
message("  Saved: ", out_preds)

message("\nDone. Output files:")
message("  ", out_train,  "  (", nrow(train_frame), " cell lines x ", ncol(train_frame) - 2, " genes)")
message("  ", out_test,   "  (", nrow(test_frame),  " patients x ",  ncol(test_frame)  - 1, " genes)")
message("  ", out_preds,  "  (placeholder, ", nrow(preds_frame), " entries)")
