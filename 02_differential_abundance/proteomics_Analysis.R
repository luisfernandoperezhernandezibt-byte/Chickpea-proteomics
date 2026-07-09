## Dufferential analysis by Luis Fernando

required_packages <- c(
  "arrow", "dplyr", "tidyr", "tibble", "stringr", "ggplot2", "ggrepel",
  "pheatmap", "RColorBrewer", "viridis", "UpSetR", "VennDiagram",
  "limma", "matrixStats", "factoextra", "FactoMineR", "corrplot",
  "gridExtra", "scales", "reshape2", "grDevices"
)

bioc_packages <- c("limma")

for (pkg in setdiff(required_packages, bioc_packages)) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, dependencies = TRUE)
}
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
for (pkg in bioc_packages) {
  if (!requireNamespace(pkg, quietly = TRUE))
    BiocManager::install(pkg)
}

invisible(lapply(required_packages, library, character.only = TRUE))
cat("All packages loaded successfully.\n")

input_dir  <- "C:/P2/DIA"
output_dir <- "C:/P2/Results"

fasta_file <- file.path(input_dir, "uniprotkb_taxonomy_id_3827_2026_02_09.fasta")

dirs <- c("QC", "DE", "Heatmaps", "Volcano", "Venn", "Tables", "PCA")
for (d in dirs) {
  dir.create(file.path(output_dir, d), recursive = TRUE, showWarnings = FALSE)
}

treatments   <- c("RH", "GH", "CD", "CS")
treat_colors <- c(RH = "#E63946", GH = "#2A9D8F", CD = "#457B9D", CS = "#E9C46A")

theme_pub <- theme_bw(base_size = 14) +
  theme(
    plot.title       = element_blank(),
    plot.subtitle    = element_blank(),
    legend.position  = "bottom",
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey95", colour = "grey70"),
    strip.text       = element_text(face = "bold")
  )

save_pub <- function(plot, name, subdir = "QC", w = 8, h = 6) {
  tiff(file.path(output_dir, subdir, paste0(name, ".tiff")),
       width = w, height = h, units = "in", res = 600,
       compression = "lzw")
  print(plot)
  dev.off()

  ggsave(file.path(output_dir, subdir, paste0(name, ".pdf")),
         plot, width = w, height = h, bg = "white", device = cairo_pdf)
}

save_pheatmap_pub <- function(pheatmap_expr, name, subdir, w = 10, h = 8) {
  tiff(file.path(output_dir, subdir, paste0(name, ".tiff")),
       width = w, height = h, units = "in", res = 600,
       compression = "lzw")
  eval(pheatmap_expr)
  dev.off()

  pdf(file.path(output_dir, subdir, paste0(name, ".pdf")),
      width = w, height = h)
  eval(pheatmap_expr)
  dev.off()
}

write_desc <- function(name, subdir, title, placement, description,
                       method, interpretation) {
  txt_path <- file.path(output_dir, subdir, paste0(name, "_DESCRIPTION.txt"))
  lines <- c(
    paste0("FILE: ", name, ".tiff / ", name, ".pdf"),
    paste0("SUGGESTED PLACEMENT: ", placement),
    "",
    paste0("TITLE (for figure legend): ", title),
    "",
    "DESCRIPTION:", description,
    "",
    "METHOD:", method,
    "",
    "HOW TO INTERPRET:", interpretation,
    "",
    paste0("Generated: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
  )
  writeLines(lines, txt_path)
}

write_table_desc <- function(name, subdir = "Tables", title, description,
                             method, interpretation) {
  txt_path <- file.path(output_dir, subdir, paste0(name, "_DESCRIPTION.txt"))
  lines <- c(
    paste0("FILE: ", name),
    "",
    paste0("TITLE: ", title),
    "",
    "DESCRIPTION:", description,
    "",
    "METHOD:", method,
    "",
    "HOW TO INTERPRET:", interpretation,
    "",
    paste0("Generated: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
  )
  writeLines(lines, txt_path)
}

cat("Directories and helper functions ready.\n")

cat("\n--- Parsing FASTA file for protein annotations ---\n")

fasta_lines <- readLines(fasta_file, warn = FALSE)
header_idx  <- which(startsWith(fasta_lines, ">"))
headers     <- fasta_lines[header_idx]

cat("Total FASTA entries:", length(headers), "\n")

parse_fasta_header <- function(h) {

  h <- sub("^>", "", h)

  accession <- str_extract(h, "(?<=\\|)[A-Za-z0-9_]+(?=\\|)")

  entry_name <- str_extract(h, "(?<=\\|)[A-Za-z0-9_]+(?=\\s)")

  protein_name <- str_extract(h, "(?<=\\s).+?(?=\\sOS=)")

  gene_name <- str_extract(h, "(?<=GN=)[^\\s]+")

  organism <- str_extract(h, "(?<=OS=).+?(?=\\sOX=)")

  data.frame(
    Accession    = ifelse(is.na(accession), "", accession),
    Entry.Name   = ifelse(is.na(entry_name), "", entry_name),
    Protein.Name = ifelse(is.na(protein_name), "", protein_name),
    Gene.Name    = ifelse(is.na(gene_name), "", gene_name),
    Organism     = ifelse(is.na(organism), "", organism),
    stringsAsFactors = FALSE
  )
}

fasta_anno <- do.call(rbind, lapply(headers, parse_fasta_header))
cat("Parsed annotations for", nrow(fasta_anno), "proteins.\n")

fasta_anno$Short.Label <- ifelse(
  fasta_anno$Gene.Name != "",
  fasta_anno$Gene.Name,
  ifelse(
    nchar(fasta_anno$Protein.Name) > 30,
    paste0(substr(fasta_anno$Protein.Name, 1, 27), "..."),
    fasta_anno$Protein.Name
  )
)

fasta_anno$Short.Label[fasta_anno$Short.Label == ""] <- fasta_anno$Accession[fasta_anno$Short.Label == ""]

write.csv(fasta_anno, file.path(output_dir, "Tables", "fasta_protein_annotations.csv"),
          row.names = FALSE)

write_table_desc(
  "fasta_protein_annotations.csv", "Tables",
  title = "Full Protein Annotation Table from UniProt FASTA",
  description = paste(
    "Complete annotation for all", nrow(fasta_anno), "proteins in the Cicer arietinum",
    "(chickpea, taxonomy ID 3827) UniProt FASTA database. Includes accession,",
    "entry name, full protein name, gene name, organism, and short label."
  ),
  method = paste(
    "FASTA headers were parsed with regex to extract structured fields.",
    "Short labels use gene name when available, otherwise truncated protein name."
  ),
  interpretation = paste(
    "Use this table to look up any protein accession and find its biological",
    "identity. The Short.Label column is used in plots for readability."
  )
)

cat("FASTA annotation table saved.\n")

report <- arrow::read_parquet(file.path(input_dir, "report.parquet"))
report <- as.data.frame(report)

cat("\nReport dimensions:", nrow(report), "rows x", ncol(report), "cols\n")
cat("Column names:\n")
print(colnames(report))

sink(file.path(output_dir, "Tables", "column_summary.txt"))
cat("DIA-NN report.parquet column structure\n")
cat("========================================\n\n")
str(report, list.len = ncol(report))
sink()

write_table_desc(
  "column_summary.txt", "Tables",
  title = "DIA-NN Report Column Summary",
  description = "Complete structure of all columns in the raw report.parquet.",
  method = "Generated using str() on the imported parquet data frame via arrow.",
  interpretation = paste(
    "Key columns: Run (sample ID), Protein.Group (protein group ID),",
    "PG.MaxLFQ (recommended normalised protein quantity)."
  )
)

cat("\n--- Cleaning Protein.Group identifiers ---\n")

report$Protein.Group.Original <- report$Protein.Group
report$Protein.Group <- sapply(
  strsplit(report$Protein.Group, ";"),
  function(x) trimws(x[1])
)

cat("Example original:", head(unique(report$Protein.Group.Original), 3), "\n")
cat("Example cleaned: ", head(unique(report$Protein.Group), 3), "\n")

detected_pgs <- unique(report$Protein.Group)
cat("Unique protein groups (cleaned):", length(detected_pgs), "\n")

pg_lookup <- data.frame(Protein.ID = detected_pgs, stringsAsFactors = FALSE) %>%
  left_join(fasta_anno, by = c("Protein.ID" = "Accession"))

pg_lookup$Protein.Name[is.na(pg_lookup$Protein.Name)] <- "Uncharacterised protein"
pg_lookup$Gene.Name[is.na(pg_lookup$Gene.Name)]       <- ""
pg_lookup$Short.Label[is.na(pg_lookup$Short.Label)]    <- pg_lookup$Protein.ID[is.na(pg_lookup$Short.Label)]

dup_labels <- pg_lookup$Short.Label[duplicated(pg_lookup$Short.Label)]
if (length(dup_labels) > 0) {
  dup_idx <- pg_lookup$Short.Label %in% dup_labels
  pg_lookup$Short.Label[dup_idx] <- paste0(
    pg_lookup$Short.Label[dup_idx], " (", pg_lookup$Protein.ID[dup_idx], ")"
  )
}

n_mapped <- sum(pg_lookup$Protein.Name != "Uncharacterised protein")
cat("Mapped to FASTA:", n_mapped, "of", nrow(pg_lookup), "proteins\n")

write.csv(pg_lookup, file.path(output_dir, "Tables", "protein_ID_to_name_lookup.csv"),
          row.names = FALSE)

write_table_desc(
  "protein_ID_to_name_lookup.csv", "Tables",
  title = "Protein ID to Name Mapping — Supplementary Table",
  description = paste(
    "Mapping table for all", nrow(pg_lookup), "detected protein groups.",
    "Links the representative UniProt accession (first ID in each group)",
    "to the full protein name, gene name, and short label used in figures.",
    n_mapped, "proteins were successfully mapped to the FASTA database."
  ),
  method = paste(
    "The first accession from each semicolon-separated Protein.Group was",
    "matched to the UniProt FASTA header annotations. Gene names are used",
    "as short labels when available; otherwise, truncated protein names are used.",
    "Duplicate labels are disambiguated by appending the accession."
  ),
  interpretation = paste(
    "Use this table to decode any protein identifier seen in the analysis.",
    "The Short.Label column matches labels used in heatmaps and volcano plots.",
    "The full Protein.Name column provides the complete functional annotation."
  )
)

id_to_label <- setNames(pg_lookup$Short.Label, pg_lookup$Protein.ID)

cat("Protein name lookup ready.\n")

runs <- unique(report$Run)
cat("\nDetected runs:\n")
print(runs)

sample_info <- data.frame(Run = runs, stringsAsFactors = FALSE) %>%
  mutate(
    BaseName  = basename(Run),
    BaseName  = str_remove(BaseName, "\\.mzML$|\\.raw$|\\.d$|\\.wiff$"),
    Treatment = str_extract(BaseName, "^[A-Za-z]+"),
    Replicate = as.integer(str_extract(BaseName, "\\d+$")),
    SampleID  = paste0(Treatment, "_R", Replicate)
  ) %>%
  mutate(
    Treatment = case_when(
      Treatment %in% treatments ~ Treatment,
      toupper(Treatment) %in% treatments ~ toupper(Treatment),
      TRUE ~ "Unknown"
    )
  ) %>%
  arrange(Treatment, Replicate)

cat("\nSample annotation:\n")
print(as.data.frame(sample_info))

cat("\nTreatment counts:\n")
print(table(sample_info$Treatment))

if (any(sample_info$Treatment == "Unknown")) {
  stop("ERROR: Some runs could not be assigned to a treatment! Edit sample_info manually.")
}

write.csv(sample_info, file.path(output_dir, "Tables", "sample_annotation.csv"),
          row.names = FALSE)

write_table_desc(
  "sample_annotation.csv", "Tables",
  title = "Sample Annotation Table",
  description = paste(
    "Maps each DIA-NN run to treatment (RH, GH, CD, CS), replicate, and SampleID.",
    "CS is the positive control."
  ),
  method = "Treatment and replicate extracted from the Run column via regex.",
  interpretation = "Verify all 12 runs (4 treatments x 3 replicates) are correctly assigned."
)

cat("\n--- Filtering report ---\n")

report_filt <- report

if ("Global.Q.Value" %in% colnames(report_filt)) {
  report_filt <- report_filt %>% filter(Global.Q.Value <= 0.01)
  cat("After Global.Q.Value <= 0.01:", nrow(report_filt), "precursors\n")
}

if ("Global.PG.Q.Value" %in% colnames(report_filt)) {
  report_filt <- report_filt %>% filter(Global.PG.Q.Value <= 0.01)
  cat("After Global.PG.Q.Value <= 0.01:", nrow(report_filt), "precursors\n")
}

if ("Lib.Q.Value" %in% colnames(report_filt)) {
  report_filt <- report_filt %>% filter(Lib.Q.Value <= 0.01)
  cat("After Lib.Q.Value <= 0.01:", nrow(report_filt), "precursors\n")
}

quant_col <- if ("PG.MaxLFQ" %in% colnames(report_filt)) {
  "PG.MaxLFQ"
} else if ("PG.Normalised" %in% colnames(report_filt)) {
  "PG.Normalised"
} else if ("PG.Quantity" %in% colnames(report_filt)) {
  "PG.Quantity"
} else {
  stop("No suitable protein-group quantity column found!")
}
cat("Using quantity column:", quant_col, "\n")

pg_long <- report_filt %>%
  select(Run, Protein.Group, Intensity = all_of(quant_col)) %>%
  distinct(Run, Protein.Group, .keep_all = TRUE) %>%
  left_join(sample_info %>% select(Run, SampleID), by = "Run")

cat("\nRows in pg_long:", nrow(pg_long), "\n")
cat("Missing SampleID after join:", sum(is.na(pg_long$SampleID)), "\n")

dup_check <- pg_long %>% group_by(Protein.Group, SampleID) %>% filter(n() > 1)
if (nrow(dup_check) > 0) {
  cat("WARNING:", nrow(dup_check), "duplicates found. Keeping max intensity.\n")
  pg_long <- pg_long %>%
    group_by(Protein.Group, SampleID) %>%
    summarise(Intensity = max(Intensity, na.rm = TRUE), .groups = "drop")
}

pg_wide <- pg_long %>%
  select(Protein.Group, SampleID, Intensity) %>%
  pivot_wider(names_from = SampleID, values_from = Intensity) %>%
  column_to_rownames("Protein.Group")

cat("Protein-group matrix:", nrow(pg_wide), "proteins x", ncol(pg_wide), "samples\n")

pg_wide[pg_wide == 0] <- NA

stopifnot(all(colnames(pg_wide) %in% sample_info$SampleID))

missing_per_sample <- colSums(is.na(pg_wide))
total_proteins     <- nrow(pg_wide)

p_miss_df <- data.frame(
  SampleID = names(missing_per_sample),
  Missing  = as.numeric(missing_per_sample),
  Detected = total_proteins - as.numeric(missing_per_sample)
) %>%
  left_join(sample_info %>% select(SampleID, Treatment), by = "SampleID") %>%
  mutate(SampleID = factor(SampleID, levels = sample_info$SampleID)) %>%
  pivot_longer(cols = c(Missing, Detected), names_to = "Status", values_to = "Count")

p1 <- ggplot(p_miss_df, aes(x = SampleID, y = Count, fill = Status)) +
  geom_bar(stat = "identity", position = "stack") +
  scale_fill_manual(values = c(Detected = "#2A9D8F", Missing = "#E76F51")) +
  labs(x = NULL, y = "Number of Protein Groups") +
  theme_pub +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
save_pub(p1, "Fig01_missing_per_sample", "QC")

write_desc(
  "Fig01_missing_per_sample", "QC",
  title = "Protein Group Detection and Missingness per Sample",
  placement = "SUPPLEMENTARY (Figure S1)",
  description = paste(
    "Stacked bar chart showing detected (teal) vs missing (orange) protein",
    "groups per sample. Provides a rapid assessment of data completeness."
  ),
  method = paste(
    "PG.MaxLFQ values from DIA-NN were assembled into a protein x sample matrix.",
    "Zero values were treated as missing (NA)."
  ),
  interpretation = paste(
    "Comparable detection rates across samples indicate consistent data quality.",
    "Samples with substantially higher missingness may have lower protein input",
    "or MS acquisition issues."
  )
)

miss_mat <- ifelse(is.na(pg_wide), 1, 0)
n_show <- min(500, nrow(miss_mat))

hm_miss_expr <- quote(
  pheatmap(miss_mat[sample(1:nrow(miss_mat), n_show), ],
           color = c("grey95", "#E63946"),
           cluster_rows = TRUE, cluster_cols = TRUE,
           show_rownames = FALSE, main = "",
           legend_breaks = c(0, 1),
           legend_labels = c("Detected", "Missing"))
)
save_pheatmap_pub(hm_miss_expr, "Fig02_missingness_heatmap", "QC", w = 10, h = 8)

write_desc(
  "Fig02_missingness_heatmap", "QC",
  title = "Missingness Pattern Heatmap",
  placement = "SUPPLEMENTARY (Figure S2)",
  description = paste(
    "Binary heatmap: grey = detected, red = missing. Up to 500 randomly",
    "sampled proteins. Rows and columns clustered."
  ),
  method = "Binarised intensity matrix; Euclidean distance, complete linkage.",
  interpretation = paste(
    "Blocks of co-missing proteins across treatments suggest treatment-specific",
    "low-abundance proteins. Scattered missingness is typical in DIA."
  )
)

keep_protein <- apply(pg_wide, 1, function(row) {
  any(sapply(treatments, function(tr) {
    cols <- sample_info$SampleID[sample_info$Treatment == tr]
    cols <- cols[cols %in% colnames(pg_wide)]
    sum(!is.na(row[cols])) >= 2
  }))
})
pg_filt <- pg_wide[keep_protein, ]
cat("After filtering (>=2 in >=1 treatment):", nrow(pg_filt), "proteins retained\n")

pg_log2 <- log2(pg_filt)

set.seed(42)
n_imputed <- 0
for (j in 1:ncol(pg_log2)) {
  col_vals <- pg_log2[[j]]
  na_idx   <- which(is.na(col_vals))
  if (length(na_idx) > 0) {
    n_imputed <- n_imputed + length(na_idx)
    col_mean <- mean(col_vals, na.rm = TRUE)
    col_sd   <- sd(col_vals, na.rm = TRUE)
    if (is.na(col_sd) || col_sd == 0) col_sd <- 1
    pg_log2[na_idx, j] <- rnorm(length(na_idx),
                                 mean = col_mean - 1.8 * col_sd,
                                 sd   = 0.3 * col_sd)
  }
}

total_cells <- nrow(pg_log2) * ncol(pg_log2)
cat("Log2 imputation complete.", n_imputed, "of", total_cells,
    paste0("(", round(100 * n_imputed / total_cells, 1), "%) imputed.\n"))

write.csv(pg_filt, file.path(output_dir, "Tables", "protein_intensities_filtered.csv"))
write.csv(pg_log2, file.path(output_dir, "Tables", "protein_log2_imputed.csv"))

write_table_desc(
  "protein_intensities_filtered.csv", "Tables",
  title = "Filtered Protein Group Intensities (Raw Scale)",
  description = paste(nrow(pg_filt), "proteins x", ncol(pg_filt),
                      "samples. PG.MaxLFQ on original scale. NA = not detected."),
  method = paste("Filtered at Global.Q.Value <= 0.01, Global.PG.Q.Value <= 0.01.",
                 "Retained if detected in >=2/3 replicates in >=1 treatment."),
  interpretation = "Use for custom analyses. NAs indicate non-detection."
)

write_table_desc(
  "protein_log2_imputed.csv", "Tables",
  title = "Log2-Transformed and Imputed Protein Intensities",
  description = paste(nrow(pg_log2), "proteins x", ncol(pg_log2),
                      "samples. Primary matrix for all downstream analyses."),
  method = paste("MinProb imputation: random draws from N(mean-1.8*SD, 0.3*SD).",
                 n_imputed, "values imputed of", total_cells, "total."),
  interpretation = paste("This is the analysis-ready matrix. High imputation",
                         "for a protein may reduce confidence in its DE result.")
)

df_box <- pg_log2 %>%
  rownames_to_column("Protein") %>%
  pivot_longer(-Protein, names_to = "SampleID", values_to = "log2Int") %>%
  left_join(sample_info %>% select(SampleID, Treatment), by = "SampleID") %>%
  mutate(SampleID = factor(SampleID, levels = sample_info$SampleID))

p2 <- ggplot(df_box, aes(x = SampleID, y = log2Int, fill = Treatment)) +
  geom_boxplot(outlier.size = 0.3, alpha = 0.8) +
  scale_fill_manual(values = treat_colors) +
  labs(x = NULL, y = expression(Log[2]~Intensity)) +
  theme_pub +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
save_pub(p2, "Fig03_intensity_boxplot", "QC")

write_desc(
  "Fig03_intensity_boxplot", "QC",
  title = "Log2 Protein Intensity Distribution per Sample",
  placement = "SUPPLEMENTARY (Figure S3)",
  description = "Box-and-whisker plots of log2-transformed protein intensities per sample.",
  method = "Log2(PG.MaxLFQ) after imputation. Box = IQR, whiskers = 1.5x IQR.",
  interpretation = paste(
    "Aligned medians indicate successful normalisation. Outlier samples with",
    "shifted medians may need investigation."
  )
)

p3 <- ggplot(df_box, aes(x = log2Int, colour = Treatment, group = SampleID)) +
  geom_density(linewidth = 0.6, alpha = 0.8) +
  scale_colour_manual(values = treat_colors) +
  labs(x = expression(Log[2]~Intensity), y = "Density") +
  theme_pub
save_pub(p3, "Fig04_intensity_density", "QC")

write_desc(
  "Fig04_intensity_density", "QC",
  title = "Log2 Intensity Density Distribution per Sample",
  placement = "SUPPLEMENTARY (Figure S4)",
  description = "Kernel density of log2 intensities coloured by treatment.",
  method = "Gaussian KDE on log2(PG.MaxLFQ) imputed values.",
  interpretation = paste(
    "Overlapping curves = good normalisation. Left-shoulder peak may reflect",
    "imputed low-abundance values."
  )
)

cor_mat <- cor(pg_log2, use = "pairwise.complete.obs", method = "pearson")

annotation_col <- data.frame(
  Treatment = sample_info$Treatment[match(colnames(cor_mat), sample_info$SampleID)],
  row.names = colnames(cor_mat)
)
ann_colors <- list(Treatment = treat_colors)

hm_cor_expr <- quote(
  pheatmap(cor_mat,
           color = colorRampPalette(c("#2166AC", "white", "#B2182B"))(100),
           display_numbers = TRUE, number_format = "%.3f", fontsize_number = 9,
           annotation_col = annotation_col,
           annotation_colors = ann_colors,
           main = "")
)
save_pheatmap_pub(hm_cor_expr, "Fig05_correlation_heatmap", "QC", w = 9, h = 8)

write_desc(
  "Fig05_correlation_heatmap", "QC",
  title = "Pearson Correlation Heatmap Between Samples",
  placement = "SUPPLEMENTARY (Figure S5)",
  description = "Pairwise Pearson r between all 12 samples with clustering.",
  method = "Pearson correlation on log2-imputed matrix. Euclidean distance, complete linkage.",
  interpretation = paste(
    "Replicates should cluster together with r > 0.95.",
    "Cross-treatment correlations reveal overall proteome similarity."
  )
)

pca_data <- t(pg_log2)
pca_res  <- PCA(pca_data, graph = FALSE, ncp = 5)

pca_df <- data.frame(
  SampleID = rownames(pca_data),
  PC1 = pca_res$ind$coord[, 1],
  PC2 = pca_res$ind$coord[, 2],
  PC3 = pca_res$ind$coord[, 3]
) %>%
  left_join(sample_info %>% select(SampleID, Treatment), by = "SampleID")

var_explained <- round(pca_res$eig[, 2], 1)

compute_hulls <- function(df, x_col, y_col, group_col) {
  df %>%
    group_by(across(all_of(group_col))) %>%
    slice(chull(get(x_col), get(y_col))) %>%
    ungroup()
}

hull_12 <- compute_hulls(pca_df, "PC1", "PC2", "Treatment")

p4 <- ggplot(pca_df, aes(x = PC1, y = PC2, colour = Treatment)) +
  geom_polygon(data = hull_12, aes(fill = Treatment), alpha = 0.12,
               linetype = 2, linewidth = 0.4, colour = NA) +
  geom_polygon(data = hull_12, aes(colour = Treatment), fill = NA,
               linetype = 2, linewidth = 0.4) +
  geom_point(size = 4) +
  geom_text_repel(aes(label = SampleID), size = 3, max.overlaps = 20,
                  show.legend = FALSE) +
  scale_colour_manual(values = treat_colors) +
  scale_fill_manual(values = treat_colors) +
  labs(x = paste0("PC1 (", var_explained[1], "%)"),
       y = paste0("PC2 (", var_explained[2], "%)")) +
  theme_pub
save_pub(p4, "Fig06_PCA_PC1_PC2", "PCA")

write_desc(
  "Fig06_PCA_PC1_PC2", "PCA",
  title = "Principal Component Analysis — PC1 vs PC2",
  placement = "MAIN MANUSCRIPT (Figure 1)",
  description = paste(
    "PCA score plot of the first two principal components. Convex hulls",
    "delineate replicate spread within each treatment group.",
    "CS = positive control (yellow)."
  ),
  method = paste(
    "PCA on", nrow(pg_log2), "proteins x", ncol(pg_log2), "samples.",
    "FactoMineR::PCA. Convex hulls used instead of confidence ellipses",
    "(n=3 per group is insufficient for reliable ellipse estimation)."
  ),
  interpretation = paste(
    "Tight clustering = high reproducibility. Treatment separation along PCs",
    "indicates systematic proteomic differences. PC1 captures the largest",
    "biological effect."
  )
)

hull_13 <- compute_hulls(pca_df, "PC1", "PC3", "Treatment")

p5 <- ggplot(pca_df, aes(x = PC1, y = PC3, colour = Treatment)) +
  geom_polygon(data = hull_13, aes(fill = Treatment), alpha = 0.12,
               linetype = 2, linewidth = 0.4, colour = NA) +
  geom_polygon(data = hull_13, aes(colour = Treatment), fill = NA,
               linetype = 2, linewidth = 0.4) +
  geom_point(size = 4) +
  geom_text_repel(aes(label = SampleID), size = 3, max.overlaps = 20,
                  show.legend = FALSE) +
  scale_colour_manual(values = treat_colors) +
  scale_fill_manual(values = treat_colors) +
  labs(x = paste0("PC1 (", var_explained[1], "%)"),
       y = paste0("PC3 (", var_explained[3], "%)")) +
  theme_pub
save_pub(p5, "Fig07_PCA_PC1_PC3", "PCA")

write_desc(
  "Fig07_PCA_PC1_PC3", "PCA",
  title = "Principal Component Analysis — PC1 vs PC3",
  placement = "SUPPLEMENTARY (Figure S6)",
  description = "PCA score plot of PC1 vs PC3 revealing secondary structure.",
  method = "Same as Fig06, plotting PC3 on the y-axis.",
  interpretation = "PC3 may capture batch effects or secondary biological signals."
)

n_pcs <- min(10, nrow(pca_res$eig))
scree_df <- data.frame(
  PC = paste0("PC", 1:n_pcs),
  Variance = pca_res$eig[1:n_pcs, 2]
)
scree_df$PC <- factor(scree_df$PC, levels = scree_df$PC)

p6 <- ggplot(scree_df, aes(x = PC, y = Variance)) +
  geom_bar(stat = "identity", fill = "#457B9D", colour = "#1D3557", width = 0.7) +
  geom_text(aes(label = paste0(round(Variance, 1), "%")), vjust = -0.5, size = 3.5) +
  labs(x = "Principal Component", y = "Variance Explained (%)") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  theme_pub
save_pub(p6, "Fig08_PCA_screeplot", "PCA")

write_desc(
  "Fig08_PCA_screeplot", "PCA",
  title = "PCA Scree Plot — Variance Explained per Component",
  placement = "SUPPLEMENTARY (Figure S7)",
  description = "Percentage of total variance explained by each principal component.",
  method = "Eigenvalues from FactoMineR::PCA converted to percentage of total variance.",
  interpretation = paste(
    "A steep drop-off (elbow) after the first PCs means most variation is",
    "captured by a few components. PC1 >50% indicates one dominant effect."
  )
)

rv <- rowVars(as.matrix(pg_log2))
names(rv) <- rownames(pg_log2)
top50_idx  <- order(rv, decreasing = TRUE)[1:min(50, length(rv))]
mat_top50  <- as.matrix(pg_log2[top50_idx, ])

rownames(mat_top50) <- ifelse(
  rownames(mat_top50) %in% names(id_to_label),
  id_to_label[rownames(mat_top50)],
  rownames(mat_top50)
)

mat_top50_scaled <- t(scale(t(mat_top50)))

hm_top50_expr <- quote(
  pheatmap(mat_top50_scaled,
           color = colorRampPalette(rev(brewer.pal(11, "RdBu")))(100),
           annotation_col = annotation_col,
           annotation_colors = ann_colors,
           clustering_distance_rows = "euclidean",
           clustering_distance_cols = "euclidean",
           clustering_method = "ward.D2",
           show_rownames = TRUE, fontsize_row = 7,
           main = "")
)
save_pheatmap_pub(hm_top50_expr, "Fig09_top50_variable_heatmap", "Heatmaps",
                  w = 12, h = 14)

write_desc(
  "Fig09_top50_variable_heatmap", "Heatmaps",
  title = "Hierarchical Clustering of Top 50 Most Variable Proteins",
  placement = "MAIN MANUSCRIPT (Figure 2)",
  description = paste(
    "Z-score heatmap of the 50 proteins with the highest variance.",
    "Protein names (gene symbols) are shown on rows.",
    "Samples are annotated by treatment (CS = positive control)."
  ),
  method = paste(
    "Row variance computed on log2-imputed matrix. Top 50 selected, Z-score",
    "normalised per row. Euclidean distance, Ward.D2 linkage."
  ),
  interpretation = paste(
    "Red = above-average abundance, Blue = below-average. Samples should",
    "cluster by treatment. Co-clustered proteins may share pathways."
  )
)

cat("\n--- Differential expression analysis (limma) ---\n")

sample_order <- colnames(pg_log2)
group <- factor(
  sample_info$Treatment[match(sample_order, sample_info$SampleID)],
  levels = treatments
)
stopifnot(!any(is.na(group)))

design <- model.matrix(~ 0 + group)
colnames(design) <- levels(group)

fit <- lmFit(as.matrix(pg_log2), design)

pairs <- combn(treatments, 2)
contrast_names    <- apply(pairs, 2, function(x) paste0(x[1], "_vs_", x[2]))
contrast_formulas <- apply(pairs, 2, function(x) paste0(x[1], " - ", x[2]))

contrast_matrix <- makeContrasts(contrasts = contrast_formulas, levels = design)
colnames(contrast_matrix) <- contrast_names

fit2 <- contrasts.fit(fit, contrast_matrix)
fit2 <- eBayes(fit2)

cs_contrasts <- grep("_vs_CS|CS_vs_", contrast_names, value = TRUE)
non_cs_contrasts <- setdiff(contrast_names, cs_contrasts)

de_results <- list()
for (i in seq_along(contrast_names)) {
  tt <- topTable(fit2, coef = i, number = Inf, sort.by = "none")
  tt$Protein.ID  <- rownames(tt)
  tt$Contrast    <- contrast_names[i]
  tt$Significant <- ifelse(tt$adj.P.Val < 0.05 & abs(tt$logFC) > 1,
                           ifelse(tt$logFC > 1, "Up", "Down"), "NS")

  tt$Protein.Name <- pg_lookup$Protein.Name[match(tt$Protein.ID, pg_lookup$Protein.ID)]
  tt$Gene.Name    <- pg_lookup$Gene.Name[match(tt$Protein.ID, pg_lookup$Protein.ID)]
  tt$Short.Label  <- pg_lookup$Short.Label[match(tt$Protein.ID, pg_lookup$Protein.ID)]

  de_results[[contrast_names[i]]] <- tt

  write.csv(tt, file.path(output_dir, "Tables",
                          paste0("DE_", contrast_names[i], ".csv")),
            row.names = FALSE)

  n_up   <- sum(tt$Significant == "Up")
  n_down <- sum(tt$Significant == "Down")
  write_table_desc(
    paste0("DE_", contrast_names[i], ".csv"), "Tables",
    title = paste("Differential Expression Results:", contrast_names[i]),
    description = paste(
      "Full limma results for", contrast_formulas[i], ".",
      nrow(tt), "proteins;", n_up, "up-regulated,", n_down, "down-regulated.",
      "Includes protein name, gene name, and short label."
    ),
    method = paste(
      "limma::lmFit + eBayes. BH-adjusted p-values.",
      "Significance: |log2FC| > 1 and adj.P.Val < 0.05."
    ),
    interpretation = paste(
      "logFC > 0 = higher in", strsplit(contrast_names[i], "_vs_")[[1]][1],
      "vs", strsplit(contrast_names[i], "_vs_")[[1]][2], ".",
      "Use adj.P.Val for significance, not raw P.Value."
    )
  )
}

de_all <- bind_rows(de_results)
write.csv(de_all, file.path(output_dir, "Tables", "DE_all_contrasts.csv"),
          row.names = FALSE)

de_summary <- de_all %>%
  filter(Significant != "NS") %>%
  group_by(Contrast, Significant) %>%
  summarise(n = n(), .groups = "drop")
cat("\nDE summary:\n")
print(as.data.frame(de_summary))
write.csv(de_summary, file.path(output_dir, "Tables", "DE_summary_counts.csv"),
          row.names = FALSE)

write_table_desc(
  "DE_summary_counts.csv", "Tables",
  title = "Summary of DEPs Across All Contrasts",
  description = paste("Count of up/down-regulated proteins per contrast.",
                      "CS is the positive control."),
  method = "Aggregated from limma DE results. |log2FC| > 1, adj.P < 0.05.",
  interpretation = paste(
    "Contrasts vs CS (positive control) are the primary comparisons.",
    "Many DEPs vs CS indicate large treatment effects relative to control."
  )
)

for (cname in contrast_names) {
  tt <- de_results[[cname]]
  tt <- tt %>% arrange(adj.P.Val)

  tt$Label <- ""
  sig_rows <- which(tt$Significant != "NS")
  top_n <- min(10, length(sig_rows))
  if (top_n > 0) {
    tt$Label[sig_rows[1:top_n]] <- tt$Short.Label[sig_rows[1:top_n]]
  }

  pv <- ggplot(tt, aes(x = logFC, y = -log10(adj.P.Val), colour = Significant)) +
    geom_point(alpha = 0.6, size = 1.8) +
    geom_text_repel(aes(label = Label), size = 2.5, max.overlaps = 20,
                    colour = "black", fontface = "italic",
                    segment.colour = "grey50", segment.size = 0.3) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", colour = "grey40") +
    geom_vline(xintercept = c(-1, 1), linetype = "dashed", colour = "grey40") +
    scale_colour_manual(values = c(Up = "#E63946", Down = "#457B9D", NS = "grey75")) +
    labs(x = expression(Log[2]~Fold~Change),
         y = expression(-Log[10]~Adjusted~italic(P)-value)) +
    theme_pub

  save_pub(pv, paste0("Fig10_volcano_", cname), "Volcano", w = 9, h = 7)

  n_up   <- sum(tt$Significant == "Up")
  n_down <- sum(tt$Significant == "Down")

  is_cs <- grepl("_vs_CS|CS_vs_", cname)
  placement <- if (is_cs) {
    "MAIN MANUSCRIPT (Figure 4 panel)"
  } else {
    "SUPPLEMENTARY"
  }

  write_desc(
    paste0("Fig10_volcano_", cname), "Volcano",
    title = paste("Volcano Plot —", cname),
    placement = placement,
    description = paste(
      "Volcano plot:", n_up, "up (red),", n_down, "down (blue).",
      "Top 10 DEPs labelled with gene/protein name (not accession).",
      if (is_cs) "This is a primary comparison vs CS positive control." else ""
    ),
    method = paste(
      "Log2FC (x) vs -log10(adj.P) (y) from limma.",
      "Dashed lines: |log2FC| = 1 and adj.P = 0.05."
    ),
    interpretation = paste(
      "Upper corners contain significant DEPs. Red = up in",
      strsplit(cname, "_vs_")[[1]][1], "; Blue = up in",
      strsplit(cname, "_vs_")[[1]][2], ".",
      "Labelled proteins are top candidates for validation."
    )
  )
}

if (nrow(de_summary) > 0) {

  de_summary$Contrast <- factor(de_summary$Contrast,
                                 levels = c(cs_contrasts, non_cs_contrasts))

  p_de_bar <- ggplot(de_summary, aes(x = Contrast, y = n, fill = Significant)) +
    geom_bar(stat = "identity", position = "dodge") +
    geom_text(aes(label = n), position = position_dodge(width = 0.9),
              vjust = -0.3, size = 3.5) +
    scale_fill_manual(values = c(Up = "#E63946", Down = "#457B9D")) +
    labs(x = NULL, y = "Number of Proteins") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    theme_pub +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  save_pub(p_de_bar, "Fig11_DE_barplot_summary", "DE")

  write_desc(
    "Fig11_DE_barplot_summary", "DE",
    title = "Summary of Differentially Expressed Proteins per Contrast",
    placement = "MAIN MANUSCRIPT (Figure 3)",
    description = paste(
      "Grouped bar chart of up (red) and down (blue) DEPs per contrast.",
      "|log2FC| > 1, adj.P < 0.05. Contrasts vs CS (positive control) listed first."
    ),
    method = "Counts from limma DE analysis across all 6 pairwise contrasts.",
    interpretation = paste(
      "Tall bars = large proteomic differences between those treatments.",
      "Compare vs-CS contrasts to assess treatment effects against control."
    )
  )
}

sig_proteins <- unique(de_all$Protein.ID[de_all$Significant != "NS"])
cat("\nTotal unique significant DEPs:", length(sig_proteins), "\n")

if (length(sig_proteins) > 1) {
  mat_sig <- as.matrix(pg_log2[sig_proteins, ])

  new_names <- ifelse(
    rownames(mat_sig) %in% names(id_to_label),
    id_to_label[rownames(mat_sig)],
    rownames(mat_sig)
  )
  rownames(mat_sig) <- new_names

  mat_sig_scaled <- t(scale(t(mat_sig)))
  mat_sig_scaled[mat_sig_scaled >  3] <-  3
  mat_sig_scaled[mat_sig_scaled < -3] <- -3

  show_rows <- length(sig_proteins) <= 80
  hm_height <- max(8, length(sig_proteins) * 0.14)

  hm_dep_expr <- quote(
    pheatmap(mat_sig_scaled,
             color = colorRampPalette(rev(brewer.pal(11, "RdBu")))(100),
             annotation_col = annotation_col,
             annotation_colors = ann_colors,
             clustering_distance_rows = "euclidean",
             clustering_method = "ward.D2",
             show_rownames = show_rows,
             fontsize_row = 5,
             main = "")
  )
  save_pheatmap_pub(hm_dep_expr, "Fig12_DEP_heatmap_all", "Heatmaps",
                    w = 12, h = hm_height)

  write_desc(
    "Fig12_DEP_heatmap_all", "Heatmaps",
    title = "Heatmap of All Differentially Expressed Proteins",
    placement = "MAIN MANUSCRIPT (Figure 5) or SUPPLEMENTARY if >80 proteins",
    description = paste(
      "Z-score heatmap of all", length(sig_proteins), "unique DEPs.",
      "Protein names (gene symbols) on rows. Z-scores capped at +/-3."
    ),
    method = paste(
      "Union of DEPs from all 6 contrasts. Row-wise Z-score normalisation.",
      "Euclidean distance, Ward.D2 linkage."
    ),
    interpretation = paste(
      "Red = high relative abundance, Blue = low. Column dendrogram shows",
      "sample clustering. Look for treatment-specific expression blocks."
    )
  )
}

dep_lists <- lapply(de_results, function(tt) tt$Protein.ID[tt$Significant != "NS"])
dep_lists_nonempty <- dep_lists[sapply(dep_lists, length) > 0]

if (length(dep_lists_nonempty) > 1) {

  tiff(file.path(output_dir, "Venn", "Fig13_UpSet_DEPs.tiff"),
       width = 12, height = 7, units = "in", res = 600, compression = "lzw")
  suppressWarnings(
    print(upset(fromList(dep_lists_nonempty), order.by = "freq",
                sets.bar.color = "#457B9D",
                main.bar.color = "#264653",
                text.scale = 1.3,
                mainbar.y.label = "Intersection Size",
                sets.x.label = "DEPs per Contrast"))
  )
  dev.off()

  pdf(file.path(output_dir, "Venn", "Fig13_UpSet_DEPs.pdf"),
      width = 12, height = 7)
  suppressWarnings(
    print(upset(fromList(dep_lists_nonempty), order.by = "freq",
                sets.bar.color = "#457B9D",
                main.bar.color = "#264653",
                text.scale = 1.3,
                mainbar.y.label = "Intersection Size",
                sets.x.label = "DEPs per Contrast"))
  )
  dev.off()

  write_desc(
    "Fig13_UpSet_DEPs", "Venn",
    title = "UpSet Plot — Overlap of DEPs Across Pairwise Contrasts",
    placement = "MAIN MANUSCRIPT (Figure 6)",
    description = paste(
      "UpSet plot of DEP intersections. Horizontal bars (left) = total DEPs",
      "per contrast. Vertical bars = intersection sizes."
    ),
    method = paste(
      "Significant DEPs (|log2FC|>1, adj.P<0.05) from each contrast.",
      "UpSetR package, ordered by frequency."
    ),
    interpretation = paste(
      "Shared intersections reveal core response proteins.",
      "Unique bars indicate contrast-specific DEPs."
    )
  )
}

venn_names <- c(cs_contrasts, non_cs_contrasts)
venn_names <- venn_names[venn_names %in% names(dep_lists_nonempty)]
venn_sets  <- dep_lists_nonempty[venn_names[1:min(4, length(venn_names))]]

if (length(venn_sets) >= 2) {
  futile.logger::flog.threshold(futile.logger::ERROR, name = "VennDiagramLogger")
  venn.diagram(
    x = venn_sets,
    filename = file.path(output_dir, "Venn", "Fig14_Venn_DEPs.tiff"),
    imagetype = "tiff",
    resolution = 600,
    fill = brewer.pal(min(length(venn_sets), 4), "Set2")[1:length(venn_sets)],
    alpha = 0.5,
    cex = 1.5, cat.cex = 1.2,
    main = ""
  )

  write_desc(
    "Fig14_Venn_DEPs", "Venn",
    title = "Venn Diagram — DEP Overlap (vs CS Control Prioritised)",
    placement = "SUPPLEMENTARY (Figure S8)",
    description = paste(
      "Venn diagram of DEP overlap. Contrasts vs CS (positive control) are",
      "prioritised in the selection of up to 4 sets."
    ),
    method = "Same DEP sets as UpSet. VennDiagram R package.",
    interpretation = "Overlapping regions = shared DEPs. See UpSet for full detail."
  )
}

id_per_treat <- sapply(treatments, function(tr) {
  cols <- sample_info$SampleID[sample_info$Treatment == tr]
  cols <- cols[cols %in% colnames(pg_filt)]
  if (length(cols) == 0) return(0)
  sum(rowSums(!is.na(pg_filt[, cols, drop = FALSE])) > 0)
})

p_ids <- data.frame(Treatment = factor(names(id_per_treat), levels = treatments),
                    Count = as.numeric(id_per_treat)) %>%
  ggplot(aes(x = Treatment, y = Count, fill = Treatment)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = Count), vjust = -0.5, size = 5) +
  scale_fill_manual(values = treat_colors) +
  labs(x = NULL, y = "Protein Groups") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  theme_pub +
  theme(legend.position = "none")
save_pub(p_ids, "Fig15_protein_IDs_per_treatment", "QC")

write_desc(
  "Fig15_protein_IDs_per_treatment", "QC",
  title = "Protein Groups Identified per Treatment",
  placement = "SUPPLEMENTARY (Figure S9)",
  description = "Total protein groups detected in at least 1 replicate per treatment.",
  method = "Count of proteins with >=1 non-NA PG.MaxLFQ value per treatment.",
  interpretation = "Similar counts indicate comparable proteome coverage across treatments."
)

cv_df <- data.frame(Treatment = character(), CV = numeric(), stringsAsFactors = FALSE)

for (tr in treatments) {
  cols <- sample_info$SampleID[sample_info$Treatment == tr]
  cols <- cols[cols %in% colnames(pg_filt)]
  if (length(cols) < 2) next
  sub      <- pg_filt[, cols, drop = FALSE]
  row_mean <- rowMeans(sub, na.rm = TRUE)
  row_sd   <- apply(sub, 1, sd, na.rm = TRUE)
  cv_vals  <- (row_sd / row_mean) * 100
  cv_vals  <- cv_vals[!is.na(cv_vals) & is.finite(cv_vals)]
  cv_df    <- rbind(cv_df, data.frame(Treatment = tr, CV = cv_vals,
                                       stringsAsFactors = FALSE))
}

p_cv <- ggplot(cv_df, aes(x = CV, fill = Treatment)) +
  geom_density(alpha = 0.5) +
  geom_vline(xintercept = 20, linetype = "dashed", colour = "red") +
  scale_fill_manual(values = treat_colors) +
  coord_cartesian(xlim = c(0, 100)) +
  labs(x = "CV (%)", y = "Density") +
  annotate("text", x = 22, y = Inf, label = "20% CV", vjust = 2,
           hjust = 0, colour = "red", size = 3.5) +
  theme_pub
save_pub(p_cv, "Fig16_CV_distribution", "QC")

write_desc(
  "Fig16_CV_distribution", "QC",
  title = "Coefficient of Variation Distribution per Treatment",
  placement = "SUPPLEMENTARY (Figure S10)",
  description = "Density plot of CV (%) per treatment. Red dashed line at 20%.",
  method = "CV = (SD/Mean) x 100 on raw PG.MaxLFQ across 3 replicates per treatment.",
  interpretation = "Median CV < 20% = good reproducibility. CS may show higher CV as positive control."
)

cv_median <- cv_df %>%
  group_by(Treatment) %>%
  summarise(Median_CV = round(median(CV, na.rm = TRUE), 1),
            Mean_CV   = round(mean(CV, na.rm = TRUE), 1),
            Pct_below_20 = round(100 * sum(CV < 20, na.rm = TRUE) / n(), 1))
cat("\nCV summary:\n")
print(as.data.frame(cv_median))
write.csv(cv_median, file.path(output_dir, "Tables", "CV_summary.csv"),
          row.names = FALSE)

write_table_desc(
  "CV_summary.csv", "Tables",
  title = "Coefficient of Variation Summary",
  description = "Median CV, mean CV, and % below 20% per treatment.",
  method = "CV on raw intensities, 3 replicates per group.",
  interpretation = "Pct_below_20 > 70% is acceptable for DIA proteomics."
)

treat_means <- sapply(treatments, function(tr) {
  cols <- sample_info$SampleID[sample_info$Treatment == tr]
  cols <- cols[cols %in% colnames(pg_log2)]
  rowMeans(pg_log2[, cols, drop = FALSE], na.rm = TRUE)
})
colnames(treat_means) <- treatments

rv_means <- apply(treat_means, 1, var)
n_top <- min(100, length(rv_means))
top100 <- names(sort(rv_means, decreasing = TRUE))[1:n_top]
mat_means_top <- treat_means[top100, ]

rownames(mat_means_top) <- ifelse(
  rownames(mat_means_top) %in% names(id_to_label),
  id_to_label[rownames(mat_means_top)],
  rownames(mat_means_top)
)

mat_means_top <- t(scale(t(mat_means_top)))

hm_treat_expr <- quote(
  pheatmap(mat_means_top,
           color = colorRampPalette(rev(brewer.pal(11, "RdBu")))(100),
           clustering_distance_rows = "euclidean",
           clustering_method = "ward.D2",
           show_rownames = TRUE, fontsize_row = 6,
           main = "")
)
save_pheatmap_pub(hm_treat_expr, "Fig17_treatment_mean_heatmap", "Heatmaps",
                  w = 8, h = max(10, n_top * 0.14))

write_desc(
  "Fig17_treatment_mean_heatmap", "Heatmaps",
  title = paste("Treatment Mean Intensities — Top", n_top, "Variable Proteins"),
  placement = "SUPPLEMENTARY (Figure S11)",
  description = paste(
    "Z-score heatmap of treatment-level means for the top", n_top,
    "variable proteins. Readable protein names on rows."
  ),
  method = paste("Mean log2 intensities per treatment (n=3). Top", n_top,
                 "by inter-treatment variance. Row Z-scored."),
  interpretation = "Simplified view removing replicate noise. Co-clustered proteins may share pathways."
)

precursors_per_pg <- report_filt %>%
  group_by(Protein.Group) %>%
  summarise(n_precursors = n_distinct(Precursor.Id), .groups = "drop")

p_prec <- ggplot(precursors_per_pg, aes(x = n_precursors)) +
  geom_histogram(binwidth = 1, fill = "#264653", colour = "white") +
  labs(x = "Number of Precursors", y = "Protein Groups") +
  coord_cartesian(xlim = c(0, quantile(precursors_per_pg$n_precursors, 0.95) + 1)) +
  theme_pub
save_pub(p_prec, "Fig18_precursors_per_PG", "QC")

write_desc(
  "Fig18_precursors_per_PG", "QC",
  title = "Distribution of Precursors per Protein Group",
  placement = "SUPPLEMENTARY (Figure S12)",
  description = "Histogram of unique precursors per protein group.",
  method = "Counted from quality-filtered report. X-axis capped at 95th percentile.",
  interpretation = "More precursors = more robust quantification. Single-precursor proteins are less reliable."
)

ids_per_run <- report_filt %>%
  group_by(Run) %>%
  summarise(
    Proteins   = n_distinct(Protein.Group),
    Peptides   = n_distinct(Stripped.Sequence),
    Precursors = n_distinct(Precursor.Id),
    .groups = "drop"
  ) %>%
  left_join(sample_info %>% select(Run, SampleID, Treatment), by = "Run") %>%
  mutate(SampleID = factor(SampleID, levels = sample_info$SampleID))

p_id_run <- ids_per_run %>%
  pivot_longer(cols = c(Proteins, Peptides, Precursors),
               names_to = "Level", values_to = "Count") %>%
  mutate(Level = factor(Level, levels = c("Proteins", "Peptides", "Precursors"))) %>%
  ggplot(aes(x = SampleID, y = Count, fill = Treatment)) +
  geom_bar(stat = "identity") +
  facet_wrap(~ Level, scales = "free_y") +
  scale_fill_manual(values = treat_colors) +
  labs(x = NULL, y = "Count") +
  theme_pub +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
save_pub(p_id_run, "Fig19_IDs_per_sample", "QC", w = 12, h = 6)

write_desc(
  "Fig19_IDs_per_sample", "QC",
  title = "Protein, Peptide and Precursor Identifications per Sample",
  placement = "SUPPLEMENTARY (Figure S13)",
  description = "Faceted bars of unique proteins, peptides, precursors per run.",
  method = "Distinct counts from quality-filtered report per Run.",
  interpretation = "Consistent counts across replicates = stable MS performance."
)

write.csv(ids_per_run, file.path(output_dir, "Tables", "IDs_per_sample.csv"),
          row.names = FALSE)

write_table_desc(
  "IDs_per_sample.csv", "Tables",
  title = "Identification Counts per Sample",
  description = "Protein, peptide, precursor counts per run.",
  method = "From filtered report (Global.Q.Value <= 0.01, Global.PG.Q.Value <= 0.01).",
  interpretation = "Outlier runs with low counts may need investigation."
)

if (length(sig_proteins) > 0) {
  dep_full_table <- de_all %>%
    filter(Significant != "NS") %>%
    select(Protein.ID, Protein.Name, Gene.Name, Short.Label,
           Contrast, logFC, adj.P.Val, Significant, AveExpr) %>%
    arrange(Contrast, adj.P.Val)

  write.csv(dep_full_table,
            file.path(output_dir, "Tables", "Supplementary_DEP_full_list.csv"),
            row.names = FALSE)

  write_table_desc(
    "Supplementary_DEP_full_list.csv", "Tables",
    title = "Supplementary Table — Complete List of Differentially Expressed Proteins",
    description = paste(
      "All significant DEPs across all contrasts with full protein annotations.",
      "Includes UniProt accession, full protein name, gene name, short label,",
      "log2FC, adjusted p-value, and regulation direction.",
      nrow(dep_full_table), "entries from", length(sig_proteins), "unique proteins."
    ),
    method = paste(
      "Filtered from limma DE results: |log2FC| > 1 and adj.P.Val < 0.05.",
      "Protein names mapped from the chickpea UniProt FASTA."
    ),
    interpretation = paste(
      "This is the primary supplementary data table for the manuscript.",
      "Sorted by contrast and significance. A protein may appear in multiple",
      "contrasts if it is significant in more than one comparison."
    )
  )

  dep_unique <- de_all %>%
    filter(Significant != "NS") %>%
    group_by(Protein.ID) %>%
    summarise(
      Protein.Name   = first(Protein.Name),
      Gene.Name      = first(Gene.Name),
      Short.Label    = first(Short.Label),
      N.Contrasts    = n_distinct(Contrast),
      Contrasts      = paste(unique(Contrast), collapse = "; "),
      Directions     = paste(unique(Significant), collapse = "; "),
      Min.adj.P      = min(adj.P.Val),
      Max.abs.logFC  = max(abs(logFC)),
      .groups = "drop"
    ) %>%
    arrange(Min.adj.P)

  write.csv(dep_unique,
            file.path(output_dir, "Tables", "Supplementary_DEP_unique_summary.csv"),
            row.names = FALSE)

  write_table_desc(
    "Supplementary_DEP_unique_summary.csv", "Tables",
    title = "Supplementary Table — Unique DEP Summary with Annotations",
    description = paste(
      "One row per unique DEP (", nrow(dep_unique), " proteins).",
      "Shows in how many contrasts each protein is significant,",
      "which contrasts, direction(s), best p-value, and largest fold change."
    ),
    method = "Aggregated from all pairwise DE results.",
    interpretation = paste(
      "Proteins significant in multiple contrasts (high N.Contrasts) are",
      "robust candidates. Max.abs.logFC indicates the strongest observed effect."
    )
  )
}

guide_lines <- c(
  "================================================================",
  "  MANUSCRIPT FIGURE PLACEMENT GUIDE",
  "  DIA-NN Proteomics — Chickpea | CS = Positive Control",
  "================================================================",
  "",
  "MAIN MANUSCRIPT FIGURES:",
  "------------------------",
  "Fig06  PCA PC1 vs PC2                        -> Figure 1",
  "Fig09  Top 50 variable proteins heatmap       -> Figure 2",
  "Fig11  DE barplot summary (all contrasts)     -> Figure 3",
  "Fig10  Volcano plots (RH/GH/CD vs CS)        -> Figure 4 (multi-panel)",
  "Fig12  All DEP heatmap                        -> Figure 5",
  "Fig13  UpSet plot DEP overlap                 -> Figure 6",
  "",
  "SUPPLEMENTARY FIGURES:",
  "-----------------------",
  "Fig01  Missing per sample                     -> Figure S1",
  "Fig02  Missingness heatmap                    -> Figure S2",
  "Fig03  Intensity boxplot                      -> Figure S3",
  "Fig04  Intensity density                      -> Figure S4",
  "Fig05  Correlation heatmap                    -> Figure S5",
  "Fig07  PCA PC1 vs PC3                         -> Figure S6",
  "Fig08  PCA scree plot                         -> Figure S7",
  "Fig14  Venn diagram                           -> Figure S8",
  "Fig15  Protein IDs per treatment              -> Figure S9",
  "Fig16  CV distribution                        -> Figure S10",
  "Fig17  Treatment mean heatmap                 -> Figure S11",
  "Fig18  Precursors per PG                      -> Figure S12",
  "Fig19  IDs per sample                         -> Figure S13",
  "Fig10  Volcano (non-CS contrasts)             -> Figure S14 (multi-panel)",
  "",
  "MAIN TABLES:",
  "-------------",
  "sample_annotation.csv                         -> Table 1",
  "DE_summary_counts.csv                         -> Table 2",
  "",
  "SUPPLEMENTARY TABLES:",
  "----------------------",
  "protein_ID_to_name_lookup.csv                 -> Table S1",
  "Supplementary_DEP_full_list.csv               -> Table S2",
  "Supplementary_DEP_unique_summary.csv          -> Table S3",
  "protein_log2_imputed.csv                      -> Table S4",
  "CV_summary.csv                                -> Table S5",
  "IDs_per_sample.csv                            -> Table S6",
  "fasta_protein_annotations.csv                 -> Table S7",
  "Individual DE_*.csv tables                    -> Tables S8-S13",
  "",
  "NOTES:",
  "- All figures: .tiff (600 DPI, LZW) + .pdf",
  "- No titles inside figures — use _DESCRIPTION.txt for legends",
  "- CS is the positive control; vs-CS contrasts are primary",
  "- Protein names (gene symbols) used in plots, not accession IDs",
  "- Full accession-to-name mapping in Table S1",
  "================================================================"
)

writeLines(guide_lines, file.path(output_dir, "MANUSCRIPT_PLACEMENT_GUIDE.txt"))

sink(file.path(output_dir, "Tables", "analysis_summary.txt"))
cat("====================================================\n")
cat("  DIA-NN PROTEOMICS ANALYSIS SUMMARY — CHICKPEA\n")
cat("  Pipeline v4.0 FINAL\n")
cat("====================================================\n\n")
cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("R version:", R.version.string, "\n")
cat("Quantity column:", quant_col, "\n")
cat("FASTA:", basename(fasta_file), "\n")
cat("Positive control: CS\n\n")

cat("--- Data dimensions ---\n")
cat("Raw report rows:", nrow(report), "\n")
cat("Filtered precursor rows:", nrow(report_filt), "\n")
cat("Protein groups (before filter):", nrow(pg_wide), "\n")
cat("Protein groups (after filter):", nrow(pg_filt), "\n")
cat("Imputed values:", n_imputed, "/", total_cells,
    paste0("(", round(100 * n_imputed / total_cells, 1), "%)"), "\n")
cat("FASTA entries:", nrow(fasta_anno), "\n")
cat("Mapped to FASTA:", n_mapped, "/", nrow(pg_lookup), "\n\n")

cat("--- Samples ---\n")
print(as.data.frame(sample_info))
cat("\n")

cat("--- IDs per run ---\n")
print(as.data.frame(ids_per_run))
cat("\n")

cat("--- CV summary ---\n")
print(as.data.frame(cv_median))
cat("\n")

cat("--- DE summary (|log2FC|>1, adj.P<0.05) ---\n")
print(as.data.frame(de_summary))
cat("\n")

cat("Total unique DEPs:", length(sig_proteins), "/", nrow(pg_log2),
    paste0("(", round(100 * length(sig_proteins) / nrow(pg_log2), 1), "%)"), "\n")
cat("====================================================\n")
sink()

write_table_desc(
  "analysis_summary.txt", "Tables",
  title = "Global Analysis Summary",
  description = "Complete pipeline summary: dimensions, filtering, imputation, DE.",
  method = "Auto-generated at pipeline completion.",
  interpretation = "Quick reference for all key metrics."
)

sink(file.path(output_dir, "Tables", "session_info.txt"))
cat("R Session Info for Reproducibility\n")
cat("===================================\n\n")
print(sessionInfo())
sink()

cat("\n\n")
cat("================================================================\n")
cat("  ANALYSIS COMPLETE (v4.0 FINAL)\n")
cat("================================================================\n")
cat("All outputs saved to:", output_dir, "\n\n")
cat("  /QC/         Quality control (.tiff 600dpi + .pdf + .txt)\n")
cat("  /PCA/        PCA figures\n")
cat("  /Heatmaps/   Heatmaps (protein names on rows)\n")
cat("  /Volcano/    Volcano plots (gene names as labels)\n")
cat("  /Venn/       UpSet & Venn diagrams\n")
cat("  /DE/         DE summary barplot\n")
cat("  /Tables/     CSVs + companion descriptions\n")
cat("  MANUSCRIPT_PLACEMENT_GUIDE.txt\n")
cat("================================================================\n")
