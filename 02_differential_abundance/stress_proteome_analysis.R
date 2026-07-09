# Stress-proteome differential abundance - chickpea (Cicer arietinum)
# Restricted to the 55 GO-classified abiotic-stress proteins. Treatments:
# RH, GH, CD, CS (CS = positive control), 3 replicates each. DE by limma;
# by Luis 
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
cat("All packages loaded.\n")

input_dir   <- "D:/P2/DIA"
stress_dir  <- "D:/P2/Stress"
output_dir  <- "D:/P2/Stress/Results"
fasta_file  <- file.path(input_dir, "uniprotkb_taxonomy_id_3827_2026_02_09.fasta")
stress_file <- file.path(stress_dir, "stress_proteins_GO_classified.csv")

dirs <- c("QC", "DE", "Heatmaps", "Volcano", "Venn", "Tables", "PCA")
for (d in dirs) {
  dir.create(file.path(output_dir, d), recursive = TRUE, showWarnings = FALSE)
}

treatments   <- c("RH", "GH", "CD", "CS")
treat_colors <- c(RH = "#E63946", GH = "#2A9D8F", CD = "#457B9D", CS = "#E9C46A")

cat_colors <- c(
  "Chaperones and HSPs"           = "#E65100",
  "Antioxidant defence"           = "#C62828",
  "Dehydrins and LEA proteins"    = "#1565C0",
  "Defence and pathogenesis"      = "#2E7D32",
  "Calcium signalling"            = "#6A1B9A",
  "Lipid stress signalling"       = "#F9A825",
  "Cold stress"                   = "#00838F",
  "Redox homeostasis"             = "#AD1457",
  "Protein folding"               = "#FF6F00",
  "Other stress"                  = "#757575"
)

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
    "", paste0("TITLE: ", title),
    "", "DESCRIPTION:", description,
    "", "METHOD:", method,
    "", "HOW TO INTERPRET:", interpretation,
    "", paste0("Generated: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
  )
  writeLines(lines, txt_path)
}

cat("Configuration ready.\n")

cat("\n--- Loading GO-classified stress proteins ---\n")
stress_df <- read.csv(stress_file, stringsAsFactors = FALSE)
stress_ids <- unique(stress_df$UniProt_ID)
cat("Stress proteins loaded:", length(stress_ids), "\n")

assign_functional_category <- function(description, sp_description) {
  desc <- tolower(paste(description, sp_description))
  if (grepl("heat shock|hsp|hsc\\d|chaperone", desc)) return("Chaperones and HSPs")
  if (grepl("glutathione|gst|dhar|dehydroascorbate", desc)) return("Antioxidant defence")
  if (grepl("superoxide|sod", desc)) return("Antioxidant defence")
  if (grepl("catalase", desc)) return("Antioxidant defence")
  if (grepl("peroxidase|peroxiredoxin|prx", desc)) return("Antioxidant defence")
  if (grepl("thioredoxin|trx", desc)) return("Antioxidant defence")
  if (grepl("ascorbate|monodehydroascorbate|mdhar", desc)) return("Antioxidant defence")
  if (grepl("dehydrin|dhn", desc)) return("Dehydrins and LEA proteins")
  if (grepl("lea |lea\\d|late embryogenesis|embryonic protein", desc)) return("Dehydrins and LEA proteins")
  if (grepl("lipoxygenase|lox", desc)) return("Lipid stress signalling")
  if (grepl("calmodulin|cam |calcium", desc)) return("Calcium signalling")
  if (grepl("annexin", desc)) return("Calcium signalling")
  if (grepl("pathogenesis|pr-\\d|pr\\d|chitinase|germin", desc)) return("Defence and pathogenesis")
  if (grepl("cold shock|cold-", desc)) return("Cold stress")
  if (grepl("peptidyl-prolyl|cyclophilin|fkbp|protein disulfide", desc)) return("Protein folding")
  if (grepl("ferredoxin|ferritin|oxidoreduct|redox|nadph", desc)) return("Redox homeostasis")
  if (grepl("isocitrate dehydrogenase", desc)) return("Redox homeostasis")
  return("Other stress")
}

stress_df$Functional_Category <- mapply(
  assign_functional_category,
  stress_df$Description,
  stress_df$SwissProt_Description
)

stress_df$Short_Label <- ifelse(
  stress_df$Gene_Name != "" & !is.na(stress_df$Gene_Name),
  stress_df$Gene_Name,
  ifelse(nchar(stress_df$Description) > 25,
         paste0(substr(stress_df$Description, 1, 22), "..."),
         stress_df$Description)
)

stress_df$Short_Label[stress_df$Short_Label == "" | is.na(stress_df$Short_Label)] <-
  stress_df$UniProt_ID[stress_df$Short_Label == "" | is.na(stress_df$Short_Label)]

dup_labels <- stress_df$Short_Label[duplicated(stress_df$Short_Label)]
if (length(dup_labels) > 0) {
  dup_idx <- stress_df$Short_Label %in% dup_labels
  stress_df$Short_Label[dup_idx] <- paste0(
    stress_df$Short_Label[dup_idx], " (", stress_df$UniProt_ID[dup_idx], ")"
  )
}

cat("Functional categories:\n")
print(table(stress_df$Functional_Category))

write.csv(stress_df, file.path(output_dir, "Tables", "stress_proteins_annotated.csv"),
          row.names = FALSE)

stress_id_to_label <- setNames(stress_df$Short_Label, stress_df$UniProt_ID)
stress_id_to_cat   <- setNames(stress_df$Functional_Category, stress_df$UniProt_ID)

cat("\n--- Parsing FASTA ---\n")
fasta_lines <- readLines(fasta_file, warn = FALSE)
header_idx  <- which(startsWith(fasta_lines, ">"))
headers     <- fasta_lines[header_idx]

parse_fasta_header <- function(h) {
  h <- sub("^>", "", h)
  accession    <- str_extract(h, "(?<=\\|)[A-Za-z0-9_]+(?=\\|)")
  protein_name <- str_extract(h, "(?<=\\s).+?(?=\\sOS=)")
  gene_name    <- str_extract(h, "(?<=GN=)[^\\s]+")
  data.frame(
    Accession    = ifelse(is.na(accession), "", accession),
    Protein.Name = ifelse(is.na(protein_name), "", protein_name),
    Gene.Name    = ifelse(is.na(gene_name), "", gene_name),
    stringsAsFactors = FALSE
  )
}
fasta_anno <- do.call(rbind, lapply(headers, parse_fasta_header))
cat("FASTA entries:", nrow(fasta_anno), "\n")

cat("\n--- Loading DIA-NN report ---\n")
report <- as.data.frame(arrow::read_parquet(file.path(input_dir, "report.parquet")))
cat("Report:", nrow(report), "rows x", ncol(report), "cols\n")

report$Protein.Group <- sapply(
  strsplit(report$Protein.Group, ";"), function(x) trimws(x[1])
)

runs <- unique(report$Run)
sample_info <- data.frame(Run = runs, stringsAsFactors = FALSE) %>%
  mutate(
    BaseName  = basename(Run),
    BaseName  = str_remove(BaseName, "\\.mzML$|\\.raw$|\\.d$|\\.wiff$"),
    Treatment = str_extract(BaseName, "^[A-Za-z]+"),
    Replicate = as.integer(str_extract(BaseName, "\\d+$")),
    SampleID  = paste0(Treatment, "_R", Replicate)
  ) %>%
  mutate(Treatment = toupper(Treatment)) %>%
  arrange(Treatment, Replicate)

cat("Samples:\n")
print(table(sample_info$Treatment))

cat("\n--- Filtering ---\n")
report_filt <- report
if ("Global.Q.Value" %in% colnames(report_filt))
  report_filt <- filter(report_filt, Global.Q.Value <= 0.01)
if ("Global.PG.Q.Value" %in% colnames(report_filt))
  report_filt <- filter(report_filt, Global.PG.Q.Value <= 0.01)
if ("Lib.Q.Value" %in% colnames(report_filt))
  report_filt <- filter(report_filt, Lib.Q.Value <= 0.01)
cat("After filtering:", nrow(report_filt), "precursors\n")

quant_col <- if ("PG.MaxLFQ" %in% colnames(report_filt)) "PG.MaxLFQ" else
  if ("PG.Normalised" %in% colnames(report_filt)) "PG.Normalised" else "PG.Quantity"
cat("Quantity column:", quant_col, "\n")

pg_long <- report_filt %>%
  select(Run, Protein.Group, Intensity = all_of(quant_col)) %>%
  distinct(Run, Protein.Group, .keep_all = TRUE) %>%
  left_join(sample_info %>% select(Run, SampleID), by = "Run")

pg_long <- pg_long %>%
  group_by(Protein.Group, SampleID) %>%
  summarise(Intensity = max(Intensity, na.rm = TRUE), .groups = "drop")

pg_wide <- pg_long %>%
  pivot_wider(names_from = SampleID, values_from = Intensity) %>%
  column_to_rownames("Protein.Group")

pg_wide[pg_wide == 0] <- NA
cat("Full matrix:", nrow(pg_wide), "proteins x", ncol(pg_wide), "samples\n")

stress_in_data <- intersect(stress_ids, rownames(pg_wide))
stress_not_found <- setdiff(stress_ids, rownames(pg_wide))
cat("\nStress proteins in quantification data:", length(stress_in_data), "/",
    length(stress_ids), "\n")
if (length(stress_not_found) > 0) {
  cat("Not found in data:", length(stress_not_found), "\n")
  cat("  ", paste(head(stress_not_found, 5), collapse = ", "), "...\n")
}

pg_stress <- pg_wide[stress_in_data, , drop = FALSE]
cat("Stress protein matrix:", nrow(pg_stress), "x", ncol(pg_stress), "\n")

keep_stress <- apply(pg_stress, 1, function(row) {
  any(sapply(treatments, function(tr) {
    cols <- sample_info$SampleID[sample_info$Treatment == tr]
    cols <- cols[cols %in% colnames(pg_stress)]
    sum(!is.na(row[cols])) >= 2
  }))
})
pg_stress_filt <- pg_stress[keep_stress, , drop = FALSE]
cat("After missingness filter:", nrow(pg_stress_filt), "stress proteins\n")

pg_log2 <- log2(pg_stress_filt)
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
cat("Imputed:", n_imputed, "/", total_cells,
    "(", round(100 * n_imputed / total_cells, 1), "%)\n")

write.csv(pg_log2, file.path(output_dir, "Tables", "stress_log2_imputed.csv"))

cat_df <- data.frame(
  UniProt_ID = rownames(pg_log2),
  Category = stress_id_to_cat[rownames(pg_log2)],
  stringsAsFactors = FALSE
)
cat_df$Category[is.na(cat_df$Category)] <- "Other stress"
cat_summary <- as.data.frame(table(cat_df$Category)) %>%
  rename(Category = Var1, Count = Freq) %>%
  mutate(Pct = round(100 * Count / sum(Count), 1)) %>%
  arrange(desc(Count))

p_cat <- ggplot(cat_summary, aes(x = reorder(Category, Count), y = Pct,
                                  fill = Category)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_text(aes(label = paste0(Pct, "% (n=", Count, ")")),
            hjust = -0.05, size = 3.2) +
  scale_fill_manual(values = cat_colors, guide = "none") +
  coord_flip() +
  labs(x = NULL, y = "Proportion of stress proteome (%)") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.25))) +
  theme_pub
save_pub(p_cat, "Fig_S01_stress_category_composition", "QC", w = 8, h = 5)
write_desc(
  "Fig_S01_stress_category_composition", "QC",
  title = "Functional Category Composition of the Abiotic Stress Proteome",
  placement = "MAIN MANUSCRIPT or SUPPLEMENTARY",
  description = paste(nrow(pg_log2), "GO-classified stress proteins grouped by",
                      "functional category."),
  method = paste("Categories assigned by description keyword mapping after",
                 "ESM-2 annotation transfer. GO:0006950 root term and descendants."),
  interpretation = "Largest categories indicate dominant stress response pathways."
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
save_pheatmap_pub(hm_cor_expr, "Fig_S02_stress_correlation_heatmap", "QC", w = 9, h = 8)

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
p_pca <- ggplot(pca_df, aes(x = PC1, y = PC2, colour = Treatment)) +
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
save_pub(p_pca, "Fig01_PCA_stress_proteins", "PCA")
write_desc(
  "Fig01_PCA_stress_proteins", "PCA",
  title = "PCA of Abiotic Stress Proteome — PC1 vs PC2",
  placement = "MAIN MANUSCRIPT",
  description = paste("PCA of", nrow(pg_log2), "GO-classified stress proteins."),
  method = paste("PCA on log2-imputed stress protein matrix. FactoMineR.",
                 "Convex hulls for n=3 replicates."),
  interpretation = paste("Treatment separation reflects stress-specific proteomic",
                         "remodelling. CS = positive control.")
)

mat_stress <- as.matrix(pg_log2)
rownames(mat_stress) <- ifelse(
  rownames(mat_stress) %in% names(stress_id_to_label),
  stress_id_to_label[rownames(mat_stress)],
  rownames(mat_stress)
)
mat_stress_scaled <- t(scale(t(mat_stress)))
mat_stress_scaled[mat_stress_scaled >  3] <-  3
mat_stress_scaled[mat_stress_scaled < -3] <- -3

row_anno <- data.frame(
  Category = stress_id_to_cat[rownames(pg_log2)],
  row.names = rownames(mat_stress)
)
row_anno$Category[is.na(row_anno$Category)] <- "Other stress"
ann_row_colors <- list(Category = cat_colors[unique(row_anno$Category)])

hm_stress_expr <- quote(
  pheatmap(mat_stress_scaled,
           color = colorRampPalette(rev(brewer.pal(11, "RdBu")))(100),
           annotation_col = annotation_col,
           annotation_row = row_anno,
           annotation_colors = c(ann_colors, ann_row_colors),
           clustering_distance_rows = "euclidean",
           clustering_method = "ward.D2",
           show_rownames = TRUE, fontsize_row = 5.5,
           main = "")
)

hm_height <- max(10, nrow(pg_log2) * 0.18)
save_pheatmap_pub(hm_stress_expr, "Fig02_stress_heatmap_all", "Heatmaps",
                  w = 12, h = hm_height)
write_desc(
  "Fig02_stress_heatmap_all", "Heatmaps",
  title = "Hierarchical Clustering of All Abiotic Stress Proteins",
  placement = "MAIN MANUSCRIPT",
  description = paste("Z-score heatmap of", nrow(pg_log2),
                      "stress proteins. Row annotation = functional category.",
                      "Column annotation = treatment."),
  method = paste("Z-score per row. Euclidean distance, Ward.D2.",
                 "Functional categories from ESM-2 annotation transfer."),
  interpretation = paste("Red = above average, Blue = below. Treatment-specific",
                         "expression blocks reveal stress pathway activation.")
)

treat_means <- sapply(treatments, function(tr) {
  cols <- sample_info$SampleID[sample_info$Treatment == tr]
  cols <- cols[cols %in% colnames(pg_log2)]
  rowMeans(pg_log2[, cols, drop = FALSE], na.rm = TRUE)
})
colnames(treat_means) <- treatments

rownames(treat_means) <- ifelse(
  rownames(treat_means) %in% names(stress_id_to_label),
  stress_id_to_label[rownames(treat_means)],
  rownames(treat_means)
)
mat_tmean_scaled <- t(scale(t(treat_means)))

row_anno_mean <- data.frame(
  Category = stress_id_to_cat[rownames(pg_log2)],
  row.names = rownames(treat_means)
)
row_anno_mean$Category[is.na(row_anno_mean$Category)] <- "Other stress"

hm_tmean_expr <- quote(
  pheatmap(mat_tmean_scaled,
           color = colorRampPalette(rev(brewer.pal(11, "RdBu")))(100),
           annotation_row = row_anno_mean,
           annotation_colors = list(Category = cat_colors[unique(row_anno_mean$Category)]),
           clustering_distance_rows = "euclidean",
           clustering_method = "ward.D2",
           show_rownames = TRUE, fontsize_row = 6,
           cluster_cols = FALSE,
           main = "")
)
save_pheatmap_pub(hm_tmean_expr, "Fig03_stress_treatment_mean_heatmap", "Heatmaps",
                  w = 8, h = hm_height)
write_desc(
  "Fig03_stress_treatment_mean_heatmap", "Heatmaps",
  title = "Treatment-Level Mean Abundance of Stress Proteins",
  placement = "MAIN MANUSCRIPT",
  description = paste("Treatment means (n=3) for", nrow(pg_log2),
                      "stress proteins. Columns NOT clustered (ordered RH, GH, CD, CS)."),
  method = "Mean log2 per treatment. Z-scored per protein. Euclidean/Ward.D2 rows.",
  interpretation = "Direct treatment comparison. Look for category-specific patterns."
)

cat("\n--- Differential expression (limma) on stress proteins ---\n")
sample_order <- colnames(pg_log2)
group <- factor(
  sample_info$Treatment[match(sample_order, sample_info$SampleID)],
  levels = treatments
)
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

cs_contrasts     <- grep("_vs_CS|CS_vs_", contrast_names, value = TRUE)
non_cs_contrasts <- setdiff(contrast_names, cs_contrasts)

de_results <- list()
for (i in seq_along(contrast_names)) {
  tt <- topTable(fit2, coef = i, number = Inf, sort.by = "none")
  tt$Protein.ID  <- rownames(tt)
  tt$Contrast    <- contrast_names[i]
  tt$Significant <- ifelse(tt$adj.P.Val < 0.05 & abs(tt$logFC) > 1,
                           ifelse(tt$logFC > 1, "Up", "Down"), "NS")

  tt$Short.Label <- stress_id_to_label[tt$Protein.ID]
  tt$Category    <- stress_id_to_cat[tt$Protein.ID]
  tt$Description <- stress_df$Description[match(tt$Protein.ID, stress_df$UniProt_ID)]
  tt$SwissProt_Description <- stress_df$SwissProt_Description[match(tt$Protein.ID, stress_df$UniProt_ID)]

  de_results[[contrast_names[i]]] <- tt
  write.csv(tt, file.path(output_dir, "Tables",
                          paste0("DE_stress_", contrast_names[i], ".csv")),
            row.names = FALSE)
}

de_all <- bind_rows(de_results)
write.csv(de_all, file.path(output_dir, "Tables", "DE_stress_all_contrasts.csv"),
          row.names = FALSE)

de_summary <- de_all %>%
  filter(Significant != "NS") %>%
  group_by(Contrast, Significant) %>%
  summarise(n = n(), .groups = "drop")

cat("\nDE summary (stress proteins):\n")
print(as.data.frame(de_summary))
write.csv(de_summary, file.path(output_dir, "Tables", "DE_stress_summary.csv"),
          row.names = FALSE)

for (cname in contrast_names) {
  tt <- de_results[[cname]] %>% arrange(adj.P.Val)

  tt$Label <- ""
  sig_rows <- which(tt$Significant != "NS")
  if (length(sig_rows) > 0) {
    tt$Label[sig_rows] <- tt$Short.Label[sig_rows]
  }

  pv <- ggplot(tt, aes(x = logFC, y = -log10(adj.P.Val), colour = Significant)) +
    geom_point(alpha = 0.7, size = 2.5) +
    geom_text_repel(aes(label = Label), size = 2.2, max.overlaps = 30,
                    colour = "black", fontface = "italic",
                    segment.colour = "grey50", segment.size = 0.3) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", colour = "grey40") +
    geom_vline(xintercept = c(-1, 1), linetype = "dashed", colour = "grey40") +
    scale_colour_manual(values = c(Up = "#E63946", Down = "#457B9D", NS = "grey75")) +
    labs(x = expression(Log[2]~Fold~Change),
         y = expression(-Log[10]~Adjusted~italic(P)-value)) +
    theme_pub

  is_cs <- grepl("_vs_CS|CS_vs_", cname)
  save_pub(pv, paste0("Fig04_volcano_stress_", cname), "Volcano", w = 9, h = 7)
  write_desc(
    paste0("Fig04_volcano_stress_", cname), "Volcano",
    title = paste("Volcano — Stress Proteins:", cname),
    placement = if (is_cs) "MAIN MANUSCRIPT" else "SUPPLEMENTARY",
    description = paste("Volcano plot of", nrow(pg_log2), "stress proteins for",
                        cname, ". All significant DEPs labelled."),
    method = "limma. |log2FC| > 1, adj.P < 0.05.",
    interpretation = paste("Red = up in", strsplit(cname, "_vs_")[[1]][1],
                           "; Blue = up in", strsplit(cname, "_vs_")[[1]][2])
  )
}

if (nrow(de_summary) > 0) {
  de_summary$Contrast <- factor(de_summary$Contrast,
                                 levels = c(cs_contrasts, non_cs_contrasts))
  p_de <- ggplot(de_summary, aes(x = Contrast, y = n, fill = Significant)) +
    geom_bar(stat = "identity", position = "dodge") +
    geom_text(aes(label = n), position = position_dodge(width = 0.9),
              vjust = -0.3, size = 3.5) +
    scale_fill_manual(values = c(Up = "#E63946", Down = "#457B9D")) +
    labs(x = NULL, y = "Number of Stress Proteins") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    theme_pub +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  save_pub(p_de, "Fig05_DE_stress_barplot", "DE")
}

sig_stress <- unique(de_all$Protein.ID[de_all$Significant != "NS"])
cat("\nSignificant stress DEPs:", length(sig_stress), "\n")

if (length(sig_stress) > 1) {
  mat_sig <- as.matrix(pg_log2[sig_stress, , drop = FALSE])
  rownames(mat_sig) <- ifelse(
    rownames(mat_sig) %in% names(stress_id_to_label),
    stress_id_to_label[rownames(mat_sig)],
    rownames(mat_sig)
  )
  mat_sig_scaled <- t(scale(t(mat_sig)))
  mat_sig_scaled[mat_sig_scaled >  3] <-  3
  mat_sig_scaled[mat_sig_scaled < -3] <- -3

  row_anno_sig <- data.frame(
    Category = stress_id_to_cat[sig_stress],
    row.names = rownames(mat_sig)
  )
  row_anno_sig$Category[is.na(row_anno_sig$Category)] <- "Other stress"

  hm_sig_expr <- quote(
    pheatmap(mat_sig_scaled,
             color = colorRampPalette(rev(brewer.pal(11, "RdBu")))(100),
             annotation_col = annotation_col,
             annotation_row = row_anno_sig,
             annotation_colors = c(ann_colors,
               list(Category = cat_colors[unique(row_anno_sig$Category)])),
             clustering_distance_rows = "euclidean",
             clustering_method = "ward.D2",
             show_rownames = TRUE, fontsize_row = 6.5,
             main = "")
  )
  sig_h <- max(8, length(sig_stress) * 0.22)
  save_pheatmap_pub(hm_sig_expr, "Fig06_stress_DEP_heatmap", "Heatmaps",
                    w = 12, h = sig_h)
  write_desc(
    "Fig06_stress_DEP_heatmap", "Heatmaps",
    title = "Heatmap of Differentially Expressed Stress Proteins",
    placement = "MAIN MANUSCRIPT",
    description = paste(length(sig_stress),
                        "significant stress DEPs. Row annotation = functional category."),
    method = "|log2FC| > 1, adj.P < 0.05. Z-score. Euclidean/Ward.D2.",
    interpretation = "Focus on category-specific blocks for pathway interpretation."
  )
}

dep_lists <- lapply(de_results, function(tt) tt$Protein.ID[tt$Significant != "NS"])
dep_lists <- dep_lists[sapply(dep_lists, length) > 0]

if (length(dep_lists) > 1) {
  tiff(file.path(output_dir, "Venn", "Fig07_UpSet_stress_DEPs.tiff"),
       width = 12, height = 7, units = "in", res = 600, compression = "lzw")
  suppressWarnings(
    print(upset(fromList(dep_lists), order.by = "freq",
                sets.bar.color = "#457B9D",
                main.bar.color = "#264653",
                text.scale = 1.3))
  )
  dev.off()
  pdf(file.path(output_dir, "Venn", "Fig07_UpSet_stress_DEPs.pdf"),
      width = 12, height = 7)
  suppressWarnings(
    print(upset(fromList(dep_lists), order.by = "freq",
                sets.bar.color = "#457B9D",
                main.bar.color = "#264653",
                text.scale = 1.3))
  )
  dev.off()
}

if (length(sig_stress) > 0) {
  dep_full <- de_all %>%
    filter(Significant != "NS") %>%
    select(Protein.ID, Short.Label, Category, Description, SwissProt_Description,
           Contrast, logFC, adj.P.Val, Significant, AveExpr) %>%
    arrange(Contrast, adj.P.Val)
  write.csv(dep_full,
            file.path(output_dir, "Tables", "Supplementary_stress_DEP_list.csv"),
            row.names = FALSE)

  dep_unique <- de_all %>%
    filter(Significant != "NS") %>%
    group_by(Protein.ID) %>%
    summarise(
      Short.Label    = first(Short.Label),
      Category       = first(Category),
      Description    = first(Description),
      N.Contrasts    = n_distinct(Contrast),
      Contrasts      = paste(unique(Contrast), collapse = "; "),
      Directions     = paste(unique(Significant), collapse = "; "),
      Min.adj.P      = min(adj.P.Val),
      Max.abs.logFC  = max(abs(logFC)),
      .groups = "drop"
    ) %>%
    arrange(Min.adj.P)
  write.csv(dep_unique,
            file.path(output_dir, "Tables", "Supplementary_stress_DEP_unique.csv"),
            row.names = FALSE)
}

sink(file.path(output_dir, "Tables", "stress_analysis_summary.txt"))
cat("====================================================\n")
cat("  STRESS PROTEOME ANALYSIS — CHICKPEA\n")
cat("  GO-classified abiotic stress proteins\n")
cat("====================================================\n\n")
cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
cat("Input stress proteins (GO-classified):", length(stress_ids), "\n")
cat("Found in DIA-NN data:", length(stress_in_data), "\n")
cat("After missingness filter:", nrow(pg_stress_filt), "\n")
cat("Imputed values:", n_imputed, "/", total_cells,
    "(", round(100 * n_imputed / total_cells, 1), "%)\n\n")
cat("Category composition:\n")
print(table(cat_df$Category))
cat("\nDE summary (|log2FC|>1, adj.P<0.05):\n")
if (nrow(de_summary) > 0) print(as.data.frame(de_summary))
cat("\nTotal unique stress DEPs:", length(sig_stress), "\n")
cat("====================================================\n")
sink()

cat("\n\n")
cat("================================================================\n")
cat("  STRESS PROTEOME ANALYSIS COMPLETE\n")
cat("================================================================\n")
cat("Output:", output_dir, "\n")
cat("Stress proteins analysed:", nrow(pg_log2), "\n")
cat("Significant stress DEPs:", length(sig_stress), "\n")
cat("================================================================\n")
