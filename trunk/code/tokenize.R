library(Matrix)
library(SingleCellExperiment)

# expr_matrix: genes x cells/samples (sparse dgCMatrix), rows named by gene ID
tokenize_expression <- function(expr_matrix, gene_median = NULL, max_len = 2048) {
  if (is.null(gene_median)) {
    gene_median <- Matrix::rowMeans(expr_matrix[expr_matrix > 0], na.rm = TRUE)
  }
  
  tokenize_cell <- function(cell_vec, genes, gene_median, max_len) {
    nz <- cell_vec > 0
    if (!any(nz)) return(character(0))
    # normalize by gene-level median expression (as in Geneformer)
    norm_val <- cell_vec[nz] / gene_median[genes[nz]]
    ord <- order(norm_val, decreasing = TRUE)
    tokens <- genes[nz][ord]
    if (length(tokens) > max_len) tokens <- tokens[seq_len(max_len)]
    tokens
  }
  genes <- rownames(expr_matrix)
  lapply(seq_len(ncol(expr_matrix)), function(j) {
    tokenize_cell(expr_matrix[, j], genes, gene_median, max_len)
  })
}

# usage
# tokens_list <- tokenize_expression(counts(sce))
# tokens_list[[1]]  # ranked gene-ID sequence for cell 1