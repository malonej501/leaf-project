library(ape)
library(treestats)
library(ggplot2)
library(ggtree)
library(dplyr)
library(svglite)
library(stringr)
library(RColorBrewer)

import_trees <- function(){
  # import trees
  tre_names <- sort(list.files("phylo_data/trees_final"))
  tre_names_filt <- tre_names[!grepl("shuff|each|cenrich", tre_names)] #
  summary = data.frame(dataset=tre_names_filt)
  return(summary)
}

import_labels <- function(){
  labels <- sort(list.files("shape_data/labels_final"))
  labels_filt <- labels[!grepl("shuff|each|cenrich", labels)]
  label_data <- data.frame()
  for (label in labels_filt){
    data <- read.delim(paste0("shape_data/labels_final/",label), header=FALSE, sep="\t")
    data$dataset = sub("^(.*class).*", "\\1", label)
    label_data <- rbind(label_data, data)
  }
  names(label_data) <- c("label","shape_num","dataset")
  label_data <- label_data[,c("dataset","label","shape_num")]
  label_data <- label_data %>%
    mutate(shape = case_when(
      shape_num == 0 ~ "unlobed",
      shape_num == 1 ~ "lobed",
      shape_num == 2 ~ "dissected",
      shape_num == 3 ~ "compound",
    ))    
  return(label_data)
}

map_higher_order_labels <- function(labels){
  # add higher order taxon classification to the tree labels using data from Naturalis
  nat_samp <- read.csv("shape_data/Naturalis/jan_zun_nat_ang_26-09-24/Naturalis_multimedia_ang_sample_26-09-24.csv") # read in full herabrium data on angiosperm sample
  tax_map <- nat_samp[,c("class","order","family","genus","specificEpithet")]
  tax_map <- tax_map[!duplicated(tax_map$genus), ] # remove rows of duplicate genera
  labels <- left_join(labels, tax_map, by = c("label" = "genus"))
  return(labels)
}

get_treeness <- function(summary){
  # calculate treeness - sum of all internal branch lengths (e.g. branches not leading to a tip) divided by the sum over all branch lengths
  for (dataset in summary$dataset) {
    tree = read.nexus(paste0("phylo_data/trees_final/", dataset))
    if (class(tree) == "phylo") {
      summary[summary$dataset == dataset, "treeness"] <- treeness(tree)  
      summary[summary$dataset == dataset, "mean_branch_length"] <- mean_branch_length(tree)
      summary[summary$dataset == dataset, "var_branch_length"] <- var_branch_length(tree)
      summary[summary$dataset == dataset, "avg_leaf_depth"] <- average_leaf_depth(tree)
      summary[summary$dataset == dataset, "var_leaf_depth"] <- var_leaf_depth(tree)
      summary[summary$dataset == dataset, "pigot_rho"] <- pigot_rho(tree)
    }
    else if(class(tree) == "multiPhylo") {
      tness_subtree_list <- c()
      mbl_list <- c()
      vbl_list <- c()
      ald_list <- c()
      vld_list <- c()
      pr_list <- c()
      for (i in seq_along(tree)) {
        tness_subtree_list <- c(tness_subtree_list, treeness(tree[[i]]))  # Append subtree treeness
        mbl_list <- c(mbl_list, mean_branch_length(tree[[i]]))
        vbl_list <- c(vbl_list, var_branch_length(tree[[i]]))
        ald_list <- c(ald_list, average_leaf_depth(tree[[i]]))
        vld_list <- c(vld_list, var_leaf_depth(tree[[i]]))
        pr_list <- c(pr_list, pigot_rho(tree[[i]]))
      }
      summary[summary$dataset == dataset, "treeness"] <- mean(tness_subtree_list)  # Calculate the mean treeness, ignoring NAs
      summary[summary$dataset == dataset, "mean_branch_length"] <- mean(mbl_list)
      summary[summary$dataset == dataset, "var_branch_length"] <- mean(vbl_list)
      summary[summary$dataset == dataset, "avg_leaf_depth"] <- mean(ald_list)
      summary[summary$dataset == dataset, "var_leaf_depth"] <- mean(vld_list)
      summary[summary$dataset == dataset, "pigot_rho"] <- mean(pr_list)

    }
  }
  return(summary)
}

plot_trees <- function(summary){
  layout(matrix(1:length(summary$dataset), ncol = 2, byrow = TRUE)) 
  par()
  for (dataset in summary$dataset) {
    tree = read.nexus(paste0("phylo_data/trees_final/", dataset))
    if (class(tree) == "phylo") {
      plot(tree, type="fan", main=dataset)
    }
    else if (class(tree) == "multiPhylo"){
      plot(tree[[1]], type="fan", main=dataset)
    }
  }
}

plot_ggtrees <- function(summary){
  tiplab_text_size = 0.2 # set the size for tip labels and heatmap labels
  tiplab_vjust = 0.25 # set the vertical justification for tip labels and heatmap labels to ensure alignmnet with tree tips
  label_data = import_labels()
  label_data = map_higher_order_labels(label_data)
  tree_names <- summary$dataset
  # tree_names <- list("jan_genus_phylo_nat_class_26-09-24.tre")
  tree_names <- list("zun_genus_phylo_nat_class_26-09-24.tre")
  # tree_names <- list("geeta_phylo_geeta_class_23-04-24.tre")
  trees <- list()  # Initialize an empty list to store trees
  heatmap_data <- c()
  
  for (dataset in tree_names) {
    data_name <- sub("^(.*class).*", "\\1", dataset)
    tree_path <- paste0("phylo_data/trees_final/", dataset)
    tree <- read.nexus(tree_path)
    labels <- label_data[label_data$dataset == data_name, ]
    label_data_ordered <- data.frame("label" = tree$tip.label) # Create a data frame with the tree tip labels in the tree order
    label_data_ordered <- left_join(label_data_ordered, labels, by="label") # Join with the labels data frame to get the order
    heatmap_data <- data.frame("order"=label_data_ordered$order) # here we specify the taxonomic level to be used for the heatmap
    rownames(heatmap_data) <- label_data_ordered$label

    if (inherits(tree, "phylo")) {
      tree <- left_join(tree, labels, by="label")
      trees[[dataset]] <- tree
    } else if (inherits(tree, "multiPhylo")) {
      tree <- left_join(tree[[1]], labels, by="label")
      trees[[dataset]] <- tree  # Select the first tree if it's a multiPhylo object
    }
  }
  class(trees) <- "multiPhylo"
  print(trees[[1]])
  p <- ggtree(trees, layout="circular", size=ifelse(length(trees) == 1, 0.1, 0.07)) +
    aes(colour=shape) +
    facet_wrap(~.id, scale="free", ncol = 4) +
    # theme_tree2() +
    # geom_tippoint(aes(colour=factor(order))) +
    geom_tiplab(size=tiplab_text_size, vjust=tiplab_vjust) + #, aes(colour=factor(order))) +
    scale_color_manual(values=c("unlobed"="#0173B2","lobed"="#DE8F05","dissected"="#029E73","compound"="#D55E00")) +
    geom_treescale(x=0, y=0, width=100, offset=5) #width=0.1 for geeta, 100 for jan, zun
  
 
  tips_plot_order <- rev(get_taxa_name(p)) # gives the tip labels in the order they appear in ggplot
  tips_ape_order <- rownames(heatmap_data)
  idx_ape_in_plot <- match(tips_ape_order, tips_plot_order) # find the index of each ape tip label in the plot
  ang <- ((360 * idx_ape_in_plot) / length(idx_ape_in_plot)) # calculate the angle for each tip label based on the plot order, not order in the ape phylo object
  ang[ang > 90 & ang < 270] <- ang[ang > 90 & ang < 270] - 180


  # extend the angle list to include NULL values for internal nodes
  total_nodes <- Ntip(trees[[1]]) + Nnode(trees[[1]])
  add_nodes <- total_nodes - length(ang) 
  ang_ext <- c(ang, rep(list(NULL), add_nodes)) 

  p <- gheatmap(p, heatmap_data, offset = 0.05, width=0.1, color=NULL, colnames=TRUE, hjust=0.5, colnames_offset_x = 5) +
    # scale_fill_brewer(palette="Set2")
    # scale_fill_manual(values = col_vector)
    scale_fill_viridis_d(option="D" ) +
    geom_text(aes(label=order, angle=ang_ext), color="white", size=tiplab_text_size, nudge_x=20, vjust=tiplab_vjust, hjust=0.5) +
    # geom_text(aes(label=as.character(1:total_nodes), angle=ang_ext), color="white", size=tiplab_text_size, nudge_x=12, vjust=tiplab_vjust, hjust=0) +
    guides(fill = "none") # remove heatmap legend
    # labs(fill = "order")
  #return(fig)
  # ggsave(file="ggtreeplot.svg", plot=p, width=10, height=10)
  ggsave(file="jan_genus_phylo_nat_class_26-09-24.pdf", plot=p, width=10, height=10, dpi=10000) #text below a certain size will not be rendered with .pdf
}

get_shape_counts <- function(summary){
  label_data = import_labels()
  tree_names <- summary$dataset
  shape_freq_df = data.frame()
  for (dataset in tree_names){
    data_name = sub("^(.*class).*", "\\1", dataset)
    labels <- label_data[label_data$dataset == data_name, ]
    shape_freq <- as.data.frame(table(labels$shape_num))
    shape_freq$dataset <- dataset
    shape_freq <- reshape(shape_freq, idvar="dataset", timevar="Var1", direction="wide")
    #shape_freq$n_tips <- shape_freq$compound + shape_freq$dissected + shape_freq$lobed + shape_freq$unlobed
    shape_freq_df <- bind_rows(shape_freq_df, shape_freq)
  }
  shape_freq_df[is.na(shape_freq_df)] <- 0
  shape_freq_df$n_tips <- rowSums(shape_freq_df[grep("^Freq", names(shape_freq_df), value = TRUE)])
  # print(shape_freq_df)
  colnames(shape_freq_df) <- c("dataset","u","l","d","c","ld","lc","ldc", "n_tips")
  summary <- merge(summary,shape_freq_df, by = "dataset")
  # print(summary)
  
}
summary <- import_trees()
summary <- get_treeness(summary)
summary <- get_shape_counts(summary)


write.csv(summary, "tree_statistics.csv", row.names = FALSE)
plot_ggtrees(summary)
#ggsave("my_ggtree_plot.svg", plot = fig, width = 8, height = 6)


