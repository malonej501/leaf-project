library(ape)
library(treestats)
library(ggplot2)
library(ggtree)
library(dplyr)

import_trees <- function(){
  # import trees
  tre_names <- sort(list.files("phylogenies/final_data/trees"))
  tre_names_filt <- tre_names[!grepl("shuff|each|cenrich", tre_names)] #
  summary = data.frame(dataset=tre_names_filt)
  return(summary)
}

import_labels <- function(){
  labels <- sort(list.files("phylogenies/final_data/labels"))
  labels_filt <- labels[!grepl("shuff|each|cenrich", labels)]
  label_data <- data.frame()
  for (label in labels_filt){
    data <- read.delim(paste0("phylogenies/final_data/labels/",label), header=FALSE, sep="\t")
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

get_treeness <- function(summary){
  # calculate treeness - sum of all internal branch lengths (e.g. branches not leading to a tip) divided by the sum over all branch lengths
  for (dataset in summary$dataset) {
    tree = read.nexus(paste0("phylogenies/final_data/trees/", dataset))
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
  print(length(summary$dataset))
  layout(matrix(1:length(summary$dataset), ncol = 2, byrow = TRUE)) 
  par()
  for (dataset in summary$dataset) {
    tree = read.nexus(paste0("phylogenies/final_data/trees/", dataset))
    if (class(tree) == "phylo") {
      plot(tree, type="fan", main=dataset)
    }
    else if (class(tree) == "multiPhylo"){
      plot(tree[[1]], type="fan", main=dataset)
    }
  }
}

plot_ggtrees <- function(summary){
  label_data = import_labels()
  tree_names <- summary$dataset
  trees <- list()  # Initialize an empty list to store trees
  
  for (dataset in tree_names) {
    data_name = sub("^(.*class).*", "\\1", dataset)
    tree_path <- paste0("phylogenies/final_data/trees/", dataset)
    tree <- read.nexus(tree_path)
    labels <- label_data[label_data$dataset == data_name, ]
    #print(labels[1,])

    if (inherits(tree, "phylo")) {
      tree <- full_join(tree, labels, by="label")
      #print(tree, n=1000)
      trees[[dataset]] <- tree
    } else if (inherits(tree, "multiPhylo")) {
      tree <- full_join(tree[[1]], labels, by="label")
      trees[[dataset]] <- tree  # Select the first tree if it's a multiPhylo object
    }
    print(tree)
  }
  class(trees) <- "multiPhylo"
  #print(trees[[1]])
  ggtree(trees, layout="circular", size=0.07) +
    aes(colour=shape) +
    facet_wrap(~.id, scale="free", ncol = 4) +
    # theme_tree2() +
    #geom_tippoint(aes(colour=factor(shape))) +
    scale_color_manual(values=c("unlobed"="#0173B2","lobed"="#DE8F05","dissected"="#029E73","compound"="#D55E00"))
  
  #return(fig)
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
  print(shape_freq_df)
  colnames(shape_freq_df) <- c("dataset","u","l","d","c","ld","lc","ldc", "n_tips")
  summary <- merge(summary,shape_freq_df, by = "dataset")
  print(summary)
  
}
summary <- import_trees()
summary <- get_treeness(summary)
summary <- get_shape_counts(summary)


print(summary)
write.csv(summary, "tree_statistics.csv", row.names = FALSE)
plot_ggtrees(summary)
#ggsave("my_ggtree_plot.svg", plot = fig, width = 8, height = 6)


