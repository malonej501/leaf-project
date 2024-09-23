library(ape)

load_tree <- function(tree_path){
  tree <- read.tree(paste0("phylogenies/", tree_path))
  return(tree)
}

load_shape_data <- function(shape_dataset){
  if (shape_dataset == "Naturalis"){
    leaf_data_occur <- read.csv("leaf_data/Naturalis/botany-20240108.dwca/Occurrence.txt")
    names(leaf_data_occur)[names(leaf_data_occur) == "id"] <- "CoreId"
    leaf_data_media <- read.csv("leaf_data/Naturalis/botany-20240108.dwca/Multimedia.txt")
    # only return records with an associated digitised image 
    shape_data <- merge(leaf_data_occur, leaf_data_media, by="CoreId")
  }
  return(shape_data)
}

load_apg <- function(){
  apg <- read.csv("leaf_data/APG_IV/APG_IV_ang_fams.csv")
}

tree_shape_intersect <- function(tree_path, tree, shape_data){
  if (grepl("Zuntini", tree_path)){
    # get the genera present in the phylogenetic tree
    tree_genera <- c()
    for (i in seq_along(tree$tip.label)){
      tree_genera[i] <- paste(strsplit(tree$tip.label[i], "_")[[1]][3], collapse = "_")
    }
  }
  # get genera from shape dataset
  shape_data_genera <- unique(shape_data$genus)
  genera_intersect <- intersect(shape_data_genera, tree_genera)
  
  return(genera_intersect)
}

get_random_rows <- function(df_full, value_list, column_name) {
  #subset the data to just the ID and taxon columns to save time
  df <- df_full[,c("CoreId",column_name)]
  result_list <- lapply(value_list, function(value) {
    # Filter the data frame based on the current value
    filtered_df <- df[df[[column_name]] == value, ]
    
    # Check if there are matching rows
    if (nrow(filtered_df) > 0) {
      # Sample a random row
      random_row <- filtered_df[sample(nrow(filtered_df), 1), ]
      return(random_row)
    } else {
      # Return NA if no matching rows
      return(NA)
    }
  })
  
  # Combine results into a data frame
  result_df <- do.call(rbind, result_list)
  return(result_df)
}

get_sample_from_shape_data <- function(sample_ids, shape_data){
  sample_full <- shape_data[shape_data$CoreId %in% sample_ids$CoreId, ]
  sample_full$genus_species <- paste0(sample_full$genus, sample_full$specificEpithet)
  sample_full_unique <- sample_full[!duplicated(sample_full$genus_species),]
  print(nrow(sample_full))
  print(nrow(sample_full_unique))
  return(sample_full_unique)
}

generate_phylo_naturais_intersect <- function(){
  
  tree_path <- "Zuntini_2024/trees/4_young_tree_smoothing_10_pruned_for_diversification_analyses.tre"
  shape_dataset <- "Naturalis"
  tree <- load_tree(tree_path)
  shape_data <- load_shape_data(shape_dataset)
  apg <- load_apg()
  genera_intersect <- tree_shape_intersect(tree_path, tree, shape_data)
  img_sample <- get_random_rows(shape_data, genera_intersect, "genus")
  sample_full_unique <- get_sample_from_shape_data(img_sample, shape_data)
  write.csv(sample_full_unique, "zuntini_naturalis_sample_16-09-24.csv")
}

compare_phylo_taxa <- function(){
  jan_labels <- read.csv("phylogenies/final_data/labels/jan_phylo_nat_class_21-1-24.txt", sep="\t", header=FALSE)
  names(jan_labels) <- c("genus_species", "shape")
  zun_labels <- read.csv("phylogenies/final_data/labels/zuntini_genera_phylo_nat_class_10-09-24.txt", sep="\t", header=FALSE)
  names(zun_labels) <- c("genus", "shape")
  print(jan_labels)
  print(zun_labels)
  jan_labels$genus <- sub("_.*", "", jan_labels$genus_species)
  genus_intersect <- intersect(jan_labels$genus, zun_labels$genus)
}


