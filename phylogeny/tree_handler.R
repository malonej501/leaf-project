library(ape)

load_tree <- function(tree_path){
  tree <- read.tree(paste0("phylogenies/", tree_path))
  return(tree)
}

load_trees <- function(path){
  # path = "phylo_data/raw_trees"
  tree_files <- list.files(path=path, pattern = "\\.nwk$|\\.tree$|\\.tre$|\\.txt$|\\.nex$", full.names = TRUE)
  trees <- c()
  for (file in tree_files){
    print(file)
    if (grepl("zuntini", file)){
      tree <- read.tree(file)
    }
    else{
      tree <- read.nexus(file)
    }
    trees[[basename(file)]] <- tree
  }
  tree_filenames <- basename(tree_files)
  return(trees)
}

load_shape_data_full <- function(shape_dataset){
  if (shape_dataset == "Naturalis"){
    leaf_data_occur <- read.csv("leaf_data/Naturalis/botany-20240108.dwca/Occurrence.txt")
    names(leaf_data_occur)[names(leaf_data_occur) == "id"] <- "CoreId"
    leaf_data_media <- read.csv("leaf_data/Naturalis/botany-20240108.dwca/Multimedia.txt")
    # only return records with an associated digitised image 
    shape_data <- merge(leaf_data_occur, leaf_data_media, by="CoreId")
  }
  return(shape_data)
}

load_naturalis_sample_data <- function(){
  path = "shape_data/Naturalis"
  #jan_nat <- read.csv(paste0(path, "/jan_nat_eud_21-01-24/Naturalis_multimedia_eud_sample_13-01-24.csv"))
  #zun_nat <- read.csv(paste0(path, "/zun_nat_eud_10-09-24/Naturalis_multimedia_eud_sample_10-09-24.csv"))
  sample <- read.csv(paste0(path, "/jan_zun_nat_ang_26-09-24/Naturalis_multimedia_ang_sample_26-09-24.csv"))
  #sample <- read.csv(paste0(path, "/jan_zun_nat_ang_09-10-24/Naturalis_multimedia_ang_sample_09-10-24.csv"))
  #print(nrow(jan_nat))
  #print(nrow(zun_nat))
  return(list(janssens_nat = sample, zuntini_nat = sample))
}

unique_tips <- function(tree){
  # Get the tip labels
  tip_labels <- tree$tip.label
  # Identify unique labels and keep the first occurrence
  unique_labels <- unique(tip_labels)
  # Create a vector to store tips to keep
  tips_to_keep <- character()
  for (label in unique_labels) {
    # Keep the first occurrence of each label
    tips_to_keep <- c(tips_to_keep, tip_labels[which(tip_labels == label)[1]])
  }
  # Drop tips that are not in tips_to_keep
  tips_to_drop <- tip_labels[!tip_labels %in% tips_to_keep]
  pruned_tree <- drop.tip(tree, tips_to_drop)
  return (pruned_tree)
}

nat_tree_intersect <- function(tree_path){
  trees <- load_trees(tree_path)
  print("Done loading trees")
  nat_samp_data <- load_naturalis_sample_data()
  #print(trees$janssens_ml_dated.tre)
  tree_names <- c("janssens","zuntini")
  genus_trees <- list()
  species_trees <- list()
  for (name in tree_names){
    print(name)
    tree <- trees[[grep(name, names(trees))]]
    tips_genus_species <- sub("^[^_]*_[^_]*_(.*)", "\\1", tree$tip.label) # return genus and species of the tip_label
    tips_genus <- sub("^(.*?)_.*", "\\1", tips_genus_species) # return just the genus
    tree$tip.label <- sub("^[^_]*_[^_]*_(.*)", "\\1", tree$tip.label) # set tip labels to genus_species
    print(paste("original tree length =",length(tree$tip.label)))
    
    nat_samp <- nat_samp_data[[grep(name, names(nat_samp_data))]] # return the naturalis data for the current tree
    nat_tree_intersect_species <- nat_samp[nat_samp$species %in% intersect(nat_samp$species, tips_genus_species),] # subset nat data to species present in the tree
    print(paste("nat phylo species intersect =", nrow(nat_tree_intersect_species)))
    
    nat_tree_intersect_genus <- nat_samp[nat_samp$genus %in% intersect(nat_samp$genus, tips_genus),] # subset nat data to genera present in the tree
    nat_tree_intersect_genus <- nat_tree_intersect_genus[!duplicated(nat_tree_intersect_genus$genus), ] # keep only the first occurring species in each genera
    print(paste("nat phylo genus intersect =", nrow(nat_tree_intersect_genus)))

    #write.csv(nat_tree_intersect_species, paste(substr(name, 1, 3), "nat_species.csv", sep="_"), row.names=FALSE)
    #write.csv(nat_tree_intersect_genus, paste(substr(name, 1, 3), "nat_genus.csv", sep="_"), row.names=FALSE)

    #subset the trees
    #print(setdiff(tree$tip.label, nat_tree_intersect_species$species))
    tree_species_sub <- drop.tip(tree, setdiff(tree$tip.label, nat_tree_intersect_species$species)) # drop tree tips not present in the nat species subset
    tree_species_sub <- unique_tips(tree_species_sub) # if some tips are duplicated, remove all copies but one
    print(paste("tree_pruned_species length =",length(tree_species_sub$tip.label)))
    #print(setdiff(tree$tip.label, nat_tree_intersect_species$species))
    
    # keep one tip per genus
    tree_tip_info <- data.frame(tip = tree$tip.label, tip_genus_species = tips_genus_species, tip_genus = tips_genus) # construct df with tip label, genus and species
    # Randomly select one species from each genus
    unique_genus <- unique(tree_tip_info$tip_genus)
    selected_tips <- sapply(unique_genus, function(genus) {
      species_options <- tree_tip_info$tip[tree_tip_info$tip_genus == genus]
      sample(species_options, 1)  # Randomly select one species
    })
    tree_pruned <- drop.tip(tree, setdiff(tree$tip.label, selected_tips)) # drop all but one tip from each genera
    
    tree_pruned$tip.label <- sub("^(.*?)_.*", "\\1", tree_pruned$tip.label) # set tip labels to genus
    print(paste("tree_pruned_1_tip_per_genus length =",length(tree_pruned$tip.label)))
    tree_genus_sub <- drop.tip(tree_pruned, setdiff(tree_pruned$tip.label, nat_tree_intersect_genus$genus)) # drop tips not present in the nat genus subset
    tree_genus_sub <- unique_tips(tree_genus_sub) # if some tips are duplicated, remove all copies but one
    print("Tree length")
    #print(length(tree_species_sub$tip.label))
    print(paste("pruned_tree intersect with naturalis tree intersect length =", length(tree_genus_sub$tip.label)))
    #write.nexus(tree_genus_sub, file = paste(substr(name, 1, 3), "nat_genus.tre", sep="_"))
    genus_trees[[name]] <- tree_genus_sub
    species_trees[[name]] <- tree_species_sub
  }
  #print(genus_trees)
  return(c(genus_trees, species_trees))
}

nat_tree_intersect(tree_path="phylo_data/raw_trees")


generate_tree_labs <- function(trees, label_data, export_labs){
  tree_names <- c("janssens","zuntini")
  tree_labs <- list()
  for (name in tree_names){
    print(name)
    tree <- trees[[grep(name, names(trees))]]
    #print(head(tree$tip.label))

    tree_label_intersect <- label_data[label_data$genus %in% tree$tip.label, ]
    print(paste("nrow filtered label data",nrow(tree_label_intersect)))
    print(paste("ntips tree", length(tree$tip.label)))
    tree_label_intersect <- tree_label_intersect[, c("genus","shape")] 
    tree_labs[[name]] <- tree_label_intersect
    if (export_labs){
      write.table(
        tree_label_intersect,
        file = paste(substr(name, 1, 3), "nat_genus.txt", sep="_"),
        sep = "\t",
        row.names = FALSE,
        col.names = FALSE,
        quote = FALSE
      )
    }
  }
  return(tree_labs)
}

drop_ambiguous_tips <- function(trees, tree_labs, export_trees){
  trees_unambig <- list()
  for (i in seq_along(trees)){
    tree <- trees[[i]]
    name <- names(trees[i])
    print(name)
    tree_lab <- tree_labs[[i]]
    tree_unambig <- drop.tip(tree, setdiff(tree$tip.label, tree_lab$genus))
    print(paste("ntips before dropping ambiguous",length(tree$tip.label)))
    print(paste("no. tips in unambiguous label data", nrow(tree_lab)))
    print(paste("ntips after dropping ambiguous",length(tree_unambig$tip.label)))
    trees_unambig[[i]] <- tree_unambig
    
    if (export_trees){
      write.nexus(tree_unambig, file=paste(substr(name, 1, 3), "nat_genus.tre", sep="_"))
    }
  }
  return(trees_unambig)
}


label_phylogenies <- function(){
  label_data <- read.csv("shape_data/Naturalis/jan_zun_nat_ang_26-09-24/jan_zun_union_nat_genus_labelled.csv")
  trees <- nat_tree_intersect(tree_path="phylo_data/raw_trees")
  tree_labs <- generate_tree_labs(trees, label_data, export_labs=TRUE)
  trees_unambig <- drop_ambiguous_tips(trees, tree_labs, export_trees=TRUE)
}

get_nat_tree_intersect <- function(){
  nat_tree_intersect(tree_path="phylo_data/raw_trees")
}

nat_tree_intersect(tree_path="phylo_data/raw_trees")
#x <- read.csv("jan_nat_species.csv")
#x <- read.csv("zun_nat_genus.csv")
#y <- read.csv("jan_nat_genus.csv")
#z <- intersect(x$genus, y$genus)
#a <- intersect(x$species, y$species)
#b <- union(x$species, y$species)
#union <- rbind(x,y)
#union <- unique(union)
#write.csv(union, "jan_zun_union_nat_genus.csv", row.names=FALSE)

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

generate_phylo_naturalis_intersect <- function(){
  
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


