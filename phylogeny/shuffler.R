library(ape)

jan_phylo_nat_class_tree <- read.nexus("./phylogenies/final_data/trees/geeta_phylo_geeta_class_23-04-24.tre")
jan_phylo_nat_class_tree <- read.nexus("./phylogenies/final_data/trees/jan_phylo_nat_class_21-01-24.tre")
jan_phylo_nat_class_labs <- read.table("./phylogenies/final_data/labels/geeta_phylo_geeta_class_23-04-24.txt", 
                                       sep = "\t", header=FALSE, col.names = 
                                         c("taxa","shape"))
jan_phylo_nat_class_labs <- read.table("./phylogenies/final_data/labels/jan_phylo_nat_class_21-01-24.txt", 
                                       sep = "\t", header=FALSE, col.names = 
                                         c("taxa","shape"))
# load the compound enriched dataset
cenrich <- read.csv("./phylogenies/Janssens_Data/compound_enriched_sub.csv")

# return phylogeny of equal number of each shape category, randomly selected
shape_freqs <- jan_phylo_nat_class_labs[jan_phylo_nat_class_labs$shape %in% c(0,1,2,3), ]
min_freq_shape <- names(which.min(table(shape_freqs$shape)))
min_freq <- min(table(shape_freqs$shape))

# generate sample
sample <- data.frame()
for (s in c(0,1,2,3)){
  sub <- subset(jan_phylo_nat_class_labs, shape == s)
  samp <- sub[sample(nrow(sub), min_freq, replace=FALSE), ]
  sample <- rbind(sample, samp)
}

#subset phylogeny
jan_phylo_nat_class_tree_sub <- drop.tip(jan_phylo_nat_class_tree, setdiff(jan_phylo_nat_class_labs$taxa, sample$taxa))
write.nexus(jan_phylo_nat_class_tree_sub, file = "jan_phylo_nat_class_21-01-24_95_each.tre")
write.table(sample, file = "jan_phylo_nat_class_21-91-24_95_each.txt", sep="\t", col.names = FALSE, row.names = FALSE, quote=FALSE)

#shuffle phylogeny
jan_phylo_nat_class_tree_shuff <- jan_phylo_nat_class_tree
jan_phylo_nat_class_tree_shuff$tip.label <- sample(jan_phylo_nat_class_tree_shuff$tip.label, replace=FALSE)
write.nexus(jan_phylo_nat_class_tree_shuff, file = "jan_phylo_nat_class_21-01-24_shuff.tre")

#shuffle multiphylo
shuffle_tree_tips <- function(tree) {
  tree$tip.label <- sample(tree$tip.label)
  return(tree)
}

# Apply the function to each tree in the multiphylo object
jan_phylo_nat_class_tree_shuff <- lapply(jan_phylo_nat_class_tree_shuff, shuffle_tree_tips)
write.nexus(jan_phylo_nat_class_tree_shuff, file = "jan_phylo_nat_class_21-01-24_shuff.tre")

#subset tree to cenrich
jan_phylo_nat_class_tree_cenrich_sub <- drop.tip(jan_phylo_nat_class_tree, setdiff(jan_phylo_nat_class_tree$tip.label, cenrich$taxa))
write.nexus(jan_phylo_nat_class_tree_cenrich_sub, file = "jan_phylo_nat_class_21-01-24_cenrich_sub.tre")
#subset labels to cenrich
jan_phylo_nat_class_labs_cenrich_sub <- merge(jan_phylo_nat_class_labs, cenrich, on="taxa")
write.table(jan_phylo_nat_class_labs_cenrich_sub, file = "jan_phylo_nat_class_21-01-24_cenrich_sub.txt", sep="\t", col.names=FALSE, row.names=FALSE, quote=FALSE)
