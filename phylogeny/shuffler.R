library(ape)

jan_phylo_nat_class_tree <- read.nexus("./phylogenies/final_data/trees/jan_phylo_nat_class_21-01-24.tre")
jan_phylo_nat_class_labs <- read.table("./phylogenies/final_data/labels/jan_phylo_nat_class_21-01-24.txt", 
                                       sep = "\t", header=FALSE, col.names = 
                                         c("taxa","shape"))

# return phylogeny of equal number of each shape category, randomly selected
min_freq_shape <- names(which.min(table(jan_phylo_nat_class_labs$shape)))
min_freq <- min(table(jan_phylo_nat_class_labs$shape))

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
