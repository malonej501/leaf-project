library(ape)

#zuntini <- readLines("zuntini_2024.tre")

#zuntini <- read.nexus("zuntini_2024.tre")
#zuntini[4] <- gsub("(?<!') (?!')", "_", zuntini[4], perl = TRUE)
#zuntini[7] <- gsub("(?<!') (?!')", "_", zuntini[7], perl = TRUE)
#writeLines(zuntini, "zuntini_2024_clean.tre")
zuntini <- read.tree("trees/4_young_tree_smoothing_10_pruned_for_diversification_analyses.tre")
#zuntini <- read.tree("Trees/2_global_tree.tre")
zuntini_tips <- data.frame(order_family_genus_species = zuntini$tip.label)
#cleaning
zuntini_tips$genus_species <- zuntini_tips$order_family_genus_species
for (i in seq_along(zuntini_tips$genus_species)){
  zuntini_tips$genus_species[i] <- paste(strsplit(zuntini_tips$genus[i], "_")[[1]][3:4], collapse = "_")
}
zuntini_tips$genus <- sub("_.*", "", zuntini_tips$genus_species)

#zuntini$tip.label <- zuntini_tips$genus_species

geeta_labs <-
  read.csv("561AngLf09_D.csv",
           header = FALSE,
           col.names = c("genus", "shape"))

# intersect between zuntini and geeta genera
zuntini_geeta_intersect <-
  zuntini_tips[zuntini_tips$genus %in% intersect(zuntini_tips$genus, geeta_labs$genus),]
zuntini_geeta_intersect <- zuntini_geeta_intersect[!duplicated(zuntini_geeta_intersect$genus), ] # keep only the first occurring species in each genera

# get the resulting subset of each dataset
geeta_labs_sub <-
  subset(geeta_labs, genus %in% zuntini_geeta_intersect$genus)
geeta_labs_sub <- merge(geeta_labs_sub, zuntini_geeta_intersect, by="genus")
geeta_labs_sub <- geeta_labs_sub[c("order_family_genus_species","shape")]
zuntini_geeta_diff <- setdiff(zuntini_tips$order_family_genus_species, geeta_labs_sub$order_family_genus_species)

zuntini_sub <- drop.tip(zuntini, zuntini_geeta_diff)
length(zuntini_sub$tip.label)

plot.phylo(zuntini_sub, type = "fan", cex = 0.2)
add.scale.bar(length = 1, lwd = 2, col = "red")


#write the subset labels
write.table(
  geeta_labs_sub,
  file = "561AngLf09_D_zuntini2024_sub.txt",
  sep = "\t",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)
# write the subset tree
write.nexus(zuntini_sub, file = "zuntini_geetasub.tre")





# zuntini intersect with naturalis data
naturalis_labs <-
  read.csv(
    "Naturalis_img_labels_unambig_full_21-1-24.csv",
    header = FALSE,
    col.names = c("genus_species", "shape")
  )
naturalis_labs$genus <- gsub("_.*", "", naturalis_labs$genus_species)
# Keep remove all but the first occurance of each genus
#naturalis_labs <- naturalis_labs[!duplicated(naturalis_labs$genus), ]
#naturalis_labs <- naturalis_labs[, c("genus", "shape")]

# intersect between zuntini and geeta genera
zuntini_naturalis_intersect <-
  zuntini_tips[zuntini_tips$genus_species %in% intersect(zuntini_tips$genus_species, naturalis_labs$genus_species), ]


# get the resulting subset of each dataset
naturalis_labs_sub <-
  subset(naturalis_labs, genus_species %in% zuntini_naturalis_intersect$genus_species)
naturalis_labs_sub <- merge(naturalis_labs_sub, zuntini_naturalis_intersect, by="genus_species")
naturalis_labs_sub <- naturalis_labs_sub[c("order_family_genus_species","shape")]

zuntini_naturalis_diff <-
  setdiff(zuntini_tips$order_family_genus_species, naturalis_labs_sub$order_family_genus_species)
zuntini_sub <- drop.tip(zuntini, zuntini_naturalis_diff)
length(zuntini_sub$tip.label)

#write the subset labels
write.table(
  naturalis_labs_sub,
  file = "Naturalis_img_labels_unambig_full_21-1-24_zuntini2024_sub.txt",
  sep = "\t",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)
# write the subset tree
write.nexus(zuntini_sub, file = "zuntini_naturalis_sub.tre")


#### This part is for generating a genus_species_tree ####
tips = zuntini_sub$tip.label

processed_tips <- sapply(tips, function(x) {
  # Split the string at underscores
  parts <- strsplit(x, "_")[[1]]
  
  # Remove the first two parts
  remaining_parts <- parts[-c(1, 2)]
  
  # Optionally, paste the remaining parts back into a single string
  result <- paste(remaining_parts, collapse = "_")
  
  return(result)
})

zuntini_sub$tip.label <- processed_tips
print(length(zuntini_sub$tip.label))
write.nexus(zuntini_sub, file = "zuntini_naturalis_sub_genus_species.tre")

#### Subsetting to naturalis data overall ####

naturalis <- read.csv("../../leaf_databases/Naturalis/sample_eud_zuntini_10-09-24/Naturalis_multimedia_eud_sample_10-09-24.csv")
zuntini_naturalis_intersect <- zuntini_tips[zuntini_tips$genus_species %in% intersect(zuntini_tips$genus_species, naturalis$species), ]
#zuntini_naturalis_intersect <- zuntini_tips[zuntini_tips$genus %in% intersect(zuntini_tips$genus, naturalis$genus), ]
length(unique(zuntini_naturalis_intersect$genus))
naturalis_zuntini_intersect <- naturalis[naturalis$species %in% intersect(naturalis$species, zuntini_tips$genus_species),]
naturalis_zuntini_intersect <- naturalis[naturalis$genus %in% intersect(naturalis$genus, zuntini_tips$genus),]
length(unique(naturalis_zuntini_intersect$species))
naturalis_zuntini_intersect <- naturalis_zuntini_intersect[!duplicated(naturalis_zuntini_intersect$genus), ]
freqtab <- data.frame(table(naturalis_zuntini_intersect$species))
write.csv(naturalis_zuntini_intersect, "Naturalis_multimedia_eud_sample_10-09-24_zuntini_intercept_genera.csv")


#### Zuntini intercept with Naturalis_multimedia_eud_sample_10-09-24_zuntini_intercept_genera_labels_unambig_full data ####

naturalis_labs <-
  read.csv(
    "Naturalis_multimedia_eud_sample_10-09-24_zuntini_intercept_labels_unambig_full.csv",
    header = FALSE,
    sep = "\t",
    col.names = c("genus_species", "shape")
  )
naturalis_labs$genus <- gsub("_.*", "", naturalis_labs$genus_species)
# Keep remove all but the first occurance of each genus
#naturalis_labs <- naturalis_labs[!duplicated(naturalis_labs$genus), ]
#naturalis_labs <- naturalis_labs[, c("genus", "shape")]

# intersect between zuntini and naturalis
zuntini_naturalis_intersect <-
  zuntini_tips[zuntini_tips$genus_species %in% intersect(zuntini_tips$genus_species, naturalis_labs$genus_species), ]

# get the resulting subset of each dataset
naturalis_labs_sub <-
  subset(naturalis_labs, genus_species %in% zuntini_naturalis_intersect$genus_species)
naturalis_labs_sub <- merge(naturalis_labs_sub, zuntini_naturalis_intersect, by="genus_species")
naturalis_labs_sub <- naturalis_labs_sub[c("order_family_genus_species","shape")]

zuntini_naturalis_diff <-
  setdiff(zuntini_tips$order_family_genus_species, naturalis_labs_sub$order_family_genus_species)
zuntini_sub <- drop.tip(zuntini, zuntini_naturalis_diff)
length(zuntini_sub$tip.label)

#write the subset labels
write.table(
  naturalis_labs_sub,
  file = "Naturalis_multimedia_eud_sample_10-09-24_zuntini_intercept_labels_unambig_full_zuntini2024_sub.txt",
  sep = "\t",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)
# write the subset tree
write.nexus(zuntini_sub, file = "zuntini_Naturalis_multimedia_eud_sample_10-09-24_zuntini_intercept_labels_unambig_full_sub.tre")


#### Zuntini intercept with Naturalis_multimedia_eud_sample_10-09-24_zuntini_intercept_genera_labels_unambig_full data ####

naturalis_labs <-
  read.csv(
    #"Naturalis_multimedia_eud_sample_10-09-24_zuntini_intercept_genera_labels_unambig_full.csv",
    "zuntini_genera_equal_genus_phylo_nat_class.txt",
    header = FALSE,
    sep = "\t",
    col.names = c("genus_species", "shape")
  )
naturalis_labs$genus <- gsub("_.*", "", naturalis_labs$genus_species)
# Keep remove all but the first occurance of each genus
#naturalis_labs <- naturalis_labs[!duplicated(naturalis_labs$genus), ]
#naturalis_labs <- naturalis_labs[, c("genus", "shape")]

# set zuntini_phylo tips to genus
extract_third_part <- function(string) {
  parts <- strsplit(string, "_")[[1]]
  if (length(parts) >= 3) {
    return(parts[3])
  } else {
    return(NA)  # Return NA if there are less than 3 parts
  }
}

zuntini$tip.label <- sapply(zuntini$tip.label, extract_third_part)

zuntini_tips <- data.frame(genus = zuntini$tip.label)

# intersect between zuntini and naturalis
zuntini_naturalis_intersect <-
  zuntini_tips[zuntini_tips$genus %in% intersect(zuntini_tips$genus, naturalis_labs$genus), ]

# get the resulting subset of each dataset
naturalis_labs_sub <-
  subset(naturalis_labs, genus %in% zuntini_naturalis_intersect)
naturalis_labs_sub <- merge(naturalis_labs_sub, zuntini_naturalis_intersect, by="genus")
naturalis_labs_sub <- naturalis_labs_sub[c("genus","shape")]

zuntini_naturalis_diff <-
  setdiff(zuntini_tips$genus, naturalis_labs_sub$genus)
zuntini_sub <- drop.tip(zuntini, zuntini_naturalis_diff)
length(zuntini_sub$tip.label)

#write the subset labels
write.table(
  naturalis_labs_sub,
  file = "Naturalis_multimedia_eud_sample_10-09-24_zuntini_intercept_genera_labels_unambig_full_zuntini2024_sub.txt",
  sep = "\t",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)
# write the subset tree
#write.nexus(zuntini_sub, file = "zuntini_Naturalis_multimedia_eud_sample_10-09-24_zuntini_intercept_labels_genera_unambig_full_sub.tre")
write.nexus(zuntini_sub, file = "zuntini_genera_equal_genus_phylo_nat_class.tre")

