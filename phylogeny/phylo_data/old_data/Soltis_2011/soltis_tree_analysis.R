library(ape)

#soltis <- read.tree("Soltis_MajRule18s26sBS100.treorg", multi=TRUE)
#soltis <- read.nexus("T31201.nex")
soltis <- read.nexus("T31199.nex")
soltis_tips <- data.frame(genus = soltis$tip.label)
geeta_labs <-
  read.csv("561AngLf09_D.csv",
           header = FALSE,
           col.names = c("genus", "shape"))

# intersect and difference between soltis and geeta genera
soltis_geeta_intersect <-
  intersect(soltis_tips$genus, geeta_labs$genus)
soltis_geeta_diff <- setdiff(soltis_tips$genus, geeta_labs$genus)
# get the resulting subset of each dataset
geeta_labs_sub <-
  subset(geeta_labs, genus %in% soltis_geeta_intersect)
soltis_sub <- drop.tip(soltis, soltis_geeta_diff)
length(soltis_sub$tip.label)

#write the subset labels
write.table(
  geeta_labs_sub,
  file = "561AngLf09_D_soltis2011_T31199_sub.txt",
  sep = "\t",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)
# write the subset tree
write.nexus(soltis_sub, file = "Soltis_T31199_geetasub.tre")





# soltis intersect with naturalis data
naturalis_labs <-
  read.csv(
    "Naturalis_img_labels_unambig_full_21-1-24.csv",
    header = FALSE,
    col.names = c("species", "shape")
  )
naturalis_labs$genus <- gsub("_.*", "", naturalis_labs$species)
# Keep remove all but the first occurance of each genus
naturalis_labs <- naturalis_labs[!duplicated(naturalis_labs$genus), ]
naturalis_labs <- naturalis_labs[, c("genus", "shape")]

# intersect and difference between soltis and geeta genera
soltis_naturalis_intersect <-
  intersect(soltis_tips$genus, naturalis_labs$genus)
soltis_naturalis_diff <-
  setdiff(soltis_tips$genus, naturalis_labs$genus)

# get the resulting subset of each dataset
naturalis_labs_sub <-
  subset(naturalis_labs, genus %in% soltis_naturalis_intersect)
soltis_sub <- drop.tip(soltis, soltis_naturalis_diff)
length(soltis_sub$tip.label)

#write the subset labels
write.table(
  naturalis_labs_sub,
  file = "Naturalis_img_labels_unambig_full_21-1-24_soltis2011_T31199_sub.txt",
  sep = "\t",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)
# write the subset tree
write.nexus(soltis_sub, file = "Soltis_T31199_naturalis_sub.tre")
