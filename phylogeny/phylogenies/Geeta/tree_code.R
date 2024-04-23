library(ape)
library(ggplot2)
library(maps)
library(ggmap)
library(ggtree)

tree <- read.nexus("Janssens_ml_dated.tre")
tree_labs <- data.frame(tree$tip.label)

geeta_tree <- read.nexus("561Data08.tre")
geeta_labs <- data.frame(geeta_tree$tip.label)

tree_labs_sub_all <-
  tree_labs[grep(paste(geeta_labs$tip.label, collapse = "|"),
                 tree_labs$tree.tip.label), ]
tree_labs_sub_first_indicies <-
  sapply(geeta_labs$tip.label, function(x)
    which(grepl(x, tree_labs$tree.tip.label))[1])
tree_labs_sub_first <-
  na.omit(data.frame(species = tree_labs[tree_labs_sub_first_indicies, ])) # remove NAs
#tree_labs_sub_first <- tree_labs_sub_first[!grepl("OUT", tree_labs_sub_first$species)] # remove outgroups
rownames(tree_labs_sub_first) <- NULL # reset index

# Create a new shape file that contains the subset of species (not genus) with shape labels
shape = read.csv("561AngLf09_D.csv",
                 header = FALSE,
                 col.names = c("genus", "shape"))
matching_species = grep(paste(shape$genus, collapse = "|"),
                        tree_labs_sub_first$species)
matching_species_labs = data.frame(species = tree_labs_sub_first$species, shape = shape$shape[matching_species])
# write.table(matching_species_labs, "Geeta_sub_species.txt", sep="\t", row.names=FALSE, col.names=FALSE, quote=FALSE)


tree_sub_labelled <-
  drop.tip(tree, tree$tip.label[!(tree$tip.label %in% tree_labs_sub_first$species)])
#tree_intersect_labelled <- drop.tip(tree_intersect_labelled, unique(tree_intersect_labelled$tiplabel))
write.nexus(tree_sub_labelled, file = "./Geeta_sub_Janssens.tre")
print(sum(table(tree_sub_labelled$tip.label)))
tips = dataframe(tip_labels = tree_sub_labelled$tip.label)

tree_minus_outgroups <-
  drop.tip(tree,
           c(
             "OUT_Pinus",
             "OUT_Taxus",
             "OUT_Ginkgo",
             "OUT_Zamia",
             "OUT_Cycas"
           ))

output_file = paste0("/home/jmalone/Documents/Leaf-Project/Janssens_Data/tree_plot.pdf")
pdf(output_file, width = 16, height = 16)
par(mar = c(1, 5, 1, 1))
#plot(tree, cex=0.25, edge.width = 0.5, show.tip.label=FALSE)
plot.phylo(tree_intersect_labelled,
           show.tip.label = FALSE,
           no.margin = TRUE)
par(xpd = TRUE)
add.scale.bar(y = -3, lcol = "red", lwd = 4)
dev.off()

ggtree(tree_minus_outgroups)
axisPhylo()

table((naturalis_sample_labelled$species %in% tree_tips$tip_labels))

# Finding geeta intersect with Naturalis
naturalis_labs <-
  read.csv(
    "Naturalis_img_labels_unambig_full_21-1-24.csv",
    header = FALSE,
    col.names = c("species", "shape")
  )
naturalis_labs$genus <- gsub("_.*", "", naturalis_labs$species)
# Keep remove all but the first occurrance of each genus
naturalis_labs <- naturalis_labs[!duplicated(naturalis_labs$genus), ]
naturalis_labs <- naturalis_labs[, c("genus", "shape")]
# reformat geeta_labs
geeta_labs <- data.frame(genus = geeta_tree$PAUP_1$tip.label)


# intersect and difference between naturalis and geeta genera
geeta_naturalis_intersect <-
  intersect(geeta_labs$genus, naturalis_labs$genus)
geeta_naturalis_diff <-
  setdiff(geeta_labs$genus, naturalis_labs$genus)


# get the resulting subset of each dataset
naturalis_labs_sub <-
  subset(naturalis_labs, genus %in% geeta_naturalis_intersect)
geeta_sub <- drop.tip(geeta_tree, geeta_naturalis_diff)
length(geeta_sub$PAUP_1$tip.label)

#write the subset labels
write.table(
  naturalis_labs_sub,
  file = "Naturalis_img_labels_unambig_full_21-1-24_Geeta_sub.txt",
  sep = "\t",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)
# write the subset tree
write.nexus(geeta_sub, file = "Geeta_naturalis_sub.tre")



# Plotting the Geographical distribution of the sample

world_map_data <- map_data("world")
naturalis_tree_intersect_geo <-
  data.frame(long = naturalis_sample_tree_intersect$longitudeDecimal,
             lat = naturalis_sample_tree_intersect$latitudeDecimal)
naturalis_tree_intersect_country <-
  data.frame(region = naturalis_sample_tree_intersect$country)
# Making country names the same between datasets
naturalis_tree_intersect_country <-
  replace(
    naturalis_tree_intersect_country,
    naturalis_tree_intersect_country == "Unknown",
    NA
  )
naturalis_tree_intersect_country <-
  replace(
    naturalis_tree_intersect_country,
    naturalis_tree_intersect_country == "United States of America",
    "USA"
  )
naturalis_tree_intersect_country <-
  replace(
    naturalis_tree_intersect_country,
    naturalis_tree_intersect_country == "United Kingdom",
    "UK"
  )
naturalis_tree_intersect_country$region[grepl("Congo", naturalis_tree_intersect_country$region)] <-
  "Democratic Republic of the Congo"
naturalis_tree_intersect_country$region[grepl("Malaysia", naturalis_tree_intersect_country$region)] <-
  "Malaysia"
naturalis_tree_intersect_country$region[grepl("Azores", naturalis_tree_intersect_country$region)] <-
  "Azores"
naturalis_tree_intersect_country$region[grepl("Canary Islands", naturalis_tree_intersect_country$region)] <-
  "Canary Islands"
naturalis_tree_intersect_country$region[grepl("Palestinian", naturalis_tree_intersect_country$region)] <-
  "Palestine"
naturalis_tree_intersect_country$region[grepl("Hawaii", naturalis_tree_intersect_country$region)] <-
  "Hawaii"
naturalis_tree_intersect_country$region[grepl("Virgin Islands", naturalis_tree_intersect_country$region)] <-
  "Virgin Islands"
naturalis_tree_intersect_country$region[grepl("Burma", naturalis_tree_intersect_country$region)] <-
  "Myanmar"


scatter <-
  ggplot(naturalis_tree_intersect_geo, aes(x = long, y = lat)) +
  geom_point(
    stroke = NA,
    size = 3,
    color = "red",
    alpha = 0.4
  )

map <- scatter +
  geom_map(
    data = world_map_data,
    map = world_map_data,
    aes(map_id = region),
    color = "black",
    fill = NA
  ) +
  coord_fixed(1.3) +
  theme_bw() +
  theme(panel.background = element_rect(fill = "white")) +
  theme(panel.border = element_blank()) +
  theme(panel.grid = element_blank()) +
  labs(x = NULL, y = NULL)
map

ggsave(
  filename = "sample_geographical_distribution1.png",
  plot = map,
  width = 16,
  height = 9,
  dpi = 600
)

country_freq <-
  data.frame(table(naturalis_tree_intersect_country$region))

gg <- ggplot()
gg <- gg + geom_map(
  data = world_map_data,
  map = world_map_data,
  aes(x = long, y = lat, map_id = region),
  fill = "lightgrey",
  color = "black"
)
gg <- gg + geom_map(
  data = country_freq,
  map = world_map_data,
  aes(map_id = Var1, fill = Freq),
  color = "black"
)
gg <- gg + scale_fill_viridis_c(direction = -1)
gg <- gg + theme_bw()
gg <- gg + labs(x = NULL, y = NULL)
gg <- gg + theme(panel.border = element_blank())
gg <- gg + theme(panel.grid = element_blank())
gg

ggsave(
  filename = "/home/jmalone/Documents/Leaf-Project/Janssens_Data/sample_geographical_distribution2.png",
  plot = gg,
  width = 16,
  height = 9,
  dpi = 600
)
