library(ape)
library(ggplot2)
library(maps)
library(ggmap)
library(ggtree)

tree <- read.nexus("Janssens_ml_dated.tre")

tree_tips <- data.frame(tip_labels = tree$tip.label)

wilf_sample <- read.csv("Wilf_eud_sample_20-02-24.csv")
wilf_sample_lab <- read.csv("./20-02-24/img_labels_unambig_full.csv")

wilf_sample_tree_intersect <- wilf_sample[wilf_sample$genus_species %in% intersect(tree_tips$tip_labels, wilf_sample$genus_species),]
# write.csv(wilf_sample_tree_intersect, file="Wilf_sample_Janssens_intersect_species_list.csv", row.names=FALSE)
length(unique(wilf_sample_tree_intersect$genus_species))
wilf_sample_lab_tree_intersect <- intersect(tree_tips$tip_labels, wilf_sample_lab$species)

tree_intersect <- drop.tip(tree, tree$tip.label[!(tree$tip.label %in% wilf_sample_tree_intersect$species)])
tree_intersect_labelled <- drop.tip(tree, tree$tip.label[!(tree$tip.label %in% wilf_sample_lab_tree_intersect)])
#tree_intersect_labelled <- drop.tip(tree_intersect_labelled, unique(tree_intersect_labelled$tiplabel))
write.nexus(tree_intersect_labelled, file="./Wilf_sample_Janssens_intersect_labelled.tre")
print(sum(table(tree_intersect_labelled$tip.label)))
tips = dataframe(tip_labels = tree_intersect_labelled$tip.label)
print(length(unique(naturalis_sample_labelled$species)))
#returns the species in the naturalis sample but not in the Janssens tree
print(naturalis_sample_labelled[!(naturalis_sample_labelled$species %in% tree$tip.label),]$species)

tree_minus_outgroups <- drop.tip(tree, c("OUT_Pinus","OUT_Taxus","OUT_Ginkgo",
                                         "OUT_Zamia","OUT_Cycas"))

output_file = paste0("/home/jmalone/Documents/Leaf-Project/Janssens_Data/tree_plot.pdf")
pdf(output_file, width=16, height=16)
par(mar = c(1, 5, 1, 1))
#plot(tree, cex=0.25, edge.width = 0.5, show.tip.label=FALSE)
plot.phylo(tree_intersect_labelled, show.tip.label=FALSE, no.margin=TRUE)
par(xpd=TRUE)
add.scale.bar(y=-3, lcol = "red", lwd = 4)
dev.off()

ggtree(tree_minus_outgroups)
axisPhylo()

table((naturalis_sample_labelled$species %in% tree_tips$tip_labels))

# Plotting the Geographical distribution of the sample

world_map_data <- map_data("world")
naturalis_tree_intersect_geo <- data.frame(long=naturalis_sample_tree_intersect$longitudeDecimal, lat=naturalis_sample_tree_intersect$latitudeDecimal)
naturalis_tree_intersect_country <- data.frame(region=naturalis_sample_tree_intersect$country)
# Making country names the same between datasets
naturalis_tree_intersect_country <- replace(naturalis_tree_intersect_country, naturalis_tree_intersect_country == "Unknown", NA)
naturalis_tree_intersect_country <- replace(naturalis_tree_intersect_country, naturalis_tree_intersect_country == "United States of America", "USA")
naturalis_tree_intersect_country <- replace(naturalis_tree_intersect_country, naturalis_tree_intersect_country == "United Kingdom", "UK")
naturalis_tree_intersect_country$region[grepl("Congo", naturalis_tree_intersect_country$region)] <- "Democratic Republic of the Congo"
naturalis_tree_intersect_country$region[grepl("Malaysia", naturalis_tree_intersect_country$region)] <- "Malaysia"
naturalis_tree_intersect_country$region[grepl("Azores", naturalis_tree_intersect_country$region)] <- "Azores"
naturalis_tree_intersect_country$region[grepl("Canary Islands", naturalis_tree_intersect_country$region)] <- "Canary Islands"
naturalis_tree_intersect_country$region[grepl("Palestinian", naturalis_tree_intersect_country$region)] <- "Palestine"
naturalis_tree_intersect_country$region[grepl("Hawaii", naturalis_tree_intersect_country$region)] <- "Hawaii"
naturalis_tree_intersect_country$region[grepl("Virgin Islands", naturalis_tree_intersect_country$region)] <- "Virgin Islands"
naturalis_tree_intersect_country$region[grepl("Burma", naturalis_tree_intersect_country$region)] <- "Myanmar"


scatter <- ggplot(naturalis_tree_intersect_geo, aes(x=long, y=lat)) +
  geom_point(stroke=NA, size=3, color="red", alpha=0.4)

map <- scatter +
  geom_map(data = world_map_data, map = world_map_data,
           aes(map_id = region), color = "black", fill = NA) +
  coord_fixed(1.3) +
  theme_bw() +
  theme(panel.background = element_rect(fill = "white")) +
  theme(panel.border=element_blank()) +
  theme(panel.grid=element_blank()) +
  labs(x=NULL, y=NULL)
map

ggsave(filename="sample_geographical_distribution1.png", plot=map, width=16, height=9, dpi=600)

country_freq <- data.frame(table(naturalis_tree_intersect_country$region))

gg <- ggplot()
gg <- gg + geom_map(data=world_map_data,
                    map=world_map_data,
                    aes(x=long, y=lat, map_id=region),
                    fill="lightgrey",
                    color="black")
gg <- gg + geom_map(data=country_freq,
                    map=world_map_data,
                    aes(map_id=Var1, fill=Freq),
                    color="black")
gg <- gg + scale_fill_viridis_c(direction=-1)
gg <- gg + theme_bw()
gg <- gg + labs(x=NULL, y=NULL)
gg <- gg + theme(panel.border=element_blank())
gg <- gg + theme(panel.grid=element_blank())
gg

ggsave(filename="/home/jmalone/Documents/Leaf-Project/Janssens_Data/sample_geographical_distribution2.png", plot=gg, width=16, height=9, dpi=600)

