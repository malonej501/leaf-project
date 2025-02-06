library(phytools)
library(ape)
library(dplyr)
library(tidyr)
library(ggplot2)

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

marginal_ancestral_states <- function(){

  tree <- read.nexus("phylo_data/trees_final/zun_genus_phylo_nat_class_26-09-24.tre")
  
  labs <- import_labels()
  labs <- labs[labs$dataset == "zun_genus_phylo_nat_class",]
  label_data_ordered <- data.frame("label" = tree$tip.label) # Create a data frame with the tree tip labels in the tree order
  label_data_ordered <- left_join(label_data_ordered, labs, by="label") # Join with the labels data frame to get the order
  rownames(label_data_ordered) <- label_data_ordered$label
  labs <- data.frame(select(label_data_ordered, shape))
  labs <- as.matrix(labs)[,1]
  
  fitER<-ace(labs,tree,model="ER",type="discrete") # fit CTMC
  round(fitER$lik.anc,3)
  
  cols = c("#D55E00","#029E73","#DE8F05","#0173B2")
  names(cols) = c("compound","dissected","lobed","unlobed")
  plotTree(tree,type="fan",fsize=0.1,ftype="i")
  nodelabels(node=1:tree$Nnode+Ntip(tree),
             pie=fitER$lik.anc,piecol=cols,cex=0.2,)
  tiplabels(pie=to.matrix(labs,sort(unique(labs))),piecol=cols,cex=0.2)
  add.simmap.legend(colors=cols,prompt=FALSE,x=0.9*par()$usr[1],
                    y=-max(nodeHeights(tree)),fsize=0.8)
}

stochastic_character_map <- function(){
  
  tree <- read.nexus("phylo_data/trees_final/zun_genus_phylo_nat_class_26-09-24.tre")
  
  labs <- import_labels()
  labs <- labs[labs$dataset == "zun_genus_phylo_nat_class",]
  label_data_ordered <- data.frame("label" = tree$tip.label) # Create a data frame with the tree tip labels in the tree order
  label_data_ordered <- left_join(label_data_ordered, labs, by="label") # Join with the labels data frame to get the order
  rownames(label_data_ordered) <- label_data_ordered$label
  labs <- data.frame(select(label_data_ordered, shape))
  labs <- as.matrix(labs)[,1]
  
  mtrees <- make.simmap(tree,labs,model="ER",nsim=100)
  mtrees_ard <- make.simmap(tree,labs,model="ARD",nsim=100)
  
  # Plot average no. transitions
  pd <- summary(mtrees,plot=FALSE)
  pd
  
  pd_ard <- summary(mtrees_ard,plot=FALSE)
  pd_ard
  
  cols = c("#D55E00","#029E73","#DE8F05","#0173B2")
  names(cols) = c("compound","dissected","lobed","unlobed")
  plot(mtrees[[1]],cols,type="fan",fsize=0.2,ftype="i")
  nodelabels(pie=pd$ace,piecol=cols,cex=0.2)
  add.simmap.legend(colors=cols,prompt=FALSE,x=0.9*par()$usr[1],
                    y=-max(nodeHeights(tree)),fsize=0.8)
  
  pd$count # get no. each transition across different trees
  df <- as.data.frame(pd$count)
  df_long <- pivot_longer(df, cols=-N, names_to="transition", values_to="count")

  
  ggplot(df_long, aes(x=transition, y=count)) + 
    geom_violin()
}

fit_ape_ctmc <- function(){
  tree <- read.nexus("phylo_data/trees_final/zun_genus_phylo_nat_class_26-09-24.tre")
  
  labs <- import_labels()
  labs <- labs[labs$dataset == "zun_genus_phylo_nat_class",]
  label_data_ordered <- data.frame("label" = tree$tip.label) # Create a data frame with the tree tip labels in the tree order
  label_data_ordered <- left_join(label_data_ordered, labs, by="label") # Join with the labels data frame to get the order
  rownames(label_data_ordered) <- label_data_ordered$label
  labs <- data.frame(select(label_data_ordered, shape))
  labs <- as.matrix(labs)[,1]
  
  
}



