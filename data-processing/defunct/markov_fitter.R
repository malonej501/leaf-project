library(msm)
library(dplyr)
library(tidyr)
library(tibble)
library(ggplot2)

# Loading and processing data
data = read.csv("../MUT2.2.csv")
data = read.csv("../MUT2.2_04-02-25.csv")
data = read.csv("../MUT1_05-02-25.csv")

# create state variable with shape as numbers
data$state = as.numeric(factor(data$shape, levels = c("u","l","d","c")))
# create id variable for each leaf for each walk
data$id = paste(data$leafid, data$walkid, sep="-")
# remove walk 3 for pc4 as only 1 step
data = data[data$id != "pc4-3", ]
data = data[order(data$walkid), ]
# Account for the first transition
rows_to_insert <- which(data$step == 0) # get row indices where step=0
new_data = list()
for(i in 1:nrow(data)) {
 if (i %in% rows_to_insert) {
   row <- data[i,]
   row$step <- -1 # create -1th step
   row$shape <- row$first_cat # the shape will be first_cat
   row$state <-as.numeric(factor(row$first_cat, levels = c("u","l","d","c"))) # make sure state agrees with shape
   new_data[[length(new_data) + 1]] <- row # append new row
 }
 new_data[[length(new_data) + 1]] <- data[i, , drop = FALSE] # append old rows
}
data <- do.call(rbind, new_data) # combine new rows into new data frame
# OPTIONAL: subset data to only include first 60 steps
data <- filter(data, step < 60)

# count-matrix - A frequency table with starting states as rows and finishing states as columns.
count_matrix = statetable.msm(shape, id, data=data) # subject is id - unique for each walk, rows

#data_step60 = subset(data, step < 60)
#write.csv(data, "MUT2.2_alt.csv", row.names=FALSE)


# define Q matix with all rates included in the search by setting to non-zero
Q <- rbind ( c(1, 1, 1, 1),
           + c(1, 1, 1, 1),
           + c(1, 1, 1, 1),
           + c(1, 1, 1, 1) )

#control = list(trace=1, REPORT=1)
# fit the model
data.msm <- msm(state ~ step, subject=id, data=data, qmatrix=Q, gen.inits=TRUE)
data.msm
# extract rate parameter MLE estimates
#Q.crude <- crudeinits.msm(state ~ step, id, data=data, qmatrix=Q)

# extract MLE rates and summary statistics
MLE_summary <- qmatrix.msm(data.msm)
Q_MLE <- data.frame(MLE_summary$estimates)
colnames(Q_MLE) <- c("u","l","d","c")
rownames(Q_MLE) <- c("u","l","d","c")
Q_MLE <- rownames_to_column(Q_MLE, var="initial_shape")
Q_MLE_long <- pivot_longer(Q_MLE, cols=2:5, names_to="final_shape", values_to="rate_msm")
# add standard error, upper and lower bounds
Q_MLE_long$SE <- pivot_longer(data.frame(MLE_summary$SE), cols=1:4)$value
Q_MLE_long$LB <- pivot_longer(data.frame(MLE_summary$L), cols=1:4)$value
Q_MLE_long$UB <- pivot_longer(data.frame(MLE_summary$U), cols=1:4)$value
# add transition column
Q_MLE_long$transition <- paste(Q_MLE_long$initial_shape, Q_MLE_long$final_shape, sep="")
# remove diagonal elements
Q_MLE_long <- Q_MLE_long[!(Q_MLE_long$transition %in% c("uu","ll","dd","cc")), ]
Q_MLE_long$rate_normalised <- Q_MLE_long$rate_msm / max(Q_MLE_long$rate_msm)
write.csv(Q_MLE_long,"MUT2.2_MLE_rates.csv", row.names=FALSE)

ggplot(Q_MLE_long, aes(x = transition, y = rate_msm)) +
  geom_bar(stat="identity")+
  geom_errorbar(aes(ymin=LB, ymax=UB, width=0.2)) +
  labs(x = "Transition", y = "MLE Rate")

# plot.prevalence.msm(data.msm)
# plot(data.msm)
# plotprog.msm(data.msm)


# combine msm rates with emcee rates for comparison
emcee <- read.csv("../markov_fitter_reports/emcee/ML_MUT2_mcmc_08-12-24.csv",header=FALSE)
emcee <- data.frame(t(emcee))
Q_MLE_long$rate_emcee <- emcee[,2] # N.B. the SE
print(Q_MLE_long, width=Inf)
Q_MLE_long <- pivot_longer(Q_MLE_long, cols=c(rate_msm, rate_emcee), names_to="method", values_to="rate")
print(Q_MLE_long, n=Inf, width=Inf)

p <- ggplot(Q_MLE_long, aes(x = transition, y = rate, fill = method)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Transition", y = "Rate", fill = "Method") +
  theme_minimal()

ggsave("msm_emcee_comparison_barplot.pdf")