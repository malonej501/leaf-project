library(msm)
library(dplyr)
library(tidyr)
library(tibble)
library(ggplot2)

# Loading and processing data
data = read.csv("MUT2.2.csv")
# create state variable with shape as numbers
data$state = as.numeric(factor(data$shape, levels = c("u","l","d","c")))
# create id variable for each leaf for each walk
data$id = paste(data$leafid, data$walkid, sep="-")
# remove walk 3 for pc4 as only 1 step
data = data[data$id != "pc4-3", ]
data = data[order(data$walkid), ]

count_matrix = statetable.msm(shape, walkid, data=data)

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
print(MLE_summary$SE)
Q_MLE <- data.frame(MLE_summary$estimates)
colnames(Q_MLE) <- c("u","l","d","c")
rownames(Q_MLE) <- c("u","l","d","c")
Q_MLE <- rownames_to_column(Q_MLE, var="initial_shape")
Q_MLE_long <- pivot_longer(Q_MLE, cols=2:5, names_to="final_shape", values_to="rate")
# add standard error, upper and lower bounds
Q_MLE_long$SE <- pivot_longer(data.frame(MLE_summary$SE), cols=1:4)$value
Q_MLE_long$LB <- pivot_longer(data.frame(MLE_summary$L), cols=1:4)$value
Q_MLE_long$UB <- pivot_longer(data.frame(MLE_summary$U), cols=1:4)$value
# add transition column
Q_MLE_long$transition <- paste(Q_MLE_long$initial_shape, Q_MLE_long$final_shape, sep="")
# remove diagonal elements
Q_MLE_long <- Q_MLE_long[!(Q_MLE_long$transition %in% c("uu","ll","dd","cc")), ]
Q_MLE_long$rate_normalised <- Q_MLE_long$rate / max(Q_MLE_long$rate)
write.csv(Q_MLE_long,"MUT2.2_MLE_rates.csv", row.names=FALSE)

ggplot(Q_MLE_long, aes(x = transition, y = rate)) +
  geom_bar(stat="identity")+
  geom_errorbar(aes(ymin=LB, ymax=UB, width=0.2)) +
  labs(x = "Transition", y = "MLE Rate")

plot.prevalence.msm(data.msm)
plot(data.msm)
plotprog.msm(data.msm)
