#### Probability of Direction test R ####
library(dplyr)
library(bayestestR)
phylo_sim_long = read.csv("phylo_sim_long.csv")

plot_order <- c(
  "MUT1_simulation",
  "MUT2_simulation",
  # "jan_phylo_nat_class_uniform0-0.1_1",
  # "zuntini_phylo_nat_class_10-09-24_genera_class_uniform0-0.1_2",
  # "geeta_phylo_geeta_class_uniform0-100_4",
  "jan_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1",
  "zun_genus_phylo_nat_26-09-24_class_uniform0-0.1_genus_1",
  "geeta_phylo_geeta_class_uniform0-100_genus_1"
)

transitions <- c(
  "l→u-u→l",
  "d→u-u→d",
  "c→u-u→c",
  "l→d-d→l",
  "l→c-c→l",
  "d→c-c→d"
)

results <- data.frame()

# Iterate through each transition
for (transition in transitions) {
  mcmc_plot_data <- phylo_sim_long %>% filter(transition == !!transition)

  for (name in unique(mcmc_plot_data$Dataset)) {
    group <- mcmc_plot_data %>% filter(Dataset == !!name)
    n <- nrow(group)
    ng0 <- nrow(filter(group, rate_norm > 0))
    
    mean <- mean(group$rate_norm, na.rm = TRUE)
    # t_test <- t.test(group$rate_norm, mu = 0)
    std <- sd(group$rate_norm, na.rm = TRUE)
    p_direction <- p_direction(group$rate_norm)
    p_significance <- p_significance(group$rate_norm, threshold = 0)
    
    results <- rbind(results, data.frame(
      dataset = name,
      transition = transition,
      mean_rate_norm = mean,
      # t_stat = t_test$statistic,
      # p_val = t_test$p.value,
      std = std,
      lb = mean - std,
      ub = mean + std,
      n = n,
      prop_over_zero = ng0 / n,
      p_direction = p_direction$pd,
      p_significance = p_significance$ps
      
    ))
  }
}

print(results)

write.csv(results, "p_direction_p_significance.csv")
