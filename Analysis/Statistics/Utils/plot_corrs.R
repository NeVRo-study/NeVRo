
library(tidyverse)
library(here)
library(rstatix)


path_tabs <- here('Results', 'Tables')

get_tab <- function(cond_str) {
  
  tab_supp <- list.files(path_tabs, pattern = str_c('_', cond_str, '_supp'))
  tab_orig <- list.files(path_tabs, pattern = str_c('_', cond_str, '.csv'))
  
  fname <- file.path(path_tabs, tab_supp)
  df_supp <- read_csv(fname)
  fname <- file.path(path_tabs, tab_orig)
  df_orig <- read_csv(fname)
  
  df <- left_join(df_orig, df_supp)
  return(df)
}


plt_corr <- function(df, v1_str, v2_str) {
  v1 <- df[[v1_str]]
  v2 <- df[[v2_str]]
  res <- cor.test(v1, v2)
  movcond <- ifelse(length(v1) == 19, 'mov', 'nomov')
  txt <- sprintf('  r = %.2f\n  p = %.3f\n  %s', res$estimate, res$p.value, movcond)
  data <- tibble(x = v1, y = v2)
  plt <- ggplot(data, aes(x = x, y = y)) + 
    geom_point() + 
    geom_smooth(method = 'lm') + 
    annotate(geom = "text", x = -Inf, y = Inf, label = txt, hjust = 0, vjust = 1) + 
    theme_classic() +
    xlab(v1_str) +
    ylab(v2_str)
  return(plt)
}


df_ <- get_tab('nomov')


plt_corr(df_, 'SPOC_CORR', 'SA_SPOC_CORR')
plt_corr(df_, 'SPOC_LAMBDA', 'SA_SPOC_LAMBDA')
plt_corr(df_, 'SPOC_LAMBDA', 'SPOC_CORR')
plt_corr(df_, 'SA_SPOC_LAMBDA', 'SA_SPOC_CORR')

plt_corr(df_, 'SA_BLOCK_CSP_auc', 'SA_SPOC_CORR')
plt_corr(df_, 'SA_BLOCK_CSP_auc', 'SA_SPOC_LAMBDA')
plt_corr(df_, 'CSP_acc', 'SPOC_LAMBDA')
plt_corr(df_, 'CSP_acc', 'SPOC_CORR')



## ANOVA: mov_cond x break_cond

df_nomov <- get_tab('nomov')
df_mov <- get_tab('mov')

# CSP:
df_anova_csp <- bind_rows(nomov = df_nomov, mov = df_mov, .id = 'movcond') %>% 
  select(matches('CSP|movcond|Subject')) %>% 
  pivot_longer(cols = (matches('CSP_auc'))) %>% 
  separate(name, into = c('breakcond', NA, NA), sep = '_')

res.aov <- anova_test(df_anova_csp, 
                      dv = value, 
                      wid = Subject, 
                      within = c(movcond, breakcond))
get_anova_table(res.aov)

# SPOC:
df_anova_spoc <- bind_rows(nomov = df_nomov, mov = df_mov, .id = 'movcond') %>% 
  select(matches('SPOC|movcond|Subject')) %>% 
  pivot_longer(cols = (matches('_CORR'))) %>% 
  separate(name, into = c('breakcond', NA, NA), sep = '_')

res.aov <- anova_test(df_anova_spoc, 
                      dv = value, 
                      wid = Subject, 
                      within = c(movcond, breakcond))
get_anova_table(res.aov)


## Post-hoc tests: 
# nomov:
t.test(df_nomov$SPOC_CORR, 
       df_nomov$SA_SPOC_CORR, 
       paired = T, 
       alternative = "two.sided")
t.test(df_nomov$SBA_BLOCK_CSP_auc, 
       df_nomov$SA_BLOCK_CSP_auc, 
       paired = T, 
       alternative = "two.sided")

# mov:
t.test(df_mov$SPOC_CORR, 
       df_mov$SA_SPOC_CORR, 
       paired = T, 
       alternative = "two.sided")
t.test(df_mov$SBA_BLOCK_CSP_auc, 
       df_mov$SA_BLOCK_CSP_auc, 
       paired = T, 
       alternative = "two.sided")


