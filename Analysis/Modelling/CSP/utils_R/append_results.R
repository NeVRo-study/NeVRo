
## Append CSP results to table with results of all methods.


library(here)
library(plotrix)
library(stringr)
library(tidyverse)

# Which movement condition(s):
mov_conds <- c('mov', 'nomov') #'nomov', 

# overwrite original file?
overwrite <- 'true'

for (mov_cond in mov_conds) {
  datCSP_path <- here('Data',
                      'EEG', 
                      '08.3_CSP_10f', 
                      mov_cond, 'SBA', 
                      'summaries')
  fname_in <- '_summary.csv'
  fname_in <- '_summary.csv'
  file_in <- file.path(datCSP_path, fname_in)
  
  tab_path <- here('Results')
  tabname_in <- str_c('results_across_methods_', mov_cond, '.csv')
  tab_in <- file.path(tab_path, tabname_in)
  if (overwrite) {
    str2add <- ''
  } else {
    str2add <- '_'
  }
  
  tabname_out <- str_c(str2add, tabname_in)
  tab_out <- file.path(tab_path, tabname_out)
  
  datCSP <- read_csv(file_in)
  datCSP <- rename(datCSP, Subject = ID)
  
  fulltable <- read_csv2(tab_in)
  fulltable %>% 
    left_join(datCSP, by = c('Subject')) %>%
    mutate(CSP = round(acc, digits = 5), 
           acc = NULL, 
           LSTM = as.numeric(LSTM)) %>%
    
    write_csv2(tab_out) 
  
  print('################')
  print(str_c('Results --- ', mov_cond))
  print('################')
  print(str_c('Mean acc: ', mean(datCSP$acc, na.rm = T)))  
  print(str_c('StdErr acc: ', std.error(datCSP$acc, na.rm = T)))  
  print(str_c('Min acc: ', min(datCSP$acc, na.rm = T)))  
  print(str_c('Max acc: ', max(datCSP$acc, na.rm = T)))  
  
}
# 
# 

# meanAcc_m <- mean(dat_m$acc, na.rm = T)
# sdAcc_m <- sd(dat_m$acc, na.rm = T)
# seAcc_m <- std.error(dat_m$acc, na.rm = T)
# 
# dat_nm <- read.csv(dat_path_nm)
# meanAcc_nm <- mean(dat_nm$acc, na.rm = T)
# sdAcc_nm <- sd(dat_nm$acc, na.rm = T)
# seAcc_nm <- std.error(dat_nm$acc, na.rm = T)
# 
# sdAcc_nm/sqrt(sum(!is.na(dat_nm$acc)))
# 
# binom.test(round(meanAcc_m*180*0.1), 180*0.1, alternative = "greater")
# 
# 
#   
#   

