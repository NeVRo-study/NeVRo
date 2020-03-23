# Participant Statistics

# getwd()
require(readxl)

table_subjects = read_xlsx("../../../Data/Table_of_Subjects.xlsx")
table_condition = read.csv2("../../../Data/Table_of_Conditions.csv")
len_table = length(table_subjects$`SUBJECT INFO`)
(n_sub = as.integer(table_subjects$SUBJECT[len_table-1]))

# Age
age = as.integer(table_subjects$`SUBJECT INFO`[4:len_table-1])
(sum_age = summary(age))  # describe(age)
(sd_age = sd(age, na.rm=T))

# Gender
gender = as.integer(table_subjects$X__1[4:len_table-1])
table(gender) # 1=female, 2=male
(n_female = table(gender)[1])
(n_male = table(gender)[2])

# Per Condition 
table(table_condition[,2:3])
  

####  Permutation test finaly results
library("perm")
require(perm)
# x <- c(12.6, 11.4, 13.2, 11.2, 9.4, 12.0)
# y <- c(16.4, 14.1, 13.4, 15.4, 14.0, 11.3)
# permTS(x,y, alternative="two.sided", method="exact.mc", control=permControl(nmc=30000))$p.value

WD = "/Users/Zimon/Desktop/Research Project II/Analysis/NeVRo_LSTM/LSTM/Random_Search_Tables/"

BiCl_SSD <- NaN 
BiCl_SPOC <- NaN 
Reg_SSD <- NaN 
Reg_SPOC <- NaN 

input_datatype = readline("Input data is 'SSD' or 'SPOC': ")

stopifnot(is.element(tolower(input_datatype), c("ssd", "spoc")))

if (tolower(input_datatype)=="ssd"){
  dtyp = ""
}else{
  dtyp = "_SPOC"
}
print(paste("dtyp is:", dtyp))

### Binary Classificatoin
## Best accuracy across subject irrespective of hyperparameter set
{
setwd(paste(WD, "Random_Search_Final_Table", dtyp, "/per_subject/", sep=""))  
accs <- read.csv2(paste("AllSub_Random_Search_Final_Table_BiCl", dtyp,".csv", sep=""))
best_accs = accs[-length(accs[,3]), 3]
best_accs = data.frame(best_accs)
best_accs = best_accs[!best_accs=="nan"]  # remove nan
best_accs = as.numeric(best_accs)  # transform into float

random_all_subjects_best = c()
for (i in 1:length(best_accs)){
  choose_best = c()
  for (j in 1:20){
    choose_best = append(choose_best, mean(sample(c(0, 1), 10*27, replace = T)))
  }
  random_all_subjects_best = append(random_all_subjects_best, max(choose_best))
}

# "p-value"
(p_above_chance = permTS(best_accs, random_all_subjects_best, alternative="greater", method="exact.mc", control=permControl(nmc=3000))$p.value)  #  p.conf.level=.99
print(paste("permut p-value:", p_above_chance))
# (p_above_chance = permTS(best_accs, random_all_subjects_best, alternative="greater", method="exact.mc", control=permControl(nmc=19))$p.value)  #  p.value = 0.05
print(paste("mean:", mean(best_accs)))
print(paste("SD:", sd(best_accs)))
print(paste("min:", min(best_accs)))
print(paste("max:", max(best_accs)))


# Save in variables 
if (tolower(input_datatype)=="ssd"){
  BiCl_SSD = best_accs
}else{
  BiCl_SPOC = best_accs
}

}

## Best hyperparameter sets across subjects
{
setwd(paste(WD, "Random_Search_Final_Table", dtyp,"/per_hp_set/", sep=""))  
besthpsets <- data.frame(read.csv2(paste("AllHPsets_Random_Search_Final_Table_BiCl", dtyp,".csv", sep="")))
n_sets <- length(besthpsets[, 1])
besthpsets <- besthpsets[1:6, 2]
# besthpsets <- besthpsets[1:n_sets, 2]

random_all_hp_best <- c()
for (i in 1:34){
  choose_best <- c()
  for (j in 1:n_sets){
    choose_best <- append(choose_best, mean(sample(c(0, 1), 10*27, replace = T)))
  }
  random_all_hp_best <- append(random_all_hp_best, mean(choose_best))
}

for (file_name in dir()){
  if (paste("AllHPsets_Random_Search_Final_Table_BiCl", dtyp, ".csv", sep="") == file_name){
    # do nothing
  }else{
    curr_file <- data.frame(read.csv2(file_name)[, c(24, 27)])
    if (is.element(curr_file[1, 1], besthpsets)){
      hpsetup <- curr_file[1,1]
      curr_file <- curr_file[, 2]
      curr_file <- (curr_file[curr_file!="nan"])
      curr_file <- as.numeric(as.character(curr_file))  # transform into float
      
      # "p-value"
      print(paste("permuted p-value of", file_name))
      print(paste("hp-setting", hpsetup))
      print(p_above_chance <- permTS(curr_file, random_all_hp_best, alternative="greater", method="exact.mc", control=permControl(nmc=3000, p.conf.level=.99))$p.value)
      #print("t.test")
      #print(paste("p-value:", t.test(curr_file, random_all_hp_best, alternative="greater")$p.value))
      #print(t.test(curr_file, random_all_hp_best, alternative="greater"))
      print(paste("set-mean:", mean(curr_file)))
      print(paste("set-sd:", sd(curr_file)))
      print("- - - - - - - - - - - - - - - -")
    }
  }
}
}

### Regression
## Best accuracy across subject irrespective of hyperparameter set
{
setwd(paste(WD, "Random_Search_Final_Table_Reg", dtyp,"/per_subject", sep=""))  
accs <- read.csv2(paste("AllSub_Random_Search_Final_Table_Reg", dtyp,".csv", sep=""))

best_accs = accs[-length(accs[, 3]), 3]
best_accs = data.frame(best_accs)
best_accs = best_accs[!best_accs=="nan"]  # remove nan
best_accs = as.numeric(best_accs)  # transform into float

zeroline_accs = accs[-length(accs[, 4]), 4]
zeroline_accs = data.frame(zeroline_accs)
zeroline_accs = zeroline_accs[!zeroline_accs=="nan"]  # remove nan
zeroline_accs = as.numeric(zeroline_accs)

# "p-value"
print("permuted p-value:")
p_above_chance = permTS(best_accs, zeroline_accs, alternative="greater", method="exact.mc", control=permControl(nmc=3000))$p.value
print(p_above_chance)
# (p_above_chance_t = t.test(best_accs, zeroline_accs, alternative="greater"))
print(paste("mean reg_val:", mean(best_accs)))
print(paste("SD reg_val:", sd(best_accs)))
print(paste("min reg_val:", min(best_accs)))
print(paste("max reg_val:", max(best_accs)))
print(paste("mean zeroline:", mean(zeroline_accs)))
print(paste("SD zeroline:", sd(zeroline_accs)))
print(paste("min zeroline:", min(zeroline_accs)))
print(paste("max zeroline:", max(zeroline_accs)))
print(paste("Difference reg_val-zeroline:", mean(best_accs)-mean(zeroline_accs)))
print(paste("SD Difference:", sd(best_accs-zeroline_accs)))
print(paste("min Difference:", min(best_accs-zeroline_accs)))
print(paste("max Difference:", max(best_accs-zeroline_accs)))


# Save in variables
if (tolower(input_datatype)=="ssd"){
  Reg_SSD = best_accs-zeroline_accs
}else{
  Reg_SPOC  = best_accs-zeroline_accs
}

}

## Best hyperparameter sets across subjects
{
  setwd(paste(WD, "Random_Search_Final_Table_Reg", dtyp,"/per_hp_set/", sep=""))  
  besthpsets <- data.frame(read.csv2(paste("AllHPsets_Random_Search_Final_Table_Reg", dtyp,".csv", sep="")))
  n_sets <- length(besthpsets[, 1])
  besthpsets <- besthpsets[1:6, 2]
  # besthpsets <- besthpsets[1:n_sets, 2]
  
  random_all_hp_best <- c()
  for (i in 1:34){
    choose_best <- c()
    for (j in 1:n_sets){
      choose_best <- append(choose_best, mean(sample(c(0, 1), 10*27, replace = T)))
    }
    random_all_hp_best <- append(random_all_hp_best, mean(choose_best))
  }
  
  for (file_name in dir()){
    if (paste("AllHPsets_Random_Search_Final_Table_Reg", dtyp, ".csv", sep="") == file_name){
      # do nothing
    }else{
      curr_file <- data.frame(read.csv2(file_name)[, c(24, 25, 26, 27)])
      if (is.element(curr_file[1, 1], besthpsets)){
        hpsetup <- curr_file[1,1]
        curr_file <- (curr_file[curr_file[, 2]!="nan", ])
        mean_val_acc <- as.numeric(as.character(curr_file[, 2]))
        zeroline_acc <- as.numeric(as.character(curr_file[, 3]))
        meanval_zeroline_acc <- as.numeric(as.character(curr_file[, 4]))
        
        # "p-value"
        print(paste("permuted p-value of", file_name))
        print(paste("hp-setting", hpsetup))
        print(p_above_chance <- permTS(mean_val_acc, zeroline_acc, alternative="greater", method="exact.mc", control=permControl(nmc=3000, p.conf.level=.99))$p.value)
        print("t.test")
        #print(paste("p-value:", t.test(curr_file, random_all_hp_best, alternative="greater")$p.value))
        #print(t.test(mean_val_acc, zeroline_acc, alternative="greater"))
        print(paste("set-mean:", mean(meanval_zeroline_acc)))
        print(paste("set-sd:", sd(meanval_zeroline_acc)))
        print("- - - - - - - - - - - - - - - -")
      }
    }
  }
}

### Test Diff SSD vs. SPoC
## Binary Classification
t.test(BiCl_SSD, BiCl_SPOC)
(p_above_chance = permTS(BiCl_SSD, BiCl_SPOC, alternative="two.sided", method="exact.mc", control=permControl(nmc=3000))$p.value)

## Regression
t.test(Reg_SSD, Reg_SPOC)
(p_above_chance = permTS(Reg_SSD, Reg_SPOC, alternative="two.sided", method="exact.mc", control=permControl(nmc=3000))$p.value)
