################################################
# Analyse HBP to compute Interoceptive Awareness
################################################

### 1. Create dataframe score (contains IAscores for each interval, as well as mean IAscore)
## 1.A. Load heartbeat perception (hbp) data
# of counted heartbeats (from questionnaire) -> create dataframe hbpc
# of ECG data of recorded heartbeats (Kubios) -> create list hbp_exg2_peaks_s_list

#--------------------------------------------------------------------------
# a) load behavioural data of counted heartbeats (hbpc)

# load package
library("R.matlab")
library("BioPhysConnectoR")


setwd("../../../Data/HBP/")
getwd()

nSub = 45
hbpc_all <- read.csv(file = "HBP_data.csv", header=TRUE, sep = ';')[0:nSub, 0:(1+5+5)]
rmv = c()  # fill number if to be removed
if (length(rmv) > 0){
  hbpc <- hbpc_all[-rmv,]
  } else {
    hbpc <- hbpc_all
  }

sub_list = hbpc[, 1]
hbpc <- hbpc[,-1] # remove first column (NeVRo_ID)
hbpc_conf <- hbpc[, 6:10] # assessment of counting
hbpc <- hbpc[, 1:5]


#--------------------------------------------------------------------------
# b) load Rpeaks from hbp Kubios files (whole hbp task from start trigger - end trigger) into R

hbp_exg2_peaks_s <- NULL
hbp_exg2_peaks_s_list <- NULL

for (part in sub_list[1:length(sub_list)]) {

 hbp_exg2_peaks_tmp <- NULL
 hbp_exg2_peaks_tmp <- readMat(paste('./HBP_Kubios_Start_End/NVR_S', sprintf("%02d", part), '_HBP_Start_End_hrv.mat', sep = ''))
 
 hbp_exg2_peaks_s <- as.vector(hbp_exg2_peaks_tmp$Res[[4]][[2]][[2]])  # occurrence of R Peaks in sec

 hbp_exg2_peaks_s_list[[part]] <- hbp_exg2_peaks_s
}

# #--------------------------------------------------------------------------
# ### 1.B. Define intervals
#
# ##Define intervals for recorded heartbeats, based on mp3-file:

# Calculated Timepoints (sec) of markers (see python-script: ECG_recrop.py):
# marker_list = c(0., 25.142, 38.244, 83.346, 96.448, 111.548, 124.65 , 179.753,  192.854)

# Hence: 
# # 1.	~25s (0 - 25.142)         | Stella: 25s (10-35)
# # 2.	~45s (38.244 - 83.346)    | Stella: 45s (48-93)
# # 3.	~15s (96.448 - 111.548)   | Stella: 15s (108-123)
# # 4.	~55s (124.65 - 179.753)   | Stella: 55s (136-191)
# # 5.	 ..s (192.854 - [end])    | Stella: 35s (206-241)


# ## Count sum of recorded Rpeaks for each interval -> create dataframe hbpr

hbpr <- data.frame()

for (part in sub_list[1:length(sub_list)]) {
  hbp_exg2_peaks_s <- hbp_exg2_peaks_s_list[[part]]
  hbpr_cur <- NULL
  hbpr_cur$hbpr_1 <- length(which(hbp_exg2_peaks_s <= 25.142))  # create new column for first interval: count sum of peaks
  hbpr_cur$hbpr_2 <- length(which((hbp_exg2_peaks_s > 38.244) & (hbp_exg2_peaks_s <= 83.346)))  # second
  hbpr_cur$hbpr_3 <- length(which((hbp_exg2_peaks_s > 96.448) & (hbp_exg2_peaks_s <= 111.548)))  # third
  hbpr_cur$hbpr_4 <- length(which((hbp_exg2_peaks_s > 124.65) & (hbp_exg2_peaks_s <= 179.753)))  # fourth
  hbpr_cur$hbpr_5 <- length(which((hbp_exg2_peaks_s > 192.854)))  # fifth
  hbpr <- rbind(hbpr, hbpr_cur)
}

## Check following subects !! Noise in Full-Length (Start-End) HBP-ECG:
#     S08: Noise from 00:26 to 0:36sec        > Ok: was during writing phase
#     S11: Noise from 03:29 (=209sec) to end  > Set on NA !!
hbpr[11, 5] = NA

# Create joint dataframe hbp for both counted and recorded Rpeaks
hbp <- data.frame()
hbp <- cbind(hbpc, hbpr)

# #--------------------------------------------------------------------------
# ### 1.C. Calculate IA score
#
# # IA score is calculated according to following formula (Schandry, 1981):
# #
# # $$ IA score = 1 / N Σ (1 -(|hbr - hbc| / hbr)) $$
# #
# # * **N** = number of intervals
# # * **hbc** = heartbeats counted
# # * **hbr** = heartbeats recorded
# #
# IA can range between 0 - 1 -> Higher IA scores indicate higher accuracy of the participants in counting their heartbeats

# create table with scores for each interval
score <- data.frame(row.names=1:nSub) # empty dataframe for loop


for (i in seq_along(hbpc)) {    # loop over columns of hbpc (=over each interval)
  score_cur <- (1-((abs(hbpr[i]-hbpc[i]))/hbpr[i])) # calculate score for each interval according to formula
  score <- cbind(score, score_cur) # create dataframe for all intervals
  score_cur <- NULL
}
colnames(score) <- c("score_1", "score_2", "score_3", "score_4", "score_5")


# calculate IA_score (for all intervals)
for (i in 1:length(score$score_1)) { # loop over rows in df score
  if(sum(is.na(score[i,])) <= 1) { # run only for rows where number of NAs <= 1
  score$IA_score[[i]] <- rowMeans(score[i,], na.rm = T)  #calculate mean for each row (all intervals per inc)
  } else {
  score$IA_score[[i]] <- NA # for rows with more than 1 NA value -> set IA_score = NA
  }
}

# Add average certainty row
score$m_confindence = rowMeans(hbpc_conf, na.rm = T) 

# create column sub_list
score$vp <- sub_list



# save score
save(score, file = "NeVRo_IA_score.RData") # save score
write.csv(score,  file = "NeVRo_IA_score.csv") # save score


# Quick and dirty plots: 
hist(score$IA_score, 20, col="lightgreen")
plot(score$IA_score)

score_IA_sort = mat.sort(mat=score, sort=6)
plot(score_IA_sort$m_confindence/7, ylim=c(0,1), ylab="IA score | confidence")
points(score_IA_sort$IA_score, col="red")

# TODO 
# ... plot(HR) in  plot(sort(score$IA_score))


# # --------------------------------------------------------------------------
# 
# ####################################
# ## 2. Calculate IA for sub_list
# 
# ### 2.A. Load function flex_medsplit
# 
# # * load flexible function: `flex_medsplit`: compute medsplit
# # * df: dataframe, containing one column to compute median split on, one column "vp"
# # * colname: column name of dataframe, containing values to calculate median split
# # * list: sub_list or subset of sub_list
# # * plot: scatterplot median split
# # * output: dataframe df with added column "medsplit"
# source(paste(path_script,"flex_medsplit.R", sep="")) # load flex_medsplit function

#--------------------------------------------------------------------------
### 2.B. median split IA for inc_clean

# a) apply function flex_medsplit
score_clean <- flex_medsplit(score, colname = "IA_score", inc_clean, plot = T)
split <- split(score_clean$vp, score_clean$medsplit)

# b) create sub_lists based on medsplit
inc_IAlow_clean <- split$below
inc_IAhigh_clean <- split$above

# c) Check if equal number in inc_IAhigh vs. inc_IAlow
if (length(inc_IAhigh_clean) != length(inc_IAlow_clean)) {
  message("groups are not equal")
}

#--------------------------------------------------------------------------
### 2.C. median split IA for inc_clean_recall_cutSignmem_ray

# a) apply function flex_medsplit
score_clean_mem <- flex_medsplit(score, colname = "IA_score", inc_clean_recall_cutSignmem_ray, plot = T)
split_mem <- split(score_clean_mem$vp, score_clean_mem$medsplit)

# b) Create inc subsgroups based on medsplits
inc_IAhigh_clean_mem <- split_mem$above
inc_IAlow_clean_mem <- split_mem$below

# c) Check if equal number in inc_IAhigh vs. inc_IAlow
if (length(inc_IAhigh_clean_mem) != length(inc_IAlow_clean_mem)) {
  message("groups are not equal")
}
#--------------------------------------------------------------------------
