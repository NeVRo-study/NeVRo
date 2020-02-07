# Topographies ------------------------------------------------------------

# Based on: https://www.mattcraddock.com/blog/2017/02/25/erp-visualization-creating-topographical-scalp-maps-part-1/   
# thanks to Sandra Naumann

# Get libs:
library(tidyverse)
library(eegUtils)


# EMOTIONS      

# Load topography information       
Topo_Emo = read_csv(file="C:/Users/Felix/Downloads/ERPs_Topo_Emotions.csv", 
                    col_names = TRUE, 
                    n_max = 1)

# Re-name to fit topoplot function
names(Topo_Emo)[names(Topo_Emo) == "Time"] = "time"

# Change from wide to long format for electrodes 
Topo_Emo = gather(Topo_Emo, electrode, amplitude, Fp1:Oz, factor_key=TRUE)

# Rename A1/A2
names(Topo_Emo)[names(Topo_Emo) == "A1"] <- "TP9"
names(Topo_Emo)[names(Topo_Emo) == "A2"] <- "TP10"

# Plot topoplots for neutral
Topo_Emo_Neu = subset(Topo_Emo, Condition == 1)

# P1s
topoplot(Topo_Emo_Neu, time_lim = c(-1000, 1),interp_limit = "head", limits = c(-5,15))+
  ggtitle("P1 (80-120 ms)")+
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))+
  annotate(geom="text", x=-1.05, y=-1, label="neutral", size=3)
  