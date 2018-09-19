#--------------------------------------------------------------------------
# create function to compute medsplits:
# df: dataframe, containing one column to compute median split on, one column "vp"
# colname: column name of dataframe, containing values to calculate median split
# list: inc_list or subset of inc_list
# plot: scatterplot median split
# output: dataframe df with added column "medsplit"
# e.g.: df = score, list = inc_clean, colname = "IA_score"

# function steps:
# 1. df_subs: create subset of dataframe based on list (inc_list)
# 2. compute medsplit: 
#    a) vec: vector from df to compute medsplit on
#    b) medsplit: elements of vec cut into 2 halfs, labeled "below", "above"
# 3. Check if medsplit is correct: sum above = sum below?
# 4. Apply function medsplit based on new subset
# 5. plot medsplit (ggplot)
# 6. return dataframe with column medsplit

#--------------------------------------------------------------------------
# function flex_medsplit

flex_medsplit <- function(df, colname, list, plot = F) {
  # 1. apply function: create subset df_subs based on list
  df_subs <- df[(df$vp[c(list)]),] # create subset of df based on list

  # 2. compute medsplit 
    vec <- df_subs[[colname]] # create vector (column named colname in df) to compute medsplit on
    names(vec) <- df_subs$vp # label vec with incIDs
    
    median_vec <- median(vec, na.rm = T)  # calculate median of vector vec
    medsplit <- cut_number(vec, n = 2, na.rm = T, labels = c("below", "above")) # cut number of vector elements in equal groups, label them "above", "below"

    # 3. check if medsplit correct
    if (length(medsplit[medsplit == "below"]) != length(medsplit[medsplit == "above"])) {
      message("medsplit is not equal:", "Below:", sum(vec < median_vec, na.rm=T), "Above:", sum(vec >= median_vec, na.rm=T))
      medsplit[which(vec == median_vec)] <-  NA # exclude median_vec from medsplit factor
    }
  
  # 4. add column medsplit to df
  df_subs$medsplit <- medsplit 
  
  # 5. plot medsplit

  if (plot == T) {
   ggplot(data = df_subs, aes(x= vp, y = vec, colour = medsplit)) +
      geom_point(na.rm = T) +
      geom_abline(intercept = median_vec, slope = 0, colour = "blue") +
      ggtitle(paste("Mediansplit", colname, "for: ", deparse(substitute(list))))
  }
  
  # 6. return dataframe df_subs with column medsplit
  return(df_subs)
}