# Topographies ------------------------------------------------------------

# Based on: https://www.mattcraddock.com/blog/2017/02/25/erp-visualization-creating-topographical-scalp-maps-part-1/   
# thanks to Sandra Naumann

# Get libs:
library(tidyverse)
library(eegUtils)
library(R.matlab)
library(grid)
library(gtable)
library(here)

norm_vec <- function(x) sqrt(sum(x^2))

plt <- NULL
plt$palette <- 'viridis'
plt$palette_dir <- 1
plt$wid_heigth_ssd  <- c(20, 10) 
plt$wid_heigth_csp  <- c(10, 10)
plt$wid_heigth_spoc <- c(20, 10)
plt$size_topo <- 1.5
  
write_out <- NULL
write_out$csp_avg  <- TRUE
write_out$spoc_avg <- TRUE
write_out$ssd_avg  <- TRUE

path_data <- here('Results', 'Patterns')
path_out <- here('Results', 'Plots', 'Patterns', plt$palette)
if (!dir.exists(path_out)) dir.create(path_out)

data_conds <- NULL
data_ID <- NULL

for (cond in c('mov', 'nomov')) {
  files <- list.files(file.path(path_data, cond))
  for (ff in files) {
  
    dat <- read_csv(file.path(path_data, cond, ff))
    
    ID_str <- str_split(ff, '\\.')[[1]][1]
    topo_nvr <- dat %>% 
      mutate(ID = ID_str, 
             Condition = cond, 
             time = 1) %>% 
      select(ID, Condition, time, 
             electrode = Row, everything()) %>% 
      filter(!electrode %in% c('HEOG', 'VEOG')) %>% 
      mutate(CSP_max_norm = CSP_max/norm_vec(CSP_max), 
             CSP_min_norm = CSP_min/norm_vec(CSP_min), 
             SPOC_norm = SPOC/norm_vec(SPOC), 
             SSD_norm = SSD/norm_vec(SSD)) 
    
      data_ID[[ID_str]] <- topo_nvr
  
  }
  data_conds[[cond]] <- bind_rows(data_ID)
}

df_full <- bind_rows(data_conds)
  

arousal_labs <- c("Power max. for \nlow arousal", "Power max. for \nhigh arousal")
names(arousal_labs) <- c("CSP_min_norm", "CSP_max_norm")

# add electrode locations:
df_full <- electrode_locations(df_full)



f <- df_full %>% 
  mutate_at(.vars = vars(electrode, Condition), funs(factor)) %>% 
  select(ID, electrode, Condition, CSP_max_norm, CSP_min_norm) %>% 
  pivot_longer(c("CSP_max_norm", "CSP_min_norm"), 
               names_to = "arousal_level",
               values_to = "CSP") %>% 
  #filter(arousal_level == 'CSP_max_norm') %>% 
  group_by(electrode, Condition, arousal_level) %>% 
  dplyr::summarise(CSP = mean(abs(CSP))) %>%
  electrode_locations() %>% 
  ungroup() %>% 
  mutate(arousal_level = recode(arousal_level, !!!arousal_labs), 
         Condition = fct_reorder(Condition, as.numeric(Condition), .desc = T))


plt_csp <- ggplot(f,
               aes(x = x,
                   y = y,
                   fill = CSP,
                   label = electrode)) +
          geom_topo(grid_res = 200,
                    interpolate = T,
                    interp_limit = "head",
                    chan_markers = "point",
                    chan_size = 0.25, 
                    head_size = 1) + 
          scale_fill_viridis_c(breaks = c(0), 
                               labels = c('0'),
                               #limits = c(0.0,0.25),
                               guide = guide_colorbar(label = TRUE, 
                                                      ticks = TRUE))+
          # scale_fill_viridis_c() +
          # scale_fill_distiller(palette = plt$palette, 
          #                      direction = plt$palette_dir) + 
          theme_void() + 
          theme(#rect = element_rect(fill = "transparent"),
                strip.text.x = element_text(size = 12), #plt$wid_heigth_ssd[1] * 2.5), 
                strip.text.y = element_text(size = 12)) + #plt$wid_heigth_csp[1] * 2.5)) +
          theme(legend.position = 'None', 
                panel.grid = element_blank(),
                axis.line = element_blank(),
                panel.background = element_rect(fill = "transparent",colour = NA),
                plot.background = element_rect(fill = "transparent",colour = NA)) + 
          coord_equal() + 
          facet_grid(arousal_level~Condition) +
          labs(fill = expression(paste("Absolute weight (a.u.)"))) 

# gtable_show_layout(gt)

# format printable:
gt <- ggplot_gtable(ggplot_build(plt_csp))
# fix size of topoplots:
gt$heights[c(8,10)] <- unit(plt$size_topo, 'cm')
gt$widths[c(5,7)] <- unit(plt$size_topo, 'cm')

h = grid::convertHeight(sum(gt$heights), "cm", TRUE)
w = grid::convertWidth(sum(gt$widths), "cm", TRUE)


ggsave(file = file.path(path_out, 'CSP_avg.png'), 
       plot = gt, 
       device = 'png', 
       width = unit(w, 'cm'), #unit(plt$wid_heigth_csp[1], 'cm'), 
       height = unit(h, 'cm'), # unit(plt$wid_heigth_csp[2], 'cm'), 
       units = 'cm')



## Plot SPoC:
f <- df_full %>% 
  mutate_at(.vars = vars(electrode, Condition), funs(factor)) %>%
  select(ID, electrode, Condition, SPOC_norm) %>% 
  group_by(ID, electrode, Condition) %>% 
  summarise(SPOC = mean(abs(SPOC_norm))) %>% 
  electrode_locations() %>% 
  ungroup() %>% 
  mutate(Condition = fct_reorder(Condition, as.numeric(Condition), .desc = T))

  

plt_spoc <- ggplot(f,
               aes(x = x,
                   y = y,
                   fill = SPOC,
                   label = electrode)) +
          geom_topo(grid_res = 200,
                    interp_limit = "head",
                    chan_markers = "point",
                    chan_size = 0.25, 
                    head_size = 1) +
  scale_fill_viridis_c(#breaks = c(0), 
                       #labels = c('0'),
                       #limits = c(0,0.3),
                       guide = guide_colorbar(label = TRUE, 
                                              ticks = TRUE))+        
  #scale_fill_viridis_c() +
          # scale_fill_distiller(palette = plt$palette, 
          #                      direction = plt$palette_dir) + 
          theme_void() +
          theme(strip.text.x = element_text(size = plt$wid_heigth_spoc[1] * 2.5)) +
          theme(legend.position = 'None', 
            panel.grid = element_blank(),
            axis.line = element_blank(),
            panel.background = element_rect(fill = "transparent",colour = NA),
            plot.background = element_rect(fill = "transparent",colour = NA)) + 
          coord_equal() + 
          facet_grid(~Condition) +
          labs(fill = expression(paste("Absolute weight (a.u.)"))) 

# gtable_show_layout(gt)

# format printable:
gt <- ggplot_gtable(ggplot_build(plt_spoc))
# fix size of topoplots:
gt$heights[c(8)] <- unit(plt$size_topo, 'cm')
gt$widths[c(5,7)] <- unit(plt$size_topo, 'cm')

h = grid::convertHeight(sum(gt$heights), "cm", TRUE)
w = grid::convertWidth(sum(gt$widths), "cm", TRUE)


if (write_out$spoc_avg) {
  ggsave(file = file.path(path_out, 'SPOC_avg.png'), 
         plot = gt, 
         device = 'png', 
         width = unit(w, 'cm'), #unit(plt$wid_heigth_spoc[1], 'cm'), 
         height = unit(h, 'cm'), #unit(plt$wid_heigth_spoc[2], 'cm'), 
         units = 'cm', 
         bg = 'transparent')
}

## Plot SSD:
f <- df_full %>% 
  mutate_at(.vars = vars(electrode, Condition), funs(factor)) %>%
  select(ID, electrode, Condition, SSD_norm) %>% 
  group_by(electrode, Condition) %>% 
  summarise(SSD = mean(abs(SSD_norm))) %>% 
  electrode_locations() %>% 
  ungroup() %>% 
  mutate(Condition = fct_reorder(Condition, as.numeric(Condition), .desc = T))

  

plt_ssd <- ggplot(f,
       aes(x = x,
           y = y,
           fill = SSD,
           label = electrode)) +
  geom_topo(grid_res = 200,
            interp_limit = "head",
            chan_markers = "point",
            chan_size = 0.25, 
            head_size = 1) + 
  scale_fill_viridis_c(breaks = c(min(f$SSD)+0.01, max(f$SSD)), 
                       labels = c('min', 'max'),
                       #limits = c(0,0.3),
                       #begin = min(f$SSD)-0.001, 
                       #end = max(f$SSD),
                       guide = guide_colorbar(label = TRUE, 
                                              ticks = FALSE, 
                                              barheight = plt$wid_heigth_ssd[1] * 0.8))+
  #scale_fill_distiller(palette = plt$palette), 
                       #direction = plt$palette_dir) + 
  theme_void() + 
  theme(strip.text.x = element_text(size = plt$wid_heigth_ssd[1] * 2.5), 
        legend.title = element_text(size = plt$wid_heigth_ssd[1] * 2.5), 
        legend.text = element_text(size = plt$wid_heigth_ssd[1] * 2.5), 
        legend.box.spacing =unit(5,'cm')) +
  theme(legend.position = 'None', 
    panel.grid = element_blank(),
        axis.line = element_blank(),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA)) +
  coord_equal() + 
  facet_grid(~Condition) +
  labs(fill = expression(paste("Absolute\nweight\n(a.u.)"))) #+ 
  #theme(legend.position = 'none')

gtable_show_layout(gt)

# format printable:
gt <- ggplot_gtable(ggplot_build(plt_ssd))
# fix size of topoplots:
gt$heights[c(8)] <- unit(plt$size_topo, 'cm')
gt$widths[c(5,7)] <- unit(plt$size_topo, 'cm')

h = grid::convertHeight(sum(gt$heights), "cm", TRUE)
w = grid::convertWidth(sum(gt$widths), "cm", TRUE)


if (write_out$spoc_avg) {
  ggsave(file = file.path(path_out, 'SSD_avg.png'), 
         plot = gt, 
         device = 'png', 
         width = unit(w, 'cm'), #unit(plt$wid_heigth_spoc[1], 'cm'), 
         height = unit(h, 'cm'), #unit(plt$wid_heigth_spoc[2], 'cm'), 
         units = 'cm', 
         bg = 'transparent')
}


if (write_out$ssd_avg) {
  ggsave(file = file.path(path_out, 'SSD_avg.png'), 
         plot = plt_ssd, 
         device = 'png', 
         width = unit(plt$wid_heigth_ssd[1], 'cm'), 
         height = unit(plt$wid_heigth_ssd[2], 'cm'))
}

# plot_all:
f <- df_full %>% 
  select(ID, electrode, Condition, SSD_norm, SPOC_norm, CSP_max_norm, CSP_min_norm) %>% 
  pivot_longer(c(SSD_norm, SPOC_norm, CSP_max_norm, CSP_min_norm), 
               names_to = "Model",
               values_to = "pattern_weights") %>% 
  group_by(ID, electrode, Condition, Model) %>% 
  summarise(avg_pattern = mean(abs(pattern_weights))) %>% 
  electrode_locations() %>% 
  ungroup() 


for (cond in c('mov', 'nomov')) {

  plt_all <- ggplot(filter(f, Condition == cond),
                    aes(x = x,
                        y = y,
                        fill = avg_pattern,
                        label = electrode)) +
    geom_topo(grid_res = 200,
              interp_limit = "head",
              chan_markers = "point",
              chan_size = 1) + 
    scale_fill_distiller(palette = "YlOrRd", 
                         direction = 1) + 
    theme_void() + 
    coord_equal() + 
    facet_grid(ID~Model) +
    labs(fill = expression(paste("Absolute weight (a.u.)"))) 
  
  ggsave(file = file.path(path_data, paste0('_all_subjects_', cond, '.pdf')), 
         plot = plt_all, 
         device = 'pdf', 
         width = unit(10, 'cm'), 
         height = unit(40, 'cm'))
}


ggdraw() + 
  draw_plot(plt_ssd, x = 0, y = 0.75, width = 0.75, height = 0.25) + 
  draw_plot(plt_spoc, x = 0, y = 0.5, width = 0.75, height = 0.25) + 
  draw_plot(plt_csp, x = 0, y = 0, width = 1, height = 0.5) +
  align_plots()


aa <- plot_grid(plt_ssd, 
                plt_spoc,
                plt_csp, 
                ncol = 1, 
                align = 'v', 
                axis = 'l', 
                rel_heights = c(1,1,1.7), 
                #rel_widths = c(1,1,2),
                labels = c('SSD', 'SPoC', 'CSP')
                )
aa
# # EMOTIONS      
# 
# # Load topography information       
# Topo_Emo = read_csv(file="C:/Users/Felix/Downloads/ERPs_Topo_Emotions.csv", 
#                     col_names = TRUE, 
#                     n_max = 1)
# 
# # Re-name to fit topoplot function
# names(Topo_Emo)[names(Topo_Emo) == "Time"] = "time"
# 
# # Change from wide to long format for electrodes 
# Topo_Emo = gather(Topo_Emo, electrode, amplitude, Fp1:Oz, factor_key=TRUE)
# 
# # Rename A1/A2
# names(Topo_Emo)[names(Topo_Emo) == "A1"] <- "TP9"
# names(Topo_Emo)[names(Topo_Emo) == "A2"] <- "TP10"
# 
# # Plot topoplots for neutral
# Topo_Emo_Neu = subset(Topo_Emo, Condition == 1)
