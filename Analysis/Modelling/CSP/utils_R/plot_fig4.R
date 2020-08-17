
## Contains the functions that each plot a subfigures of Figure 4 
## in the paper.

# 2020 Felix Klotzsche



plot_fig4_csp <- function(df_full, plt, path_out) {

  ## Plot the 2x2 topoplot subfigure in Figure 4 in the paper. 
  ## Takes the data frame w/ all normalized (!) pattern weights, 
  ## filters out the CSP relevant parts, claculates the mean of the
  ## absolute values (across subjects) and plots these into the 4
  ## topoplots sorted by movement condition (mov vs nomov, columns)
  ## and whether the patterns corresponds to the filter which maximizes 
  ## variance for episodes of low or high arousal. 
  
  ## Arguments:
  #
  # df_full: df with normalized (!) pattern weights, needs following columns: 
  #              - ID
  #              - Condition
  #              - electrode
  #              - CSP_max_norm
  #              - CSP_min_norm
  # plt: named list with plotting params ("palette", "palette_dir",
  #                                       "wid_heigth_ssd", "wid_heigth_csp", 
  #                                       "wid_heigth_spoc", "size_topo")
  # path_out: path to directory where the result shall be stored
  # save_png: boolean indicating whether the results sould be stored 
  #           (defaults to TRUE)
  #
  ## RETURNS:
  # plt_csp: grob containing the info used for saving the png
  
  
  
  arousal_labs <- c("Power max. for \nlow arousal", "Power max. for \nhigh arousal")
  names(arousal_labs) <- c("CSP_min_norm", "CSP_max_norm")
  
  # add electrode locations:
  df_full <- electrode_locations(df_full)
  
  
  
  f <- df_full %>% 
    mutate_at(.vars = vars(electrode, Condition), factor) %>% 
    select(ID, electrode, Condition, CSP_max_norm, CSP_min_norm) %>% 
    pivot_longer(c("CSP_max_norm", "CSP_min_norm"), 
                 names_to = "arousal_level",
                 values_to = "CSP") %>% 
    group_by(electrode, Condition, arousal_level) %>% 
    dplyr::summarise(CSP = mean(abs(CSP))) %>%
    electrode_locations() %>% 
    ungroup() %>% 
    mutate(arousal_level = recode(arousal_level, 
                                  !!!arousal_labs), 
           Condition = fct_reorder(Condition, 
                                   as.numeric(Condition), 
                                   .desc = T))
  
  
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
    theme(
          strip.text.x = element_text(size = 12, 
                                      margin = margin(b = 0.25, 
                                                      unit = 'cm')), 
          strip.text.y = element_text(size = 12, 
                                      margin = margin(r = 0.5, 
                                                      l = 0.5, 
                                                      unit = 'cm')),
          legend.position = 'None', 
          panel.grid = element_blank(),
          axis.line = element_blank(),
          panel.background = element_rect(fill = "transparent",
                                          colour = NA),
          plot.background = element_rect(fill = "transparent",
                                         colour = NA)) + 
    coord_equal() + 
    facet_grid(arousal_level~Condition) +
    labs(fill = expression(paste("Absolute weight (a.u.)"))) 
  
  
  # format printable:
  gt <- ggplot_gtable(ggplot_build(plt_csp))
  # fix abs. size of topoplots:
  gt$heights[c(8,10)] <- unit(plt$size_topo, 'cm')
  gt$widths[c(7,5)] <- unit(plt$size_topo, 'cm')
  
  ## show layout (only un-comment for debugging):
  # gtable_show_layout(gt)
  
  h = grid::convertHeight(sum(gt$heights), "cm", TRUE)
  w = grid::convertWidth(sum(gt$widths), "cm", TRUE)
  
  
  fname = 'CSP_avg.png'
  ggsave(file = file.path(path_out, fname), 
         plot = gt, 
         device = 'png', 
         width = unit(w, 'cm'), #unit(plt$wid_heigth_csp[1], 'cm'), 
         height = unit(h, 'cm'), # unit(plt$wid_heigth_csp[2], 'cm'), 
         units = 'cm')
  print(str_glue('Saved {fname} to\n{path_out}.'))
  
  return(gt)
}



plot_fig4_spoc <- function(df_full, plt, path_out, save_png = TRUE) {
  
  ## Plot the 1x2 SPoC topoplot subfigure in Figure 4 in the paper. 
  ## Takes the data frame w/ all normalized (!) pattern weights, 
  ## filters out the SPoC relevant parts, calculates the mean of the
  ## absolute values (across subjects) and plots these into the 2
  ## topoplots sorted by movement condition (mov vs nomov). 
  
  ## ARGUMENTS:
  # df_full: df with normalized (!) pattern weights, needs following columns: 
  #              - ID
  #              - Condition
  #              - electrode
  #              - SPOC_norm
  # plt: named list with plotting params ("palette", "palette_dir",
  #                                       "wid_heigth_ssd", "wid_heigth_csp", 
  #                                       "wid_heigth_spoc", "size_topo")
  # path_out: path to directory where the result shall be stored
  # save_png: boolean indicating whether the results sould be stored 
  #           (defaults to TRUE)
  #
  ## RETURNS:
  # plt_spoc: grob containing the info used for saving the png
  
  
  
  ## Plot SPoC:
  f <- df_full %>% 
    mutate_at(.vars = vars(electrode, Condition), funs(factor)) %>%
    select(ID, electrode, Condition, SPOC_norm) %>% 
    group_by(ID, electrode, Condition) %>% 
    summarise(SPOC = mean(abs(SPOC_norm))) %>% 
    electrode_locations() %>% 
    ungroup() %>% 
    mutate(Condition = fct_reorder(Condition, 
                                   as.numeric(Condition), 
                                   .desc = T))
  
  
  
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
    scale_fill_viridis_c(breaks = c(0), 
                         labels = c('0'),
                         #limits = c(0.0,0.25),
                         guide = guide_colorbar(label = TRUE, 
                                                ticks = TRUE))+
    theme_void() +
    theme(
      strip.text.x = element_text(size = 12, 
                                  margin = margin(b = 0.25, 
                                                  unit = 'cm')), 
      strip.text.y = element_text(size = 12, 
                                  margin = margin(r = 0.5, 
                                                  l = 0.5, 
                                                  unit = 'cm')),
      legend.position = 'None', 
      panel.grid = element_blank(),
      axis.line = element_blank(),
      panel.background = element_rect(fill = "transparent",
                                      colour = NA),
      plot.background = element_rect(fill = "transparent",
                                     colour = NA)) + 
    
    coord_equal() + 
    facet_grid(~Condition) +
    labs(fill = expression(paste("Absolute weight (a.u.)"))) 
  
  # gtable_show_layout(gt)
  
  # format printable:
  gt <- ggplot_gtable(ggplot_build(plt_spoc))
  # fix size of topoplots:
  gt$heights[c(8)] <- unit(plt$size_topo, 'cm')
  gt$widths[c(5,7)] <- unit(plt$size_topo, 'cm')
  
  ## show layout (only un-comment for debugging):
  # gtable_show_layout(gt)
  
  h = grid::convertHeight(sum(gt$heights), "cm", TRUE)
  w = grid::convertWidth(sum(gt$widths), "cm", TRUE)
  
  
  if (write_out$spoc_avg) {
    fname = 'SPOC_avg.png'
    ggsave(file = file.path(path_out, fname), 
           plot = gt, 
           device = 'png', 
           width = unit(w, 'cm'),  
           height = unit(h, 'cm'),
           units = 'cm', 
           bg = 'transparent')
    
    print(str_glue('Saved {fname} to\n{path_out}.'))
  }
  return(gt)
}

