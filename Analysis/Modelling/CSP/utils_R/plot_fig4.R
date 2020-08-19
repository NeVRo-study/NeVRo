
## Contains the functions that each plot a subfigures of Figure 4 
## in the paper.

# 2020 Felix Klotzsche



plot_fig4_csp <- function(df_full, plt, path_out, show_legend = FALSE) {

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
  
  
  
  arousal_labs <- c("power max. for \nlow arousal", "power max. for \nhigh arousal")
  names(arousal_labs) <- c("CSP_min_norm", "CSP_max_norm")
  
  # add electrode locations:
  df_full <- electrode_locations(df_full)
  
  # show legend?
  leg_pos <- ifelse(show_legend, 'bottom', 'None')
  
  
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
    scale_fill_viridis_c(breaks = c(0, 0.1, 0.2),
                         #labels = c('0'),
                         #limits = c(0.0,0.25),
                         guide = guide_colorbar(label = TRUE,
                                                ticks = FALSE))+
    # scale_fill_viridis_c(breaks = c(0.1),
    #                      guide = guide_colorbar(label = TRUE, 
    #                                             ticks = FALSE)) +
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
          #text = element_text(family = "Arial"),
          legend.position = leg_pos, 
          legend.box.margin = margin(25, 15, 15, 160),
          panel.grid = element_blank(),
          axis.line = element_blank(),
          panel.background = element_rect(fill = "transparent",
                                          colour = NA),
          plot.background = element_rect(fill = "transparent",
                                         colour = NA)) + 
    coord_equal() + 
    facet_grid(arousal_level~Condition) +
    guides(fill = guide_colorbar(title = expression(paste("absolute normalized\nweights (a.u.)")), 
                                 title.position = 'right', 
                                 ticks = FALSE, 
                                 barheight = 0.5)) 
  
  
  # format printable:
  gt <- ggplot_gtable(ggplot_build(plt_csp))
  # fix abs. size of topoplots:
  gt$heights[c(8,10)] <- unit(plt$size_topo, 'cm')
  gt$widths[c(7,5)] <- unit(plt$size_topo, 'cm')
  
  ## show layout (only un-comment for debugging):
  # gtable_show_layout(gt)
  
  h = grid::convertHeight(sum(gt$heights), "cm", TRUE)
  w = grid::convertWidth(sum(gt$widths), "cm", TRUE)
  
  
  fname = 'CSP_avg.pdf'
  ggsave(file = file.path(path_out, fname), 
         plot = gt, 
         #device = 'eps', 
         width = unit(w, 'cm'), #unit(plt$wid_heigth_csp[1], 'cm'), 
         height = unit(h, 'cm'), # unit(plt$wid_heigth_csp[2], 'cm'), 
         units = 'cm', 
         bg = 'transparent')
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
    mutate_at(.vars = vars(electrode, Condition), list(factor)) %>%
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
    fname = 'SPOC_avg.pdf'
    ggsave(file = file.path(path_out, fname), 
           plot = gt, 
           #device = 'png', 
           width = unit(w, 'cm'),  
           height = unit(h, 'cm'),
           units = 'cm', 
           bg = 'transparent')
    
    print(str_glue('Saved {fname} to\n{path_out}.'))
  }
  return(gt)
}



plot_fig4_ssd <- function(df_full, plt, path_out, save_png = TRUE) {
  
  ## Plot the 4x2 SSD topoplot subfigure in Figure 4 in the paper. 
  ## Takes the data frame w/ all normalized (!) pattern weights, 
  ## filters out the SSD relevant parts, calculates the mean of the
  ## absolute values (across subjects) and plots these into the 2x4
  ## topoplots sorted by movement condition (mov vs nomov, columns) 
  ## and number of the SPoC component. 
  
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
  
  
  ## Plot SSD:
  f <- df_full %>% 
    mutate_at(.vars = vars(electrode, Condition), list(factor)) %>%
    select(ID, electrode, Condition, matches('SSD.*_norm')) %>% 
    pivot_longer(cols = matches('^SSD'), 
                 names_to = c("kick", "ssd_comp_nr", "me"),
                 names_sep = "_",
                 values_to = "SSD") %>% 
    select(-c('kick', 'me')) %>% 
    group_by(electrode, Condition, ssd_comp_nr) %>% 
    # summarise_at(.vars = vars(matches('SSD.*_norm')), 
    #              .funs = ~mean(abs(.))) %>% 
    dplyr::summarise(SSD = mean(abs(SSD))) %>% 
    electrode_locations() %>% 
    ungroup() %>% 
    mutate(Condition = fct_reorder(Condition, 
                                   as.numeric(Condition), 
                                   .desc = T))
  
  
  
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
      scale_fill_viridis_c(breaks = c(0), 
                           labels = c('0'),
                           #limits = c(0.0,0.25),
                           guide = guide_colorbar(label = TRUE, 
                                                  ticks = TRUE))+
    #scale_fill_distiller(palette = plt$palette), 
    #direction = plt$palette_dir) + 
    coord_equal() + 
    facet_grid(ssd_comp_nr~Condition, 
               switch = 'y') +
      theme_void() + 
      theme(
        strip.text.x = element_text(size = 12, 
                                    margin = margin(b = 0.25, 
                                                    unit = 'cm')),
        # strip.text.y.left = element_text(angle = 0),
        # strip.text.y = element_text(size = 12, 
        #                             margin = margin(r = 0.5, 
        #                                             l = 0.5, 
        #                                             unit = 'cm')),
        strip.text.y = element_blank(), 
        legend.position = 'None', 
        panel.grid = element_blank(),
        axis.line = element_blank(),
        panel.background = element_rect(fill = "transparent",
                                        colour = NA),
        plot.background = element_rect(fill = "transparent",
                                       colour = NA)) +
    labs(fill = expression(paste("Absolute\nweight\n(a.u.)"))) #+ 
  #theme(legend.position = 'none')
 
  
  # format printable:
  gt <- ggplot_gtable(ggplot_build(plt_ssd))
  # fix size of topoplots:
  row_idx_dep_on_nr_ssd_comps <- seq(8,8+2*length(unique(f$ssd_comp_nr)) - 1, 2)
  gt$heights[row_idx_dep_on_nr_ssd_comps] <- unit(plt$size_topo, 'cm')
  gt$widths[c(6,8)] <- unit(plt$size_topo, 'cm')
  
  ## Uncomment only for debugging:
  # gtable_show_layout(gt)
  
  h = grid::convertHeight(sum(gt$heights), "cm", TRUE)
  w = grid::convertWidth(sum(gt$widths), "cm", TRUE)
  
  
  if (save_png) {
    fname = 'SSD_avg.pdf'
    ggsave(file = file.path(path_out, fname), 
           plot = gt, 
           #device = 'png', 
           width = unit(w, 'cm'), #unit(plt$wid_heigth_spoc[1], 'cm'), 
           height = unit(h, 'cm'), #unit(plt$wid_heigth_spoc[2], 'cm'), 
           units = 'cm', 
           bg = 'transparent')
    print(str_glue('Saved {fname} to\n{path_out}.'))
  }
  return(gt)
}
  

plot_fig4_supp <- function(df_full, cond, plt, path_out, save_png = TRUE) {
  
  ## Plot the supplements for Figure 4 in the paper. 
  ## Takes the data frame w/ all normalized (!) pattern weights, 
  ## filters out the relevant parts and plots these into the Nx7
  ## topoplots sorted by decomposition algorithm and subject. 
  
  ## ARGUMENTS:
  # df_full: df with normalized (!) pattern weights, needs following columns: 
  #              - ID
  #              - Condition
  #              - electrode
  #              - SPOC_norm
  # cond: (string) either 'mov' or 'nomov'; specifies the movemnet condition
  # plt: named list with plotting params ("palette", "palette_dir",
  #                                       "wid_heigth_ssd", "wid_heigth_csp", 
  #                                       "wid_heigth_spoc", "size_topo")
  # path_out: path to directory where the result shall be stored
  # save_png: boolean indicating whether the results sould be stored 
  #           (defaults to TRUE)
  #
  ## RETURNS:
  # plt_spoc: grob containing the info used for saving the png
  
  plt$size_topo <- 2 * 0.4
  
  ## Plot supp fig:
  f <- df_full %>% 
    mutate_at(.vars = vars(electrode, Condition), list(factor)) %>%
    select(ID, electrode, Condition, 
           matches('SSD.*_norm'), 
           matches('SPOC.*_norm'), 
           matches('CSP.*_norm')) %>% 
    pivot_longer(cols = matches(c('SSD', 'SPOC', 'CSP')), 
                 names_to = c('pattern', 'kick_me'),
                 names_sep = "_nor",
                 values_to = 'weights') %>% 
    select(-c('kick_me')) %>% 
    filter(Condition == cond) %>% 
    #filter(ID == 'NVR_S02') %>% 
    mutate(ID = str_remove(ID, 'NVR_')) %>% 
    mutate(pattern = as_factor(pattern),
           pattern = fct_relevel(pattern, 
                                 'SSD_1', 
                                 'SSD_2', 
                                 'SSD_3', 
                                 'SSD_4', 
                                 'SPOC', 
                                 'CSP_min', 
                                 'CSP_max')) %>% 
    group_by(ID, electrode, pattern) %>% 
    #dplyr::summarise(SSD = mean(abs(SSD))) %>% 
    electrode_locations() %>% 
    ungroup() %>% 
    mutate(Condition = fct_reorder(Condition, 
                                   as.numeric(Condition), 
                                   .desc = T))
  
  
  
  plt_supp <- ggplot(f,
                    aes(x = x,
                        y = y,
                        fill = weights,
                        label = electrode)) +
    geom_topo(grid_res = 20,
              interp_limit = "head",
              chan_markers = "point",
              chan_size = 0.25, 
              head_size = 0.1) + 
    # scale_fill_viridis_c(#breaks = c(0), 
    #                      #labels = c('0'),
    #                      #limits = c(0.0,0.25),
    #                      guide = guide_colorbar(label = TRUE, 
    #                                             ticks = TRUE))+
    scale_fill_distiller(palette = 'RdBu') + 
    #direction = plt$palette_dir) + 
    #coord_equal() + 
    facet_grid(ID~pattern, 
               switch = 'y', 
               scales = 'free') +
    theme_void() + 
    theme(
      strip.text.x = element_blank(),#element_text(size = 12, 
                      #            margin = margin(b = 0.25, 
                      #                            unit = 'cm')), 
      strip.text.y = element_text(size = 12, 
                                  margin = margin(r = 0.5,
                                                  l = 0.5,
                                                  unit = 'cm')),
      #strip.text.y.left = element_text(angle = 0),
      legend.position = 'bottom', 
      legend.box.margin = margin(20, 20, 20, 20),
      panel.grid = element_blank(),
      axis.line = element_blank(),
      panel.background = element_rect(fill = "transparent",
                                      colour = NA),
      plot.background = element_rect(fill = "transparent",
                                     colour = NA)) +
    labs(fill = expression(paste("Normalized weights\n(a.u.)"))) #+ 
  #theme(legend.position = 'none')
  
  
  # format printable:
  gt <- ggplot_gtable(ggplot_build(plt_supp))
  # fix size of topoplots:
  row_idx <- str_detect(as.character(gt$heights), 'null')
  col_idx <- str_detect(as.character(gt$widths), 'null')
  gt$heights[row_idx] <- unit(plt$size_topo, 'cm')
  gt$widths[col_idx] <- unit(plt$size_topo, 'cm')
  # fix distance between columns:
  idx_btw_mods <- which(col_idx) + 1
  idx_btw_mods <- idx_btw_mods[c(4, 5)]
  gt$widths[idx_btw_mods] <- unit(plt$size_topo * 2/3, 'cm')
  
  
  ## Uncomment only for debugging:
  #gtable_show_layout(gt)
  
  h = grid::convertHeight(sum(gt$heights), "cm", TRUE)
  w = grid::convertWidth(sum(gt$widths), "cm", TRUE)
  
  
  if (save_png) {
    fname <- str_glue('fig4_supp_{cond}.png')
    ggsave(file = file.path(path_out, fname), 
           plot = gt, 
           device = 'png', 
           width = unit(w, 'cm'), #unit(plt$wid_heigth_spoc[1], 'cm'), 
           height = unit(h, 'cm'), #unit(plt$wid_heigth_spoc[2], 'cm'), 
           units = 'cm', 
           bg = 'transparent')
    print(str_glue('Saved {fname} to\n{path_out}.'))
  }
  return(gt)
}

