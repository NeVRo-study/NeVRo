

df_full %>% filter(Condition == 'nomov') %>% 
  mutate(elec_side = if_else(as.logical(parse_number(electrode) %% 2), 'left', 'right'), 
         numm = parse_number(electrode), 
         electrode = ) %>% 
    group_by(ID, elec_side, electrode) %>% 
  summarize(meanCSP1 = mean(abs(SPOC_norm), na.rm=T)) -> he
 
left <- he %>% filter(elec_side == 'left') %>%  pull(meanCSP1)
right <- he %>% filter(elec_side == 'right') %>%  pull(meanCSP1)
boxplot(meanCSP1 ~ elec_side, data = he)


df_full %>% filter(Condition == 'nomov') %>% 
  mutate(strr = str_extract(electrode, '[^0-9]+'), 
         numm = parse_number(electrode), 
         elec_side = if_else(as.logical(parse_number(electrode) %% 2), 'left', 'right'), 
         numm_c = if_else(elec_side == 'right', numm-1, numm), 
         elec_site = str_c(strr, numm_c)) %>% 
  select(ID, elec_side, elec_site, matches('.*_norm')) %>% 
  drop_na() %>%
  pivot_longer(cols = matches('.*_norm.*')) %>% 
  group_by(name, elec_site, elec_side) %>% 
  nest()  %>% 
  spread(key = elec_side, value = data) %>% 
  filter(name == 'SPOC_norm')  %>% 
  mutate(
    t_test_res = map2(right, left, ~{t.test(abs(.x$value), abs(.y$value), paired=TRUE) %>% broom::tidy() }),
    diff = unlist(map2(right, left, ~{mean(abs(.x$value) - abs(.y$value))})),
    right = map(right, nrow),
    left = map(left, nrow)) %>% 
  unnest(cols = c(left, right, t_test_res))-> ha



df_full %>% filter(Condition == 'nomov') %>%  
  select(ID, electrode, matches('.*_norm')) %>% 
  drop_na() %>%
  pivot_longer(cols = matches('.*_norm.*')) %>% 
  group_by(name, electrode) %>% 
  mutate(value = abs(value)) %>% 
  t_test(value~1, mu = 0) %>%  filter(p < 0.05) -> ress



## CBP test:
tmp <- tibble()

for (i in 1:1000) {
  
  df_full %>% filter(Condition == 'nomov') %>%  
    select(ID, electrode, matches('.*_norm')) %>% 
    group_by(ID) %>% 
    mutate(electrode = sample(electrode)) %>% 
    mutate(across("CSP_max_norm":"SSD_4_norm",  ~ abs(.))) %>% 
    group_by(electrode) %>% 
    summarize(across("CSP_max_norm":"SSD_4_norm", ~ mean(.x))) %>% 
    bind_rows(tmp) -> tmp
}

# true vals:
df_full %>% filter(Condition == 'nomov') %>%  
  select(ID, electrode, matches('.*_norm')) %>% 
  mutate(across("CSP_max_norm":"SSD_4_norm",  ~ abs(.))) %>% 
  group_by(electrode) %>% 
  summarize(across("CSP_max_norm":"SSD_4_norm", ~ mean(.x))) -> true_vals

true_vals_c <- true_vals %>%  pivot_longer(matches("*_norm"))
tmp_c <- tmp
tmp_c <- tmp_c %>%  pivot_longer(matches("*_norm"))

tt <- left_join(tmp_c, true_vals_c, by = c("electrode", "name"), suffix = c('_perm', '_true')) %>% 
  mutate(perm_higher = value_perm > value_true)

tt %>%  select(electrode, perm_higher, name) %>% 
  summarize(
  %>% pivot_wider(id_cols = c(electrode, name), values_from = perm_higher)
  



ha <- ha %>% 
    mutate(time = 1, 
           electrode = elec_site, 
           label = electrode) %>% 
    electrode_locations() # %>%   filter(electrode == 'P3')

ha <- df_full %>% 
  electrode_locations() %>% 
  filter(Condition == 'nomov')
    
# plt_csp <- ggplot(ha,
#                   aes(x = x,
#                       y = y,
#                       z = estimate,
#                       fill = estimate)) +
#     geom_topo(grid_res = 200,
#               interpolate = FALSE,
#               interp_limit = "head",
#               #chan_markers = "text",
#               chan_size = 0.25,
#               head_size = 1,
#               bins = 10) +
#   coord_equal()
# plt_csp


ggplot(ha, 
       aes(x = x, 
           y = y,
           fill = p.value)) + 
           # fill = statistic)) + 
           #fill = abs(CSP_min_norm))) + 
  stat_scalpmap(grid_res = 200,
                # r = 140, 
                # interpolate = T
                ) +  #based on geom_raster()
  geom_mask(size = 20) +    #based on geom_path()
  geom_head(interp_limit = 'skirt') +                   #based on geom_path() and geom_curve()
  geom_channels(geom = "text", aes(label=electrode)) +  #based on geom_point() and geom_text()
  scale_fill_viridis_c(limits = c(-0.0, 0.99),  
                       #limits = c(-2.0, 2),
                       oob = scales::squish, 
                       direction = -1) +
  coord_equal()

