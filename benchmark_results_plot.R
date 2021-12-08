library(tidyverse)

benchmark_results <- readr::read_csv(
  'https://raw.githubusercontent.com/Rosenkrands/sar-path-planning/main/benchmark_results.csv'
)

instance_size <- tibble(
  instance = c("132167","3e559b","4fceab","506fa3","802616","88182c","8ea3cb","cd97cf","f79242"),
  tiles = c(90,50,60,20,30,70,100,80,40),
  size = factor(tiles/2)
)

levels(instance_size$size) <- vapply(levels(instance_size$size), function(size) paste0(size,'x',size,' km (',as.numeric(size)*2,'x',as.numeric(size)*2,')'), "20x20 km (40x40)")         

plot_data <- benchmark_results %>% 
  inner_join(instance_size, by = c("instance"))

         
plot_data %>%
  mutate(algorithm = factor(algorithm, levels = c("Greedy", "Hillclimbing", "GRASP"))) %>%
  ggplot(aes(x= algorithm , y = `objective mean`)) +
  geom_point() +
  geom_errorbar(aes(ymin = `objective mean` - `objective std`,
                    ymax = `objective mean` + `objective std`), width = .2) +
  facet_wrap(~size, scales = "free")
