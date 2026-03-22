library(ggplot2)
library(dplyr)
library(ggridges)
library(viridis)
library(extrafont)
library(ggtext)
library(patchwork)


subject_names <- c("Human","Nash Equilibrium","GPT-4","GPT-3.5","Claude3-Opus","Claude3-Sonnet",
                   "Llama3-70b","Llama3-8b","Llama2-13b","Llama2-7b"
                   )
data_list <- list( # Distribution List
  c(4,0,3,6,1,6,32,30,12,6),
  c(0,0,0,0,25,25,20,15,10,5),
  c(0,0,0,0,0,0,0,0,100,0),
  c(0.7,22.1,0.9,4.7,28.9,13.7,0.4,0.1,27.9,0),
  c(0,0,0,0,0,0,0,0,99.1,0.9),
  c(0,0,0,0,0,0,0,0,0.8,99.2),
  c(7,0,0,0,0,0,0,0,82.6,10.4),
  c(0,5.8,0,0,0,0,3,4.4,86.8,0),
  c(0,0,0,0,0,0,0,0,100,0),
  c(0,0,0,0,0,0,0,1.4,98.6,0)
)

feature_values <- data.frame(
  Subjects = subject_names,
  Feature = c(1, 0.8343,0.4975, 0.5833, 0.5104,0.4369,0.5767,0.6065,0.4975,0.5253)  # Similarity Score
)

df <- data.frame(
  Number = rep(11:20, 10),
  Frequency = unlist(data_list),
  Subjects = rep(subject_names, each = 10)
)

df_long <- df %>%
  group_by(Subjects) %>%
  mutate(Count = Frequency * 100) %>%
  ungroup()


expand_data <- function(df) {
  expanded_df <- data.frame()
  for (i in 1:nrow(df)) {
    expanded_df <- rbind(expanded_df, data.frame(
      Number = rep(df$Number[i], df$Count[i]),
      Subjects = rep(df$Subjects[i], df$Count[i])
    ))
  }
  return(expanded_df)
}


df_expanded <- expand_data(df_long)

df_expanded$Subjects <- factor(df_expanded$Subjects, levels = rev(subject_names))


bold_and_larger_selected <- function(x) {
  ifelse(x %in% c("Human", "Nash Equilibrium"), 
         paste0("<span style='font-size:21pt; font-family:\"Times New Roman\";'><strong><em>", x, "</em></strong></span>"), 
         paste0("<span style='font-size:18pt; font-family:\"Times New Roman\";'>", x, "</span>"))
}
p1 <- ggplot(df_expanded, aes(x = Number, y = Subjects, fill = stat(quantile))) +
  stat_density_ridges(
    geom = "density_ridges_gradient",
    calc_ecdf = TRUE,
    quantiles = c(0.25, 0.5, 0.75),
    quantile_lines = TRUE,
    scale = 2,
    bandwidth = 0.5
  ) +
  scale_fill_manual(
    values = c("#FAEBDDFF","#841E5AFF","#CB1B4FFF","#F06043FF"),
    name = "Quantile",
    labels = c("(0, 0.25]", "(0.25, 0.5]", "(0.5, 0.75]", "(0.75, 1]"),
    breaks = c(1, 2, 3, 4)
  ) +
  #scale_fill_viridis(
  #  discrete = TRUE,
  #  name = "Quantile",
  #  option = "rocket",
  #  labels = c("(0, 0.25]", "(0.25, 0.5]", "(0.5, 0.75]", "(0.75, 1]"),
  #  breaks = c(1, 2, 3, 4),
  #  direction = -1
  #) +
  scale_x_continuous(breaks = 11:20, limits = c(11, 20)) +
  labs(
    x = "Response (%)", y = "Subject Density"
  ) +
  theme_minimal() +
  theme(
    text = element_text(family = "Times New Roman", size = 21),
    legend.position = "top",
    panel.grid.major.x = element_line(color = "gray90", size = 0.2),
    panel.grid.major.y = element_line(color = "gray90", size = 0.6),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    plot.title = element_text(face = "bold", size = 24),
    plot.subtitle = element_text(face = "italic", color = "grey50", size = 22),
    plot.caption = element_text(color = "grey50", size = 18),
    axis.text = element_text(color = "gray20", size = 18),
    axis.title = element_text(color = "gray20", size = 22),
    axis.text.y = element_markdown(color = "gray20", size = 20),
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 16)
  ) +
  scale_y_discrete(labels = bold_and_larger_selected)

p2 <- ggplot(feature_values, aes(x = Subjects, y = Feature, fill = "Similarity")) +
  geom_col(width = 0.3) +
  #geom_text(aes(label = Feature), vjust = -0.3, color = "black", size = 3.5) +
  coord_flip() +
  scale_y_continuous(
    limits = c(0, 1),  
    expand = c(0, 0),
    breaks = seq(0, 1, by = 0.2),  
    labels = function(x) ifelse(x %in% c(0, 1), as.character(x), "")  
  ) +
  scale_x_discrete(limits = rev(subject_names)) +
  scale_fill_manual(values = "#708090FF") +
  labs(y = NULL) +  
  theme_void() +
  theme(
    text = element_text(family = "Times New Roman", size = 21),
    legend.position = "none",
    axis.title.x = element_text(size = 21, color = "black"),  
    axis.text.x = element_text(size = 12, color = "black"),   
    axis.line.x = element_line(color = "black"), 
    axis.line.y = element_line(color = "black"), 
    axis.ticks.x = element_line(color = "black", size = 0.5), 
    axis.ticks.length.x = unit(-0.4, "cm"), 
    panel.grid.major = element_line(color = "gray90", size = 0.1), 
  )+
  guides(y = guide_axis(check.overlap = TRUE))  

combined_plot <- p1 + p2 + plot_layout(widths = c(4, 1))

print(combined_plot)