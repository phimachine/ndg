require(dplyr)
require(ggplot2)
require(stringr)
require(tidyr)
require(data.table)
require(lubridate)
library(RColorBrewer)


# set your codel path here
hidden_path <- "/drive/Git/ndg" #getwd()
plot_path <- file.path(hidden_path, "plots")

# load csvs
confu <- file.path(plot_path, "confu.csv")
alpha <- file.path(plot_path, "alpha.csv")
confu <- fread(confu)
alpha <- fread(alpha)

datasets <- unique(confu$dataset)
ds <- "code"
for (ds in datasets) {
  slice <- confu[dataset == ds]
  g <- ggplot()
  if (ds == "cub200") {
    g <- g +
      geom_hline(yintercept = 0.01, color = "dodgerblue") +
      geom_text(aes(0.75, 0.01, label = 0.01, vjust = -1), color = "dodgerblue")
  }
  if (ds == "code" || ds == "allnli" || ds == "cub200") {
    alpha <- 0.5
    size <- 0.5
  }else {
    alpha <- 0.4
    size <- 1
  }
  g <- g +
    geom_line(data = slice, aes(x = precision, y = recall, group = paired), color = "grey", alpha = 0.2, size = 0.2) +
    geom_point(data = slice, aes(x = precision, y = recall, color = split), alpha = alpha, size = size) +
    theme(legend.position = "bottom")

  if (ds == "code") {
    g <- g + xlim(0.95, 1)
  }
  # g
  ggsave(file.path(plot_path, paste("confu_", ds, ".png", sep = "")), g, width = 4, height = 4)
}


#
# df <- bind_rows(list(reranker = rerank_df, original = orig_df), .id = "score") %>%
#   mutate(score = replace(score, score == "reranker", "reranker score   \n   ")) %>%
#   mutate(score = replace(score, score == "original", "original beam search score    \n   "))
#
#
# g <- ggplot(data = df, aes(x = sensitivity, y = precision, color = str_wrap(score, 9)))
# g <- g + geom_line(size = 1)
# g <- g +
#   theme(text = element_text(size = 12), legend.position = "right", legend.key.height=unit(3,"line")) +
#   xlab("recall") +
#   ylab('precision') +
#   labs(color = "Confidence\nmetric") +
#   ylim(0.2, NA) +
#   geom_hline(yintercept = 0.2404, linetype = 'dashed', colour = "#00bfc4") +
#   geom_text(aes(0.1, 0.2404, label = 0.2404, vjust = -0.5), colour = "#00bfc4") +
#   geom_hline(yintercept = 0.2235, linetype = 'dashed', colour = "#f8766d") +
#   geom_text(aes(0.1, 0.2235, label = 0.2235, vjust = 1.5), colour = "#f8766d") +
#   # geom_hline(yintercept = 0.5, linetype='dashed', colour="dodgerblue") +
#   # geom_text(aes(0, 0.5, label = 0.5, vjust = -1), colour="dodgerblue")+
#   geom_point(aes(x = 0.384, y = 0.5), size = 3, colour = "#00bfc4") +
#   geom_text(aes(0.384, 0.5, label = "(0.384,0.5)", hjust = -0.1, vjust = -0.35), colour = "#00bfc4") +
#   geom_point(aes(x = 0.224, y = 0.5), size = 3, colour = "#f8766d") +
#   geom_text(aes(0.224, 0.5, label = "(0.224,0.5)", hjust = 0.8, vjust = 2.5), colour = "#f8766d")
# g
#
# ggsave(file.path(plot_path, "pr_small.png"), g, width = 5, height = 3)
#
#
#
# df <- bind_rows(list(reranker = rerank_df, original = orig_df), .id = "score") %>%
#   mutate(score = replace(score, score == "reranker", "reranker score")) %>%
#   mutate(score = replace(score, score == "original", "original beam search score"))
# g <- ggplot(data = df, aes(x = sensitivity, y = precision, color = score))
# g <- g + geom_line(size = 1)
# g <- g +
#   theme(text = element_text(size = 12), legend.position = "bottom") +
#   xlab("recall") +
#   ylab('precision') +
#   labs(color = "Confidence metric") +
#   ylim(0.2, NA) +
#   geom_hline(yintercept = 0.2404, linetype = 'dashed', colour = "#00bfc4") +
#   geom_text(aes(0.1, 0.2404, label = 0.2404, vjust = -0.5), colour = "#00bfc4") +
#   geom_hline(yintercept = 0.2235, linetype = 'dashed', colour = "#f8766d") +
#   geom_text(aes(0.1, 0.2235, label = 0.2235, vjust = 1.5), colour = "#f8766d") +
#   # geom_hline(yintercept = 0.5, linetype='dashed', colour="dodgerblue") +
#   # geom_text(aes(0, 0.5, label = 0.5, vjust = -1), colour="dodgerblue")+
#   geom_point(aes(x = 0.384, y = 0.5), size = 3, colour = "#00bfc4") +
#   geom_text(aes(0.384, 0.5, label = "(0.384,0.5)", hjust = -0.1, vjust = -0.35), colour = "#00bfc4") +
#   geom_point(aes(x = 0.224, y = 0.5), size = 3, colour = "#f8766d") +
#   geom_text(aes(0.224, 0.5, label = "(0.224,0.5)", hjust = 0.8, vjust = 2.5), colour = "#f8766d")
# g
#
# ggsave(file.path(plot_path, "pr.png"), g, width = 5.5, height = 4)
