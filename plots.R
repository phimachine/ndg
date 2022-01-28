# cloud

require(dplyr)
require(ggplot2)
require(stringr)
require(tidyr)
require(data.table)
require(lubridate)
# require(ragg)

change_names <- function(name) {
  if (name == "allnli") {
    "AllNLI"
  }else if (name == "mnist") {
    "MNIST"
  }else if (name == "mnist2") {
    "MNIST even"
  }else if (name == "cub200") {
    "CUB200"
  }else if (name == "sst2") {
    "SST2"
  }else if (name == "code") {
    "Devign"
  }else {
    stop("hmm")
  }
}

# set your path here
figure_path <- "/drive/Git/cache/figures"
out_path <- "/drive/Git/ndg/plots"
# load csvs
cloud_path <- file.path(figure_path, "cloud.csv")
df <- fread(cloud_path)
df <- df %>% select(-V1)

df <- df %>%
  rowwise() %>%
  mutate(dataset = change_names(dataset)) %>%
  ungroup() %>%
  setDT()
#
# # ggplot
# g <- ggplot()
# g <- g + geom_point(data = df, aes(x = train, y = valid, color = dataset, alpha=0.01))
# g <- g +
#   theme(text = element_text(size = 16), legend.position = "bottom") +
#   xlab("training accuracy") +
#   ylab("validation accuracy")
#   labs(color = "dataset")
# g
#
# g<- ggplot(df,aes(x=train,color=dataset)) +
#   stat_count(data=df,aes(y=cumsum(..count..)),geom="step")
# g
#
# g <- ggplot(df, aes(x=valid, color=dataset)) +
#   geom_density()+
#   geom_histogram(binwidth=.001, alpha=.5, position="identity")
# g


df2 <- df %>% pivot_longer(!dataset, names_to = "split", values_to = "accuracy")
# g <- ggplot(df2, aes(x=dataset, y=accuracy, color = split)) +
#   geom_boxplot()
#   # scale_y_continuous(limits = c(0.9,1))
# g

g <- ggplot(df2, aes(x = dataset, y = accuracy, color = split)) +
  geom_boxplot(outlier.shape = 1) +
  theme(legend.position = "bottom")
g
ggsave(file.path(out_path, "box.png"), g, width = 5, height = 4)


g <- ggplot(df2, aes(x = dataset, y = accuracy, color = split)) +
  geom_boxplot(outlier.shape = 1) +
  scale_y_continuous(limits = c(0.99, 1)) +
  theme(legend.position = "bottom")
g

ggsave(file.path(out_path, "box_zoom.png"), g, width = 5, height = 4)


### alpha ribbon
alpha_path <- file.path(figure_path, "alpha_ribbon.csv")
df <- fread(alpha_path)

df <- df %>% rowwise() %>% mutate(dataset = change_names(dataset)) %>% ungroup()


g <- ggplot(df) +
  # geom_ribbon(aes(x=alpha, y=q2, ymin = q1, ymax = q3, fill=dataset,alpha=0.1))+
  #   geom_label(aes(x = alpha, y = q2, label = paste(alpha, signif(q2,4), sep = ", "), color = dataset)) +
  geom_line(aes(x = alpha, y = q2, color = dataset)) +
  geom_point(aes(x = alpha, y = q2, color = dataset)) +
  scale_x_log10() +
  scale_y_continuous(trans = "logit", n.breaks = 20) +
  annotation_logticks()+ theme(legend.position="bottom")

my_label <- function(a){
  if (is.integer(a)){
    sprintf("%d", a)
  }else{
    sprintf("%.2f", a)
  }
}

g <- ggplot(df) +
  # geom_ribbon(aes(x=alpha, y=q2, ymin = q1, ymax = q3, fill=dataset,alpha=0.1))+
  #   geom_label(aes(x = alpha, y = q2, label = paste(alpha, signif(q2,4), sep = ", "), color = dataset)) +
  geom_line(aes(x = alpha, y = average, color = dataset, linetype = split)) +
  geom_point(aes(x = alpha, y = average, color = dataset)) +
  scale_x_log10(breaks=c(1.01, 2, 3, 4, 5, 10, 20, 30, 50, 80, 100, 200, 500,1000,5000,100000),
                labels = c(1.01, 2, 3, 4, 5, 10, 20, 30, 50, 80, 100, 200, 500,1000,5000,100000), guide = guide_axis(n.dodge=2))+
  scale_y_continuous(trans = "logit", breaks = c(0.3,0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999))+
  theme(legend.position="bottom",legend.box = "vertical")+
  labs(y="accuracy")
# +annotation_logticks()
# geom_line(aes(x=alpha, y=q1, color = dataset))+
# geom_line(aes(x=alpha, y=q3, color = dataset))+
g
ggsave(file.path(out_path, "alpha_acc.png"), g, width = 5, height = 6)

g <- ggplot(df) +
  # geom_ribbon(aes(x=alpha, y=q2, ymin = q1, ymax = q3, fill=dataset,alpha=0.1))+
  #   geom_label(aes(x = alpha, y = q2, label = paste(alpha, signif(q2,4), sep = ", "), color = dataset)) +
  geom_line(aes(x = alpha, y = average, color = dataset, linetype = split)) +
  geom_point(aes(x = alpha, y = average, color = dataset)) +
  scale_x_log10(breaks=c(1.01, 2, 3, 4, 5, 10, 20, 30, 50, 80, 100, 200, 500,1000,5000,100000),
                labels = c(1.01, 2, 3, 4, 5, 10, 20, 30, 50, 80, 100, 200, 500,1000,5000,100000), guide = guide_axis(n.dodge=2))+
  scale_y_continuous(trans = "logit", breaks = c(0.3,0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999))+
  theme(legend.box = "horizontal", legend.justification=c(1,0), legend.position=c(1,0))+
  guides(linetype = guide_legend(title.position="top", title.hjust = 0.5),
         color = guide_legend(title.position="top", title.hjust = 0.5))+
  labs(y="accuracy")
# +annotation_logticks()
# geom_line(aes(x=alpha, y=q1, color = dataset))+
# geom_line(aes(x=alpha, y=q3, color = dataset))+
g
ggsave(file.path(out_path, "alpha_acc_main.png"), g, width = 6, height = 4)

edges <-df %>% select(dataset, alpha, edges, all_edges)
edges <- edges %>% rename(excluded = edges, included=all_edges) %>% setDT()
edges <- edges %>% melt(id.vars=c("dataset","alpha"), measure.vars=c("excluded", "included"))
edges <- edges %>% rename(edges = value, mutual_neuron_dependency = variable)

g <- ggplot(edges) +
  # geom_ribbon(aes(x=alpha, y=q2, ymin = q1, ymax = q3, fill=dataset,alpha=0.1))+
  #   geom_label(aes(x = alpha, y = q2, label = paste(alpha, signif(q2,4), sep = ", "), color = dataset)) +
  geom_line(aes(x = alpha, y = edges, color = dataset, linetype=mutual_neuron_dependency)) +
  geom_point(aes(x = alpha, y = edges, color = dataset))+
  scale_x_log10(breaks=c(1.01, 2, 3, 4, 5, 10, 20, 30, 50, 80, 100, 200, 500,1000,5000,100000),
                labels =c(1.01, 2, 3, 4, 5, 10, 20, 30, 50, 80, 100, 200, 500,1000,5000,100000), guide = guide_axis(n.dodge=2))+
  scale_y_log10()  +
  labs(linetype="mutual neuron dependency")+
  theme(legend.position="bottom", legend.box = "vertical")
g
ggsave(file.path(out_path, "alpha_edges.png"), g, width = 5, height = 6)

g <- ggplot(edges) +
  # geom_ribbon(aes(x=alpha, y=q2, ymin = q1, ymax = q3, fill=dataset,alpha=0.1))+
  #   geom_label(aes(x = alpha, y = q2, label = paste(alpha, signif(q2,4), sep = ", "), color = dataset)) +
  geom_line(aes(x = alpha, y = edges, color = dataset, linetype=mutual_neuron_dependency)) +
  geom_point(aes(x = alpha, y = edges, color = dataset))+
  scale_x_log10(breaks=c(1.01, 2, 3, 4, 5, 10, 20, 30, 50, 80, 100, 200, 500,1000,5000,100000),
                labels =c(1.01, 2, 3, 4, 5, 10, 20, 30, 50, 80, 100, 200, 500,1000,5000,100000), guide = guide_axis(n.dodge=2))+
  scale_y_log10()  +
  labs(linetype="mutual neuron dependency")+
  theme(legend.position="bottom", legend.box = "horizontal")+
  guides(linetype = guide_legend(title.position="top", title.hjust = 0.5),
         color = guide_legend(title.position="top", title.hjust = 0.5))
g
ggsave(file.path(out_path, "alpha_edges_main.png"), g, width = 5, height = 5)

