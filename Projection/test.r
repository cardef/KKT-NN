#| echo: false
library(tidyverse)
library(ggpubr)
library(ggthemes)
library(svglite)
library(GGally)
library(akima)


params_df <- read_csv("Projection/param_analysis.csv", show_col_types=FALSE)
grid <- with(params_df, interp(x = action_p, y = action_q, z = sum, linear = TRUE, extrap = FALSE,
                            xo = seq(-1, 1, length = 100), 
                            yo = seq(-1, 1, length = 100)))

df <- as.data.frame(interp2xyz(grid))

#ggplot(df, aes(x=x, y=y, fill=z)) + geom_tile()  + theme_minimal()
ggally_mysmooth <- function(data, mapping, ...){
    vars = unique(unlist(lapply(mapping, all.vars)))
    grid <- with(data, interp(x = data[[vars[1]]], y = data[[vars[2]]], z = data[[vars[3]]], linear = TRUE, extrap = FALSE,
                            xo = seq(-1, 1, length = 100), 
                            yo = seq(-1, 1, length = 100)))

    df <- as.data.frame(interp2xyz(grid))
    ggplot(data = df, mapping=aes(x,y,fill=z)) +geom_raster(interpolate=TRUE, show.legend=TRUE)  +  scale_fill_gradient2_tableau(palette = "Orange-Blue Diverging", transform=c("log10", "reverse")) + theme_classic()
}
plot <- ggpairs(params_df, mapping=aes(fill=sum),columns = 1:7, upper = list(continuous = "blank"), diag = list(continuous = "blankDiag"), lower = list(continuous = ggally_mysmooth), legend=7) + theme(legend.position="bottom")
ggsave("Projection/param_analysis.svg", plot)