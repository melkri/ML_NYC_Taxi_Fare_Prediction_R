library(mlbench)
library(caret)
library(tidymodels)
library(lightgbm)
library(parallel)
library(doParallel)
library(yaml)
library(readr)
library(yaml)
library(tidyverse)

config <- yaml.load_file("config.yaml")
df_train <- read_csv(config$dataset$train)
df_train <- subset(df_train, select = -c(id,pickup_time, pickup_date))
df_train <- df_train %>%
  mutate(zero_indicator = as.integer(zero_indicator))

library(mlbench)
library(caret)

control <- trainControl(method="cv", number=5, allowParallel = FALSE)

# Model training
model <- train(
  fare_amount ~ ., 
  data = df_train, 
  method = "rf",
  metric = "RMSE", 
  preProcess = "scale", 
  trControl = control
)
importance <- varImp(model, scale=FALSE)
print(importance)
plot(importance)

control2 <- rfeControl(functions=rfFuncs, method="cv", number=10, verboseIter = TRUE)
results <- rfe(df_train[,c(1:2, seq(4:length(df_train)))], df_train[,3], sizes=c(1:17), rfeControl=control2)
predictors(results)
plot(results, type=c("g", "o"))