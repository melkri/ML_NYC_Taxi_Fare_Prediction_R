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

control2 <- rfeControl(functions=rfFuncs, method="cv", number=4)
results <- rfe(df_train[, c(1:2, seq(4, ncol(df_train)))], as.matrix(df_train[,3]), sizes=c(1:22), rfeControl=control2)

predictors(results)
plot(results, type=c("g", "o"))

recipe_obj <- recipe(fare_amount~ ., data = df_train) %>%
  step_scale(all_predictors()) %>%  # Standardize the predictors
  step_pca(all_predictors(), threshold = 0.96)  # Perform PCA with 90% variance threshold

df_train_pca <- bake(prep(recipe_obj), new_data = df_train)
