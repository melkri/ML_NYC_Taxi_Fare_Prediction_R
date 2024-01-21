library(yaml)
library(readr)
library(dplyr)
library(geosphere)
library(leaflet)
library(htmltools)
library(GGally)
library(lubridate)
library(ggplot2)
library(corrplot)
library(qmap)


source('functions.R')
config <- yaml.load_file("config.yaml")

set.seed(config$seed)
df <- read_csv(config$dataset$raw)

head(df)

# Split key column to get date
df$key_formatted <- gsub("\\..*", "", as.character(df$key))
df$key_formatted <- as.Date(df$key_formatted)

df$pickup_datetime <- as.POSIXct(df$pickup_datetime)

# Format columns
date_format <- "%Y-%m-%d %H:%M:%S"

df$key_formatted <- format(df$key_formatted, date_format)
df$pickup_datetime_formatted <- format(df$pickup_datetime, date_format)

# Check if there are any rows where key_formatted and pickup_datetime_formatted are different
checker <- df$key_formatted != df$pickup_datetime_formatted

# Filter the data frame to get rows where key_formatted and pickup_datetime_formatted are different
result_df <- df[checker, ]
result_df
# We can see that both columns here are the seme,let's abandon key column and format time column
# Extract pickup_date and pickup_time from pickup_datetime

df$pickup_date <- format(as.Date(df$key), "%Y-%m-%d")
df$pickup_time <- format(df$key, "%H:%M:%S")

# Drop the pickup_datetime column
df <- df[, !(names(df) %in% c("pickup_datetime","pickup_datetime_formatted","key","key_formatted")), drop = FALSE]

colnames(df)


na_zero_stats <- t(sapply(df, calculate_na_zero_stats))
  
summary_df <- as.data.frame(na_zero_stats)
summary_df$column_names <- rownames(summary_df)
summary_df <- summary_df[, c("column_names", "na_count", "zero_count", "na_percentage", "zero_percentage")]

print(summary_df)

df %>%
  filter(pickup_latitude == 0, pickup_longitude == 0, dropoff_longitude == 0)

# As all of the non-meaningful 0 values are in longitude/lattitude columns, we have decided to train separate models on values with and without zeros.

df$zero_indicator <- ifelse(df$pickup_latitude == 0 | df$pickup_longitude == 0 | df$dropoff_latitude == 0 | df$dropoff_longitude == 0, TRUE, FALSE)

df$straight_dist <- ifelse(df$zero_indicator == 1, 0, 
                                as.numeric(distHaversine(df[,c("pickup_longitude", "pickup_latitude")], 
                                                         df[,c("dropoff_longitude", "dropoff_latitude")])/1000))
# Saving the files
split_and_save(df, TRUE, 'df')

# Summary of the data
summary(df_train)
head(df_train)
# Correlation matrix
# Exclude 'id' column for correlation analysis
correlation_matrix <- cor(df_train[, !(names(df_train)  %in% c("id", "pickup_date", "pickup_time", "zero_indicator"))], use = "complete.obs") 
corrplot(correlation_matrix, method = "circle")


# Scatterplot for geolocation values
ggplot(df_train[df_train$zero_indicator != 1,], aes(x = pickup_longitude, y = pickup_latitude)) +
  geom_point(aes(color = fare_amount), alpha = 0.5) +
  scale_color_gradient(low = "blue", high = "red") +
  labs(title = 'Pickup locations colored by fare amount', x = 'Longitude', y = 'Latitude') +
  theme_minimal()

ggplot(df_train, aes(x = dropoff_longitude, y = dropoff_latitude)) +
  geom_point(aes(color = fare_amount), alpha = 0.5) +
  scale_color_gradient(low = "blue", high = "red") +
  labs(title = 'Dropoff locations colored by fare amount', x = 'Longitude', y = 'Latitude') +
  theme_minimal()

# Boxplot for fare_amount against passenger_count
ggplot(df_train, aes(x = as.factor(passenger_count), y = fare_amount)) +
  geom_boxplot() +
  labs(title = 'Boxplot of fare amount by passenger count', x = 'Passenger Count', y = 'Fare Amount') +
  theme_minimal()

# Additional: Time series plot of fare_amount
df_train$pickup_date <- as.Date(df_train$pickup_date)
ggplot(df_train, aes(x = pickup_date, y = fare_amount)) +
  geom_line() +
  labs(title = 'Time series plot of fare amount', x = 'Date', y = 'Fare Amount') +
  theme_minimal()


# Filter out rows where zero_indicator = 1
df_train_filtered <- df_train[df_train$zero_indicator != 1,]

# Create a color palette for the points
pal <- colorQuantile("Greens", df_train_filtered$fare_amount, n = 5)

# Create a map centered around New York for pickup locations
pickup_map <- leaflet() %>%
  setView(lng = -73.935242, lat = 40.730610, zoom = 10) %>%
  addTiles() %>%
  addCircleMarkers(data = df_train_filtered,
                   lng = ~pickup_longitude, lat = ~pickup_latitude,
                   color = ~pal(fare_amount),
                   radius = 0.3,
                   opacity = 0.5)

# Create a map centered around New York for dropoff locations
dropoff_map <- leaflet() %>%
  setView(lng = -73.935242, lat = 40.730610, zoom = 10) %>%
  addTiles() %>%
  addCircleMarkers(data = df_train_filtered,
                   lng = ~dropoff_longitude, lat = ~dropoff_latitude,
                   color = ~pal(fare_amount),
                   radius = 0.3,
                   opacity = 0.5)


ggplot(df_train, aes(x = pickup_time, y = fare_amount)) +
  geom_point(alpha = 0.1, position = position_jitter(width = 0.3, height = 0)) +
  theme_minimal() +
  labs(x = "Pickup Time", y = "Fare Amount", title = "Fare Amount vs Pickup Time")


# Display the two maps side by side
browsable(
  tags$div(
    style = "display: flex; justify-content: space-between;",
    tags$div(
      style = "width: 50%;",
      pickup_map
    ),
    tags$div(
      style = "width: 50%;",
      dropoff_map
    )
  )
)

### Modelling ###
# Exclude the 'id' column

df_train <- read_csv(config$dataset$train)


df_train <- subset(df_train, select = -id)
df_train <- subset(df_train, select = -pickup_time)
# Include only the 'pickup_date' column
df_train <- subset(df_train, select = -pickup_date)
df_train$zero_indicator <- as.integer(df_train$zero_indicator)

params <- yaml::read_yaml("config.yaml")


recipe <- recipe(fare_amount ~ ., data = df_train) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  prep(data = df_train)


rec <- recipe(fare_amount ~ ., data = df_train) %>%
  step_normalize(all_predictors(), -all_nominal()) %>%
  step_dummy(all_nominal(), -all_outcomes())

# 2) Model Specification
# Define the models
models <- list(
  linear_reg = linear_reg(penalty = params$linear_reg$penalty, mixture = params$linear_reg$mixture) %>% set_engine("glmnet"),
  rand_forest = rand_forest(mtry = tune(), min_n = tune()) %>% set_engine("ranger"),
  decision_tree = decision_tree(cost_complexity = tune(), min_n = tune()) %>% set_engine("rpart"),
  svm = svm_rbf(cost = tune(), rbf_sigma = tune()) %>% set_engine("kernlab"),
  light_gbm = boost_tree(trees = 1000, min_n = tune()) %>% set_engine("lightgbm")
)


# 3) Feature Selection and Model Tuning
# Define the resampling method
cv <- vfold_cv(df_train, v = 10)

# Tune the models
tuned_models <- lapply(models, function(model) {
  tune_grid(
    model,
    rec,
    resamples = cv,
    grid = 20,
    metrics = metric_set(mape)
  )
})

# 4) Hyperparameter Tuning
# Tune the models
tuned_models <- lapply(tuned_models, function(tuned_model) {
  tune_bayes(
    tuned_model,
    initial = 20,
    iter = 50,
    metrics = metric_set(mape)
  )
})

# 5) Model Evaluation
# Evaluate the models
evaluated_models <- lapply(tuned_models, function(tuned_model) {
  collect_metrics(tuned_model)
})

# Print the results
lapply(evaluated_models, print)

# 6) Explainability with Shapley values
# Compute Shapley values for the best model
best_model <- tuned_models[[which.min(sapply(evaluated_models, function(x) mean(x$mape)))]]
shap_values <- shapley(best_model, X = df_train)

# Print Shapley values
print(shap_values)




# Load necessary libraries
library(tidymodels)
library(parsnip)
library(recipes)
library(tune)
library(glmnet)
library(ranger)
library(rpart)
library(kernlab)
library(lightgbm)
library(shapley)
library(yaml)

# Load model parameters from YAML file
params <- yaml::read_yaml("config.yaml")

# Define a recipe for preprocessing
preprocess_data <- function(df) {
  recipe(fare_amount ~ ., data = df) %>%
    step_normalize(all_predictors(), -all_nominal()) %>%
    step_dummy(all_nominal(), -all_outcomes())
}

# Define the models
define_models <- function() {
  list(
    linear_reg = linear_reg(penalty = tune(), mixture = tune()) %>% set_mode("regression") %>% set_engine("glmnet"),
    rand_forest = rand_forest(mtry = tune(), min_n = tune()) %>% set_mode("regression") %>% set_engine("ranger"),
    decision_tree = decision_tree(cost_complexity = tune(), min_n = tune()) %>% set_mode("regression") %>% set_engine("rpart"),
    svm = svm_rbf(cost = tune(), rbf_sigma = tune()) %>% set_mode("regression") %>% set_engine("kernlab"),
    light_gbm = boost_tree(trees = 1000, min_n = tune()) %>% set_mode("regression") %>% set_engine("lightgbm")
  )
}
# Tune the models
tune_models <- function(models, rec, cv) {
  lapply(models, function(model) {
    tune_grid(
      model,
      rec,
      resamples = cv,
      grid = 20,
      metrics = metric_set(mape)
    )
  })
}

# Hyperparameter Tuning
tune_hyperparameters <- function(tuned_models) {
  lapply(tuned_models, function(tuned_model) {
    tune_bayes(
      tuned_model,
      initial = 20,
      iter = 50,
      metrics = metric_set(mape)
    )
  })
}

# Model Evaluation
evaluate_models <- function(tuned_models) {
  lapply(tuned_models, function(tuned_model) {
    collect_metrics(tuned_model)
  })
}

# Compute Shapley values for the best model
compute_shapley <- function(tuned_models, df_train) {
  best_model <- tuned_models[[which.min(sapply(evaluated_models, function(x) mean(x$mape)))]]
  shapley(best_model, X = df_train)
}

# Main function
run_models <- function() {
  df_train <- df_train # replace with your actual data
  rec <- preprocess_data(df_train)
  models <- define_models()
  cv <- vfold_cv(df_train, v = 10)
  tuned_models <- tune_models(models, rec, cv)
  tuned_models <- tune_hyperparameters(tuned_models)
  evaluated_models <- evaluate_models(tuned_models)
  lapply(evaluated_models, print)
  shap_values <- compute_shapley(tuned_models, df_train)
  print(shap_values)
}

# Check if the script is being run directly
if (length(commandArgs(trailingOnly = TRUE)) == 0) {
  run_models()
}

# Run the main function
run_models()
