# Install and load necessary packages
install.packages(c("tidymodels", "lightgbm"))

library(tidymodels)
library(lightgbm)
library(parallel)
library(doParallel)
library(yaml)
library(readr)
library(yaml)
library(tidyverse)

#> Registered S3 method overwritten by 'tune':
#>   method                   from   
#>   required_pkgs.model_spec parsnip
options(tidymodels.dark = TRUE)

config <- yaml.load_file("config.yaml")
df_train <- read_csv(config$dataset$train)


df_train <- subset(df_train, select = -c(id,pickup_time, pickup_date))
df_train <- subset(df_train, select = -pickup_time)
# Include only the 'pickup_date' column
df_train <- subset(df_train, select = -pickup_date)



cl <- makeCluster(11)  
registerDoParallel(cl)  
 # Assuming df_train is your training dataset and df_test is your testing dataset

# Section 1: Data Preprocessing

# Step 1: Handle missing values, if any

df_train <- df_train %>% drop_na()

# Step 2: Scale variables using min_max scaler
df_train_scaled <- df_train %>%
  mutate(zero_indicator = as.integer(zero_indicator)) %>% 
  recipe(fare_amount ~ .) %>%
  step_normalize(all_predictors()) %>%
  prep() %>%
  bake(new_data = df_train)

# Section 2: Feature Selection

# Step 1: Split the data into training and validation sets
set.seed(22)
split <- initial_split(df_train_scaled, prop = 0.8)
df_train_split <- training(split)
df_val_split <- testing(split)

# Step 2: Create models
models <- list(
  #rf = rand_forest(mtry = tune(), trees = tune(), min_n = tune())%>%
   #set_mode("regression"),
  dt = decision_tree(cost_complexity = tune(),
                     tree_depth = tune(),
                     min_n = tune()) %>% 
    set_engine("rpart") %>% 
    set_mode("regression"),
  lm = linear_reg(penalty = tune(), mixture = tune()) %>%
    set_mode("regression") %>%
    set_engine("glmnet"),

  
  #svm = svm_rbf(cost = tune(), rbf_sigma = tune())%>%
   #set_mode("regression"),
  lgbm = boost_tree(trees = tune(), learn_rate = tune())%>%
                      set_mode("regression")
)

# Step 3: Backward elimination to find the optimal features
feature_selection_results <- list()
tree_grid <- grid_regular(cost_complexity(), tree_depth(), min_n(), levels = 3)


for (model_name in names(models)) {
  print(paste("Processing model:", model_name))
  
  wf <- workflow() %>%
    add_model(models[[model_name]]) %>%
    add_recipe(recipe(fare_amount ~ ., data = df_train_split))
  
  # Tune parameters
  
  set.seed(100)
  tune_results <- tune_grid(
    wf,
    resamples = vfold_cv(df_train_split, v = 3),
    metrics = metric_set(mape),
    control = control_grid(verbose = TRUE),
    grid = grid_latin_hypercube(parameters(models[[model_name]]), size = 10)
  )
   
  feature_selection_results[[model_name]] <- tune_results
}






# Print the results
for (model_name in names(models)) {
  print(paste("Results for model:", model_name))
  print(feature_selection_results[[model_name]])
}



# Access the results for each model
for (model_name in names(feature_selection_results)) {
  print(paste("Results for", model_name))
  print(feature_selection_results[[model_name]])
  cat("\n")
}
# You can then access the results for each model like this:
for (model_name in names(feature_selection_results)) {
  print(paste("Results for", model_name))
  print(feature_selection_results[[model_name]])
  cat("\n")
}

# Section 3: Hyperparameter Tuning

# Step 1: Create a grid for hyperparameter tuning
hyper_grid <- expand.grid(
  trees = c(100, 200),
  mtry = c(2, 4),
  sigma = c(0.1, 0.5)
)

# Step 2: Tune the models
config <- yaml::read_yaml("config.yaml")

tuned_models <- list()

for (model_name in names(models)) {
  # Retrieve hyperparameter grid for the current model
  hyper_grid_model <- config[[model_name]]
  
  tune_result <- workflow() %>%
    add_model(models[[model_name]]) %>%
    add_recipe(recipe(target ~ .)) %>%
    tune_grid(
      resamples = cv(df_train_split, num = 10),
      grid = hyper_grid_model
    ) %>%
    collect_metrics() %>%
    filter(.metric == "mape") %>%
    arrange(mean) %>%
    slice(1)
  
  tuned_models[[model_name]] <- finalize_model(
    models[[model_name]],
    tune_result
  )
}


# Section 4: Evaluate the best models on the validation set

results <- tibble(
  Model = character(),
  Features = character(),
  MAPE_Training = double(),
  MAPE_Validation = double()
)

for (model_name in names(tuned_models)) {
  # Extract features
  features <- tidy(tuned_models[[model_name]]) %>%
    filter(term != "(Intercept)") %>%
    pull(term)
  
  # Evaluate on training and validation sets
  mape_training <- df_train_split %>%
    bind_cols(tidy(tuned_models[[model_name]]) %>%
                filter(term != "(Intercept)")) %>%
    predict(tuned_models[[model_name]]) %>%
    metrics(truth = target, estimate = .pred) %>%
    filter(.metric == "mape") %>%
    pull(mean)
  
  mape_validation <- df_val_split %>%
    bind_cols(tidy(tuned_models[[model_name]]) %>%
                filter(term != "(Intercept)")) %>%
    predict(tuned_models[[model_name]]) %>%
    metrics(truth = target, estimate = .pred) %>%
    filter(.metric == "mape") %>%
    pull(mean)
  
  results <- bind_rows(results, tibble(
    Model = model_name,
    Features = paste(features, collapse = ", "),
    MAPE_Training = mape_training,
    MAPE_Validation = mape_validation
  ))
}

# Print the results table
print(results)

# Section 5: Predict on df_test using the best models

# Assuming df_test_scaled is your scaled testing dataset
df_test_scaled <- df_test %>%
  recipe(target ~ .) %>%
  step_normalize(all_predictors(), method = "range") %>%
  prep() %>%
  bake(new_data = df_test)

# Make predictions
predictions <- tibble(Model = character(), Predictions = list())

for (model_name in names(tuned_models)) {
  model_predictions <- df_test_scaled %>%
    bind_cols(tidy(tuned_models[[model_name]]) %>%
                filter(term != "(Intercept)")) %>%
    predict(tuned_models[[model_name]]) %>%
    select(.pred)
  
  predictions <- bind_rows(predictions, tibble(
    Model = model_name,
    Predictions = list(model_predictions)
  ))
}

# Print the predictions table
print(predictions)
