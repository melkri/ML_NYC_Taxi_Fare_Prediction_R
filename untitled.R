# Load required libraries
library(tidymodels)
library(ranger)
library(kernlab)
library(lightgbm)
library(caret)

df_train <- read_csv(config$dataset$train)


df_train <- subset(df_train, select = -id)
df_train <- subset(df_train, select = -pickup_time)
# Include only the 'pickup_date' column
df_train <- subset(df_train, select = -pickup_date)
df_train$zero_indicator= as.integer(df_train$zero_indicator)

# Define K-Fold Cross Validation
folds <- vfold_cv(df_train, v = 10)

# Initialize empty results dataframe
results_df <- tibble(
  Model = character(),
  Features = character(),
  Avg_MAPE_Train = numeric(),
  Avg_MAPE_Validation = numeric()
)

# Define models
models <- list(
  rf_model = rand_forest(trees = tune()) %>%
    set_engine("ranger")%>%
    set_mode("regression"),
  dt_model = decision_tree() %>%
    set_engine("rpart")%>%
    set_mode("regression"),
  svm_model = svm_rbf(cost = tune()) %>%
    set_engine("kernlab")%>%
    set_mode("regression"),
  lgbm_model = boost_tree(trees = tune()) %>%
    set_engine("lightgbm")%>%
    set_mode("regression")
)

# Feature ranking loop
for (model_name in names(models)) {
  print(paste("Processing model:", model_name))
  
  # Create recipe for each model
  rec <- recipe(fare_amount ~ ., data = df_train)
  
  # Create model specification
  model_spec <- workflow() %>%
    add_recipe(rec) %>%
    add_model(models[[model_name]])
  
  # Extract feature importance using mutual information
  print("Extracting feature importance using mutual information...")
  feature_ranking_mi <- model_spec %>%
    fit_resamples(
      resamples = folds,
      metrics = metric_set(mape),
      control = control_resamples(save_pred = TRUE)
    ) %>%
    collect_metrics() %>%
    group_by(.metric, .model) %>%
    summarize(mean = mean(.estimate, na.rm = TRUE)) %>%
    ungroup() %>%
    arrange(mean)
  
  # Extract feature importance using correlation threshold
  print("Extracting feature importance using correlation threshold...")
  feature_importance_corr <- df_train %>%
    select(-fare_amount) %>%
    cor() %>%
    abs() %>%
    rownames_to_column(var = "feature") %>%
    gather(key = "variable", value = "correlation", -feature) %>%
    filter(correlation > 0.9) %>%
    pull(feature)
  
  # Combine feature sets based on mutual information and correlation threshold
  selected_features <- union(feature_ranking_mi$Model, feature_importance_corr)
  
  # Update recipe with selected features
  rec <- rec %>%
    update_role(all_predictors(), new_role = selected_features)
  
  # Tune parameters
  print("Tuning parameters...")
  tune_results <- tune_grid(
    model_spec,
    resamples = folds,
    grid = 5
  )
  
  # Select best tuning parameters
  best_params <- select_best(tune_results, "mape")
  
  # Train model with selected parameters
  final_model <- finalize_model(model_spec, best_params) %>%
    fit(df_train)
  
  # Evaluate on training and validation sets
  print("Evaluating on training and validation sets...")
  train_eval <- predict(final_model, df_train) %>%
    bind_cols(df_train) %>%
    metrics(truth = fare_amount, estimate = .pred) %>%
    select(-.estimate)
  
  validation_eval <- resamples %>%
    resample_summary(metrics = metric_set(mape), statistics = "mean")
  
  # Save results to dataframe
  results_df <- results_df %>%
    add_row(
      Model = model_name,
      Features = toString(selected_features),
      Avg_MAPE_Train = train_eval$mape$mean,
      Avg_MAPE_Validation = validation_eval$mape$mean
    )
}

# Print results
print(results_df)

# ... (continue with hyperparameter tuning and prediction on df_test)
