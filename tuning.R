library(tidymodels)

# Function to perform two-step tuning for a given model
tune_model <- function(model_name, model_spec, tuning_params_stage1, tuning_params_stage2, df_train, cv_folds) {
  cat("Tuning", model_name, "Step 1...\n")
  
  # Step 1: Tune the first parameters
  tuning_grid_stage_1 <- grid_random(
    !!!rlang::syms(tuning_params_stage1),
    size = 10
  )
  
  tune_result_stage_1 <- tune_grid(
    model_spec,
    resamples = cv_folds,
    grid = tuning_grid_stage_1,
    metrics = metric_set(mape),
    control = control_grid(verbose = TRUE)
  )
  
  best_params_stage_1 <- tune_result_stage_1 %>% collect_metrics() %>% arrange(mean) %>%
    filter(row_number() == 1)
  
  # Extract the tuned parameters and their values
  tuned_params_stage_1 <- setNames(
    best_params_stage_1[[tuning_params_stage1]],
    tuning_params_stage1
  )
  
  # Step 2: Tune the remaining parameters using the best from Step 1
  cat("Tuning", model_name, "Step 2...\n")
  model_spec_stage_2 <- model_spec
  
  # Set the tuned parameters for the second step
  model_spec_stage_2 <- model_spec_stage_2 %>% set_args(!!!tuned_params_stage_1)
  
  # Generate the tuning grid for the second step
  tuning_grid_stage_2 <- grid_random(
    !!!rlang::syms(tuning_params_stage2),
    size = 10
  )
  
  tune_result_stage_2 <- tune_grid(
    model_spec_stage_2,
    resamples = cv_folds,
    grid = tuning_grid_stage_2,
    metrics = metric_set(mape),
    control = control_grid(verbose = TRUE)
  )
  
  cat("Top 5 configurations for", model_name, "Step 2:\n")
  top_configs_stage_2 <- tune_result_stage_2 %>% collect_metrics() %>% arrange(mean) %>%
    filter(row_number() <= 5)
  
  print(top_configs_stage_2)
  
  return(top_configs_stage_2)
}

# Function to iterate through a list of models and perform tuning
tune_models <- function(models, df_train, cv_folds, tuning_params_stage1, tuning_params_stage2) {
  results <- list()
  
  for (model_name in names(models)) {
    model_spec <- models[[model_name]]
    
    results[[model_name]] <- tune_model(
      model_name,
      model_spec,
      tuning_params_stage1[[model_name]],
      tuning_params_stage2[[model_name]],
      df_train,
      cv_folds
    )
  }
  
  return(results)
}

# Example usage
models_tune1 <- list(
  decision_tree = c("cost_complexity", "tree_depth"),
  linear_model = c("penalty"),
  lgbm = c("learn_rate")
)

models_tune2 <- list(
  decision_tree = c("tree_depth", "loss_reduction", "stop_iter"),
  linear_model = c("mixture"),
  lgbm = c("tree_depth", "loss_reduction", "stop_iter")
)

models <- list(
  decision_tree = decision_tree(cost_complexity = tune(), tree_depth = tune(), min_n = tune()) %>% 
    set_engine("rpart") %>% 
    set_mode("regression"),
  linear_model = linear_reg(penalty = tune(), mixture = tune()) %>%
    set_mode("regression") %>%
    set_engine("glmnet"),
  lgbm = boost_tree(trees = tune(), learn_rate = tune()) %>%
    set_mode("regression")
)

set.seed(123)
cv_folds <- vfold_cv(df_train, v = 5)

results <- tune_models(models, df_train, cv_folds, models_tune1, models_tune2)
