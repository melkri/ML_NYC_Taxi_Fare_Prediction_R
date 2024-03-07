split_and_save <- function(data, is_zero, file_prefix) {
  indices <- sample(1:nrow(data), size = floor(nrow(data) * config$test_size))
  
  df_train <- data[-indices, ]
  df_test <- data[indices, ]
  
  write.csv(df_train, paste0('data/input_split/', file_prefix, '_train.csv'), row.names = FALSE)
  write.csv(df_test, paste0('data/input_split/', file_prefix, '_test.csv'), row.names = FALSE)
}


calculate_na_zero_stats <- function(column) {
  na_count <- sum(is.na(column))
  zero_count <- sum(column == 0, na.rm = TRUE)
  total_count <- length(column)
  
  na_percentage <- (na_count / total_count) * 100
  zero_percentage <- (zero_count / total_count) * 100
  
  return(c(na_count = na_count, zero_count = zero_count, na_percentage = na_percentage, zero_percentage = zero_percentage))
}

tune_two_stages <- function(models, model_name) {
  # Extract parameters based on the model_name
  model_params <- models[[model_name]]
  
  # Stage 1 tuning
  set.seed(123)
  wflw_model_1 <- workflow() %>%
    add_model(model_params$model_1) %>%
    add_recipe(model_params$rec_spec)
  
  set.seed(123)
  cv_folds <- vfold_cv(df_train, v = 3)
  tune_stage_1 <- tune_grid(
    wflw_model_1,
    resamples = cv_folds,
    grid      = model_params$grid_1,
    metrics   = metric_set(mape),
    control   = control_grid(verbose = TRUE)
  )
  autoplot(tune_stage_1, metric = 'mape')
  best_params_model_1 <- tune_stage_1 %>% collect_metrics() %>% arrange(mean) %>%
    filter(row_number() == 1)
  print(best_params_model_1)
  
  # Stage 2 tuning
  set.seed(123)
  param_name <- names(model_params$grid_1)[1]
  param_value <- best_params_model_1[[1]]
  
  # Set up model_2
  model_params$model_2_args[[param_name]] <- param_value
  model_2 <- model_params$model_1 %>%
    set_args(
      !!!model_params$model_2_args  
    )
  wflw_model_2 <- wflw_model_1 %>%
    update_model(model_2)
  
  # Tune stage 2
  set.seed(123)
  tune_stage_2 <- tune_grid(
    wflw_model_2,
    resamples = cv_folds,
    grid      = model_params$grid_2,
    metrics   = metric_set(mape),
    control   = control_grid(verbose = TRUE)
  )
  autoplot(tune_stage_2, metric = 'mape')
  grid_last <- tune_stage_2
  all_results <- tune_stage_2 %>% collect_metrics() %>% arrange(mean) 
  
  # Get top 3 models
  top_3_models <- all_results %>% filter(row_number() <= 3)
  
  
  return(list(result = top_3_models, all_results = all_results, grid = tune_stage_2))
}

set_seed_grid <- function(grid, seed) {
  set.seed(seed)
  grid
}
