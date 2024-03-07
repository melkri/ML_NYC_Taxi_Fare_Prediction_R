
library(caret)
library(tidymodels)
library(lightgbm)
library(parallel)
library(doParallel)
library(yaml)
library(readr)
library(yaml)
library(tidyverse)
library(bonsai)

config <- yaml.load_file("config.yaml")


df_train_new <- read_csv(config$dataset$train)
df_train <- read_csv(config$dataset$train)
df_test <- read_csv(config$dataset$test)

df_test_mutated <- df_test %>%
  mutate(
    time_of_day = as.POSIXct(pickup_time, format = '%T'),
    tenmin_interval = format(ceiling_date(time_of_day, '10 mins'), "%H%M"),
    hour_numeric = hour(pickup_time),
    hour4 = as.integer(hour_numeric == 4),
    hour5 = as.integer(hour_numeric == 5),
    hour19 = as.integer(hour_numeric == 19),
    hour20 = as.integer(hour_numeric == 20)
  )

# Now, perform group_by, summarize, and left_join
df_test_final <- df_test_mutated %>%
  group_by(tenmin_interval) %>%
  summarize(count = n(), avg_fare = mean(fare_amount)) %>%
  arrange(tenmin_interval) %>%
  left_join(df_test_mutated, by = "tenmin_interval") %>%
  select(-tenmin_interval, -avg_fare, -id, -pickup_time, -pickup_date, -hour_numeric, -time_of_day) %>%
  mutate(zero_indicator = as.integer(zero_indicator))

df_test <- df_test_final
# View the updated DataFrame
glimpse(df_test_final)





# Create a new column 'time_of_day' to represent the time within the day
df_train <- df_train %>%
  mutate(time_of_day = as.POSIXct(pickup_time, format = '%T'))

# Create a new column 'tenmin_interval' to represent the 10-minute interval
df_train <- df_train %>%
  mutate(tenmin_interval = format(ceiling_date(time_of_day, '10 mins'), "%H%M"))

# Calculate average fare_amount and count for each 10-minute interval
tenmin_interval_stats <- df_train %>%
  group_by(tenmin_interval) %>%
  summarize(count = n(), avg_fare = mean(fare_amount)) %>%
  arrange(tenmin_interval)  # Order by tenmin_interval

# Add a new column 'tenmin_count' to represent the count for each 10-minute interval
df_train <- df_train %>%
  left_join(tenmin_interval_stats, by = "tenmin_interval") %>%
  select(-tenmin_interval, -avg_fare)  # Remove unnecessary columns






# Create dummy columns for the specified hours
df_train <- df_train %>%
  mutate(hour_numeric = hour(pickup_time))

# Create dummy columns for the specified hours
df_train <- df_train %>%
  mutate(
    hour4 = as.integer(hour_numeric == 4),
    hour5 = as.integer(hour_numeric == 5),
    hour19 = as.integer(hour_numeric == 19),
    hour20 = as.integer(hour_numeric == 20)
  ) 
# View the updated DataFrame
glimpse(df_train)


df_train <- subset(df_train, select = -c(id,pickup_time, pickup_date, hour_numeric, time_of_day))
df_train <- df_train %>%
  mutate(zero_indicator = as.integer(zero_indicator))



lgbm = boost_tree(trees = tune(), learn_rate = tune())%>%
  set_mode("regression")


rec_spec <- recipe(fare_amount ~ ., df_train)


xgb_spec_stage_1 <- boost_tree(
  mode   = "regression",
  engine = "lightgbm",
  learn_rate = tune()
)

set.seed(123)
grid_stage_1 <- grid_random(
  learn_rate(),
  size = 10
)

wflw_xgb_stage_1 <- workflow() %>%
  add_model(xgb_spec_stage_1) %>%
  add_recipe(rec_spec)

set.seed(123)
cv_folds <- vfold_cv(df_train, v = 5)
tune_stage_1 <- tune_grid(
  wflw_xgb_stage_1,
  resamples = cv_folds,
  grid      = grid_stage_1,
  metrics   = metric_set(mape),
  control   = control_grid(verbose = TRUE)
)


best_params_xgb_1 <- tune_stage_1 %>% collect_metrics() %>% arrange(mean) %>%
  filter(row_number() == 1)

xgb_spec_stage_2 <- xgb_spec_stage_1 %>%
  set_args(
    learn_rate     = best_params_xgb_1$learn_rate,
    tree_depth     = tune(),
    loss_reduction = tune(),
    stop_iter      = tune()
  )
wflw_xgb_stage_2 <- wflw_xgb_stage_1 %>%
  update_model(xgb_spec_stage_2)

# * Define Stage 2 grid ----
set.seed(123)
grid_stage_2 <- grid_random(
  tree_depth(),
  loss_reduction(),
  stop_iter(),
  size = 10
)

# * Tune stage 2 -----
tune_stage_3 <- tune_grid(
  wflw_xgb_stage_2,
  resamples = cv_folds,
  grid      = grid_stage_2,
  metrics   = metric_set(mape),
  control   = control_grid(verbose = TRUE)
)

tune_stage_3 %>% collect_metrics() %>% arrange(mean) %>%  filter(row_number() <= 3)


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
  result <- all_results %>% filter(row_number() <= 3)
  result[[param_name]] <- best_params_model_1[[1]]

  return(list(result = result, all_results = all_results, grid = tune_stage_2))
}





set.seed(123)
# Example usage
set_seed_grid <- function(grid, seed) {
  set.seed(seed)
  grid
}
mtryval <- sqrt(length(df_train))

models <- list(
  lgbm = list(
    model_name = 'lgbm',
    model_1 = boost_tree(mode = "regression", engine = "lightgbm", learn_rate = tune()),
    model_2_args = list(tree_depth = tune(), loss_reduction = tune(), stop_iter = tune()),
    rec_spec = recipe(fare_amount ~ ., df_train),
    grid_1 = set_seed_grid(grid_latin_hypercube(learn_rate(), size = 10), 123),
    grid_2 = set_seed_grid(grid_latin_hypercube(tree_depth(), loss_reduction(), stop_iter(), size = 10), 123)
  ),
  xgboost = list(
    model_name = 'xgboost',
    model_1 = boost_tree(mode = "regression", engine = "xgboost", learn_rate = tune()),
    model_2_args = list(tree_depth = tune(), loss_reduction = tune(), stop_iter = tune()),
    rec_spec = recipe(fare_amount ~ ., df_train),
    grid_1 = set_seed_grid(grid_latin_hypercube(learn_rate(), size = 10), 123),
    grid_2 = set_seed_grid(grid_latin_hypercube(tree_depth(), loss_reduction(), stop_iter(), size = 10), 123)
  ),
  decision_tree = list(
    model_name = 'decision_tree',
    model_1 = decision_tree(mode = "regression", engine = "rpart", cost_complexity = tune()),
    model_2_args = list(tree_depth = tune(), min_n = tune()),
    rec_spec = recipe(fare_amount ~ ., df_train),
    grid_1 = set_seed_grid(grid_latin_hypercube(cost_complexity(), size = 10), 123),
    grid_2 = set_seed_grid(grid_latin_hypercube(tree_depth(), min_n(), size = 10), 123)
  ),
  random_forest = list(
    model_name = 'random_forest',
    model_1 = rand_forest(mode = "regression", engine = "ranger", mtry = 5, trees = tune()),
    model_2_args = list(min_n = tune()),
    rec_spec = recipe(fare_amount ~ ., df_train),
    grid_1 = set_seed_grid(grid_latin_hypercube(trees(), size = 10), 123),
    grid_2 = set_seed_grid(grid_latin_hypercube(min_n(), size = 10), 123)
  )
)



model_names <- c('lgbm', 'xgboost', 'decision_tree', 'random_forest')


# Loop over models
for (model_name in model_names) {
  result <- tune_two_stages(models, model_name)
  
  # You can use the 'result' object as needed, for example, printing or saving the results
  print(paste("Result for", model_name, ":", result))
  
  # If you want to save the results, you can use a list or another appropriate data structure
  results_list[[model_name]] <- result
}


# If you want to save all the results in a list
results_list6 <- list(
  lgbm = tune_two_stages(models, 'lgbm'),
  decision_tree = tune_two_stages(models, 'decision_tree'),
  xgboost = tune_two_stages(models, 'xgboost')
)
results_list6 <- list(
  random_forest = tune_two_stages(models, 'random_forest')
)

Sresults_rf <- data.frame()

for (i in 1:3) {
  tuned_params <- results_list3$random_forest$result[i, c("min_n", "trees")]
  

  rounded_params <- round(as.numeric(tuned_params), 2)
  

  lgbm_model <- rand_forest(
    mode = "regression",
    engine = "ranger",
    min_n = rounded_params[1],
    trees = as.integer(rounded_params[2]),
    mtry = 6
    #stop_iter = as.integer(rounded_params[4])
  )

  rec_spec <- recipe(fare_amount ~ ., df_train)
  
  wflw_model <- workflow() %>%
    add_model(lgbm_model) %>%
    add_recipe(rec_spec)
  

  trained_model <- wflw_model %>% fit(data = df_train)
  predictions <- predict(trained_model, new_data = df_test)
  mape_test <- mean(abs(df_test$fare_amount - predictions$.pred) / df_test$fare_amount) * 100
  
  iteration_df <- data.frame(
    iteration = i,
    mean_train = results_list$lgbm$result[i, "mean"],
    Mape_test = mape_test,
    models_spec = paste(names(tuned_params), "=", rounded_params, collapse = ", "),
    model_name = "decision_tree"
  )
  
  results_rf <- bind_rows(
    , iteration_df)
}

print(results_df)

results_rf$mean <- results_list3$random_forest$result$mean
results_rf <- results_rf[, c("iteration", "mean", "Mape_test", "models_spec", "model_name")]
results_rf$models_spec <- paste(results_rf$models_spec, "mtry = 6", sep = ", ")


