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