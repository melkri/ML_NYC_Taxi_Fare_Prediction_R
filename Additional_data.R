# Create a new column 'time_of_day' to represent the time within the day
df_train_new <- df_train_new %>%
  mutate(time_of_day = as.POSIXct(pickup_time, format = '%T'))

# Create a new column 'tenmin_interval' to represent the 10-minute interval
df_train_new <- df_train_new %>%
  mutate(tenmin_interval = format(ceiling_date(time_of_day, '10 mins'), "%H%M"))

# Calculate average fare_amount and count for each 10-minute interval
tenmin_interval_stats <- df_train_new %>%
  group_by(tenmin_interval) %>%
  dplyr::summarize(count = n(), avg_fare = mean(fare_amount, na.rm = TRUE), .groups = "drop") %>%
  arrange(tenmin_interval)

# Plot count of records per 10-minute interval
ggplot(tenmin_interval_stats, aes(x = tenmin_interval, y = count)) +
  geom_bar(stat = "identity", fill = "lightgreen", color = "black") +
  labs(title = "Count of Records Per 10-Minute Interval",
       x = "10-Minute Interval",
       y = "Count of Records") +
  theme(axis.text.x = element_blank(), plot.background = element_rect(fill = "#f2ebe6"),
        panel.background = element_rect(fill = "#f2ebe6"),  # Remove x-axis tick text
        axis.ticks.x = element_blank(),  # Remove x-axis ticks
        axis.title.x = element_text(margin = margin(t = 20)))  # Adjust the title margin for better visibility

SS# Plot average fare_amount per 10-minute interval
ggplot(tenmin_interval_stats, aes(x = tenmin_interval, y = avg_fare)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  labs(title = "Average Fare Amount Per 10-Minute Interval",
       x = "10-Minute Interval",
       y = "Average Fare Amount") +
    theme(axis.text.x = element_blank(), plot.background = element_rect(fill = "#f2ebe6"),
        panel.background = element_rect(fill = "#f2ebe6"),  # Remove x-axis tick text
        axis.ticks.x = element_blank(),  # Remove x-axis ticks
        axis.title.x = element_text(margin = margin(t = 20)))   #  # Rotate x-axis labels for better visibility

S# Assuming your pickup_time is of class hms
df_train_new <- df_train_new %>%
  mutate(hour = hour(pickup_time),
         minute = minute(pickup_time),
         tenmin_interval = make_datetime(2000, 1, 1, hour, minute - minute %% 10, 0))

# Create a new column 'tenmin_count' to represent the count of records in each 10-minute interval
df_train_new <- df_train_new %>%
  group_by(hour, tenmin_interval) %>%
  mutate(tenmin_count = n()) %>%
  ungroup()

# Display the modified DataFrame
df_train_new %>% glimpse()

library(corrplot)

# Assuming df_train contains only numerical variables

# Calculate correlation matrix
cor_matrix <- cor(df_train)

# Create a modern, readable plot
corrplot(cor_matrix, method = "color", tl.col = "black", tl.srt = 45, addrect = 3)

# Add a title
title("Correlation Plot for df_train", cex.main = 1.5)

library(ggplot2)
library(corrplot)

# Assuming df_train contains only numerical variables

# Calculate correlation matrix
cor_matrix <- cor(df_train)

# Filter variables based on absolute correlation with fare_amount > 0.1
selected_variables <- names(which(abs(cor_matrix[, "fare_amount"]) > 0.1))


# Create a subset of df_train with selected variables
df_train_filtered <- df_train[, selected_variables]

# Calculate correlation matrix for the filtered variables
cor_matrix_filtered <- cor(df_train_filtered)

# Create a ggplot correlation plot with a custom background color
cor_matrix_long <- reshape2::melt(cor_matrix_filtered)

# Create a ggplot correlation plot with a custom background color
ggplot(data = cor_matrix_long, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "#fcbba1", high = "#3288bd", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", 
                       name = "Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_rect(fill = "#f2ebe6"),plot.background = element_rect(fill = "#f2ebe6")) +
  ggtitle("Filtered Correlation Plot for df_train")
SSS