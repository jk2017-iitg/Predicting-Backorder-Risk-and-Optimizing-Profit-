# Importing the libraries
library(tidyquant)
library(unbalanced)
library(h2o)

# Load training and test sets 
train_raw_df    <- read.csv("Kaggle_Training_Dataset_v2.csv")
test_raw_df     <- read.csv("Kaggle_Test_Dataset_v2.csv")

# Unbalanced data set
train_raw_df$went_on_backorder %>% table() %>% prop.table()

# train set: Percentage of complete cases
train_raw_df %>% complete.cases() %>% sum() / nrow(train_raw_df)

# test set: Percentage of complete cases
test_raw_df %>% complete.cases() %>% sum() / nrow(test_raw_df)

# Train/Validation Set Split
split_pct <- 0.85
n <- nrow(train_raw_df)
sample_size <- floor(split_pct * n)

set.seed(159)
idx_train <- sample(1:n, size = sample_size)

valid_raw_df <- train_raw_df[-idx_train,]
train_raw_df <- train_raw_df[idx_train,]

# Custom pre-processing function
preprocess_raw_data <- function(data) {
  # data = data frame of backorder data
  data %>%
    select(-sku) %>%
    drop_na(national_inv) %>%
    mutate(lead_time = ifelse(is.na(lead_time), -99, lead_time)) %>%
    mutate_if(is.character, .funs = function(x) ifelse(x == "Yes", 1, 0)) %>%
    mutate(went_on_backorder = as.factor(went_on_backorder))
}

# Apply the preprocessing steps
train_df <- preprocess_raw_data(train_raw_df) 
valid_df <- preprocess_raw_data(valid_raw_df) 
test_df  <- preprocess_raw_data(test_raw_df)

# Inspect the processed data
glimpse(train_df)

# Use SMOTE sampling to balance the dataset
input  <- train_df %>% select(-went_on_backorder)
output <- train_df$went_on_backorder 
train_balanced <- ubSMOTE(input, factor(output), perc.over = 200, perc.under = 200, k = 5)

# Recombine the synthetic balanced data
train_df <- bind_cols(as.tibble(train_balanced$X), tibble(went_on_backorder = train_balanced$Y))
train_df

# Inspect class balance after SMOTE
train_df$went_on_backorder %>% table() %>% prop.table() 

# Initiate h2o
h2o.init()
h2o.no_progress()

# Convert to H2OFrame
train_h2o <- as.h2o(train_df)
valid_h2o <- as.h2o(valid_df)
test_h2o  <- as.h2o(test_df)

# Automatic Machine Learning
y <- "went_on_backorder"
x <- setdiff(names(train_h2o), y)

automl_models_h2o <- h2o.automl(
  x = x, 
  y = y,
  training_frame    = train_h2o,
  validation_frame  = valid_h2o,
  leaderboard_frame = test_h2o,
  max_runtime_secs  = 45
)

automl_leader <- automl_models_h2o@leader

pred_h2o <- h2o.predict(automl_leader, newdata = test_h2o)
as.tibble(pred_h2o)

perf_h2o <- h2o.performance(automl_leader, newdata = test_h2o) 

# Getting performance metrics
h2o.metric(perf_h2o) %>%
  as.tibble() %>%
  glimpse()

# Plot ROC Curve
left_join(h2o.tpr(perf_h2o), h2o.fpr(perf_h2o)) %>%
  mutate(random_guess = fpr) %>%
  select(-threshold) %>%
  ggplot(aes(x = fpr)) +
  geom_area(aes(y = tpr, fill = "AUC"), alpha = 0.5) +
  geom_point(aes(y = tpr, color = "TPR"), alpha = 0.25) +
  geom_line(aes(y = random_guess, color = "Random Guess"), size = 1, linetype = 2) +
  theme_tq() +
  scale_color_manual(
    name = "Key", 
    values = c("TPR" = palette_dark()[[1]],
               "Random Guess" = palette_dark()[[2]])
  ) +
  scale_fill_manual(name = "Fill", values = c("AUC" = palette_dark()[[5]])) +
  labs(title = "ROC Curve", 
       subtitle = "Model is performing much better than random guessing") +
  annotate("text", x = 0.25, y = 0.65, label = "Better than guessing") +
  annotate("text", x = 0.75, y = 0.25, label = "Worse than guessing")

# AUC Calculation
h2o.auc(perf_h2o)