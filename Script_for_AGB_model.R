# Load necessary libraries
library(dplyr)
library(xgboost)
library(caret)
library(randomForest)
library(hydroGOF)
library(terra)
library(parallel)

# Set working directory
setwd("F:/wangjiaxi/数据/GF7_rainstorm/9_AGB_factors")

# Data paths
beijing_path <- "F:/wangjiaxi/数据/GF7_rainstorm/9_AGB_factors/BJ/30m/BJ_point.csv"
luochuan_path <- "F:/wangjiaxi/数据/GF7_rainstorm/9_AGB_factors/LC/30m/LC_point.csv"
zixing_path <- "F:/wangjiaxi/数据/GF7_rainstorm/9_AGB_factors/ZX/30m/ZX_point.csv"
output_path <- "F:/wangjiaxi/数据/GF7_rainstorm/9_AGB_factors/cleaned_data.csv"

# Read datasets
beijing_data <- read.csv(beijing_path)
luochuan_data <- read.csv(luochuan_path)
zixing_data <- read.csv(zixing_path)

# Check structure
str(beijing_data)
str(luochuan_data)
str(zixing_data)

# Data cleaning: remove rows with -9999 or 0 values
clean_data <- function(data) {
  data <- data %>% filter_all(all_vars(. != -9999& . != 0))
  return(data)
}

beijing_data_clean <- clean_data(beijing_data)
luochuan_data_clean <- clean_data(luochuan_data)
zixing_data_clean <- clean_data(zixing_data)

# Combine all data
all_data <- bind_rows(beijing_data_clean, luochuan_data_clean, zixing_data_clean)

# Check combined data structure
str(all_data)

# Save cleaned data
write.csv(all_data, output_path, row.names = FALSE)

# Reload cleaned data if needed
all_data <- read.csv(output_path)


# ===========================
# Build AGB model (XGBoost)
# ===========================

set.seed(66)

# Random seeds were fixed to ensure reproducibility.
# Parallel computation was not enforced within XGBoost or RF,
# ensuring deterministic model behavior across runs.

train_index <- createDataPartition(all_data$AGB, p = 0.7, list = FALSE)
train_data <- all_data[train_index, ]
test_data <- all_data[-train_index, ]

dtrain <- xgb.DMatrix(
  data = as.matrix(train_data[, !colnames(train_data) %in% "AGB"]),
  label = log1p(train_data$AGB)
)

dtest <- xgb.DMatrix(
  data = as.matrix(test_data[, !colnames(test_data) %in% "AGB"]),
  label = log1p(test_data$AGB)
)


param_grid <- expand.grid(
  nrounds = c(300, 500),
  eta = c(0.05, 0.1),
  max_depth = c(6, 8, 10),
  min_child_weight = c(1, 5),
  gamma = c(0, 1),
  colsample_bytree = c(0.8, 1.0),
  subsample = c(0.9, 1.0)
)

best_rmse <- Inf
best_params <- NULL
best_nrounds <- 0

for (i in 1:nrow(param_grid)) {
  cat("Processing parameter set", i, "of", nrow(param_grid), "\n")
  params <- list(
    eta = param_grid$eta[i],
    max_depth = param_grid$max_depth[i],
    min_child_weight = param_grid$min_child_weight[i],
    gamma = param_grid$gamma[i],
    colsample_bytree = param_grid$colsample_bytree[i],
    subsample = param_grid$subsample[i],
    objective = "reg:squarederror",
    eval_metric = "rmse"
  )
  
  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = param_grid$nrounds[i],
    nfold = 10,
    early_stopping_rounds = 10,
    verbose = 0
  )
  
  min_rmse <- min(cv$evaluation_log$test_rmse_mean)
  if (min_rmse < best_rmse) {
    best_rmse <- min_rmse
    best_params <- params
    best_nrounds <- which.min(cv$evaluation_log$test_rmse_mean)
  }
}

# Save best params and rounds
saveRDS(list(best_params = best_params, best_nrounds = best_nrounds), "best_model_params.rds")
cat("Best parameters and rounds saved\n")

# Load best params
model_params <- readRDS("best_model_params.rds")
best_params <- model_params$best_params
best_nrounds <- model_params$best_nrounds

# Train final AGB model
model_agb <- xgb.train(
  params = best_params,
  data = dtrain,
  nrounds = best_nrounds
)

xgb.save(model_agb, "agb_model.xgb")

# Feature importance
importance_agb <- xgb.importance(model = model_agb)
write.csv(importance_agb, "importance_agb.csv", row.names = TRUE)


# Model evaluation
train_predictions_log <- predict(model_agb, dtrain)
test_predictions_log  <- predict(model_agb, dtest)

train_predictions_agb <- expm1(train_predictions_log)
test_predictions_agb  <- expm1(test_predictions_log)

# Metrics (train)
train_r_squared <- 1 - sum((train_data$AGB - train_predictions_agb)^2) / sum((train_data$AGB - mean(train_data$AGB))^2)
train_rmse <- rmse(train_data$AGB, train_predictions_agb)
train_mae <- mae(train_data$AGB, train_predictions_agb)
train_pbias <- 100 * sum(train_predictions_agb - train_data$AGB) / sum(train_data$AGB)
train_nse <- NSE(train_predictions_agb, train_data$AGB)

# Metrics (test)
test_r_squared <- 1 - sum((test_data$AGB - test_predictions_agb)^2) / sum((test_data$AGB - mean(test_data$AGB))^2)
test_rmse <- rmse(test_data$AGB, test_predictions_agb)
test_mae <- mae(test_data$AGB, test_predictions_agb)
test_pbias <- 100 * sum(test_predictions_agb - test_data$AGB) / sum(test_data$AGB)
test_nse <- NSE(test_predictions_agb, test_data$AGB)

# NSE was computed following the standard Nash–Sutcliffe definition
# (hydroGOF::NSE), with predictions as "sim" and observations as "obs"

metrics <- data.frame(
  Metric = c("R-squared", "RMSE", "MAE", "PBIAS", "NSE"),
  Train = c(train_r_squared, train_rmse, train_mae, train_pbias, train_nse),
  Test = c(test_r_squared, test_rmse, test_mae, test_pbias, test_nse)
)

write.csv(metrics, "agb_model_metrics.csv", row.names = FALSE)

# Save predictions
write.csv(data.frame(Actual = train_data$AGB, Predicted = train_predictions_agb), "train_predictions.csv", row.names = FALSE)
write.csv(data.frame(Actual = test_data$AGB, Predicted = test_predictions_agb), "test_predictions.csv", row.names = FALSE)


# ================================
# Error model (Random Forest)
# ================================

set.seed(66)
train_predictions <- read.csv("train_predictions.csv")
test_predictions <- read.csv("test_predictions.csv")

train_data$error <- train_predictions$Actual - train_predictions$Predicted
test_data$error <- test_predictions$Actual - test_predictions$Predicted

write.csv(train_data, "train_data.csv", row.names = FALSE)
write.csv(test_data, "test_data.csv", row.names = FALSE)

all_error_data <- bind_rows(train_data, test_data)

num_cores <- detectCores() - 1

param_grid_rf <- expand.grid(
  ntree = c(100, 200, 300),
  mtry = c(3, 5, 7),
  nodesize = c(5, 10, 20)
)

best_rf_rmse <- Inf
best_rf_params <- NULL

for (i in 1:nrow(param_grid_rf)) {
  cat("Processing RF parameter set", i, "of", nrow(param_grid_rf), "\n")
  
  params <- param_grid_rf[i, ]
  
  rf_model <- randomForest(error ~ ., 
                           data = all_error_data[, !colnames(all_error_data) %in% c("AGB")],
                           ntree = params$ntree, mtry = params$mtry, nodesize = params$nodesize,
                           importance = TRUE)
  
  preds <- predict(rf_model, newdata = all_error_data)
  rmse_rf <- sqrt(mean((all_error_data$error - preds)^2))
  
  if (rmse_rf < best_rf_rmse) {
    best_rf_rmse <- rmse_rf
    best_rf_params <- params
  }
}

cat("Best RF parameters:\n")
print(best_rf_params)

final_rf_model <- randomForest(error ~ ., 
                               data = all_error_data[, !colnames(all_error_data) %in% c("AGB")],
                               ntree = best_rf_params$ntree, mtry = best_rf_params$mtry,
                               nodesize = best_rf_params$nodesize, importance = TRUE)

saveRDS(final_rf_model, "final_rf_model.rds")
cat("Final RF model saved\n")

factor_importance <- importance(final_rf_model)
importance_table <- as.data.frame(factor_importance)
importance_table$Feature <- rownames(importance_table)
colnames(importance_table) <- c("%IncMSE", "IncNodePurity", "Feature")

write.csv(importance_table, "factor_importance_rf.csv", row.names = FALSE)

error_predictions <- predict(final_rf_model, newdata = all_error_data)

rmse_error <- sqrt(mean((all_error_data$error - error_predictions)^2))
mae_error <- mean(abs(all_error_data$error - error_predictions))
r_squared_error <- 1 - sum((all_error_data$error - error_predictions)^2) / sum((all_error_data$error - mean(all_error_data$error))^2)
nse_error <- 1 - sum((all_error_data$error - error_predictions)^2) / sum((all_error_data$error - mean(all_error_data$error))^2)
pbias_error <- 100 * sum(all_error_data$error - error_predictions) / sum(all_error_data$error)

metrics_error <- data.frame(
  Metric = c("RMSE", "MAE", "R-squared", "NSE", "PBIAS"),
  Value = c(rmse_error, mae_error, r_squared_error, nse_error, pbias_error)
)

write.csv(metrics_error, file = "error_model_metrics.csv", row.names = FALSE)

write.csv(data.frame(Actual = all_error_data$error, Predicted = error_predictions), "uncertainty_error_rf_model.csv", row.names = FALSE)


# ================================
# Predict AGB and error rasters
# ================================

# Raster prediction was implemented using a chunk-wise approach
# rather than terra::predict(), to ensure strict control over
# predictor order, memory usage, and compatibility with XGBoost models.

library(terra)

path.rasters <- list(
  Beijing = "F:/wangjiaxi/data/GF7_rainstorm/9_AGB_factors/BJ/data",
  Luochuan = "F:/wangjiaxi/data/GF7_rainstorm/9_AGB_factors/LC/data",
  Zixing = "F:/wangjiaxi/data/GF7_rainstorm/9_AGB_factors/ZX/data"
)

for (region in names(path.rasters)) {
  path.plot <- paste0("F:/wangjiaxi/data/GF7_rainstorm/9_AGB_factors/", region)
  rasterFiles <- list.files(path = path.rasters[[region]], pattern = '.tif$', full.names = TRUE)
  
  # Only GeoTIFF predictor layers (.tif) were loaded to avoid
  # accidental inclusion of auxiliary or temporary files
  
  cat("Reading raster files for", region, "\n")
  print(rasterFiles)
  
  predictors <- rast(rasterFiles)
  feature_names <- gsub(".tif", "", basename(rasterFiles))
  
  template_raster <- rast(predictors[[1]])
  
  model_agb <- xgb.load("agb_model.xgb")
  final_rf_model <- readRDS("final_rf_model.rds")
  
  predict_chunk <- function(start_row, end_row, model, predictors) {
    chunk_values <- terra::values(predictors, row = start_row, nrows = end_row - start_row + 1)
    colnames(chunk_values) <- feature_names
    predict(model, newdata = as.matrix(chunk_values))
  }
  
  chunk_size <- 1000
  
  pred_raster_agb <- template_raster
  values(pred_raster_agb) <- NA
  
  pred_raster_error <- template_raster
  values(pred_raster_error) <- NA
  
  n_rows <- nrow(pred_raster_agb)
  for (start_row in seq(1, n_rows, by = chunk_size)) {
    end_row <- min(start_row + chunk_size - 1, n_rows)
    pred_chunk_agb <- predict_chunk(start_row, end_row, model_agb, predictors)
    pred_raster_agb[start_row:end_row, ] <- pred_chunk_agb
    pred_chunk_error <- predict_chunk(start_row, end_row, final_rf_model, predictors)
    pred_raster_error[start_row:end_row, ] <- pred_chunk_error
  }
  
  output_file_agb <- file.path(path.plot, paste0("predicted_AGB_", region, ".tif"))
  output_file_error <- file.path(path.plot, paste0("predicted_error_", region, ".tif"))
  
  writeRaster(pred_raster_agb, filename = output_file_agb, overwrite = TRUE)
  writeRaster(pred_raster_error, filename = output_file_error, overwrite = TRUE)
  
  cat("Predicted AGB saved to:", output_file_agb, "\n")
  cat("Predicted error saved to:", output_file_error, "\n")
}
