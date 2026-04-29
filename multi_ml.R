library(caret)
library(e1071)       # SVR
library(glmnet)      # Ridge
library(ranger)      # Random Forest
library(xgboost)     # GBDT
library(nnet)        # FNN
library(keras)       # For ResNet if you really need deep learning
library(tensorflow)
library(torch)
library(rio)
library(dplyr)
library(tidyr)

rm(list=ls())
 
save_dir <- "~/GlobaS/model_train/ML_multi/"

ml.data <- import("~data_raw/rf_input_data.xlsx") %>%
  mutate(water_mana = as.factor(water_mana)) 
 
## Base Setting 
set.seed(1234)

my_folds <- createFolds(y = ml.data$water_mana, k = 10, returnTrain = TRUE)
ctrl <- trainControl(
  method = "cv",
  number = 10,
  index = my_folds,
  savePredictions = "final"
)

#1: SVR #####################################################################

svm_model <- train(
  Total_Yield ~ ., 
  data = ml.data,
  method = "svmRadial",
  trControl = ctrl,
  preProcess = c("center", "scale"),
  tuneLength = 5
)

svm_best <- svm_model$bestTune
svm_pred <- svm_model$pred
for (col in names(svm_best)) {
  svm_pred <- svm_pred[svm_pred[[col]] == svm_best[[col]], ]
}
metrics_vec <- postResample(svm_pred$pred, svm_pred$obs)
svm_metrics <- data.frame(t(metrics_vec)) %>%
  mutate(Model = "SVR") %>%      
  relocate(Model) %>%         
  rename(R2 = Rsquared)

saveRDS(svm_model, file = file.path(save_dir, "svr_model.rds"))
write.csv(svm_pred, file.path(save_dir, "svr_pred_results.csv"), row.names = FALSE)


#2: XGB ####################################################################

gbdt_model <- train(
  Total_Yield ~ ., 
  data = ml.data,
  method = "xgbTree",
  trControl = ctrl,
  preProcess = c("center", "scale"),
  tuneLength = 5)

gbdt_best <- gbdt_model$bestTune
gbdt_pred <- gbdt_model$pred
for (col in names(gbdt_best)) {
  gbdt_pred <- gbdt_pred[gbdt_pred[[col]] == gbdt_best[[col]], ]
}
gbdt_vec <- postResample(gbdt_pred$pred, gbdt_pred$obs)
gbdt_metrics <- data.frame(t(gbdt_vec)) %>%
  mutate(Model = "GBDT") %>%      
  relocate(Model) %>%         
  rename(R2 = Rsquared)

saveRDS(gbdt_model, file = file.path(save_dir, "gbdt_xgbTree_model.rds"))
write.csv(gbdt_pred, file.path(save_dir, "gbdt_pred_results.csv"), row.names = FALSE)

#3: Ridge ##############################

ridge_model <- train(
  Total_Yield ~ ., 
  data = ml.data,
  method = "glmnet",
  trControl = ctrl,
  preProcess = c("center", "scale"),
  tuneLength = 5
)

ridge_best <- ridge_model$bestTune
ridge_pred <- ridge_model$pred
for (col in names(ridge_best)) {
  ridge_pred <- ridge_pred[ridge_pred[[col]] == ridge_best[[col]], ]
}
ridge_vec <- postResample(ridge_pred$pred, ridge_pred$obs)
ridge_metrics <- data.frame(t(ridge_vec )) %>%
  mutate(Model = "Ridge") %>%      
  relocate(Model) %>%         
  rename(R2 = Rsquared)

saveRDS(ridge_model, file = file.path(save_dir, "ridge_glmnet_model.rds"))
write.csv(ridge_pred, file.path(save_dir, "ridge_results.csv"), row.names = FALSE)

#4: FNN ######################################################

 
grid_nnet <- expand.grid(
  size = c(5, 10, 15),   
  decay = c(0, 0.01, 0.1)   
)

 
set.seed(1234)
fnn_model <- caret::train(
  Total_Yield ~ ., 
  data = ml.data,
  method = "nnet",
  trControl = ctrl,
  preProcess = c("center", "scale"),
  tuneGrid = grid_nnet,
  trace = FALSE,
  linout = TRUE   
)

bt <- fnn_model$bestTune
fnn_pred <- fnn_model$pred
for (col in names(bt)) {
  fnn_pred <- fnn_pred[fnn_pred[[col]] == bt[[col]], ]
}

fnn_vec <- caret::postResample(fnn_pred$pred, fnn_pred$obs)
fnn_metrics <- data.frame(t(fnn_vec)) %>%
  mutate(Model = "FNN") %>%      
  relocate(Model) %>%         
  rename(R2 = Rsquared)

saveRDS(fnn_model, file = file.path(save_dir, "fnn_model_nnet.rds"))
write.csv(fnn_pred, file.path(save_dir, "fnn_model_results.csv"), row.names = FALSE)


#5: ResNet ###############################################

rm(list = ls())
save_dir <- "~/GlobaS/model_train/ML_multi/"

ml.data <- import("~data_raw/rf_input_data.xlsx") %>%
  mutate(water_mana = as.factor(water_mana)) 
 
#### (1)  ResNet Block 
ResNetBlock <- nn_module(
  "ResNetBlock",
  initialize = function(input_dim, hidden_dim, dropout = 0.2) {
    self$linear1 <- nn_linear(input_dim, hidden_dim)
    self$bn1     <- nn_batch_norm1d(hidden_dim)
    self$relu    <- nn_relu()
    self$dropout <- nn_dropout(dropout)
    self$linear2 <- nn_linear(hidden_dim, input_dim) 
    self$bn2     <- nn_batch_norm1d(input_dim)
  },
  forward = function(x) {
    identity <- x 
    out <- self$linear1(x)
    out <- self$bn1(out)
    out <- self$relu(out)
    out <- self$dropout(out)
    out <- self$linear2(out)
    out <- self$bn2(out)
    out + identity  
  }
)
#### (2) Tabular ResNet 
TabularResNet <- nn_module(
  "TabularResNet",
  initialize = function(input_dim, hidden_dim = 64) {
    self$input_layer <- nn_linear(input_dim, hidden_dim)
    self$relu <- nn_relu()
    self$block1 <- ResNetBlock(hidden_dim, hidden_dim)
    self$block2 <- ResNetBlock(hidden_dim, hidden_dim)
    
    self$output_layer <- nn_linear(hidden_dim, 1)
  },
  forward = function(x) {
    x <- self$input_layer(x) %>% self$relu()
    x <- self$block1(x) %>% self$relu()
    x <- self$block2(x) %>% self$relu()
    x <- self$output_layer(x)
    x
  }
)

dummies_mana <- model.matrix(~ water_mana - 1, data = ml.data)
 
continuous_data <- ml.data %>% select(-Total_Yield, -water_mana)
continuous_cols_idx <- 1:ncol(continuous_data)  
 
x_full_matrix <- cbind(as.matrix(continuous_data), dummies_mana)
y_full_vector <- as.matrix(ml.data$Total_Yield)

k_folds <- 10
folds <- createFolds(ml.data$water_mana, k = k_folds, list = TRUE, returnTrain = FALSE)

all_predictions <- data.frame()
 

# ==========================================
#  Cross-validation
# ==========================================
for(i in 1:k_folds) {
 
  test_idx <- folds[[i]]
  train_idx <- setdiff(1:nrow(ml.data), test_idx)
 
  x_train_raw <- x_full_matrix[train_idx, ]
  x_test_raw  <- x_full_matrix[test_idx, ]
  y_train_t   <- torch_tensor(y_full_vector[train_idx], dtype = torch_float())
  y_test_t    <- torch_tensor(y_full_vector[test_idx], dtype = torch_float())
 
  train_means <- apply(x_train_raw[, continuous_cols_idx], 2, mean)
  train_sds   <- apply(x_train_raw[, continuous_cols_idx], 2, sd)
  train_sds[train_sds == 0] <- 1 
  partial_scale <- function(mat, means, sds, c_idx) {
 
    out <- mat 
    out[, c_idx] <- sweep(sweep(mat[, c_idx], 2, means, "-"), 2, sds, "/")
    return(out)
  }
 
  x_train_scaled <- partial_scale(x_train_raw, train_means, train_sds, continuous_cols_idx)
  x_test_scaled  <- partial_scale(x_test_raw, train_means, train_sds, continuous_cols_idx)
 
  x_train_t <- torch_tensor(x_train_scaled, dtype = torch_float())
  x_test_t  <- torch_tensor(x_test_scaled, dtype = torch_float())
 
  input_dim <- ncol(x_train_t) 
  model <- TabularResNet(input_dim) 
  optimizer <- optim_adam(model$parameters, lr = 0.01)
  loss_fn <- nn_mse_loss()
  
  # --- training ---
  epochs <- 100 
  for (epoch in 1:epochs) {
    model$train()
    optimizer$zero_grad()
    y_p <- model(x_train_t)
    loss <- loss_fn(y_p, y_train_t$view(c(-1, 1)))
    loss$backward()
    optimizer$step()
  }
  
  # --- Prediction ---
  model$eval()
  with_no_grad({
    preds_t <- model(x_test_t)
  })
  
  # --- results---
  fold_res <- data.frame(
    obs = as.numeric(y_test_t),
    pred = as.numeric(preds_t),
    Fold = i,
    Original_Index = test_idx
  )
  all_predictions <- rbind(all_predictions, fold_res)
  
  cat(sprintf("Fold %d/%d Done.\n", i, k_folds))
}

final_r2 <- caret::R2(all_predictions$pred, all_predictions$obs)
final_rmse <- caret::RMSE(all_predictions$pred, all_predictions$obs)
final_mae  <- caret::MAE(all_predictions$pred, all_predictions$obs)
ResNet_metrics <- data.frame(
  Model = "ResNet",
  R2 = final_r2,
  RMSE = final_rmse,
  MAE = final_mae)

write.csv(all_predictions, file.path(save_dir, "resnet_results.csv"), row.names = FALSE)
 
full_means <- apply(x_full_matrix[, continuous_cols_idx], 2, mean)
full_sds   <- apply(x_full_matrix[, continuous_cols_idx], 2, sd)
full_sds[full_sds == 0] <- 1
 
final_scale_func <- function(mat) {
  out <- mat
  out[, continuous_cols_idx] <- sweep(sweep(mat[, continuous_cols_idx], 2, full_means, "-"), 2, full_sds, "/")
  return(out)
}

 
x_final_tensor <- torch_tensor(final_scale_func(x_full_matrix), dtype = torch_float())
y_final_tensor <- torch_tensor(y_full_vector, dtype = torch_float())

 
input_dim <- ncol(x_final_tensor)
final_model <- TabularResNet(input_dim)
optimizer <- optim_adam(final_model$parameters, lr = 0.01)
loss_fn <- nn_mse_loss()

 
final_epochs <- 150 
for (epoch in 1:final_epochs) {
  final_model$train()
  optimizer$zero_grad()
  
  y_pred <- final_model(x_final_tensor)
  loss <- loss_fn(y_pred, y_final_tensor$view(c(-1, 1)))
  
  loss$backward()
  optimizer$step()
  
  if (epoch %% 50 == 0) cat(sprintf("Final Training Epoch %d: Loss %.2f\n", epoch, loss$item()))
}
torch_save(final_model, file.path(save_dir, "final_resnet_model.pt"))

 
preprocess_params <- list(
  means = full_means,           
  sds = full_sds,               
  cont_idx = continuous_cols_idx, 
  levels_mana = levels(ml.data$water_mana)  
)

saveRDS(preprocess_params, file.path(save_dir, "model_preprocess_params.rds"))

 
f_metrics_all <- bind_rows(svm_metrics, gbdt_metrics,ridge_metrics,fnn_metrics,ResNet_metrics)
write.csv(f_metrics_all, file.path(save_dir, "multi_model_metics.csv"), row.names = FALSE)

