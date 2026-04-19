library(rio)
library(dplyr)
library(caret)
library(ranger)
library(fastshap)
library(shapviz)

# RF modeling ################################################################
 
set.seed(1234)
rf.data <- import("~data_raw/rf_input_data.xlsx") 
   
folds <- createFolds(y = rf.data$water_mana, k = 10, returnTrain = TRUE)

ctrl <- trainControl(method = "cv",number = 10,index = folds, savePredictions = "final")

rf_cv <- train(Total_Yield ~ ., data = rf.data,
               method = "ranger", 
               trControl = ctrl,
               tuneGrid = expand.grid(
                 mtry = c(3, 5, 7, 10),   
                 splitrule = c("variance", "extratrees"),
                 min.node.size = c(1, 5, 10)),  
               num.trees = 500,  
               importance = "permutation")

print(rf_cv$resample)
mean(rf_cv$resample$Rsquared)
final_model <- rf_cv$finalModel

## save model
#saveRDS(rf_cv, file = "~/Research/GlobaS/model_train/YP_ranger_cv10.rds")


#SHAP (using parallel) ----------------------------------
library(doParallel)
library(future.apply)

vars_expected <- rf_cv$finalModel$forest$independent.variable.names

pred_fun_caret <- function(object, newdata) {
  predict(object, newdata = newdata) 
}

X_all <- rf.data %>% dplyr::select(-Total_Yield)

# run parallel
n_cores <- parallel::detectCores() - 1
cl <- makeCluster(n_cores)
clusterSetRNGStream(cl, 1234) 

registerDoParallel(cl)

sh <- fastshap::explain(
  object        = rf_cv,   
  X             = X_all,
  pred_wrapper  = pred_fun_caret,
  nsim          = 256,                
  adjust        = TRUE)

imp_df_shap <- data.frame(
  Feature = colnames(sh),
  Importance = colMeans(abs(sh), na.rm = TRUE)
) %>% arrange(desc(Importance))

saveRDS(sh, file = "~/Research/GlobaS/model_train/YP_RF_shap_matrix.rds", compress = "xz")

## summarize 
rm(list = ls())
rf_cv <- readRDS("~/Research/GlobaS/model_train/YP_ranger_cv10.rds")
bt <- rf_cv$bestTune
pred_cv <-  subset(rf_cv$pred, mtry == bt$mtry)

metrics_vec <- postResample(pred_cv$pred, pred_cv$obs)
rf_metrics <- data.frame(t(metrics_vec)) %>%
  mutate(Model = "RF") %>%      
  relocate(Model) %>%         
  rename(R2 = Rsquared)

save_dir <- "~/model_train/ML_multi/"

write.csv(pred_cv, file.path(save_dir, "rf_pred_results.csv"), row.names = FALSE)
write.csv(rf_metrics, file.path(save_dir, "rf_model_metics.csv"), row.names = FALSE)


