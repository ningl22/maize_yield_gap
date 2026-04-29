library(caret)
library(ranger)
library(dplyr)
library(ggplot2)
library(patchwork) 
library(rio)

rm(list = ls())

rf.YGC.data <- import("~/country_level/YGC_country_all.xlsx") %>%
rf.YGC.data$water_mana <- as.factor(rf.YGC.data$water_mana) 

set.seed(1234)
 
folds <- createFolds(y = rf.YGC.data$water_mana, k = 10, returnTrain = TRUE)

ctrl <- trainControl(
  method = "cv",
  number = 10,
  index = folds, 
  savePredictions = "final"
)
 
tune_grid <- expand.grid(
  mtry = c(2, 4, 6, 8, 10), 
  splitrule = c("variance", "extratrees"),
  min.node.size = c(1, 3, 5, 10)  
)
 
rf_cv_yrr <- train(
  YRR ~ ., 
  data = rf.YGC.data,
  method = "ranger",
  trControl = ctrl,
  tuneGrid = tune_grid,
  num.trees = 500,
  importance = "permutation"
)

mean(rf_cv_yrr$resample$Rsquared)
cv_rmse <- mean(rf_cv_yrr$resample$RMSE)
final_model <- rf_cv_yrr$finalModel
 
saveRDS(rf_cv_yrr, file = ".../rf_cv_ygc_cv10.rds")
 
