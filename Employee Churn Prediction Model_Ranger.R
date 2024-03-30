## Employee Churn Prediction Model ##

library(caret)
library(ranger)
library(pROC)
library(ggplot2)
library(readxl)

#Random Forest using Ranger package on employee retention
#Load the data
dataset<- read_excel("Pharmaceutical Data.xlsx")
str(dataset)

#Remove irrelevant variables
dataset <- dataset[,-c(8,9,19,24)]

# Renaming columns to remove spaces and non-standard characters
names(dataset) <- gsub(" ", "_", names(dataset))
names(dataset) <- make.names(names(dataset))  # This will ensure all names are valid R variable names


# Convert character columns to factors
categorical_columns <- c("Leaving_the_company", "BusinessTravel", "Department", 
                         "Education_Field", "Gender", "Job_Role", 
                         "Marital_Status", "OverTime")

for (col in categorical_columns) {
  dataset[[col]] <- as.factor(dataset[[col]])
}


# Set the seed for reproducibility
set.seed(123)

# Split the data into training and testing sets
splitIndex <- createDataPartition(dataset$Leaving_the_company, p = .70, list = FALSE, times = 1)
trainData <- dataset[splitIndex, ]
testData  <- dataset[-splitIndex, ]

# Model fitting with ranger
rf_model <- ranger(
  formula         = Leaving_the_company ~ ., 
  data            = trainData, # Use trainData for training
  num.trees       = 280, 
  importance      = 'impurity', 
  classification  = TRUE,
  probability     = TRUE  # Enable probability prediction
)

# Variable Importance Visualization
var_importance <- importance(rf_model)
importance_df <- data.frame(
  Variable = names(var_importance),
  Importance = var_importance
)
importance_df <- importance_df[order(-importance_df$Importance), ]
ggplot(importance_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  xlab("Variable") +
  ylab("Importance") +
  ggtitle("Variable Importance in Random Forest Model")

### Model Tuning Step for 'mtry'
# Create a range of mtry values to explore
mtry_values <- round(seq(1, sqrt(ncol(trainData)), length.out = 8))
tuning_results <- data.frame(mtry = integer(), AUC = numeric())

for (mtry_val in mtry_values) {
  set.seed(123)
  # Fit the model with the current mtry value
  model <- ranger(
    formula = Leaving_the_company ~ ., 
    data = trainData, 
    num.trees = 500, 
    mtry = mtry_val, 
    importance = 'impurity',
    classification = TRUE,
    probability = TRUE
  )
  # Predict on the test set
  predictions <- predict(model, data = testData, type = "response")$predictions
  # Calculate AUC
  roc_res <- roc(response = as.factor(testData$Leaving_the_company), predictor = predictions[,2])
  auc_res <- auc(roc_res)
  tuning_results <- rbind(tuning_results, data.frame(mtry = mtry_val, AUC = auc_res))
}

# Determine the best 'mtry' value
best_mtry <- tuning_results[which.max(tuning_results$AUC), "mtry"]
print(best_mtry)

##########################################################
##########################################################
#Refit the model with the best_mtry

#class_weights <- c("No" = 1, "Yes" = 2)  # For example, weighting 'Yes' class three times more than 'No'

# Refit the model using the best mtry value obtained from tuning
refitted_rf_model <- ranger(
  formula = Leaving_the_company ~ ., 
  data = trainData,
  num.trees = 500,
  mtry = best_mtry, # Using the best mtry value from tuning
  importance = 'impurity',
  classification = TRUE,
  probability = TRUE # Ensuring probability estimation is enabled
  #class.weights = class_weights  # Applying the class weights
)

# Predicting with the refitted model
predictions_prob_refit <- predict(refitted_rf_model, data = testData)

# Accessing the predicted probabilities
probabilities_refit <- predictions_prob_refit$predictions

# Convert probabilities to factor predictions based on a threshold
predicted_classes <- ifelse(probabilities_refit[, "Yes"] > 0.3, "Yes", "No")
predicted_classes <- factor(predicted_classes, levels = c("No", "Yes"))

# Actual classes
actual_classes <- as.factor(testData$Leaving_the_company)

# Confusion Matrix
conf_mat <- confusionMatrix(predicted_classes, actual_classes)
print(conf_mat)


# Calculate AUC
roc_res <- roc(response = actual_classes, predictor = probabilities_refit[, "Yes"])
auc_res <- auc(roc_res)
print(auc_res)

# Plot ROC Curve
plot(roc_res, main = "ROC Curve", col = "blue", lwd = 2, legacy.axes = TRUE)


########################################################
#####################CROSS VALIDATION###################

set.seed(123) # For reproducibility

# Create 10-fold cross-validation indices
folds <- createFolds(dataset$Leaving_the_company, k = 10, list = TRUE, returnTrain = TRUE)

# Placeholder for each fold's AUC
auc_list <- numeric(length(folds))

for(i in seq_along(folds)) {
  # Splitting the data into training and testing sets based on folds
  train_indices <- folds[[i]]
  trainData <- dataset[train_indices, ]
  testData <- dataset[-train_indices, ]
  
  # Train the model
  rf_model <- ranger(
    formula = Leaving_the_company ~ ., 
    data = trainData,
    num.trees = 500,
    mtry = 5,
    importance = 'impurity',
    classification = TRUE,
    probability = TRUE
  )
  
  # Predict on the test set
  predictions_prob <- predict(rf_model, data = testData)
  
  # Apply custom threshold to classify predictions
  predicted_classes <- ifelse(predictions_prob$predictions[, "Yes"] > 0.5, "Yes", "No")
  
  # Evaluate the model
  roc_res <- roc(testData$Leaving_the_company, predictions_prob$predictions[, "Yes"])
  auc_list[i] <- auc(roc_res)
}

# Calculate the mean AUC across all folds
mean_auc <- mean(auc_list)
print(paste("Mean AUC across all folds:", mean_auc))
