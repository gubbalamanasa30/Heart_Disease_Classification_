# Set the file path
file_path <- "D:/Users/Heart Disease classification/heart_disease_health_indicators_BRFSS2015.csv"
# Import the CSV file
heart_data <- read.csv(file_path, stringsAsFactors = FALSE)

# View the structure of the data
str(heart_data)
table(heart_data$HeartDiseaseorAttack)

#undersampling (Balancing)
# Load required package
library(dplyr)

# Separate class 0 and class 1
class0 <- filter(heart_data, HeartDiseaseorAttack == 0)
class1 <- filter(heart_data, HeartDiseaseorAttack == 1)

# Undersample class 0 to match class 1 count
set.seed(123)  # For reproducibility
class0_sampled <- sample_n(class0, nrow(class1))

# Combine both classes
balanced_data <- bind_rows(class0_sampled, class1)

# Shuffle the rows
balanced_data <- balanced_data[sample(nrow(balanced_data)), ]

# Check class distribution
table(balanced_data$HeartDiseaseorAttack)

#Exploration
dim(balanced_data)
str(balanced_data)
colSums(is.na(balanced_data))
summary(balanced_data)
cor_matrix <- cor(balanced_data)
round(cor_matrix[, "HeartDiseaseorAttack"], 2)  # Correlation with target

#Visual Analysis
library(ggplot2)
library(dplyr)
library(corrplot)
library(RColorBrewer)


x11()
hist(balanced_data$Age, breaks = 10, col = "skyblue", main = "Age Distribution", xlab = "Age Category")
x11()
boxplot(BMI ~ HeartDiseaseorAttack, data = balanced_data,
        col = c("lightgreen", "tomato"), names = c("No", "Yes"),
        main = "BMI vs Heart Disease")

x11()
ggplot(balanced_data, aes(x = factor(HighBP), fill = factor(HeartDiseaseorAttack))) +
  geom_bar(position = "fill") +
  labs(x = "High Blood Pressure", y = "Proportion", fill = "Heart Disease",
       title = "HighBP vs Heart Disease") +
  scale_fill_manual(values = c("steelblue", "darkred"))
x11()
ggplot(balanced_data, aes(x = factor(Sex), fill = factor(HeartDiseaseorAttack))) +
  geom_bar(position = "fill") +
  labs(title = "Heart Disease vs Gender", x = "Sex (0 = Female, 1 = Male)", y = "Proportion", fill = "Heart Disease") +
  scale_fill_manual(values = c("#66c2a5", "#fc8d62"))

x11()
ggplot(balanced_data, aes(x = factor(PhysActivity), fill = factor(HeartDiseaseorAttack))) +
  geom_bar(position = "fill") +
  labs(title = "Heart Disease vs Physical Activity", x = "Physical Activity", y = "Proportion", fill = "Heart Disease") +
  scale_fill_manual(values = c("#8dd3c7", "#fb8072"))
x11()
ggplot(balanced_data, aes(x = factor(GenHlth), fill = factor(HeartDiseaseorAttack))) +
  geom_bar(position = "fill") +
  labs(title = "Heart Disease vs General Health Rating", x = "General Health (1 = Excellent to 5 = Poor)", y = "Proportion") +
  scale_fill_manual(values = c("#a6cee3", "#b2df8a"))

# Reattach target
cor_with_target <- cor(balanced_data)[, "HeartDiseaseorAttack"]
cor_with_target <- sort(cor_with_target[-which(names(cor_with_target) == "HeartDiseaseorAttack")])
top_features <- c(head(cor_with_target, 5), tail(cor_with_target, 5))
cor_df <- data.frame(Feature = names(top_features),
                     Correlation = as.numeric(top_features))

# Reorder for bar plot
cor_df$Feature <- factor(cor_df$Feature, levels = cor_df$Feature[order(cor_df$Correlation)])

x11(width = 8, height = 6)
ggplot(cor_df, aes(x = Feature, y = Correlation, fill = Correlation > 0)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  coord_flip() +
  scale_fill_manual(values = c("#fc8d62", "#66c2a5")) +
  labs(title = "Top Positive & Negative Correlations with Heart Disease",
       x = "Feature", y = "Correlation Coefficient")

x11()
ggplot(balanced_data, aes(x = Age, y = BMI, color = factor(HeartDiseaseorAttack))) +
  geom_point(alpha = 0.5) +
  labs(title = "Age vs BMI by Heart Disease", x = "Age Category", y = "BMI", color = "Heart Disease") +
  scale_color_manual(values = c("#1f78b4", "#e31a1c"))

#Model biulding
library(dplyr)
library(caret)
library(e1071)
library(MASS)
library(class)
library(pROC)
library(ggplot2)
library(ROCR)
library(rpart)
library(rpart.plot)
library(reshape2)

# Convert ordinal/factor variables
balanced_data <- balanced_data %>%
  mutate(
    HeartDiseaseorAttack = factor(HeartDiseaseorAttack),
    HighBP = factor(HighBP),
    HighChol = factor(HighChol),
    CholCheck = factor(CholCheck),
    Smoker = factor(Smoker),
    Stroke = factor(Stroke),
    Diabetes = factor(Diabetes),
    PhysActivity = factor(PhysActivity),
    Fruits = factor(Fruits),
    Veggies = factor(Veggies),
    HvyAlcoholConsump = factor(HvyAlcoholConsump),
    AnyHealthcare = factor(AnyHealthcare),
    NoDocbcCost = factor(NoDocbcCost),
    DiffWalk = factor(DiffWalk),
    Sex = factor(Sex),
    Education = factor(Education, ordered = TRUE),
    Income = factor(Income, ordered = TRUE),
    GenHlth = factor(GenHlth, ordered = TRUE)
  )

# Split data
set.seed(42)
split_idx <- createDataPartition(balanced_data$HeartDiseaseorAttack, p = 0.8, list = FALSE)
train_data <- balanced_data[split_idx, ]
test_data <- balanced_data[-split_idx, ]

# Scale numeric columns
num_vars <- c("BMI", "MentHlth", "PhysHlth", "Age")
preproc <- preProcess(train_data[, num_vars], method = c("center", "scale"))
train_scaled <- predict(preproc, train_data[, num_vars])
test_scaled <- predict(preproc, test_data[, num_vars])

# Combine scaled and categorical
train_full <- cbind(train_scaled, dplyr::select(train_data, -all_of(num_vars)))
test_full <- cbind(test_scaled, dplyr::select(test_data, -all_of(num_vars)))

# Add 2nd order polynomial terms
poly_terms <- function(data) {
  data %>%
    mutate(
      BMI2 = BMI^2,
      Age2 = Age^2,
      MentHlth2 = MentHlth^2,
      PhysHlth2 = PhysHlth^2,
      BMI_Age = BMI * Age,
      BMI_MH = BMI * MentHlth,
      BMI_PH = BMI * PhysHlth,
      Age_MH = Age * MentHlth,
      Age_PH = Age * PhysHlth,
      MH_PH = MentHlth * PhysHlth
    )
}

train_int <- poly_terms(train_scaled)
test_int <- poly_terms(test_scaled)

# Combine with original categorical variables
train_full_int <- cbind(train_full, dplyr::select(train_int, ends_with("2"), contains("_")))
test_full_int <- cbind(test_full, dplyr::select(test_int, ends_with("2"), contains("_")))


evaluate_model <- function(model, X_train, y_train, X_test, y_test, model_name) {
  if (model_name == "knn") {
    pred <- knn(train = X_train, test = X_test, cl = y_train, k = 5, prob = TRUE)
    prob <- attr(pred, "prob")
    pred_prob <- ifelse(pred == "1", prob, 1 - prob)
  } else if (model_name == "lda") {
    model <- lda(y_train ~ ., data = X_train)
    pred_prob <- predict(model, X_test)$posterior[,2]
    pred <- predict(model, X_test)$class
  } else if (model_name == "naivebayes") {
    model <- naiveBayes(y_train ~ ., data = X_train)
    pred_prob <- predict(model, X_test, type = "raw")[,2]
    pred <- predict(model, X_test)
  } else if (model_name == "tree") {
    pred_prob <- predict(model, X_test, type = "prob")[,2]  # Probability of class "1"
    pred <- predict(model, X_test, type = "class")
  } else {
    # Logistic regression
    pred_prob <- predict(model, X_test, type = "response")
    pred <- ifelse(pred_prob > 0.5, "1", "0")
  }
  
  pred <- factor(pred, levels = c("0", "1"))
  y_test <- factor(y_test, levels = c("0", "1"))
  
  cat("\nModel:", toupper(model_name), "\n")
  print(confusionMatrix(pred, y_test, positive = "1"))
  
  precision <- posPredValue(pred, y_test, positive = "1")
  recall <- sensitivity(pred, y_test, positive = "1")
  f1 <- (2 * precision * recall) / (precision + recall)
  cat(sprintf("Precision: %.3f | Recall: %.3f | F1: %.3f\n", precision, recall, f1))
  
  roc_obj <- roc(y_test, as.numeric(pred_prob))
  cat(sprintf("AUC: %.3f\n", auc(roc_obj)))
  
  plot(roc_obj, main = paste("ROC -", toupper(model_name)), col = "darkgreen")
  abline(0, 1, lty = 2)
}

logit_model1 <- glm(HeartDiseaseorAttack ~ ., data = train_full, family = binomial)
summary(logit_model1)
tree_model1 <- rpart(HeartDiseaseorAttack ~ ., data = train_full)
lda_model1 <- lda(HeartDiseaseorAttack ~ ., data = train_full)
lda_pred <- predict(lda_model1, newdata = test_full)
pred_class <- lda_pred$class
pred_prob <- lda_pred$posterior[, "1"]  # Probability of class "1"
#Actual labels
actual <- test_full$HeartDiseaseorAttack
#Convert to factors with correct levels
pred_class <- factor(pred_class, levels = c("0", "1"))
actual <- factor(actual, levels = c("0", "1"))
#Confusion matrix
cm <- confusionMatrix(pred_class, actual, positive = "1")
print(cm)
#Precision, Recall, F1
precision <- cm$byClass["Precision"]
recall <- cm$byClass["Recall"]
f1 <- 2 * (precision * recall) / (precision + recall)
cat(sprintf("Precision: %.3f | Recall: %.3f | F1 Score: %.3f\n", precision, recall, f1))
# ROC & AUC
roc_obj <- roc(actual, as.numeric(pred_prob))
auc_val <- auc(roc_obj)
cat(sprintf("AUC: %.3f\n", auc_val))
#Plot ROC Curve
plot(roc_obj, main = "ROC Curve - LDA", col = "blue")
abline(0, 1, lty = 2)

evaluate_model(logit_model1, train_full, train_full$HeartDiseaseorAttack, test_full, test_full$HeartDiseaseorAttack, "logit")
evaluate_model(tree_model1, train_full, train_full$HeartDiseaseorAttack, test_full, test_full$HeartDiseaseorAttack, "tree")
evaluate_model(NULL, train_full[, -1], train_full$HeartDiseaseorAttack, test_full[, -1], test_full$HeartDiseaseorAttack, "naivebayes")
#evaluate_model(lda_model1, train_full[, -1], train_full$HeartDiseaseorAttack, test_full[, -1], test_full$HeartDiseaseorAttack, "lda")
evaluate_model(NULL, train_full[, -1], train_full$HeartDiseaseorAttack, test_full[, -1], test_full$HeartDiseaseorAttack, "knn")

#models on original unbalanced data (main effects)
# Convert categorical variables properly
heart_data <- heart_data %>%
  mutate(
    HeartDiseaseorAttack = factor(HeartDiseaseorAttack),
    HighBP = factor(HighBP),
    HighChol = factor(HighChol),
    CholCheck = factor(CholCheck),
    Smoker = factor(Smoker),
    Stroke = factor(Stroke),
    Diabetes = factor(Diabetes),
    PhysActivity = factor(PhysActivity),
    Fruits = factor(Fruits),
    Veggies = factor(Veggies),
    HvyAlcoholConsump = factor(HvyAlcoholConsump),
    AnyHealthcare = factor(AnyHealthcare),
    NoDocbcCost = factor(NoDocbcCost),
    DiffWalk = factor(DiffWalk),
    Sex = factor(Sex),
    Education = factor(Education, ordered = TRUE),
    Income = factor(Income, ordered = TRUE),
    GenHlth = factor(GenHlth, ordered = TRUE)
  )

# Split into Train and Test sets (80%-20%)
set.seed(42)
split_idx <- createDataPartition(heart_data$HeartDiseaseorAttack, p = 0.8, list = FALSE)
train_data_orig <- heart_data[split_idx, ]
test_data_orig <- heart_data[-split_idx, ]

# Scale only numeric columns
num_vars_orig <- c("BMI", "MentHlth", "PhysHlth", "Age")
preproc_orig <- preProcess(train_data_orig[, num_vars_orig], method = c("center", "scale"))
train_scaled_orig <- predict(preproc_orig, train_data_orig[, num_vars_orig])
test_scaled_orig <- predict(preproc_orig, test_data_orig[, num_vars_orig])

# Combine scaled and categorical variables
train_full_orig <- cbind(train_scaled_orig, dplyr::select(train_data_orig, -all_of(num_vars_orig)))
test_full_orig <- cbind(test_scaled_orig, dplyr::select(test_data_orig, -all_of(num_vars_orig)))

# Logistic Regression
logit_model_orig <- glm(HeartDiseaseorAttack ~ ., data = train_full_orig, family = "binomial")

# Decision Tree
tree_model_orig <- rpart(HeartDiseaseorAttack ~ ., data = train_full_orig)

# LDA
lda_model_orig <- lda(HeartDiseaseorAttack ~ ., data = train_full_orig)
lda_pred_orig <- predict(lda_model_orig, newdata = test_full_orig)
pred_class_orig <- lda_pred_orig$class
pred_prob_orig <- lda_pred_orig$posterior[, "1"]  # Probability of class "1"
#Actual labels
actual <- test_full_orig$HeartDiseaseorAttack
#Convert to factors with correct levels
pred_class <- factor(pred_class_orig, levels = c("0", "1"))
actual_orig = test_full_orig$HeartDiseaseorAttack
actual <- factor(actual_orig, levels = c("0", "1"))
#Confusion matrix
cm <- confusionMatrix(pred_class, actual, positive = "1")
print(cm)
#Precision, Recall, F1
precision <- cm$byClass["Precision"]
recall <- cm$byClass["Recall"]
f1 <- 2 * (precision * recall) / (precision + recall)
cat(sprintf("Precision: %.3f | Recall: %.3f | F1 Score: %.3f\n", precision, recall, f1))
# ROC & AUC
roc_obj <- roc(actual, as.numeric(pred_prob_orig))
auc_val <- auc(roc_obj)
cat(sprintf("AUC: %.3f\n", auc_val))

# KNN and Naive Bayes will be handled through the evaluate function
evaluate_model(logit_model_orig, train_full_orig, train_full_orig$HeartDiseaseorAttack, 
               test_full_orig, test_full_orig$HeartDiseaseorAttack, "logit")

evaluate_model(tree_model_orig, train_full_orig, train_full_orig$HeartDiseaseorAttack, 
               test_full_orig, test_full_orig$HeartDiseaseorAttack, "tree")

evaluate_model(NULL, train_full_orig[, -1], train_full_orig$HeartDiseaseorAttack, 
               test_full_orig[, -1], test_full_orig$HeartDiseaseorAttack, "naivebayes")

evaluate_model(NULL, train_full_orig[, -1], train_full_orig$HeartDiseaseorAttack, 
               test_full_orig[, -1], test_full_orig$HeartDiseaseorAttack, "knn")

#Comparison plot
# Create data frames for balanced and unbalanced metrics
library(tidyr)
balanced <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Naive Bayes", "LDA", "KNN"),
  Precision = c(0.762, 0.689, 0.975, 0.758, 0.948),
  Recall = c(0.792, 0.849, 0.996, 0.800, 0.973),
  Accuracy = c(0.7726, 0.7326, 0.9852, 0.7722, 0.9599),
  F1_Score = c(0.777, 0.761, 0.985, 0.778, 0.960),
  AUC = c(0.849, 0.762, 0.999, 0.848, 0.991),
  Dataset = "Balanced"
)

unbalanced <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Naive Bayes", "LDA", "KNN"),
  Precision = c(0.538, NA, 0.900, 0.435, 0.996),
  Recall = c(0.123, 0.000, 0.980, 0.226, 0.774),
  Accuracy = c(0.9075, 0.9058, 0.9879, 0.8995, 0.9784),
  F1_Score = c(0.201, NA, 0.939, 0.298, 0.871),
  AUC = c(0.846, 0.500, 0.999, 0.839, 0.986),
  Dataset = "Unbalanced"
)

# Combine both
combined <- bind_rows(balanced, unbalanced)

# Reshape into long format
long_data <- combined %>%
  pivot_longer(cols = c(Precision, Recall, Accuracy, F1_Score, AUC), 
               names_to = "Metric", values_to = "Value")

# Plot
ggplot(long_data, aes(x = Model, y = Value, fill = Dataset)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ Metric, scales = "free_y") +
  labs(title = "Comparison of Model Performance (Balanced vs Unbalanced Data)",
       x = "Model", y = "Metric Value") +
  scale_fill_manual(values = c("#66c2a5", "#fc8d62")) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Models on interaction terms
logit_model2 <- glm(HeartDiseaseorAttack ~ ., data = train_full_int, family = binomial)
summary(logit_model2 )
tree_model2 <- rpart(HeartDiseaseorAttack ~ ., data = train_full_int)

#LDA model 2
# Train LDA model on interaction-enhanced data
lda_model2 <- lda(HeartDiseaseorAttack ~ ., data = train_full_int)
lda_pred2 <- predict(lda_model2, newdata = test_full_int)
pred_class2 <- lda_pred2$class
pred_prob2 <- lda_pred2$posterior[, "1"]
# Actual labels
actual2 <- test_full_int$HeartDiseaseorAttack
# Convert to factors with correct levels
pred_class2 <- factor(pred_class2, levels = c("0", "1"))
actual2 <- factor(actual2, levels = c("0", "1"))
# Confusion matrix
cm2 <- confusionMatrix(pred_class2, actual2, positive = "1")
print(cm2)
# Precision, Recall, F1
precision2 <- cm2$byClass["Precision"]
recall2 <- cm2$byClass["Recall"]
f1_2 <- 2 * (precision2 * recall2) / (precision2 + recall2)
cat(sprintf("Precision: %.3f | Recall: %.3f | F1 Score: %.3f\n", precision2, recall2, f1_2))

# ROC & AUC
roc_obj2 <- roc(actual2, as.numeric(pred_prob2))
auc_val2 <- auc(roc_obj2)
cat(sprintf("AUC: %.3f\n", auc_val2))

# Plot ROC Curve
plot(roc_obj2, main = "ROC Curve - LDA with Interaction Terms", col = "blue")
abline(0, 1, lty = 2)

evaluate_model(logit_model2, train_full_int, train_full_int$HeartDiseaseorAttack, test_full_int, test_full_int$HeartDiseaseorAttack, "logit")
evaluate_model(tree_model2, train_full_int, train_full_int$HeartDiseaseorAttack, test_full_int, test_full_int$HeartDiseaseorAttack, "tree")
evaluate_model(NULL, train_full_int[, -1], train_full_int$HeartDiseaseorAttack, test_full_int[, -1], test_full_int$HeartDiseaseorAttack, "naivebayes")
#evaluate_model(NULL, train_full_int[, -1], train_full_int$HeartDiseaseorAttack, test_full_int[, -1], test_full_int$HeartDiseaseorAttack, "lda")
evaluate_model(NULL, train_full_int[, -1], train_full_int$HeartDiseaseorAttack, test_full_int[, -1], test_full_int$HeartDiseaseorAttack, "knn")


###Model tuning
# Ensure valid class labels (No, Yes) instead of 0 and 1
train_full$HeartDiseaseorAttack <- factor(train_full$HeartDiseaseorAttack, levels = c(0, 1), labels = c("No", "Yes"))
test_full$HeartDiseaseorAttack  <- factor(test_full$HeartDiseaseorAttack,  levels = c(0, 1), labels = c("No", "Yes"))

train_full_int$HeartDiseaseorAttack <- factor(train_full_int$HeartDiseaseorAttack, levels = c(0, 1), labels = c("No", "Yes"))
test_full_int$HeartDiseaseorAttack  <- factor(test_full_int$HeartDiseaseorAttack,  levels = c(0, 1), labels = c("No", "Yes"))

levels(train_full$HeartDiseaseorAttack)
# [1] "No"  "Yes"

#1. Logistic Regression – AIC Stepwise (no grid)
logit_orig <- step(glm(HeartDiseaseorAttack ~ ., data = train_full, family = "binomial"),
                   direction = "both", trace = FALSE)

# 2. Decision Tree – cp tuning
set.seed(42)
tree_orig_tuned <- train(
  HeartDiseaseorAttack ~ ., 
  data = train_full,
  method = "rpart",
  trControl = trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  metric = "ROC",
  tuneGrid = expand.grid(cp = seq(0.001, 0.05, by = 0.005))
)

# 3. Naive Bayes – laplace, usekernel, adjust
nb_orig <- train(
  HeartDiseaseorAttack ~ ., data = train_full,
  method = "naive_bayes",
  trControl = trainControl(
    method = "cv", 
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  metric = "ROC",
  tuneGrid = expand.grid(
    laplace = c(0, 0.5, 1),
    usekernel = c(TRUE, FALSE),
    adjust = c(0.5, 1, 2)
  )
)

#4. KNN – k tuning (odd values)
set.seed(42)
knn_orig <- train(
  HeartDiseaseorAttack ~ ., data = train_full,
  method = "knn",
  trControl = trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  metric = "ROC",
  tuneGrid = expand.grid(k = seq(3, 12, by = 2))
)

#Tuning on Original + Interactions
#1. Logistic Regression – AIC Stepwise
logit_int <- step(glm(HeartDiseaseorAttack ~ ., data = train_full_int, family = "binomial"),
                  direction = "both", trace = FALSE)

#2. Decision Tree – cp tuning
set.seed(42)
tree_int_tuned <- train(
  HeartDiseaseorAttack ~ ., 
  data = train_full_int,
  method = "rpart",
  trControl = trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  metric = "ROC",
  tuneGrid = expand.grid(cp = seq(0.001, 0.05, by = 0.005))
)

#3. Naive Bayes – same grid
nb_int <- train(
  HeartDiseaseorAttack ~ ., data = train_full_int,
  method = "naive_bayes",
  trControl = trainControl(
    method = "cv", 
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  metric = "ROC",
  tuneGrid = expand.grid(
    laplace = c(0, 0.5, 1),
    usekernel = c(TRUE, FALSE),
    adjust = c(0.5, 1, 2)
  )
)

# 4. KNN – same grid
set.seed(42)
knn_int <- train(
  HeartDiseaseorAttack ~ ., data = train_full_int,
  method = "knn",
  trControl = trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  metric = "ROC",
  tuneGrid = expand.grid(k = seq(3, 12, by = 2))
)

evaluate_best_model <- function(model, test_data, model_name) {
  # Check if glm (Logistic Regression) or caret::train model
  if (inherits(model, "glm")) {
    # Logistic regression (glm): get probabilities and predicted class manually
    pred_prob <- predict(model, newdata = test_data, type = "response")
    pred_class <- ifelse(pred_prob > 0.5, "Yes", "No")
    pred_class <- factor(pred_class, levels = c("No", "Yes"))
  } else {
    # For all caret models (rpart, knn, naive_bayes)
    pred_prob <- predict(model, newdata = test_data, type = "prob")[, "Yes"]
    pred_class <- predict(model, newdata = test_data)
  }
  
  actual <- test_data$HeartDiseaseorAttack
  
  cat("\n🔹 Model:", model_name, "\n")
  cm <- confusionMatrix(pred_class, actual, positive = "Yes")
  print(cm)
  
  precision <- cm$byClass["Precision"]
  recall <- cm$byClass["Recall"]
  f1 <- 2 * (precision * recall) / (precision + recall)
  cat(sprintf("Precision: %.3f | Recall: %.3f | F1 Score: %.3f\n", precision, recall, f1))
  
  roc_obj <- roc(actual, as.numeric(pred_prob))
  auc_val <- auc(roc_obj)
  cat(sprintf("AUC: %.3f\n", auc_val))
  
  plot(roc_obj, main = paste("ROC -", model_name), col = "darkgreen")
  abline(0, 1, lty = 2)
}


evaluate_best_model(logit_orig, test_full, "Logistic Regression (Original)")
evaluate_best_model(tree_orig_tuned, test_full, "Decision Tree (Original)")
evaluate_best_model(nb_orig, test_full, "Naive Bayes (Original)")
evaluate_best_model(knn_orig, test_full, "KNN (Original)")

evaluate_best_model(logit_int, test_full_int, "Logistic Regression (Interaction)")
evaluate_best_model(tree_int_tuned, test_full_int, "Decision Tree (Interaction)")
evaluate_best_model(nb_int, test_full_int, "Naive Bayes (Interaction)")
evaluate_best_model(knn_int, test_full_int, "KNN (Interaction)")

