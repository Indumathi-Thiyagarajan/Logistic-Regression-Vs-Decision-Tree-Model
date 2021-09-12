install.packages("dpylr")
library(dplyr)

customer_churn <- read.csv("E:/ISM 645/Assignment 3/customer_churn.csv")
head(customer_churn)
str(customer_churn)

#Build a logistic regression model to predict customer churn by using all possible variables, except customerID.


glm(Churn~ gender + SeniorCitizen + Partner+ Dependents+ tenure+ PhoneService+  MultipleLines + InternetService+
          OnlineSecurity+ OnlineBackup + DeviceProtection+ TechSupport+StreamingTV+ StreamingMovies+ Contract+PaperlessBilling+
          PaymentMethod+ MonthlyCharges+ TotalCharges, data = customer_churn,
    family = "binomial") %>%
  summary()
#Calculate the Pseudo R2 for the logistic regression.

install.packages("DescTools")
library(DescTools)
glm(Churn~ gender + SeniorCitizen + Partner+ Dependents+ tenure+ PhoneService+  MultipleLines + InternetService+
      OnlineSecurity+ OnlineBackup + DeviceProtection+ TechSupport+StreamingTV+ StreamingMovies+ Contract+PaperlessBilling+
      PaymentMethod+ MonthlyCharges+ TotalCharges, data = customer_churn,
    family = "binomial") %>%
  PseudoR2()


#===================================================================


# Split data into 70% train and 30% test datasets

install.packages("caTools")
library(caTools)
set.seed(123)
split <- sample.split(customer_churn, SplitRatio = 0.7)
train <- subset(customer_churn, split==TRUE)
test <- subset(customer_churn, split==FALSE)

# Train the same logistic regression on only "train" data.


library(dplyr)

glm(Churn~ gender + SeniorCitizen + Partner+ Dependents+ tenure+ PhoneService+  MultipleLines + InternetService+
      OnlineSecurity+ OnlineBackup + DeviceProtection+ TechSupport+StreamingTV+ StreamingMovies+ Contract+PaperlessBilling+
      PaymentMethod+ MonthlyCharges+ TotalCharges, data = train,
    family = "binomial") %>%
  summary()


#===================================================================



#For "test" data, make prediction using the logistic regression
#Set the cutoff value and create a confusion matrix
install.packages("broom")
library(broom)
library(dplyr)


customerchurn_test_lr <- glm(Churn~ gender + SeniorCitizen + Partner+ Dependents+ tenure+ PhoneService+  MultipleLines + InternetService+
                         OnlineSecurity+ OnlineBackup + DeviceProtection+ TechSupport+StreamingTV+ StreamingMovies+ Contract+PaperlessBilling+
                         PaymentMethod+ MonthlyCharges+ TotalCharges, data = train,
                       family = "binomial") %>%
  augment(type.predict = "response", newdata = test) %>%
  mutate(predicted = ifelse(.fitted > 0.5, 1,0))


#.fitted contains probability of customer churn from 0 to 1. we set cutoff value to diff customer churn or not.
#mutate variable to create positive (>0.5) and negative response.
#ifesle format is (condition. if true vaule, if else that value)


table(test$Churn  , customerchurn_test_lr$predicted)

#(actual, predicted)

#===================================================================



# Based on prediction results in Question 3, draw a ROC curve and calculate AUC (Hint: pROC package).

install.packages("pROC")
library(pROC)

roc_churn_lr <-roc(test$Churn  , customerchurn_test_lr$.fitted)
plot(roc_churn_lr)

auc(roc_churn_lr)



#===================================================================
# For "train" data, build a decision tree to predict customer churn by using all possible variables, except customerID (which is same as previous questions) (Hint: rpart package).
#  For "test" data, draw a ROC curve and calculate AUC (Hint: pROC package).


install.packages("rpart")
library(rpart)

churn_tree <- rpart(Churn~ gender + SeniorCitizen + Partner+ Dependents+ tenure+ PhoneService+  MultipleLines + InternetService+
                      OnlineSecurity+ OnlineBackup + DeviceProtection+ TechSupport+StreamingTV+ StreamingMovies+ Contract+PaperlessBilling+
                      PaymentMethod+ MonthlyCharges+ TotalCharges, data = train,
                    method = "class")
churn_tree

install.packages("rpart.plot")
library(rpart.plot)

prp(churn_tree, extra = 1)


predicted_churn_tree <- predict(churn_tree,newdata = test)

library(pROC)
roc_churn_tree <- roc(test$Churn, predicted_churn_tree [,2])

plot(roc_churn_tree)

auc(roc_churn_tree)


#===================================================================



#Prune your decision tree with cp = 0.01.



tree_trainrose_churn <- rpart(Churn~ gender + SeniorCitizen + Partner+ Dependents+ tenure+ PhoneService+  MultipleLines + InternetService+
                                OnlineSecurity+ OnlineBackup + DeviceProtection+ TechSupport+StreamingTV+ StreamingMovies+ Contract+PaperlessBilling+
                                PaymentMethod+ MonthlyCharges+ TotalCharges, data = train,
                              method = "class", cp = 0.01)

printcp(tree_trainrose_churn)
plotcp(tree_trainrose_churn)


pruned_tree <- prune(tree_trainrose_churn, cp = 0.01)

prp(pruned_tree, cex =1, extra = 1)

predicted_churn_tree_cp <- predict(tree_trainrose_churn,newdata = test)

# For "test" data, draw a ROC curve of the pruned decision tree and calculate AUC.

library(pROC)
roc_churn_tree_cp <- roc(test$Churn, predicted_churn_tree_cp [,2])

plot(roc_churn_tree_cp)

auc(roc_churn_tree_cp)



#I prefer Logistic regression as it is simple and also AOC is higher with Logistic regression model than decision tree

#AUC of Logistic Regression model = 0.8426
#AUC of Decision Tree model = 0.8096



