#### Importing the installed libraries ####
library(e1071)
library(rpart)
library(rpart.plot)
library(randomForest)
library(class)
library(caTools)
library(neuralnet)
library(C50)
library(ElemStatLearn)
library(caret)
library(kknn)
library(ROSE)
library(pROC)
library(qqplot2)

# Clear the global environment
rm(list = ls())
dev.off()
# ctrl + L

# Finding working directory
getwd()

# Please follow the steps mentioned  
#### Importing the Telco Customer Churn dataset --(1st Step) #### 
data <- read.csv("Churn-dataset.csv", na.strings = c("?", "NA", "NaN"), header = TRUE)

#### IQR on dataIQR to remove the outliers ####

# There are not much outliers in the dataset.
dataIQR <- data  # Taking a copy of the data
# Working on Tenure
x <- dataIQR$tenure
qnt <- quantile(x, probs=c(.25, .75), na.rm = T)
caps <- quantile(x, probs=c(.05, .95), na.rm = T)
H <- 1.5 * IQR(x, na.rm = T)
x[x < (qnt[1] - H)] <- caps[1]
x[x > (qnt[2] + H)] <- caps[2]
boxplot(dataIQR[,c('tenure')], horizontal=FALSE, axes= TRUE,main = "Tenure", col = "light blue")
hist(data$tenure,col = "orange")

# Working on TotalCharges
dataIQR <- data  # Taking a copy of the data
x <- dataIQR$TotalCharges
qnt <- quantile(x, probs=c(.25, .75), na.rm = T)
caps <- quantile(x, probs=c(.05, .95), na.rm = T)
H <- 1.5 * IQR(x, na.rm = T)
x[x < (qnt[1] - H)] <- caps[1]
x[x > (qnt[2] + H)] <- caps[2]
boxplot(dataIQR[,c('TotalCharges')], horizontal=FALSE, axes= TRUE,main = "Total Charges", col = "blue")
hist(data$tenure,col = "red",main ="Total Charges" )
qplot(data$TotalCharges)

# View dataset
View(data)

# Summary of the dataset
summary(data)

# Attributes present in the dataset
attributes(data)

# Structure of the data set
str(data)

#### Identifying missing values ####
is.na(data)
missingValues<-data[is.na(data)]
missingValues


#### Drop cutomerID column - its not going to help us in the prediction --(2nd Step) ####  
data = data[,-1]

#### Replacing the missing values of the TotalCharges column by the mean --(3rd Step) ####   
data$TotalCharges = ifelse(is.na(data$TotalCharges), ave(data$TotalCharges, FUN = function(x) mean(x,na.rm = TRUE)), data$TotalCharges)
data = na.omit(data)

#### Outcome variable being a categorical variable is made into factor --(4th Step) ####  
data$gender = as.numeric(factor(data$gender, levels = c('Male','Female'), labels = c(1,2)))
data$Partner = as.numeric(factor(data$Partner, levels = c('Yes','No'), labels = c(1,2)))
data$Dependents= as.numeric(factor(data$Dependents, levels = c('Yes','No'), labels = c(1,2)))
data$PhoneService = as.numeric(factor(data$PhoneService, levels = c('Yes','No'), labels = c(1,2)))
data$MultipleLines = as.numeric(factor(data$MultipleLines, levels = c('Yes','No','No phone service'), labels = c(1,2,3)))
data$PaperlessBilling = as.numeric(factor(data$PaperlessBilling, levels = c('Yes','No'), labels = c(1,2)))
data$InternetService = as.numeric(factor(data$InternetService, levels = c('DSL','Fiber optic','No'), labels = c(1,2,3)))
data$OnlineSecurity = as.numeric(factor(data$OnlineSecurity, levels = c('Yes','No internet service','No'), labels = c(1,2,3)))
data$OnlineBackup = as.numeric(factor(data$OnlineBackup, levels = c('Yes','No internet service','No'), labels = c(1,2,3)))
data$DeviceProtection = as.numeric(factor(data$DeviceProtection, levels = c('Yes','No internet service','No'), labels = c(1,2,3)))
data$TechSupport = as.numeric(factor(data$TechSupport, levels = c('Yes','No internet service','No'), labels = c(1,2,3)))
data$StreamingTV = as.numeric(factor(data$StreamingTV, levels = c('Yes','No internet service','No'), labels = c(1,2,3)))
data$StreamingMovies = as.numeric(factor(data$StreamingMovies, levels = c('Yes','No internet service','No'), labels = c(1,2,3)))
data$Contract = as.numeric(factor(data$Contract, levels = c('Month-to-month','One year','Two year'), labels = c(1,2,3)))
data$PaymentMethod = as.numeric(factor(data$PaymentMethod, levels = c('Electronic check','Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'), labels = c(1,2,3,4)))
data$Churn = factor(data$Churn, levels = c('Yes','No'), labels = c(1,2))


#### Spliting the data into test and training dataset --(5th Step) ####  
split <- sample.split(data$Churn, SplitRatio = 0.80)
train <- subset(data, split == TRUE)
test <- subset(data, split == FALSE)
testActual <- test$Churn


#### Feature scaling  Z-Scores --(6th Step) ####  
train[-20] = scale(train[-20])
test[-20] = scale(test[-20])


#### Applying Random Forest to find the importace of variable --(7th Step) ####  
rf <- randomForest(Churn~., data=data, importance=TRUE, ntree=300,na.action = na.omit)
Prediction <- predict(rf, test,type ="class")
table(actual=test[,20],Prediction)
# Error rate - Random Forest
wrong<- (test[,20]!=Prediction )
errorRate<-sum(wrong,na.rm = TRUE)/length(wrong)
errorRate 
# Accuracy rate - Random Forest
wrong <- (test[,20]== Prediction) *100
accuracy_rf <- sum(wrong)/length(wrong)
accuracy_rf
#Finding the importance of the variables
importance(rf)
order(importance(rf))
varImpPlot(rf)
#Calculating Precision, recall & F1 score
precision <- posPredValue(Prediction, test[,20], positive="1")
recall <- sensitivity(Prediction, test[,20], positive="1")
F1 <- (2 * precision * recall) / (precision + recall)
precision
recall
F1
# Plotting the ROC curve - Multiple thresholds TPR & FPR
PredictionProb_rf <- predict(rf, test, type = "prob") # Using probability
auc <- auc(test$Churn,PredictionProb_rf[,2])
auc
plot(roc(test$Churn,PredictionProb_rf[,2]),colorize = TRUE,col = "red")


# Following are the different Classification models
#### Applying C5.0 ####
C50Alg <- C5.0(Churn~tenure+TotalCharges+Contract+MonthlyCharges,data=train )
C50Predict <-predict( C50Alg,test, type="class")
table(actual=test[,20],C50=C50Predict)
# Error rate - C5.0
wrong<- (test[,20]!=C50Predict)
errorRate<-sum(wrong,na.rm = TRUE)/length(wrong)
errorRate 
# Accuracy rate - C5.0
wrong <- (test[,20]== C50Predict) *100
accuracy_c50 <- sum(wrong)/length(wrong)
accuracy_c50
plot(C50Alg)
# install.packages('caret', dependencies = TRUE)
#Calculating Precision, Recall $ F1
precision <- posPredValue(C50Predict, test[,20], positive="1")
recall <- sensitivity(C50Predict, test[,20], positive="1")
F1 <- (2 * precision * recall) / (precision + recall)
precision
recall
F1
# Plotting the ROC curve - Multiple thresholds TPR & FPR
PredictionProb_c50 <- predict(C50Alg, test, type = "prob") # Using probability
auc <- auc(test$Churn,PredictionProb_c50[,1])
auc
plot(roc(test$Churn,PredictionProb_c50[,1]))



#### Applying NaiveBayes ####    
# Not included Contract
nb <- naiveBayes(Churn~tenure+TotalCharges+ MonthlyCharges+Contract, data = train)
nbPredict<- predict(nb,test)
table(nb=nbPredict,Class=test$Churn)
# Error rate - NaiveBayes
wrong<- (test[,20]!=nbPredict)
errorRate<-sum(wrong,na.rm = TRUE)/length(wrong)
errorRate 
# Accuracy rate - NaiveBayes
wrong <- (test[,20]== nbPredict) *100
accuracy_nb <- sum(wrong)/length(wrong)
accuracy_nb
str(nb)
#Calculating Precision, Recall $ F1
precision <- posPredValue(nbPredict, test[,20], positive="1")
recall <- sensitivity(nbPredict, test[,20], positive="1")
F1 <- (2 * precision * recall) / (precision + recall)
precision
recall
F1
# Plotting the ROC curve - Multiple thresholds TPR & FPR
PredictionProb_nb <- predict(nb, test, type = "raw") # Using probability
auc <- auc(test$Churn,PredictionProb_nb[,2])
auc
plot(roc(test$Churn,PredictionProb_nb[,2]),col = "green")



#### Applying Decision Tree ####
CART <- rpart(Churn~tenure+TotalCharges+ MonthlyCharges+Contract,data=train)
CARTPredict <-predict( CART,test, type="class") # Using Class
table(actual=test[,20],CART=CARTPredict)
# Error rate - Desicion Tree
wrong<- (test[,20]!=CARTPredict)
errorRate<-sum(wrong,na.rm = TRUE)/length(wrong)
errorRate 
# Accuracy rate - Decsion Tree
wrong <- (test[,20]== CARTPredict) *100
accuracy <- sum(wrong)/length(wrong)
accuracy
summary(CART)
prp(CART)
rpart.plot(CART, type = 5, extra = 110)
#Calculating Precision, Recall $ F1
precision <- posPredValue(CARTPredict, test[,20], positive="1")
recall <- sensitivity(CARTPredict, test[,20], positive="1")
F1 <- (2 * precision * recall) / (precision + recall)
precision
recall
F1
# Plotting the ROC curve - Multiple thresholds TPR & FPR
PredictionProb_CART <- predict(CART, test, type = "prob") # Using probability
auc <- auc(test$Churn,PredictionProb_CART[,2])
auc
plot(roc(test$Churn,PredictionProb_CART[,2]), col = "yellow")



#### Applying SVM (Not considered for model comparison)####
svm = svm(formula = Churn~tenure+TotalCharges+ MonthlyCharges + Contract, data =train, type = 'C-classification', kernel = 'linear')
Prediction <- predict(svm, test)
table(actual=test[,20],Prediction)
# Error rate - Random Forest
wrong<- (test[,20]!=Prediction )
errorRate<-sum(wrong,na.rm = TRUE)/length(wrong)
errorRate
# Accuracy rate - Random Forest
wrong <- (test[,20]== Prediction) *100
accuracy <- sum(wrong)/length(wrong)
accuracy
#Calculating Precision, Recall $ F1
precision <- posPredValue(Prediction, test[,20], positive="1")
recall <- sensitivity(Prediction, test[,20], positive="1")
F1 <- (2 * precision * recall) / (precision + recall)
accuracy
precision
recall
F1
table(actual=test[,20],Prediction)



#### Applying KNN ####
# When K = 1
knn_1 <- knn(train[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],test[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],train[,20],k=1)
confusionMatrix <- table(Prediction=knn_1,Actual=test[,20])
Acc_1 <- sum(test[,20]==knn_1)/nrow(test)*100 
#Calculating Precision, Recall $ F1
precision <- posPredValue(knn_1, test[,20], positive="1")
recall <- sensitivity(knn_1, test[,20], positive="1")
F1 <- (2 * precision * recall) / (precision + recall)
confusionMatrix
Acc_1
precision
recall
F1


# When K = 2
knn_2 <- knn(train[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],test[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],train[,20],k=2)
confusionMatrix2 <- table(Prediction=knn_2,Actual=test[,20])
Acc_2 <- sum(test[,20]==knn_2)/nrow(test)*100
#Calculating Precision, Recall $ F1
precision <- posPredValue(knn_2, test[,20], positive="1")
recall <- sensitivity(knn_2, test[,20], positive="1")
F1 <- (2 * precision * recall) / (precision + recall)
confusionMatrix2
Acc_2
precision
recall
F1


# When K = 5
knn_3 <- knn(train[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],test[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],train[,20],k=5)
confusionMatrix3 <- table(Prediction=knn_3,Actual=test[,20])
Acc_3 <- sum(test[,20]==knn_3)/nrow(test)*100
#Calculating Precision, Recall $ F1
precision <- posPredValue(knn_3, test[,20], positive="1")
recall <- sensitivity(knn_3, test[,20], positive="1")
F1 <- (2 * precision * recall) / (precision + recall)
confusionMatrix3
Acc_3
precision
recall
F1


# When K = 30
knn_4 <- knn(train[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],test[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],train[,20],k=30)
confusionMatrix4 <- table(Prediction=knn_4,Actual=test[,20])
Acc_4 <- sum(test[,20]==knn_4)/nrow(test)*100
#Calculating Precision, Recall $ F1
precision <- posPredValue(knn_4, test[,20], positive="1")
recall <- sensitivity(knn_4, test[,20], positive="1")
F1 <- (2 * precision * recall) / (precision + recall)
confusionMatrix4
Acc_4
precision
recall
F1



#### ROC for all the models ####
# List of predictions
plot(roc(test$Churn,PredictionProb_rf[,2]),colorize = TRUE,col = "red", main = "Comparing RF, CART, C50 & NB")
plot(roc(test$Churn,PredictionProb_CART[,1]), add = TRUE, colorize = TRUE, col = "yellow")
plot(roc(test$Churn,PredictionProb_c50[,1]), add = TRUE, colorize = TRUE, col = "blue")
plot(roc(test$Churn,PredictionProb_nb[,2]), add = TRUE, colorize = TRUE,col = "green")

