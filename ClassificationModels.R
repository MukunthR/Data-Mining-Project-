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

# Clear the global environment
rm(list = ls())
dev.off()
# ctrl + L

# Finding working directory
getwd()


#### Importing the Black Friday dataset ####
data <- read.csv("Churn-dataset.csv", na.strings = c("?", "NA", "NaN"), header = TRUE)

#### IQR on dataIQR to remove thhe outliers ####
dataIQR <- data  # Takning a copy of the data
x <- dataIQR$tenure
qnt <- quantile(x, probs=c(.25, .75), na.rm = T)
caps <- quantile(x, probs=c(.05, .95), na.rm = T)
H <- 1.5 * IQR(x, na.rm = T)
x[x < (qnt[1] - H)] <- caps[1]
x[x > (qnt[2] + H)] <- caps[2]
boxplot(dataIQR[,c('tenure')], horizontal=FALSE, axes= TRUE)

# view dataset
View(data)

# summary of the dataset
summary(data)

# attributes present in the dataset
attributes(data)

# structure of the data set
str(data)

#### Identifying missing values ####
is.na(data)
missingValues<-data[is.na(data)]
missingValues
data = na.omit(data)

#### Drop cutomerID column - its not going to help us in the prediction ####

data = data[,-1]

#### Replacing the missing values of the TotalChargescolumn by the mean ####
data$TotalCharges = ifelse(is.na(data$TotalCharges), ave(data$TotalCharges, FUN = function(x) mean(x,na.rm = TRUE)), data$TotalCharges)


#### Outcome variable being a categorical variable is made into factor ####
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



#### spliting the data into test and training dataset ####
split <- sample.split(data$Churn, SplitRatio = 0.80)
train <- subset(data, split == TRUE)
test <- subset(data, split == FALSE)
testActual <- test$Churn



#### Feature scaling ####
train[-20] = scale(train[-20])
test[-20] = scale(test[-20])



#### Applying Random Forest to find the importace of variable ####
rf <- randomForest(Churn~., data=data, importance=TRUE, ntree=300,na.action = na.omit)
Prediction <- predict(rf, test)
table(actual=test[,20],Prediction)
# Error rate - Random Forest
wrong<- (test[,20]!=Prediction )
errorRate<-sum(wrong,na.rm = TRUE)/length(wrong)
errorRate 
# Accuracy rate - Random Forest
wrong <- (test[,20]== Prediction) *100
accuracy <- sum(wrong)/length(wrong)
accuracy
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
accuracy <- sum(wrong)/length(wrong)
accuracy
plot(C50Alg)
# install.packages('caret', dependencies = TRUE)
#Calculating Precision, Recall $ F1
precision <- posPredValue(C50Predict, test[,20], positive="1")
recall <- sensitivity(C50Predict, test[,20], positive="1")
F1 <- (2 * precision * recall) / (precision + recall)
precision
recall
F1



#### Applying NaiveBayes ####    
# Not included Contract
nb <- naiveBayes(Churn~tenure+TotalCharges+ MonthlyCharges, data = train)
nbPredict<- predict(nb,test)
table(nb=nbPredict,Class=test$Churn)
# Error rate - NaiveBayes
wrong<- (test[,20]!=nbPredict)
errorRate<-sum(wrong,na.rm = TRUE)/length(wrong)
errorRate 
# Accuracy rate - NaiveBayes
wrong <- (test[,20]== nbPredict) *100
accuracy <- sum(wrong)/length(wrong)
accuracy
str(nb)
#Calculating Precision, Recall $ F1
precision <- posPredValue(nbPredict, test[,20], positive="1")
recall <- sensitivity(nbPredict, test[,20], positive="1")
F1 <- (2 * precision * recall) / (precision + recall)
precision
recall
F1



#### Applying Decision Tree ####
CART <- rpart(Churn~tenure+TotalCharges+ MonthlyCharges+Contract,data=train)
CARTPredict <-predict( CART,test, type="class")
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



#### Applying SVM ####
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
precision
recall
F1



#### Applying KNN ####
# When K = 1
knn_1 <- knn(train[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],test[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],train[,20],k=1)
confusionMatrix <- table(Prediction=knn_1,Actual=test[,20])
pre_1 <- sum(test[,20]==knn_1)/nrow(test)*100 
#Calculating Precision, Recall $ F1
precision <- posPredValue(knn_1, test[,20], positive="1")
recall <- sensitivity(knn_1, test[,20], positive="1")
F1 <- (2 * precision * recall) / (precision + recall)
confusionMatrix
pre_1
precision
recall
F1

# When K = 2
knn_2 <- knn(train[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],test[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],train[,20],k=2)
confusionMatrix2 <- table(Prediction=knn_2,Actual=test[,20])
pre_2 <- sum(test[,20]==knn_2)/nrow(test)*100
#Calculating Precision, Recall $ F1
precision <- posPredValue(knn_2, test[,20], positive="1")
recall <- sensitivity(knn_2, test[,20], positive="1")
F1 <- (2 * precision * recall) / (precision + recall)
confusionMatrix2
pre_2
precision
recall
F1

# When K = 5
knn_3 <- knn(train[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],test[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],train[,20],k=5)
confusionMatrix3 <- table(Prediction=knn_3,Actual=test[,20])
pre_3 <- sum(test[,20]==knn_3)/nrow(test)*100
#Calculating Precision, Recall $ F1
precision <- posPredValue(knn_3, test[,20], positive="1")
recall <- sensitivity(knn_3, test[,20], positive="1")
F1 <- (2 * precision * recall) / (precision + recall)
confusionMatrix3
pre_3
precision
recall
F1

# When K = 10
knn_4 <- knn(train[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],test[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],train[,20],k=10)
confusionMatrix4 <- table(Prediction=knn_4,Actual=test[,20])
pre_4 <- sum(test[,20]==knn_4)/nrow(test)*100
#Calculating Precision, Recall $ F1
precision <- posPredValue(knn_4, test[,20], positive="1")
recall <- sensitivity(knn_4, test[,20], positive="1")
F1 <- (2 * precision * recall) / (precision + recall)
confusionMatrix4
pre_4
precision
recall
F1



#visualization  - Not yet completed
xtrain <- train[0:10,c("tenure","TotalCharges","Churn")]
xtrain
train[,'Churn']
set[,'Churn']
library(ElemStatLearn)
set <- train[0:100,c("tenure","TotalCharges")]
X1 = seq(min(set[0:100,'tenure']) - 1, max(set[, 'tenure']) + 1, by = 0.01)
X2 = seq(min(set[0:100,'TotalCharges']) - 1, max(set[,'TotalCharges' ]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('tenure', 'TotalCharges')
y_grid = predict(svm, newdata = grid_set)
plot(set[,c('tenure', 'TotalCharges')],
     main = 'SVM (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[0:100,'Churn'] == 1, 'green4', 'red3'))
