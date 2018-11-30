#### Importing the installed libraries ####
library(e1071)
library(rpart)
library(rpart.plot)
library(randomForest)
library(class)
library(caTools)
library(neuralnet)
library(C50)
library(ROCR)

# Clear the global environment
rm(list = ls())
dev.off()
#Ctrl +L
# Finding working directory
getwd()

# Please follow the steps mentioned 
#### Importing the Telco Customer Churn dataset --(1st Step) ####
data <- read.csv("Churn-dataset.csv", na.strings = c("?", "NA", "NaN"), header = TRUE)

# view dataset
View(data)

# summary of the dataset
summary(data)

# attributes present in the dataset
attributes(data)

# structure of the data set
str(data)

#### Identifying missing values & Removing them ####
is.na(data$PaymentMethod)
missingValues<-data[is.na(data)]
missingValues


#### Drop cutomerID column - its not going to help us in the prediction --(2nd Step)####
data = data[,-1]

#### Replacing the missing values of the TotalChargescolumn by the mean --(3rd Step)####
data$TotalCharges = ifelse(is.na(data$TotalCharges), ave(data$TotalCharges, FUN = function(x) mean(x,na.rm = TRUE)), data$TotalCharges)
data = na.omit(data)

#Finding the unique values in the categorical variables
unique(data)

#### Encoding the categorical variabl as factors --(4th Step) ####
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
data$Churn = as.numeric(factor(data$Churn, levels = c('Yes','No'), labels = c(1,2)))


#### spliting the data into test and training dataset --(5th Step) ####
split <- sample.split(data$TotalCharges,SplitRatio = 0.80)
train <- subset(data, split == TRUE)
test <- subset(data, split == FALSE)
testActual <- test$TotalCharges


#### Feature scaling - Z Scores --(6th Step)####
train[-20] = scale(train[-20])
test[-20] = scale(test[-20])


#### Appling ANN --(7th Step)####
nn  <- neuralnet(Churn~ tenure + TotalCharges + MonthlyCharges + Contract,train, hidden=8, threshold=0.10, stepmax = 1e6)
nn$result.matrix
plot(nn)
#### Prediction using neural network --(8th Step) ####
nnResults <-compute(nn, test[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')])
nnResults
ANN=as.numeric(nnResults$net.result)
ANN
# Rounding the Generated values 
ANN_round<-round(ANN)
ANN_round
# Confusion Matrix
ANN_cat<-ifelse(ANN<1.5,1,2)
table(Actual=test$Churn,ANN_round)
# Finding the Accuracy rate
wrong<- (test$Churn!=ANN_cat)
wrong
ErrorRate<-sum(wrong)/length(wrong)
ErrorRate
accuracy <- 1 - ErrorRate
accuracy

#### Plotting the ROC curve --(9th Step) ####
detach(package:neuralnet,unload = T)

nn.pred = prediction(ANN, test$Churn)
pref <- performance(nn.pred, "tpr", "fpr")
plot(pref,col = "red", main = "Plotting the Neural Net curve Using ROCR package")

