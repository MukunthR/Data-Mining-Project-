#### Importing the installed libraries ####
library(e1071)
library(rpart)
library(rpart.plot)
library(randomForest)
library(class)
library(caTools)
library(neuralnet)
library(C50)

# Clear the global environment
rm(list = ls())
dev.off()
#Ctrl +L
# Finding working directory
getwd()

#### Importing the Black Friday dataset ####
data <- read.csv("Churn-dataset.csv", na.strings = c("?", "NA", "NaN"), header = TRUE)

# view datase
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
data = na.omit(data)

#### Drop cutomerID column - its not going to help us in the prediction ####
data = data[,-1]

#### Replacing the missing values of the TotalChargescolumn by the mean ####
data$TotalCharges = ifelse(is.na(data$TotalCharges), ave(data$TotalCharges, FUN = function(x) mean(x,na.rm = TRUE)), data$TotalCharges)

#Finding the unique values in the categorical variables
unique(data)

# Encoding the categorical variabl as factors
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


#### spliting the data into test and training dataset ####
split <- sample.split(data$TotalCharges,SplitRatio = 0.80)
train <- subset(data, split == TRUE)
test <- subset(data, split == FALSE)
testActual <- test$TotalCharges


# Feature scaling
train[-20] = scale(train[-20])
test[-20] = scale(test[-20])


#### Applying Random Forest to find the importace of variable ####
rf <- randomForest(Churn~., data = train, importance=TRUE, ntree=300,na.action = na.omit)
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
round(importance(rf), 2)



# #### Linear regression model ####
# lm <- lm(formula = Churn ~ tenure, data = train)
# summary(lm)
# influence(lm)
# anova(lm)
# ggplot() + 
#   geom_point(aes(x = train $ tenure, y = train $tenure),colour = 'red') +
#   geom_line(aes(x = train$tenure, y = predict(lm, train)),colour = 'blue')
#   ggtitle('Total Charges vs Tenure')
#   xlab('Tenure')
#   ylab('Total Charges')


#### Appling ANN ####
?neuralnet

nn  <- neuralnet(Churn~ tenure + TotalCharges + MonthlyCharges + Contract,train, hidden=10, threshold=0.01, stepmax = 1e6)
nn$result.matrix
plot(nn)

## Prediction using neural network

predict_testNN = compute(nn, test[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')])
predict_testNN = (predict_testNN$net.result * (max(data$Churn) - min(data$Churn))) + min(data$Churn)
plot(test$Churn, predict_testNN, col='blue', pch=20, ylab = "Predicted Churn NN", xlab = "real Churn")




