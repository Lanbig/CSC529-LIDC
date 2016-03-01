setwd("~/Dropbox/CSC529/Project")

require(caret)
require(ggplot2)
require(randomForest)
require(e1071)
require(pROC)
require("foreach")
require("doSNOW")
require(party)

myd = read.csv("Lung Cancer dataset/LIDC_Data_SEIDEL_01-05-2015.csv",header = T, sep = ',')
mycl = data.frame(matrix(NA, ncol = 1, nrow = nrow(myd)) )
names(mycl)[1]<-paste("class")

for(i in 1:nrow(myd))
{
  row <- myd[i,]
  mycl[i,1] <- round((row[5]+row[6]+row[7]+row[8])/4)
}

qplot(mycl$class, geom="bar",ylab = "count", main = "Class Distribution ", xlab = "Class", fill=as.factor(mycl$class)) 
###############################################################

data = cbind(mycl$class,myd[,9:72])
names(data)[1]<-paste("class")

# define an 66%/34% train/test split of the dataset
trainIndex <- createDataPartition(data$class, p=0.66, list=FALSE)
data_train <- data[ trainIndex,]
data_test <- data[-trainIndex,]

Sys.time()->start;
fit.rf = train(as.factor(class) ~ . ,data_train , method= "rf", ntree=200 , tuneGrid = data.frame(mtry = 4), 
               trControl=trainControl(method="cv",number=5) )
fit.rf
print(Sys.time()-start);


pred <- predict(fit.rf, data_train[,2:65])
xtab <- table(pred, as.factor(data_train$class))
confusionMatrix(xtab)

# test

pred <- predict(fit.rf, data_test[,2:65])
xtab <- table(pred, as.factor(data_test$class))
confusionMatrix(xtab)

###################################################################

Sys.time()->start;
fit.rf = train(as.factor(class) ~ . ,data_train , method= "ctree", trControl=trainControl(method="cv",number=5) )
fit.rf
print(Sys.time()-start);


pred <- predict(fit.rf, data_train[,2:65])
xtab <- table(pred, as.factor(data_train$class))
confusionMatrix(xtab)

# test

pred <- predict(fit.rf, data_test[,2:65])
xtab <- table(pred, as.factor(data_test$class))
confusionMatrix(xtab)


