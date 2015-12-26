# Practical Machine Learning Course Project


## Overview
The devices for quantifing self movement are becoming more popular. They are able to collect a large amount of data about people and their personal health activities. The goal of this project is to utilize some sample data on the quality of certain exercises to predict the manner in which they did the exercise. This analysis will build a machine learning model from the sample data that is attempting to accurately predict the manner in which the exercise was performed. This is a classification problem into discrete categories.

## Analysis

### Load libraries
The following libraries are required to reproduce the result of this report.

```r
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)
```

### Download and load data

```r
trainCVFname <- "pml-training.csv"
if (!file.exists(trainCVFname)) {
  trainURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
  download.file(trainURL, trainCVFname)
}

testFname <- "pml-testing.csv"
if (!file.exists(testFname)) {
  testURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
  download.file(testURL, testFname )
}

trainCV <- read.csv(trainCVFname, na.strings=c("","NA","#DIV/0!"))
testing <- read.csv(testFname, na.strings=c("","NA","#DIV/0!"))
```

### Split Data into training and cross-validation sets
Partioning the data set from pml-training.csv into two data sets: 60% in the training set and 40% in the cross-validation set.

```r
set.seed(1234) # set seed for reproducibility
inTrain <- createDataPartition(y=trainCV$classe, p=0.6, list=FALSE)

training <- trainCV[inTrain, ]
CVset <- trainCV[-inTrain, ]
```

### Clean the data
The data have no dependence on time and participants, so the first 7 columns in the data sets are removed. The variables related with mean or stddev contains many NAs. Those columns are also removed.

```r
features <- names(training[,colSums(is.na(training)) == 0])[8:59]
training <- training[,c(features,"classe")]
CVset <- CVset[,c(features,"classe")]
testing <- testing[,c(features,"problem_id")]
```


### Machine learning
#### 1. Decision Tree: Generating Model


```r
modelDT <- rpart(classe ~ ., data=training, method="class")
```

#### 2. Random Forests: Generating Model


```r
modelRF <- randomForest(classe ~ . , data=training)
```

### Evaluate models via the cross-validation set
#### 1 Decision Tree: Evaluatation


```r
predictionsDT <- predict(modelDT, CVset, type = "class")
confDT <- confusionMatrix(predictionsDT, CVset$classe)
confDT 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1980  212   21   72   31
##          B   85  862   72   90   98
##          C   56  153 1086  209  175
##          D   71  101  110  823   89
##          E   40  190   79   92 1049
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7392          
##                  95% CI : (0.7294, 0.7489)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6699          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8871   0.5679   0.7939   0.6400   0.7275
## Specificity            0.9401   0.9455   0.9085   0.9434   0.9374
## Pos Pred Value         0.8549   0.7142   0.6468   0.6893   0.7234
## Neg Pred Value         0.9544   0.9012   0.9543   0.9304   0.9386
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2524   0.1099   0.1384   0.1049   0.1337
## Detection Prevalence   0.2952   0.1538   0.2140   0.1522   0.1848
## Balanced Accuracy      0.9136   0.7567   0.8512   0.7917   0.8324
```

#### 2 Random Forests: Evaluatation


```r
predictionsRF <- predict(modelRF, CVset, type = "class")
confRF <- confusionMatrix(predictionsRF, CVset$classe)
confRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232   10    0    0    0
##          B    0 1503   12    0    0
##          C    0    5 1354   20    2
##          D    0    0    2 1264    2
##          E    0    0    0    2 1438
## 
## Overall Statistics
##                                           
##                Accuracy : 0.993           
##                  95% CI : (0.9909, 0.9947)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9911          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9901   0.9898   0.9829   0.9972
## Specificity            0.9982   0.9981   0.9958   0.9994   0.9997
## Pos Pred Value         0.9955   0.9921   0.9804   0.9968   0.9986
## Neg Pred Value         1.0000   0.9976   0.9978   0.9967   0.9994
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1916   0.1726   0.1611   0.1833
## Detection Prevalence   0.2858   0.1931   0.1760   0.1616   0.1835
## Balanced Accuracy      0.9991   0.9941   0.9928   0.9911   0.9985
```


## Conclusions
Compared predictions with the cross-validation set, the accuracy of the Decision-Tree and Random-Forests models is 0.739 and 0.993, respectively. Therefore, the Random Forests model is selected to the 20 test cases available in the testing set.

## Applying Selected Model to Test Set
There are 20 samples in the testing set. The predictions is printed below and written into files for assignment submissions.


```r
predictionsTest <- predict(modelRF, testing, type = "class")

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predictionsTest)
predictionsTest
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
