library(readr)
setwd("C:\\Users\\kdoyl\\OneDrive\\Documents\\IST 707")

myFile="deception_data_converted_final (1).csv"
MyData<- read.csv(myFile)
print(str(MyData))
LIE=MyData$lie
SENT=MyData$sentiment
MyData<-MyData[-c(1,2)]
print(head(MyData))
library(tidyr)
library(tidyverse)
Reviews<- MyData %>%unite(review, X,X.1,X.2,X.3,X.4,X.5,X.6,X.7,X.8,X.9,X.10,X.11,X.12,X.13,X.14,X.15,X.16,X.17,X.18,X.19,X.20,sep = " ",remove = TRUE )
dim(Reviews)

library(tm)
corpus <- Corpus(VectorSource(Reviews$review))


library(slam)
Reviews_dtm <- DocumentTermMatrix(corpus,
                                 control = list(
                                   stopwords = TRUE, 
                                   wordLengths=c(3, 15),
                                   removePunctuation = T,
                                   removeNumbers = T,
                                   tolower=T,
                                   stemming = T,
                                   remove_separators = T))
(WordFreq <- colSums(as.matrix(Reviews_dtm)))
(head(WordFreq))
(length(WordFreq))
ord <- order(WordFreq)
(WordFreq[head(ord)])
(WordFreq[tail(ord)])
(Row_Sum_Per_doc <- rowSums((as.matrix(Reviews_dtm))))


Reviews_M <- as.matrix(Reviews_dtm)
Reviews_M_N1 <- apply(Reviews_M, 1, function(i) i/sum(i))
Reviews_Matrix_Norm <- t(Reviews_M_N1)
(Reviews_M[c(1:15),c(1:15)])
(Reviews_Matrix_Norm[c(1:15),c(1:15)])


## Convert to matrix
Reviews_dtm_matrix = as.matrix(Reviews_dtm)
str(Reviews_dtm_matrix)
(Reviews_dtm_matrix[c(1:3),c(2:4)])

## convert to DF
Reviews_DF <- as.data.frame(as.matrix(Reviews_dtm))
str(Reviews_DF)
Reviews_DF[1:5,1:5]

library(wordcloud)
wordcloud(colnames(Reviews_dtm_matrix), Reviews_dtm_matrix[13, ], max.words = 70)
(head(sort(as.matrix(Reviews_dtm)[13,], decreasing = TRUE), n=20))

install.packages("pastecs")
library(pastecs)

Reviews_DF_LIE <- cbind(LIE, Reviews_DF)
Reviews_DF_LIE[1:5,1:5]

(n <- round(nrow(Reviews_DF_LIE)/5))
(s <- sample(1:nrow(Reviews_DF_LIE), n))
## The test set is the sample
Reviews_DF_LIE_TestSet <- Reviews_DF_LIE[s,]
## The trainng set is the not sample
Reviews_DF_LIE_TrainSet <- Reviews_DF_LIE[-s,]
Reviews_DF_LIE_TrainSet_nolabel <- Reviews_DF_LIE_TrainSet[,-1]
Reviews_DF_LIE_Trainset_labels <- Reviews_DF_LIE_TrainSet[,1]
Reviews_DF_LIE_Testset_nolabel <- Reviews_DF_LIE_TestSet[,-1]
Reviews_DF_LIE_Testset_label <- Reviews_DF_LIE_TestSet[,1]

Reviews_DF_SENT <- cbind(SENT, Reviews_DF)
Reviews_DF_SENT[1:5,1:5]

Reviews_DF_SENT_TestSet <- Reviews_DF_SENT[s,]
## The trainng set is the not sample
Reviews_DF_SENT_TrainSet <- Reviews_DF_SENT[-s,]
Reviews_DF_SENT_TrainSet_nolabel <- Reviews_DF_SENT_TrainSet[,-1]
Reviews_DF_SENT_Trainset_labels <- Reviews_DF_SENT_TrainSet[,1]
Reviews_DF_SENT_Testset_nolabel <- Reviews_DF_SENT_TestSet[,-1]
Reviews_DF_SENT_Testset_label <- Reviews_DF_SENT_TestSet[,1]

library(naivebayes)
NBLIEclassfier <- naive_bayes(LIE ~.,
                                     data=Reviews_DF_LIE_TrainSet, na.action = na.pass)
NBLIEClassifier_Prediction <- predict(NBLIEclassfier, Reviews_DF_LIE_Testset_nolabel)
print(NBLIEClassifier_Prediction)
CHI_table_1 <-table(NBLIEClassifier_Prediction,Reviews_DF_LIE_Testset_label)
plot(NBLIEClassifier_Prediction)

NBSENTclassfier <- naive_bayes(SENT ~.,
                              data=Reviews_DF_SENT_TrainSet, na.action = na.pass)
NBSENTClassifier_Prediction <- predict(NBSENTclassfier, Reviews_DF_SENT_Testset_nolabel)
print(NBSENTClassifier_Prediction)
CHI_table_2 <-table(NBSENTClassifier_Prediction,Reviews_DF_SENT_Testset_label)
plot(NBSENTClassifier_Prediction)

chi1<- chisq.test(CHI_table_1)
chi2<- chisq.test(CHI_table_2)

#Test with Normalized data
Reviews_Norm_DF <- as.data.frame(as.matrix(Reviews_Matrix_Norm))
Reviews_Norm_DF[1:5,1:5]

Reviews_Norm_DF_LIE <- cbind(LIE, Reviews_Norm_DF)
Reviews_Norm_DF_LIE[1:5,1:5]

(n <- round(nrow(Reviews_Norm_DF_LIE)/5))
(s <- sample(1:nrow(Reviews_Norm_DF_LIE), n))
## The test set is the sample
Reviews_Norm_DF_LIE_TestSet <- Reviews_Norm_DF_LIE[s,]
## The trainng set is the not sample
Reviews_Norm_DF_LIE_TrainSet <- Reviews_Norm_DF_LIE[-s,]
Reviews_Norm_DF_LIE_TrainSet_nolabel <- Reviews_Norm_DF_LIE_TrainSet[,-1]
Reviews_Norm_DF_LIE_Trainset_labels <- Reviews_Norm_DF_LIE_TrainSet[,1]
Reviews_Norm_DF_LIE_Testset_nolabel <- Reviews_Norm_DF_LIE_TestSet[,-1]
Reviews_Norm_DF_LIE_Testset_label <- Reviews_Norm_DF_LIE_TestSet[,1]

Reviews_Norm_DF_SENT <- cbind(SENT, Reviews_Norm_DF)
Reviews_Norm_DF_SENT[1:5,1:5]

Reviews_Norm_DF_SENT_TestSet <- Reviews_Norm_DF_SENT[s,]
## The trainng set is the not sample
Reviews_Norm_DF_SENT_TrainSet <- Reviews_Norm_DF_SENT[-s,]
Reviews_Norm_DF_SENT_TrainSet_nolabel <- Reviews_Norm_DF_SENT_TrainSet[,-1]
Reviews_Norm_DF_SENT_Trainset_labels <- Reviews_Norm_DF_SENT_TrainSet[,1]
Reviews_Norm_DF_SENT_Testset_nolabel <- Reviews_Norm_DF_SENT_TestSet[,-1]
Reviews_Norm_DF_SENT_Testset_label <- Reviews_Norm_DF_SENT_TestSet[,1]

NBLIEclassfier_Norm <- naive_bayes(LIE ~.,
                              data=Reviews_Norm_DF_LIE_TrainSet, na.action = na.pass)
NBLIEClassifier_Prediction_Norm <- predict(NBLIEclassfier_Norm, Reviews_Norm_DF_LIE_Testset_nolabel)
print(NBLIEClassifier_Prediction_Norm)
CHI_table_1_Norm <-table(NBLIEClassifier_Prediction_Norm,Reviews_Norm_DF_LIE_Testset_label)
plot(NBLIEClassifier_Prediction_Norm)

NBSENTclassfier_Norm <- naive_bayes(SENT ~.,
                               data=Reviews_Norm_DF_SENT_TrainSet, na.action = na.pass)
NBSENTClassifier_Prediction_Norm <- predict(NBSENTclassfier_Norm, Reviews_Norm_DF_SENT_Testset_nolabel)
print(NBSENTClassifier_Prediction_Norm)
CHI_table_2_Norm <-table(NBSENTClassifier_Prediction_Norm,Reviews_Norm_DF_SENT_Testset_label)
plot(NBSENTClassifier_Prediction_Norm)

chi1_Norm<- chisq.test(CHI_table_1_Norm)
chi2_Norm<- chisq.test(CHI_table_2_Norm)

#SVMs with norm data

library(e1071)
##Polynomial

tuned_costp <- tune(svm,LIE~., data=Reviews_DF_LIE_TrainSet,
                   kernel="polynomial", 
                   ranges=list(cost=c(.01,.1,1,10,100,100)))
summary(tuned_costp) 


SVM_fit_P <- svm(LIE~., data=Reviews_DF_LIE_TrainSet, 
                 kernel="polynomial", cost=.1, 
                 scale=FALSE)
print(SVM_fit_P)
##Prediction --
(pred_P <- predict(SVM_fit_P, Reviews_DF_LIE_TestSet, type = "class"))
length(pred_P)

(Ptable <- table(pred_P, Reviews_DF_LIE_Testset_label))
chisq.test(Ptable)
(MR_P <- 1 - sum(diag(Ptable))/sum(Ptable))



tuned_costpS <- tune(svm,SENT~., data=Reviews_DF_SENT_TrainSet,
                    kernel="polynomial", 
                    ranges=list(cost=c(.01,.1,1,10,100,100)))
summary(tuned_costpS) 


SVM_fit_PS <- svm(SENT~., data=Reviews_DF_SENT_TrainSet, 
                 kernel="polynomial", cost=.01, 
                 scale=FALSE)
print(SVM_fit_PS)
##Prediction --
(pred_PS <- predict(SVM_fit_PS, Reviews_DF_SENT_TestSet, type = "class"))
length(pred_PS)

(PStable <- table(pred_PS, Reviews_DF_SENT_Testset_label))
chisq.test(PStable)
(MR_PS <- 1 - sum(diag(PStable))/sum(PStable))


## Linear Kernel...

tuned_costS <- tune(svm,SENT~., data=Reviews_DF_SENT_TrainSet,
                   kernel="linear", 
                   ranges=list(cost=c(.01,.1,1,10,100,100)))
summary(tuned_costS) 

SVM_fit_LS <- svm(SENT~., data=Reviews_DF_SENT_TrainSet, 
                 kernel="linear", cost=1, 
                 scale=FALSE)
print(SVM_fit_LS)
##Prediction --
(pred_LS <- predict(SVM_fit_LS, Reviews_DF_SENT_Testset_nolabel, type="class"))
(L_tableS<-table(pred_LS, Reviews_DF_SENT_Testset_label))
chisq.test(L_tableS)
## Misclassification Rate for Linear
(MR_LS <- 1 - sum(diag(L_tableS))/sum(L_tableS))


## Radial Kernel...

tuned_costrS <- tune(svm,SENT~., data=Reviews_DF_SENT_TrainSet,
                    kernel="radial", 
                    ranges=list(cost=c(.01,.1,1,10,100,100)))
summary(tuned_costrS)


SVM_fit_RS <- svm(SENT~., data=Reviews_DF_SENT_TrainSet, 
                 kernel="radial", cost=100, 
                 scale=FALSE)
print(SVM_fit_RS)
##Prediction --
(pred_RS <- predict(SVM_fit_RS, Reviews_DF_SENT_Testset_nolabel, type="class"))
(R_tableS<-table(pred_RS, Reviews_DF_SENT_Testset_label))
chisq.test(R_tableS)

## Misclassification Rate for Radial
(MR_RS <- 1 - sum(diag(R_tableS))/sum(R_tableS))

