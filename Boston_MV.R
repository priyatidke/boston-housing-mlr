### Cleare the Environment ###

rm(list = ls(all=TRUE))

### Read data ###
getwd()
setwd("//Users//priyavivekbhandarkar//Desktop//INSOFE//ALL MODULES INSOFE//HOTe//20181223_Batch55_CSE7212c_MLR_HOTe")
Boston_MV <- read.csv("boston.csv", header = T, sep=",")
Boston_MV

### Understand The Data ###

View(Boston_MV)

dim(Boston_MV)

str(Boston_MV)

head(Boston_MV)

tail(Boston_MV)
 
### Load all libraries ###

install.packages("ggcorrplot")
install.packages("reshape2")
library(reshape2)
library(ggcorrplot)
library(corrplot)
library(DMwR)
library(dummies)
library(MASS)
library(car)
library(caret)
library(ggplot2)

### Descriptive Analysis ###

summary(Boston_MV)
names(Boston_MV)

### Exploratory Data Analysis - Scatterplot ###

pairs(Boston_MV)

###

require(ggplot2)
require(reshape2)
Boston_MV1 = melt(Boston_MV, id.vars='MV')
ggplot(Boston_MV1) +
  geom_jitter(aes(value,MV, colour=variable)) + geom_smooth(aes(value, MV, colour=variable), method=lm, se=FALSE) +
  facet_wrap(~variable, scales="free_x") +
  labs(x = "Percentage cover (%)", y = "Number of individuals (N)")

### Count Null Values ###

sum(is.na(Boston_MV))

### Covariance ###

cov(Boston_MV)

### Correlation ###

cor(Boston_MV)
cor(Boston_MV, Boston_MV$MV)

### Plot ###

library(corrplot)
M <- cor(Boston_MV)
corrplot(M, method = "number", tl.cex = 1, number.cex = .6)

#*mv decreases with increase in crim (medium), 
#*indus (High),nox(low),age(low),rad(low),tax(low),
#*ptratio(high), lstat (Highest) and increases with
#*increase in zn(medium),rm(Highest), b(low). 

### Histogram ###

hist(Boston_MV$MV, xlab = "Median Value", main = "Boston House Pricing", col = "blue")

#*Right skew Distribution log Transformation for crim dis , nox,zn
#*left skew Distribution sqrt Transformation for PT

### SIMPLE LINEAR REGRESSION ###

  ##Analyse the goodness of model ##

model <- lm(MV~CRIM,Boston_MV)
summary(model) 

model <- lm(MV~ZN,Boston_MV)
summary(model) 

summary(lm(MV~INDUS,Boston_MV))

summary(lm(MV~NOX,Boston_MV))

summary(lm(MV~RM,Boston_MV))

summary(lm(MV~AGE,Boston_MV))

summary(lm(MV~DIS,Boston_MV))

summary(lm(MV~RAD,Boston_MV))

summary(lm(MV~TAX,Boston_MV))

summary(lm(MV~PT,Boston_MV))

summary(lm(MV~B,Boston_MV))

summary(lm(MV~LSTAT,Boston_MV))

# Model Building

### Train Test Split (70:30) ###

rows = seq(1,nrow((Boston_MV)))
trainRows = sample(rows,(70*nrow(Boston_MV))/100)
Home_train = Boston_MV[trainRows,] 
Home_test = Boston_MV[-trainRows,]
dim(Home_train)
dim(Home_test)
names(Home_train)

### Building the Linear Regression Model ###

LinReg = lm(MV~CRIM,data=Home_train)
summary(LinReg)

summary(lm(MV~ZN,data=Home_train))

summary(lm(MV~INDUS,data=Home_train))

summary(lm(MV~CHAS,data=Home_train))

summary(lm(MV~NOX,data=Home_train))

summary(lm(MV~RM,data=Home_train))

summary(lm(MV~AGE,data=Home_train))

summary(lm(MV~DIS,data=Home_train))

summary(lm(MV~RAD,data=Home_train))

summary(lm(MV~TAX,data=Home_train))

summary(lm(MV~PT,data=Home_train))

summary(lm(MV~B,data=Home_train))

summary(lm(MV~LSTAT,data=Home_train))

### To extract the residuals: ###

  # CRIM residuals #

head(LinReg$residuals)

### To extract the train predictions: ###

head(LinReg$fitted.values)

### Plot the data points and the line of best fit ###

plot(Boston_MV$CRIM,Boston_MV$MV, col = "brown",lwd = 1,
     xlab="CRIM",ylab="MV ($)",main="CRIM vs MV")
abline(LinReg,col="blue",lty=1,lwd=2)
grid(10,10,lwd=1,col='Green')

### Some most affected plot ###

par(mfrow = c(1,2))
### Positive ###

plot(Boston_MV$RM,Boston_MV$MV, col = "brown",lwd = 1,
     xlab="RM",ylab="MV ($)",main="RM vs MV")
abline(LinReg,col="blue",lty=1,lwd=2)
grid(10,10,lwd=1,col='Green')

### Negative ##

plot(Boston_MV$LSTAT,Boston_MV$MV, col = "brown",lwd = 1,
     xlab="LSTAT",ylab="MV ($)",main="LSTAT vs MV")
abline(LinReg,col="blue",lty=1,lwd=2)
grid(10,10,lwd=1,col='Green')


### Residual Analysis ###

## Plot residuals ##

plot(LinReg$residuals,ylab="Residuals",main="Residuals",col = 'brown', lwd = 1)
grid(10,10,lwd = 1)


### Residual plots with R plot function ###

# par(mfrow = c(50,50)) 
plot(LinReg,lwd =1,col = 'brown')

### Test Data ###

test_prediction = predict(LinReg, Home_test)
test_actual = Home_test$MV

### Evaluating the Model ###

library(DMwR)

# Error verification on train data #
regr.eval(Home_train$MV, LinReg$fitted.values)

# Error verification on test data #
regr.eval(test_actual, test_prediction)


# Confidence and Prediction Intervals
#- Confidence Intervals talk about the average values intervals
#- Prediction Intervals talk about the all individual values intervals

Conf_Pred = data.frame(predict(LinReg, Home_test, interval="confidence",level=0.95))
Pred_Pred = data.frame(predict(LinReg, Home_test, interval="prediction",level=0.95))
Conf_Pred
Pred_Pred 
names(Conf_Pred)

              ###############################################


######### MULTILINEAR REGRESSION ##########

### Train and Test the Model ###
set.seed(12345)
inTrain <- createDataPartition(y = Boston_MV$MV, p = 0.80, list = FALSE)
Train_dat <- Boston_MV[inTrain,]
Test_dat <- Boston_MV[-inTrain,]

#%states <- as.data.frame(Boston_MV)
#%fit <- lm(MV ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PT + B + LSTAT, data = states)
#%summary(fit)##summary(lm(MV~.,Train_dat))

### Regression Model - linear model###

states <- as.data.frame(Train_dat)
fit_a <- lm(MV~., data = states)
summary(fit_a)
### or ###
fit <- lm(MV~., data = Train_dat)
summary(fit)
data.frame(coef = round(fit$coefficients, 2))
summary(fit)

#*lm(formula = MV ~ ., data = Train_dat)
#*Coefficients:
#*  (Intercept)         CRIM           ZN        INDUS         CHAS          NOX           RM  
#*4.296e+01   -9.636e-02    4.671e-02   -8.387e-04    2.504e+00   -1.998e+01    3.212e+00  
#*AGE          DIS          RAD          TAX           PT            B        LSTAT  
#*8.585e-03   -1.590e+00    3.320e-01   -1.233e-02   -1.039e+00    9.506e-03   -5.242e-01  

### Predict ###

predict_lm <- predict(fit, newdata = Test_dat)
RMSE_lm <- sqrt(sum((predict_lm - Test_dat$MV)^2)/length(Test_dat$MV))
c(RMSE = RMSE_lm, R2 = summary(fit)$r.squared )
  ###
predict_lmtr <- predict(fit, newdata = Train_dat)
RMSE_lmtr <- sqrt(sum((predict_lmtr - Train_dat$MV)^2)/length(Train_dat$MV))

plot(Test_dat$MV, predict_lm, xlab="Actual Price", ylab = "Predicted Price", col = "blue")

### after stepAIC ###

states <- as.data.frame(Train_dat)
fit_aic <- lm(MV ~ CRIM + ZN  + CHAS + NOX + RM + DIS + RAD + TAX + PT + B + LSTAT, data = states)
summary(fit_aic)


### Linear Model 2 - Using log Transformation ###

set.seed(12345)
fit1 <- lm(log(MV)~., data = Train_dat)

set.seed(12345)
predict_lm1 <- predict(fit1, newdata = Test_dat)
RMSE_lm1 <- sqrt(sum((exp(predict_lm1) - Test_dat$MV)^2)/length(Test_dat$MV))
c(RMSE = RMSE_lm1, R2 = summary(fit1)$r.squared )
summary(fit1)
##
plot(Test_dat$MV, predict_lm1, xlab="Actual Price", ylab = "Predicted Price", col = "blue")


### RIGHT SKEWED DISTRIBUTION - LOG TRANSFORMATION ##

  ##1
log_CRIM<-log(Boston_MV$CRIM)
summary(log_CRIM)
plot(log_CRIM)

  #%CRIM_log = transform(Boston_MV$CRIM, method = "log")
  #%result of transformation
  #%head(CRIM_log)
  # %summary of transformation
  #%summary(CRIM_log)
  #%plot(CRIM_log, xlab="CRIM", ylab="MV")

  ##2

log_DIS<-log(Boston_MV$DIS)
summary(log_DIS)
plot(log_DIS)

  ##3

log_NOX<-log(Boston_MV$NOX)
summary(log_NOX)
plot(log_NOX)


  ##4

log_ZN<-log(Boston_MV$ZN)
summary(log_ZN)
plot(log_ZN)

### Left Skewed - SQRT Transformation ###

sqrt_pt <- sqrt(Boston_MV$PT)
summary(sqrt_pt)
plot(sqrt_pt)

   #or#
summary(sqrt(Boston_MV$PT))

#%PT_log = transform(Boston_MV$PT, method = "sqrt")
#%summary(PT_log)

### linear model 3- using selected features###

set.seed(12345)
fit2<- lm(formula = log(MV)~CRIM + CHAS + NOX + RM + DIS + PT + RAD + B + LSTAT, data = Train_dat)

set.seed(12345)
predict_lm2 <- predict(fit2, newdata = Test_dat)
RMSE_lm2 <- sqrt(sum((exp(predict_lm2) - Test_dat$MV)^2)/length(Test_dat$MV))
c(RMSE = RMSE_lm2, R2 = summary(fit2)$r.squared )

##

plot(Test_dat$MV, predict_lm2, xlab="Actual Price", ylab = "Predicted Price", col = "blue")

### Diagnostic plots ###

#including all features
layout(matrix(c(1,2,3,4),2,2))
plot(fit)

  #or
#using selected features
layout(matrix(c(1,2,3,4),2,2))
plot(fit2)

### bining ###

bins1 <- cut(Boston_MV$CHAS, 2, include.lowest = TRUE)
bins
summary.bins(bins1) 
bins <- cut(Boston_MV$CHAS, 2, include.lowest = TRUE, labels = c('not_river','River'))
bins


### Variance Inflation Factor ###

model <- lm(MV~.,Boston_MV)
vif(model)

# Stepwise Regression

### 1. Backward


step <- stepAIC(lm(MV~.,Boston_MV), direction="backward")
step

### 2. Forward

step <- stepAIC(lm(MV~1,Boston_MV), direction="forward")# scope = ~PriorExperience + PriorSalary + Gender + Holidays)
step

### 3. Both

step <- stepAIC(lm(MV~.,Boston_MV), direction="both")
step

# Influential Points


### 1. Outliers

#### a. TAX ##


model = lm(MV~TAX,Boston_MV)

cook = cooks.distance(model)
cook
plot(cook,ylab="Cooks distances")

### b. NOX ###


model = lm(MV~NOX,Boston_MV)

cook = cooks.distance(model)
cook
plot(cook,ylab="Cooks distances")


### 2. Residuals

model = lm(MV~TAX,Boston_MV)
residuals = model$residuals
outliers <- boxplot(residuals,plot=T)$out

sort(outliers)
length(outliers)



model = lm(MV~RM,Boston_MV)

residuals = model$residuals
outliers <- boxplot(residuals,plot=T)$out

sort(outliers)
length(outliers)


### 3. Leverage

#### a. 

model = lm(MV~TAX,Boston_MV)
lev= hat(model.matrix(model))
lev
plot(lev)


#### b.

model = lm(MV~RM,Boston_MV)
lev= hat(model.matrix(model))
lev
plot(lev)

# Model with Influential Points


model = lm(MV~TAX,Boston_MV)
summary(model)


### Residual Plots


plot(model)


### Plot regression model

plot(Boston_MV$TAX, Boston_MV$MV, 
     main="Regression Model",
     xlab="TAX",
     ylab="MV"
)

abline(model, 
       col="steelblue",
       lty=1,
       lwd=4
)


# Model without Influential Points


model = lm(MV~TAX, Boston_MV)
summary(model)


### Residual Plots


plot(model)


### Plot regression model

plot(Boston_MV$TAX, Boston_MV$MV, 
     main="Regression Model",
     xlab="TAX",
     ylab="MV"
)

abline(model, 
       col="steelblue",
       lty=1,
       lwd=4
)

#################################################################################################
