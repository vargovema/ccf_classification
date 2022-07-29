# Detecting Fradulent Credit Card Transactions

## Outline of the problem
The task that was undertaken here was detecting fraudulent transactions, their type and frequency, by using different models and methods applied to a data set that has been given. By looking at all the past transactions and finding any anomalies or patterns, we were able to obtain valuable information that can be used in risk assessment and fraud prevention.


## Why is fraud detection a relevant?
Billions of dollars are lost annually due to credit card fraud, and this is an issue that is constantly increasing since it has become more accessible because of the digitalization of the payment processes. Fraud is one of the biggest causes for financial losses and the first step in trying to understand and try to stop it from happening is risk assessment. The rise of these transactions is what makes it a matter that needs to be taken more seriously into consideration and an efficient fraud detection model is of the highest importance. Fraud detection is a central part in the prevention of future fraud from taking place, by analyzing the past data and finding certain patterns and information that can aid into the further understanding of the fraudulent processes. By detecting and looking at the previous fraudulent activity, we can implement certain strategies that will minimize future losses. By incorporating their findings, companies can learn from fraud that has taken place and create a system which is more able to detect fraud as soon as it happens, create a warning, or even prevent it from happening.

## Descriptive analysis of the data set

```{r, echo=FALSE, message=FALSE, warning=FALSE, results='hide'}
require(corrplot)
require(glmnet)
rm(list = ls())

set.seed(100)
setwd("~/OneDrive - Wirtschaftsuniversität Wien - IT-SERVICES/Semester 3/Business Analytics II/Lecture 7")
ccf <- read.csv("ccf.csv", sep=",")
```

```{r, echo=FALSE, message=FALSE, warning=FALSE, results='hide'}
lapply(ccf, class)
levels(ccf$Fraud)
with(ccf, levels(Fraud))
ccf <- transform(ccf, Fraud = factor(Fraud, levels = c("yes", "no"), labels = c(1, 0)))
with(ccf, levels(Fraud))

metric.index <- which(lapply(ccf, class) == "numeric")
factor.index <- which(lapply(ccf, class) == "factor")

n_vars <- ncol(ccf)
names_vars <- names(ccf)
```

The dataset provided is data on a number of transactions, which include data on whether or not it was a fraud, the amount of the transaction, the time at which the transaction took place (this is given as the minutes after the first transaction of the data set) as well as 28 variables which due to privacy reasons could not be disclosed. The dataset provided has some limitations which made modeling it more difficult. One of the limitations of the dataset is that we have variables which we do not know what they stand for (V1 - V28), and this makes the modeling harder as it is hard to interpret the potential relationship which the given variables have between themselves. However, we can still achieve that by looking at the correlation between the explanatory variables.

The correlation plot below provides us with insights on how the different variables correlate to each other, in a graphically very descriptive way. The numbers given on the left show us the strength of the correlation between the variables, a negative coefficient meaning that the bigger one of the variables is the smaller the other one is, and a positive coefficient meaning the opposite. On the top a nice visualization of the relationship can be observed, with the way of the relationship is portrayed by the shape of the ball and the colour corresponds to the strength of it. The correlation of the variables is also important for conducting the logistic regression, as one of the assumptions of this model is that there little or no multicollinearity among the variables, and from this plot we can observe that there is not a strong correlation between the variables, hence we can preform logistic regression.

```{r, echo=FALSE, message=FALSE, warning=FALSE, fig.height=6}
op <- par(cex = .5, mar = c(4.5, 2, 2, 1))
corrplot.mixed(cor(ccf[metric.index]),  upper = "ellipse",
               number.cex = .7,
               tl.col = "black")
par(op)
```

From the marginal distribution below, several things can be observed. One of them is that the frequency of not fraudulent transactions is a lot higher than the one of fraudulent transactions. It can also be seen that the density of the amount in those transactions is usually in the same range. When it comes to timing, two peaks can be seen, with a minimum situated in between them. We can see that most variables seem to have the highest density around the value 0 and that for most of them the density does not vary that much, only when it is close to the point with the highest density.
As the frequency of fraudulent transactions is very low when compared to the frequency of non fraudulent transactions, it is difficult to model predictions for it as the data is highly unbalanced. To deal with the problem of unbalanced data, we shall resample it. We will do this by implementing a function in R, that allows us to resample data in a way where we can redefine the ratio of fraudulent to non fraudulent transactions as well as the number of frauds which will be included in the training dataset. The function is also useful because it allows us to split the data into a test and a training data set, which will come useful when we create models and out of sample predictions.

```{r, echo=FALSE, message=FALSE, warning=FALSE}
op <- par(mfrow = c(4, 8), cex = .4, mar = c(2, 2, 2.5, 2.5), mgp = c(1.2, .5, 0))
for(i in 1 : n_vars) {
  if(i %in% metric.index) {
    hist(ccf[[i]], freq = FALSE, main = names_vars[i], xlab = "")
  } else {
    plot(table(ccf[[i]]), main = names_vars[i], ylab = "Frequency")
  }
}
par(op)
```


## Models used for testing 

Since the data is highly unbalanced, we use undersampling to resample the data where we obtain training and test data sets. We use the training data sets to estimates 3 different models and predict the test data sets based on the models. For each model, we resample the data for $n$ amount of times in order get more reliable models and predictions. However, the predictions of response variable we get from the models are continuous variables $\hat Y_i^{cont}$ which we need to classify as either 0 or 1 to turn the continuous response variable into binary variable. To achieve that, we need use a threshold $\alpha$ based on which we classify the prediction $i$ as 0 if  $\hat Y_i^{cont}<\alpha$ and we classify the prediction $i$ as 1 if $\hat Y_i^{cont}≥\alpha$ where natural choice for the threshold is $\alpha = 0.5$.

```{r, echo=FALSE, , message=FALSE, warning=FALSE, results='hide'}
lapply(ccf, class)
levels(ccf$Fraud)

ccf$Fraud <- as.integer(as.character(ccf$Fraud))
with(ccf, levels(Fraud))
```

```{r, echo=FALSE, , message=FALSE, warning=FALSE, results='hide'}
resample_data <- function(data, n_success, ratio, response_variable = 1, 
                          success_value = "1", verbose = FALSE) {
  response <- data[[response_variable]]
  success.index <- which(response == success_value)
  failure.index <- which(response != success_value)
  success.index_train <- sort(sample(success.index, n_success))
  failure.index_train <- sort(sample(failure.index, n_success * ratio))
  success.index_test <- success.index[!(success.index %in% success.index_train)]
  failure.index_test <- sort(sample(failure.index[
    !(failure.index %in% failure.index_train)], ratio * length(success.index_test)))
  sanity.check <- any(c(success.index_train, failure.index_train) %in% 
                        c(success.index_test, failure.index_test))
  if(!sanity.check) {
    if(verbose) cat("The training and test data sets do not overlap.\n") }
  else {cat("The training and test data sets do overlap!!!\n")}
  
  out <- list(train = data[c(success.index_train, failure.index_train), ],
    test = data[c(success.index_test, failure.index_test), ])
  return(out)
}
```

### Linear probability model

The classical model for binary outcomes is the Bernoulli distribution where we assume that $Y_i ∼ Ber(p_i)$ for $i = 1,...,n$. Based on the Bernoulli random variables we arrive to the linear probability model, given by the following formula: 
\[
Y_i = \beta_0 + \sum_{k = 1}^d \beta_k X_{i,k} + \epsilon_i,
\]
where $\beta_0$ is the intercept of the the linear function and $\beta_k$ is the change in the probability that $Y_i = 1$ while holding the other aggressors constant. We estimate $\beta_0,\beta_1,...,\beta_d$ via linear regression function using R. As in common multiple regression, we can use OLS estimation (ordinary least squares which is a type of linear least squares method for estimating the unknown parameters in a linear regression model) for inference about the parameters $\beta_0, . . . , \beta_d$.

The benefit of this model is that it is very simple to use and understand. It is also very easy to interpret the significance of the different coefficients. One of the shortcoming of this model is that we obtain continuous random variables possibly with values in all of real numbers set $\mathbb{R}$ which is not desirable as the response variable is binary. Another crucial downside of linear regression is that outliers have a huge effect on the model, and can make it worthless.

### Logistic regression model

Logistic regression model uses log-odds (also called logit) which describe the probability of response variable $Y_i$ increasing or decreasing based on a regressor $X_i$ where the logit is linear in X. Logit also describes the logistic regression given by:
\[
\mathbb{P}(Y_i = 1 \mid X) = \frac{e^{\beta_0 + \sum_{k = 1}^d \beta_k X_{i,k}}}{1 + e^{\beta_0 + \sum_{k = 1}^d \beta_k X_{i,k}}}
\]
where increasing $X_k$ by one unit changes the log-odds by $\beta_k$ (it multiplies the odds by $e^{\beta_k}$). Estimation of coefficients $\beta_0, \beta_1, ..., \beta_d$ is done via Maximum Likelihood estimation (MLE).

The benefit of using the logistic regression is that the results which we get, unlike in the linear regression model, will always be between 0 and 1, so they will never fall of outside that range. It also gives us a good idea of how important a certain regressor is, and to what extent it influences the probability of the explanatory variable being 1, or in this case of a fraud occurring. It is also less affected by outliers than the linear probability model. The downside is that it assumes linearity between dependent and independent variables, and also that it is very difficult to predict multivariate relationships with a big number of regressors using the logistic regression model, such as we potentially have here because the model becomes less accurate and predicts many independent variables as insignificant.

### Regularized logistic regression in Lagrangian form

The regularized logistic regression model works similarly to the logistic regression model, just that now we add a constraint to the beta coefficients of the different explanatory variables. We do this, so that we can end up with a constrained optimization problem, which will then help us to "kick out" certain explanatory variables which do not explain the response variable sufficiently enough. The formula is given by:
\[
\max_{\beta_0, \beta} \, \left\{ \sum_{i = 1}^n Y_i (\beta_0 + \beta'X_i) - \log(1 + e^{\beta_0 + \beta' X_i})  - \lambda \sum_{k = 1}^d |\beta_k| \right\}
\]
The term $\lambda$ is chosen by us. As said previously, the penalty term will result in some variables being equal to zero, and this result in a variable selection.

As the penalized regression model is similar to the logistic regression model the same pros and cons apply, with the addition that the penalized regression model allows for a bigger bias in the data by kicking out certain regressors if the model deems to not be "relevant enough". However, this can also be viewed as a benefit, as it provides us with less explanatory variables to look at, and a simpler model to predict.

Based on the lined out reasons, we believe the regularized logistic regression to be the best model to use in order to detect credit card fraud. It does take a lot of computational power to run it, but as credit card fraud is a serious issue, we believe that this model regardless of its complexity, provides the best results which will help us to identify which transactions where indeed fraudulent, by providing us only the most important variables, also defined as regressors previously, which help distinguish fraudulent transactions from non-fraudulent ones.


## Estimation results

Now we are going to run the regularized logistic regression in order to show the results, explain the significance of certain variables which the model will provide us with as well as the out of sample prediction performance of the model.

```{r, echo=FALSE, message=FALSE, warning=FALSE, results='hide'}
resampled_ccf <- resample_data(ccf, n_success = 300, ratio = 40, success_value = 1)
train <- resampled_ccf$train
test <- resampled_ccf$test
fit <- glmnet(x = as.matrix(train[, -1]), y = train[, 1], family = "binomial")
lambdas <- fit$lambda
```

```{r, echo=FALSE, warning=FALSE, message=FALSE, fig.height=3}
lambda.min <- min(lambdas)
lambda.max <- max(lambdas)
lambda.med <- median(lambdas)
vars.names <- c("(Intercept)", colnames(ccf))

op <- par(cex = .5, mar = c(4.5, 2, 2, 1))
plot(coef(fit, s = lambda.min),
pch = 16,
main = bquote("Coefficients for different lambda"),
ylab = expression(beta), xaxt = "n")
axis(1, at = 1 : length(vars.names),
labels = vars.names, las = 2)
points(coef(fit, s = lambda.med),
pch = 2, col = "dodgerblue3")
points(coef(fit, s = lambda.max),
pch = 3, col = "darkorange")
legend('topright', pch = c(16, 2, 3),
col = c("black", "dodgerblue3", "darkorange"),
legend = c("low", "medium", "high"))
par(op)
```

This plot highlights how for different values of $\lambda$, the number of non-zero coefficient changes. As we see, the bigger the lambda value is the more and more beta coefficients will be zero, meaning that the variables will not be important for our model. 
Hence, it is important to choose the right $\lambda$ value so that the model is not too biased. Luckily, the $cv.glmnet$ function in R provides us with the optimal $\lambda$ value.

```{r, echo=FALSE, message=FALSE, warning=FALSE, results='hide'}
fit.cv1 <- cv.glmnet(x = as.matrix(train[, -1]), y = train[, 1],
                     type.measure = "deviance", family = "binomial")

fit.cv2 <- cv.glmnet(x = as.matrix(train[, -1]), y = train[, 1],
                    type.measure = "class", family = "binomial")

fit.cv3 <- cv.glmnet(x = as.matrix(train[, -1]),  y = train[, 1],
                    type.measure = "auc", family = "binomial")
```

To outline the model, we shall firstly present three different methods of cross validation (using $cv.glmnet$ function) with which R provides us with, and see which one preforms the best, and that one we shall implement for further analysis.

The three plots below represent three different error type measures for cross validation: binomial deviance, misclasification error and area under the curve. Binomial deviance is defined as a quality of fit statistic for model fitting and it generalizes the idea of using sum of squared residuals. The misclassification error type measure provides us with the amount of misclassification errors and area under the curve tells us how well the model can distinguish between the two different groups (0 and 1). The first plot is binomial deviance type measure, the second one is misclassification error type measure and the last one is area under the curve type measure.
From the plots we can see that the all of them preform well; for misclassification error and binomial deviance the desired output is that the value is as close to zero as it can be, and we can see that for both of them the value is around 0.05. For the area under the curve type measure, the desired outcome is for the value to be as close to 1 as possible, and from the third graph we can see that our model preforms extremely well, with most of the areas being above 0.95. 
Another thing the graph provides us with is how many non-zero beta coefficients are present across different values of alpha; this is represented on the top of the graph by the different number values. The binomial deviance type measure has the highest variable loss out of the three, as we can see when we go from the recommended $\lambda$ value (the first dotted line to the left) to the $\lambda$ 1se the number of non-zero beta coefficients decreases more than with the other two models. In this respect, the misclassification error performs the best.
Overall, we choose to further analyze the area under the curve model as this model preforms the best when it comes to consistent and satisfactory values, and we believe it will be the best one to choose for further model predictions.

```{r, echo=FALSE, message=FALSE, warning=FALSE, fig.height=3.6}
op <- par(mfrow=c(1,3),cex = .5, mar = c(4.5, 4, 2, 1))
plot(fit.cv1)
plot(fit.cv2)
plot(fit.cv3)
par(op)
```

```{r, echo=FALSE, message=FALSE, warning=FALSE, results='hide'}
n_ests <- 10
n_vars <- ncol(ccf)
ratio <- 40
alpha <- 0.4
table(ccf[[1]])
n_success <- 300
betas <- matrix(NA_real_, ncol = n_vars, nrow = n_ests)
accuracy <- rep(NA_real_, n_ests)

vars.names <- c("(Intercept)", names(ccf[, -1]))
colnames(betas) <- vars.names

for(i in 1 : n_ests) {
  if(i %% 1 == 0) cat("Round", i, "out of", n_ests, "\n")
  
  resampled_default.tmp <- resample_data(ccf, n_success = n_success,
                                         ratio = ratio, success_value = 1)
  train <- resampled_default.tmp$train
  test <- resampled_default.tmp$test
  logr_reg.cv <- cv.glmnet(x = as.matrix(train[, -1]), y = train[, 1],
                           type.measure = "auc", family = "binomial")
  Y.cont <- predict(logr_reg.cv, newx = as.matrix(test[, -1]),
                    type = "response", s = "lambda.min")
  Y.cont[is.na(Y.cont)] <- 0
  Y.hat <- ifelse(Y.cont > alpha, 1, 0)
  out.tmp <- list(accuracy = mean(test$Fraud == Y.hat),
                  beta = coef(logr_reg.cv, s = "lambda.min"))
  accuracy[i] <- out.tmp$accuracy
  betas[i, ] <- as.numeric(out.tmp$beta)
}
```

For the next step, we used the output of the resampling which we mentioned earlier. From this we received two different data sets; a train and a test data set. Using the train data set, we created a regression model, which we tested using the test data set. What we did, is we put in the values of the different explanatory variables from the test dataset into the established model, in order to receive "predictions" on what the Y value should be in this case. Then we compared the modeled Y value to the actual Y value of the test data set. We ran this 10 times, resampling the data every time, in order to ensure maximum accuracy.

```{r, echo=FALSE, fig.height=3}
beta_lim <- range(betas, na.rm = TRUE)
mean_accuracy <- round(mean(accuracy),4)

op <- par(cex = .5, mar = c(4.5, 2, 2, 1))
plot(accuracy, pch = 16, main = "Accuracy")
abline(h = mean_accuracy, col = "dodgerblue3", lwd = 2)
text(1,mean_accuracy+0.00008, labels = mean_accuracy, cex = 1, col="dodgerblue3")

par(op)
```

From the accuracy plot, we can see that the mean accuracy is 99.46%. As mentioned above, this is the ratio to how many times we predicted the Y value correctly. The value of 100% would be ideal, however, this value is very close to 100% so we can judge that the model predicts the dataset very well.

```{r, echo=FALSE, fig.height=3}
beta_lim <- range(betas, na.rm = TRUE)
mean_accuracy <- round(mean(accuracy),4)

op <- par(cex = .5, mar = c(4.5, 2, 2, 1))
plot(1 : length(vars.names), 1 : length(vars.names), type = "n",
     ylim = beta_lim, xlab = "", ylab = expression(beta[i]),
     xaxt = "n", main = "Robustness Beta Coefficients")
axis(1, at = 1 : length(vars.names), labels = vars.names, las = 2)
for(i in 1 : n_ests) {
  points(1 : length(vars.names), betas[i, ], pch = 16)
}
abline(h = 0, col = "dodgerblue3", lwd = 2)
par(op)
```

From the robustness of the different beta coefficients, we can see how the beta coefficients for the different variables preform across the iterations. The beta coefficients tell us in what way the variable influences the probability of Y being 1, and thereby the transaction being fraudulent. The significant variables in this case will be the ones for which the beta coefficients are not equal to 0 more than 8 times.

```{r, echo=FALSE, fig.height=3}
signif_vars <- apply(betas, 2, function(x) sum(x != 0, na.rm = TRUE) > 8)

op <- par(cex = .5, mar = c(4.5, 2, 2, 1))
plot(1 : length(vars.names[signif_vars]), 1 : length(vars.names[signif_vars]), type = "n",
     ylim = beta_lim, xlab = "", ylab = expression(beta[i]),
     xaxt = "n", main = "Robustness Beta Coefficients (significant only)")
axis(1, at = 1 : length(vars.names[signif_vars]), labels = vars.names[signif_vars], las = 2)
for(i in seq_along(vars.names[signif_vars])) {
  points(1 : length(vars.names[signif_vars]), betas[i, signif_vars], pch = 16)
}
abline(h = 0, col = "dodgerblue3", lwd = 2)
par(op)
```

The second plot provides us with only significant beta coefficients and we can see that the variables our model predicted to be significant are: V4, V8, V10, V12, V13, V14 and V16. Only the variable V4 has a positive beta coefficient, meaning that an increase in this variable increases the probability of our transaction being fraudulent. The others have a average beta coefficient which is negative, meaning that they decrease the probability of our transaction being fraudulent.
