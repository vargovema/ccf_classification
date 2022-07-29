require(corrplot)
require(glmnet)
rm(list = ls())

set.seed(100)
ccf <- read.csv("data/ccf.csv", sep=",")

## Descriptive analysis of the data set

lapply(ccf, class)
levels(ccf$Fraud)
with(ccf, levels(Fraud))
ccf <- transform(ccf, Fraud = factor(Fraud, levels = c("yes", "no"), labels = c(1, 0)))
with(ccf, levels(Fraud))

metric.index <- which(lapply(ccf, class) == "numeric")
factor.index <- which(lapply(ccf, class) == "factor")

n_vars <- ncol(ccf)
names_vars <- names(ccf)

op <- par(cex = .5, mar = c(4.5, 2, 2, 1))
corrplot.mixed(cor(ccf[metric.index]),  upper = "ellipse",
               number.cex = .7,
               tl.col = "black")
par(op)

op <- par(mfrow = c(4, 8), cex = .4, mar = c(2, 2, 2.5, 2.5), mgp = c(1.2, .5, 0))
for(i in 1 : n_vars) {
  if(i %in% metric.index) {
    hist(ccf[[i]], freq = FALSE, main = names_vars[i], xlab = "")
  } else {
    plot(table(ccf[[i]]), main = names_vars[i], ylab = "Frequency")
  }
}
par(op)

## Resampling data

lapply(ccf, class)
levels(ccf$Fraud)

ccf$Fraud <- as.integer(as.character(ccf$Fraud))
with(ccf, levels(Fraud))

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

## Estimation results

resampled_ccf <- resample_data(ccf, n_success = 300, ratio = 40, success_value = 1)
train <- resampled_ccf$train
test <- resampled_ccf$test
fit <- glmnet(x = as.matrix(train[, -1]), y = train[, 1], family = "binomial")
lambdas <- fit$lambda

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

fit.cv1 <- cv.glmnet(x = as.matrix(train[, -1]), y = train[, 1],
                     type.measure = "deviance", family = "binomial")

fit.cv2 <- cv.glmnet(x = as.matrix(train[, -1]), y = train[, 1],
                     type.measure = "class", family = "binomial")

fit.cv3 <- cv.glmnet(x = as.matrix(train[, -1]),  y = train[, 1],
                     type.measure = "auc", family = "binomial")

op <- par(mfrow=c(1,3),cex = .5, mar = c(4.5, 4, 2, 1))
plot(fit.cv1)
plot(fit.cv2)
plot(fit.cv3)
par(op)

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

beta_lim <- range(betas, na.rm = TRUE)
mean_accuracy <- round(mean(accuracy),4)

op <- par(cex = .5, mar = c(4.5, 2, 2, 1))
plot(accuracy, pch = 16, main = "Accuracy")
abline(h = mean_accuracy, col = "dodgerblue3", lwd = 2)
text(1,mean_accuracy+0.00008, labels = mean_accuracy, cex = 1, col="dodgerblue3")

par(op)

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
