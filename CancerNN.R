#install.packages("caret")
library(caret)

args = commandArgs(trailingOnly=TRUE)

filename <- args[1]

dataset <- read.csv(filename, header = TRUE)

# View dimensions of imported dataset
dim(dataset)

# Copy the data for formatting
dataset2 <- dataset

# Convert the integer IC50 values to factors
dataset2$IC50 <- as.factor(dataset2$IC50)

# Rewrite the original dataset variable
dataset <- dataset2

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$IC50, p=0.80, list=FALSE)

# select 20% of the data for validation
validation <- dataset[-validation_index,]

# use the remaining 80% of data to training and testing the models
dataset <- dataset[validation_index,]

# Get the dimensions of the training dataset
dim(dataset)

# list types of data for each attribute
sapply(dataset, class)

# View the data
head(dataset)

# summarize the class distribution
percentage <- prop.table(table(dataset$IC50)) * 100
cbind(freq=table(dataset$IC50), percentage=percentage)

# Summarize the data
summary(dataset)

# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# a) linear algorithms
set.seed(7)
fit.lda <- train(IC50~., data=dataset, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(IC50~., data=dataset, method="rpart", metric=metric, trControl=control)

# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart))
summary(results)

# Compare accuracy of models
dotplot(results)

# Summarize the best model
print(fit.cart)

# Estimate skill of Cart on the validation dataset
predictions <- predict(fit.cart, validation)
confusionMatrix(predictions, validation$IC50)

