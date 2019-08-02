#Natural Language Processing

#Importing the dataset

dataset_original = read.delim("Restaurant_Reviews.tsv", quote= "", stringsAsFactors = FALSE)

#Cleaning the texts
#install.packages("tm")
install.packages("SnowballC")
library(tm)
corpus = VCorpus(VectorSource(dataset$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
library(SnowballC)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace) #removes the extra white spaces

#Creating the bag of words model
dtm = DocumentTermMatrix(corpus)
#to remove only words that appear only once
dtm = removeSparseTerms(dtm, 0.999) #be careful not to remove many words
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting classifier to the Training set
#install.packages("randomForest")
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)

# Predicting the Test set results
#y_pred = predict(classifier, newdata = test_set[-3])
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)


