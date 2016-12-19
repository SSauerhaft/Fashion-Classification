#-- This script is used to load in training articles, build a model to classify these articles, 
#   and then test the accuracy of the model.

#-- install and load in required packages
install.packages("RTextTools")
install.packages("plyr")
install.packages("e1071")
install.packages("SnowballC")
install.packages("SparseM")
install.packages("RWeka")

lib <- c("RTextTools", "tm","plyr","class", "e1071", "SnowballC","Matrix","SparseM","RWeka")
lapply(lib, require, character.only = TRUE)

## DOCUMENT PREP
#-- create clean corpus function-- 
#   this removes all unwanted characters and stopwords, makes everything lower case and 
#   stems the words so that words with the same root will be counted as the same (i.e. run vs running)

#-- list of stopwords to be removed from corpus (in addition to stopwords("english"))
#   These were found recursively after running the model a few times
myStopwords <- c("can", "say","one","way","use", "also","becaus","come","howev","tell",
                 "will","much","need","take","tend","even","like","need","last", "two",
                 "one","particular","rather","said","get","well","make","much","ask",
                 "come","end", "first","two","help","often","may","might","see","someth",
                 "thing","photo","point","post","right","now","think","’ve ","’re ",
                 "around", "day", "feel", "good", "back", "just", "know", "lot", "made", 
                 "new", "part", "start", "want", "work", "york", "befor", "includ", "month",
                 "peopl", "pleas", "still", "year", "high", "call", "time", "week")

cleancorpus <- function(corpus) {
  corpus.tmp <- tm_map(corpus, removePunctuation)
  corpus.tmp <- tm_map(corpus.tmp, stripWhitespace)
  corpus.tmp <- tm_map(corpus.tmp, stemDocument) 
  corpus.tmp <- tm_map(corpus.tmp, content_transformer(tolower))
  corpus.tmp <- tm_map(corpus.tmp, removeNumbers)
  corpus.tmp <- tm_map(corpus.tmp, removeWords, c(stopwords("english"), myStopwords))
  return(corpus.tmp)
}

#-- create a corpus for each "topic", in this case "FASHION" and "OTHER_random".
#  I had a separate folder for each "topic" so I just made two separate corpi.
#  once the corpus is created apply the cleancorpus function to format the 
#  corpus to be more manageable. These corpuses will then be combined into 
#  one big corpus. (this way its easier to keep track of which are from which topic)

#-- create the "fashion" corpus
fash_dir <- "/Users/stephaniesauerhaft/Desktop/content/FASHION"
f.COR <- Corpus(DirSource(directory = fash_dir))
f.COR <- cleancorpus(f.COR)
#-- create the "other_random" corpus
oth_dir <- "/Users/stephaniesauerhaft/Desktop/content/OTHER_random"
oth.COR <- Corpus(DirSource(directory = oth_dir))
oth.COR <- cleancorpus(oth.COR)

#-- combine the "fashion" and "other_random" corpus into one big corpus
COR <- c(f.COR,oth.COR)

#-- create dataframe for all the actual "targetTopic" labels 
#   these labels will be used to verify the accuracy of our model
f.list <- as.data.frame(rep(1, length(f.COR)))              #"FASHION" articles will be represented with label "1"
oth.list <- as.data.frame(rep(2, length(oth.COR)))          #"OTHER_random" articles will be represented with label "2"
colnames(f.list) <- "targetTopic"
colnames(oth.list) <- "targetTopic"
doc.Topic <- rbind(f.list, oth.list)

#-- create a data frame of the document names. 
f.doc.names <- as.data.frame(list.files(fash_dir))
oth.doc.names <- as.data.frame(list.files(oth_dir)) 
colnames(f.doc.names) <- "docName"
colnames(oth.doc.names) <- "docName"
doc.Names <- rbind(f.doc.names,oth.doc.names)
#targetTopic$docNames <- doc.names

# Sets the default number of threads to use
options(mc.cores=1)

#-- create a function that allows the DTM to account for bigrams
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 2))
#-- create a document term matrix,and remove sparse terms to bring it to a manageable size
DTM <- DocumentTermMatrix(COR, control = list(tokenize = BigramTokenizer,weighting = function(x)
  weightTfIdf(x, normalize = FALSE)))
DTM <- removeSparseTerms(DTM, 0.999)

## MODEL BUILDING
#-- split data into training and testing data sets-- randomly select which documents will be used for each
train.idx <- sample(nrow(DTM), ceiling(nrow(DTM) * 0.7))     #uses random sample, 70% of number of docs to use as training docs
test.idx <- (1:nrow(DTM)) [-train.idx]                       #uses all rows that were not used in train.idx as test documents (30%)

#-- create new matrix of just the training or testing rows
#   this just has to do with how one needs to input training/testing data into 
#   the package that build the algorithm.  
trDTM <- DTM[train.idx, ]
teDTM <- DTM[test.idx, ]
trTOPIC <- doc.Topic[train.idx, ]
teTOPIC <- doc.Topic[test.idx, ]

#-- stack the training/testing matrices so the randomly selected articles are still matched up with their correct label
#  really just reordering the rows of the newDTM
newDTM <- rbind(trDTM, teDTM)
newTOPIC <- c(trTOPIC,teTOPIC)

#-- create container for inital training/testing data
container <- create_container(newDTM, newTOPIC, trainSize = 1:length(train.idx), testSize = (length(train.idx)+1):nrow(newDTM),  virgin = FALSE)

#-- build the actual models 
SVM_MAX_models <- train_models(container, algorithms = c("SVM", "MAXENT"), kernel = "linear")

#-- build results table which shows predicted topic for each document
test_results <- classify_models(container, SVM_MAX_models)
test_score <- create_scoreSummary(container,test_results)
test_analytics <- create_analytics(container, test_results)
summary.analytics(test_analytics)

#-- create data frame summaries of different analytic specs.
#summary_topic <- test_analytics@label_summary
#summary_alg <- test_analytics@algorithm_summary
#summary_ens <- test_analytics@ensemble_summary
#summary_doc <- test_analytics@document_summary


#-- attach the document names and their "targetTopic" labels to the results/score tables
test_results$targetTopic <- doc.Topic[test.idx,]
test_results$docName <- doc.Names[test.idx,]
test_score$targetTopic <- doc.Topic[test.idx,]
test_score$docName <- doc.Names[test.idx,]

#-- create confusion matrices to look at the accuracy of each model separately, and then the accuracy of the "Best_Label"
conf.mat_SVM <- table("Predictions" = test_results$SVM_LABEL, "Actual" = test_results$targetTopic)
conf.mat_SVM 
accuracy_SVM <- (sum(diag(conf.mat_SVM)) / length(test.idx) * 100)
sprintf( "Accuracy (SVM): %f", accuracy_SVM )
#-- look at what docs have been incorrectly labeled and determine if labels are valid
incorrect_docs_SVM <- test_results[which(test_results$SVM_LABEL != test_results$targetTopic), ]

conf.mat_MAX <- table("Predictions" = test_results$MAXENTROPY_LABEL, "Actual" = test_results$targetTopic)
conf.mat_MAX 
accuracy_MAX <- (sum(diag(conf.mat_MAX)) / length(test.idx) * 100)
sprintf( "Accuracy (MAX): %f", accuracy_MAX )
#-- look at what docs have been incorrectly labeled and determine if labels are valid
incorrect_docs_MAX <- test_results[which(test_results$MAXENTROPY_LABEL != test_results$targetTopic), ]

conf.mat_BEST <- table("Predictions" = test_score$BEST_LABEL, "Actual" = test_score$targetTopic)
conf.mat_BEST 
accuracy_BEST <- (sum(diag(conf.mat_BEST)) / length(test.idx) * 100)
sprintf( "Accuracy (BEST): %f", accuracy_BEST )
#-- look at what docs have been incorrectly labeled and determine if labels are valid
incorrect_docs_BEST <- test_score[which(test_score$BEST_LABEL != test_score$targetTopic), ]

#-- if satisfied with results of this model be sure to save the DTM and the model
#save(DTM,file="DTM.Rd")
#save(SVM_MAX_models,file="SVM_MAX_models.Rd")





