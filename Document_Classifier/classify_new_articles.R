#-- This script is to classify new articles coming into the system using the
#  already trained and tested algorithm (built in creating_model.R)

#-- load in required packages
install.packages("RTextTools")
install.packages("plyr")
install.packages("e1071")
install.packages("SnowballC")
install.packages("SparseM")

lib <- c("RTextTools", "tm","plyr","class", "e1071", "SnowballC","Matrix","SparseM")
lapply(lib, require, character.only = TRUE)

#-- Load saved SVM model and original DTM-- these files must be saved in the same directory
load("DTM.Rd")
load("SVM_MAX_models.Rd")

##DOCUMENT PREP-- these are exactly the same as in the build_models.R script, they are included here
#  too so that new documents can be formatted EXACTLY as they were in the building of the model. 
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

#-- create clean corpus function-- 
#   this removes all unwanted characters and stopwords, makes everything lower case and 
#   stems the words so that words with the same root will be counted as the same (i.e. run vs running)

cleancorpus <- function(corpus) {
  corpus.tmp <- tm_map(corpus, removePunctuation)
  corpus.tmp <- tm_map(corpus.tmp, stripWhitespace)
  corpus.tmp <- tm_map(corpus.tmp, stemDocument) 
  corpus.tmp <- tm_map(corpus.tmp, content_transformer(tolower))
  corpus.tmp <- tm_map(corpus.tmp, removeNumbers)
  corpus.tmp <- tm_map(corpus.tmp, removeWords, c(stopwords("english"), myStopwords))
  return(corpus.tmp)
}

#-- set the pathname to where the new articles you want to classify are saved
pathname_new <-  "/Users/stephaniesauerhaft/Desktop/content/cat_content_30000 copy///"

#-- create corpus from new documents that we want to classify-- they need to go through EXACT same processing
# as training docs went through to make the model. 
docs <- Corpus(DirSource(directory = pathname_new, encoding = "UTF-8"))
docs.cl <- cleancorpus(docs)

#-- create DTM by putting new corpus data into same DTM used to make the model
docs.df <- data.frame(text = unlist(lapply(sapply(docs.cl, `[`, "content"),paste,collapse="\n")), stringsAsFactors = FALSE)
docs.DTM <- create_matrix(docs.df["text"], originalMatrix = DTM)
# ***IF YOU GET AN ERROR FROM THE LINE ABOVE RUN THE LINE OF CODE BELOW and CHANGE "Acronym" on line 42 to "acronym"
#trace("create_matrix",edit=T)

#-- get number of documents to be classified
newdoc_size <- nrow(docs.DTM)

#-- run new documents through already trained models and build table showing classification of each doument
newdoc_container <- create_container(docs.DTM, labels = rep(0,newdoc_size), testSize = 1:newdoc_size, virgin = TRUE)
doc_results <- classify_models(newdoc_container, SVM_MAX_models)
doc_score <- create_scoreSummary(newdoc_container,doc_results)

#-- include doc names in the doc_results data frame in order to determine which articles are classified as what
doc_names <- as.data.frame(list.files(pathname_new))
doc_results$doc_name <- doc_names
colnames(doc_results$doc_name) <- "doc_names"

#-- look at which documents were labeled as fashion by BOTH algorithms-- and then the ones with a confidence over 85%
fashion_docs <- doc_results[which(doc_results$SVM_LABEL == 1), ]
fashion_docs <- fashion_docs[which(fashion_docs$MAXENTROPY_LABEL == 1), ]
high_conf_fashion <- fashion_docs[which(fashion_docs$SVM_PROB > 0.85), ]
high_conf_fashion <- high_conf_fashion[which(high_conf_fashion$MAXENTROPY_PROB > 0.85), ]

#-- these high confidence fashion articles are what should officially be labeled as FASHION articles. 
