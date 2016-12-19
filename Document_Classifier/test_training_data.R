## Testing the Training Documents used to build model

#-- It is important to make sure the data set being used to train the model is very clean and representive of
# the actual topics they are pre-labeled as. It is possible that there are some non-fashion articles that are
# in the "OTHER_random" folder, and therefore could skew the model results. 

#--This script should be used recursively after doing an initial build of the model, and then run the very 
# same articles that were used to train the model, through the models they trained in order to see if any 
# stand out as outliers from the rest of the category. The files that are found to be in the wrong category
# should then be moved to the proper folder, and the model should be re-trained. 

#-- create container of just the articles used for training and run them through the models. 
tr_container <- create_container(newDTM, newTOPIC, testSize = 1:length(train.idx), virgin = FALSE)
tr_results <- classify_models(tr_container, SVM_MAX_models)

tr_results$targetTopic <- doc.Topic[train.idx,]
tr_results$docName <- doc.Names[train.idx,]

# how many articles that have been used to train the model are correctly classified by the model?
tr_conf.mat_SVM <- table("Predictions" = tr_results$SVM_LABEL, "Actual" = tr_results$targetTopic)
tr_conf.mat_SVM 
tr_accuracy_SVM <- sum(diag(tr_conf.mat_SVM)) / length(train.idx) * 100
sprintf( "tr_Accuracy (SVM): %f", tr_accuracy_SVM)

tr_conf.mat_MAX <- table("Predictions" = tr_results$MAXENTROPY_LABEL, "Actual" = tr_results$targetTopic)
tr_conf.mat_MAX 
tr_accuracy_MAX <- sum(diag(tr_conf.mat_MAX)) / length(train.idx) * 100
sprintf( "tr_Accuracy (MAX): %f", tr_accuracy_MAX )

# which articles have been incorrectly labeled?
# move these articles to the proper folder, and then retrain model.
tr_incorrect_SVM <- tr_results[which(tr_results$SVM_LABEL != tr_results$targetTopic), ]
tr_incorrect_MAX <- tr_results[which(tr_results$MAXENTROPY_LABEL != tr_results$targetTopic), ]
