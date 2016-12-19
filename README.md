# Fashion-Classification
Classification of fashion articles 
Document Classification
This project builds a model to classify articles as either fashion or non-fashion, and lays out the steps to apply these techniques to other topics. In the case of fashion, 88% of the articles classified as fashion were actually fashion. I chose to focus on minimizing false positives, recognizing that this results in some true fashion articles being missed. 

The scripts themselves give pretty thorough step-by-step instructions but here one can find a quick description of all the files included in the “Document_Classifier” folder, as well as a more generalized explanation of each of the major parts in the process. 

Document_Classifier Folder Includes: 

	1. “Document_Classifier.RProj” : This is the R project which contains all the scripts related to building, testing and applying a classification model to documents. (“build_models.R”, “training_data_testing.R”, “classify_new_articles.R”) 

	2. “build_models.R” : This script formats the initial training and testing using data with known categories and then builds the SVM, and Maximum Entropy models. It also does initial accuracy testing on the models.

	3. “training_data_testing.R” : This script is used to recursively test the training articles to make sure they have been labeled as the proper topic.

	4. “classify_new_articles.R” : This is the script that will be used regularly to classify new articles being added to the system. Once an initial model and DTM have been made and saved it is easy to just run new articles through the system without having to use the “build_model.R” or “training_data_testing.R” scripts. 

	5. “SVM_MAX_models.Rd”  : As of right now, this is the best version of the models that will be loaded into the “classify_new_articles.R” script and applied to the new articles.

	6. “DTM.Rd” : This is the document term matrix used to train the models in “SVM_MAX_models.Rd”. This file will also be loaded into the “classify_new_articles.R” script and applied to the new articles to be classified. 

	7. “workspace.RData” : A saved version of the most recent workspace running all of these scripts. 

B. Model Building
	“build_model.R”

	This process is only applied when you want to update the models themselves. This is a good thing to do periodically as the articles on any given topic become more diverse. For example, right now all the articles used to train the “FASHION” category all came from one publisher, so for now the “SVM_MAX_models.Rd” model is very representative of that publishers style and tendencies. As the number of publishers contributing to this category increases one should build a new model to be more representative of the more general set of fashion articles. 

	Training Data: In order to build a classification model one needs to have a set of articles where the topics are known. I had a folder of known “FASHION” articles, and another of known “OTHER_random” articles. The documents in each of these folders will be tagged in the script itself based on what folder they reside in. It is important that each of the topic folders contain a similar number of documents so the models don't favor one topic over another. 

	Pre-Processing: The documents need to be converted to a corpus, and then formatted and cleaned up for further analysis. This means all unwanted characters (numbers, spaces, punctuation) are removed, letters are converted to lowercase, and “stopwords” are removed. Stopwords are words that occur commonly but don't hold significance to any particular topic (ex. for, of, by, and, etc.) The remaining words are then stemmed, bringing words down to their root so that a word having multiple forms counts as the same thing (run, runner, running).  We are left with essentially just a list of words contained in each document. 

myStopwords: In addition to the “stopword” dictionary included in the R text mining package, I created a second dictionary of words to remove from the corpus. These words were found both recursively by examining the DTM (which is the next step), and by looking at the articles themselves. For example, I noticed that every “FASHION” article contained the word “photo” at least once. This is due to the fact that all these training fashion articles came from the same publication and therefore were formatted similarly so that every picture was converted to “photo” in the plain text version of an article. This wasn't true for articles containing photos from other publications, so it caused the model to put a heavy emphasis on the word “photo” as a “FASHION” word, when the reason the word was present had very little to do with the actual text of the article. 

	Document Term Matrix: The cleaned up corpus is then used to create a document term matrix. A DTM represents a corpus by giving each term a column, and each document a row. This script uses a term frequency-inverse document frequency (Tf-Idf) DTM so the values filling this matrix use a numerical statistic to relate how important each term is to each document. 
The use of bigrams was also incorporated into this DTM, so it not only takes into account individual words but can include two words that are often used together. This helps for words that can mean very different things when paired up with something else (i.e. “good” vs. “not good”). All of these steps result in a huge matrix that is almost impossible to deal with, and that is made up of mostly zeros. In order to cut the size down, and get rid of some terms that don't provide much insight to the topic of a document, we remove terms that are present in less than 1% of the document. 

	Testing and Training Datasets: The data then needs to be split into two groups: the data that will be used to train the models and the data used to do initial testing of the model. I chose to randomly select 70% of the data to use for training and 30% for testing. 

	Building the Model: The training and testing data sets then need to be reordered due to the required inputs to the model building function. I chose to use a combination of support vector machine and maximum entropy algorithms because I found that using them together results in a higher accuracy than using either individually. 

	Testing the Model: Next we run the testing documents through the model, and get a list of the suggested topic for each document with a confidence or probability level. A confusion matrix is made for the results of each algorithm individually so one can get an idea about how well they are performing. Then you can do the same for both algorithms together using the “create_scoreSummary” function which takes into account the probabilities of each algorithm and if they agree on the topic of the documents. 

	Saving the Model: Save the model and DTM so that they can be loaded in to the script to classify new articles so this whole process doesn't need to be rebuilt each time you want to classify new articles. The model only needs to be updated periodically as more variety in the types of articles within the desired topic increases. 

C. Testing the Training Data
	“training_data_testing.R”— this script is used together with the “build_models.R” script.

	In order to catch additional mislabeled documents it is helpful to recursively run the training data through the already built model in order to see which of these documents the model predicts incorrectly. These mislabeled documents should then be moved to the proper category folder, and the model should be rebuilt. Any incorrectly labeled training data will result in a skewing of the models accuracy, so do this process recursively several time until the accuracy of the training data comes back as 100%, or as close as possible. 

D. Classifying New Articles
	“classify_new_articles.R”— If you just want to classify new articles based on existing models you only need this script. 

	Load in Saved Model and DTM: In order to not have to remake a model and training DTM every time you want to classify new articles you just load in the already trained model and formatted DTM.

	Pre-Processing: The new documents are then processed exactly the same way as the previous documents. This include the DTM of the new documents containing all the same terms as in the training DTM. This is easy using the “create_matrix”* function.
	**There is a typo in the source code for this function, and there are directions for how to fix it included in the script at this step	

	Classify Documents: Then just run the new documents through the model just as was done in the previous scripts and get the predicted classifications of each document by each of the algorithms. 
	
	Filter Results: Generally, the results when classifying new “virgin” documents will never come close to the accuracy of the testing documents. This is because there is more variety in the new documents being classified. In order to get the best accuracy I selected only the documents that were categorized as “FASHION” by both algorithms. After testing many different combinations I filtered even further so that it only keeps articles where both algorithms agree to be “FASHION” and each algorithm does so at a confidence level above 85%. This does cause many actual fashion articles to be left out of the results, but also weeds out enough of the non-fashion to get an accuracy of over 80%. This confidence level value may need to change slightly as more fashion articles are bing contributed.  

	I used a file of about 30,000 articles to test the model. None of these articles were coming from actual fashion publications, so one wouldn't expect there to be many fashion articles. Using this model and filtering method it resulted in 104 articles being classified as “high confidence fashion”, and after going through each article 92 of them were actually fashion related, resulting in an 88% accuracy. 
	


	
