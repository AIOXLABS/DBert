# What is DarijaBert

DBert is the first Open Source BERT model for the Moroccan Arabic dialect called “Darija”. It is based on the same architecture as Bert-base, but without the Next Sentence Prediction (NSP) objective. This model was trained on a total of ~3 Million sequences of Darija dialect representing 691MB of text or a total of ~100M tokens.
What has been released in this repository?
#We are releasing the following :

PyTorch training code based on HuggingFace trainer 
Pre-processing code
WordPiece tokenization code
Pre-trained model in both PyTorch and TensorFlow versions
Example notebook to finetune the model

# Pretraining data

The model was trained on a dataset issued from three different sources:
Stories written in Darija scrapped from a dedicated website
Youtube comments from 40 different Moroccan channels
Tweets crawled based on a list of Darija keywords. 

Concatenating these datasets sums up to 691MB of text.

# Data preprocessing: 

* Replacing repeated characters with one occurrence of this character
* Replacing hashtags, user mentions and URLS respectively with following words: HASHTAG, USER, URL. 
* Keeping sequences with at least two arabic words
* Removing Tatweel character '\\u0640'
* Removing diacritics
