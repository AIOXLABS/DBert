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
* Stories written in Darija scrapped from a dedicated website
* Youtube comments from 40 different Moroccan channels
* Tweets crawled based on a list of Darija keywords. 

Concatenating these datasets sums up to 691MB of text.

# Data preprocessing: 

* Replacing repeated characters with one occurrence of this character
* Replacing hashtags, user mentions and URLS respectively with following words: HASHTAG, USER, URL. 
* Keeping sequences with at least two arabic words
* Removing Tatweel character '\\u0640'
* Removing diacritics
# Pretraining Process

Same architecture as  [BERT-base] (https://arxiv.org/pdf/1810.04805.pdf) was used, but without the Next Sentence Prediction objective.
Whole Word Masking (WWM)  with a probability of 15% was adopted
The sequences were tokenized with a WordPiece Tokenizer from the [Huggingface Transformer library](https://huggingface.co/transformers/). We chose 128 as the maximum length of the input for the model.

The vocabulary size is 80.000 wordpiece token

The whole training was done on GCP Compute Engine using free cloud TPU v3.8 offered by Google's TensorFlow Research Cloud (TRC) program. It took 49 hours to run the 40 epochs of pretraining.

# Downstream tasks 

DarijaBert was fine tuned on 3 downstream tasks, namely Dialect Identification (DI), Sentiment Analysis (SA), and Topic Modeling (TM). The results were compared to 6 other models, which support arabic either fully or partially:

Vocabulary size (ar/all)

Data Size
#Params
#Steps


| Model | Authors  | Arabic composition | Vocabulary size (Ar/all) | Num Tokens (Ar/all) | Data Size | Num of parameters | Num of Steps | 


