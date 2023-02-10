<p align="center">
  <img  src="./Logo.jpeg" width="350">
</p>

# What is DarijaBERT ?
DarijaBERT is the first Open Source BERT model for the Moroccan Arabic dialect called “Darija”. It is based on the same architecture as BERT-base, but without the Next Sentence Prediction (NSP) objective. This model was trained on a total of ~3 Million sequences of Darija dialect representing 691MB of text or a total of ~100M tokens.

# What has been released in this repository?

We are releasing the following :

* Pre-processing code
* WordPiece tokenization code
* Pre-trained model in both PyTorch and TensorFlow versions(future plan)
* Example notebook to finetune the model
* MTCD dataset

# Pretraining data

The model was trained on a dataset issued from three different sources:
* Stories written in Darija scrapped from a dedicated website
* Youtube comments from 40 different Moroccan channels
* Tweets crawled based on a list of Darija keywords. 

Concatenating these datasets sums up to 691MB of text.

# Data preprocessing: 

* Replacing repeated characters with one occurrence of this character
* Replacing hashtags, user mentions and URLs respectively with following words: HASHTAG, USER, URL. 
* Keeping sequences with at least two arabic words
* Removing Tatweel character '\\u0640'
* Removing diacritics
# Pretraining Process

* Same architecture as  [BERT-base](https://github.com/google-research/bert)  was used, but without the Next Sentence Prediction objective.

* Whole Word Masking (WWM)  with a probability of 15% was adopted

* The sequences were tokenized using the  WordPiece Tokenizer from the [Huggingface Transformer library](https://huggingface.co/transformers/). We chose 128 as the maximum length of the input for the model.

* The vocabulary size is 80.000 wordpiece token

The whole training was done on GCP Compute Engine using free cloud TPU v3.8 offered by Google's TensorFlow Research Cloud (TRC) program. It took 49 hours to run the 40 epochs of pretraining.
# Masking task
Since DarijaBERT was trained  using Whole Word Masking, it  is capable of predicting  missing word  in sentence.
```python
from transformers import pipeline
unmasker = pipeline('fill-mask', model='Kamel/DarijaBERT')
unmasker(" اشنو [MASK] ليك ")

{'score': 0.02539043314754963,
  'sequence': 'اشنو سيفطو ليك',
  'token': 25722,
  'token_str': 'سيفطو'},
```

# Downstream tasks 


**UPCOMING**



********* DarijaBERT models were transfered on the SI2M Lab HuggingFace repo : Juin 20th,2022 ********

## Loading the model

The model can be loaded directly using the Huggingface library:
```python
from transformers import AutoTokenizer, AutoModel
DarijaBert_tokenizer = AutoTokenizer.from_pretrained("SI2M-Lab/DarijaBERT")
DarijaBert_model = AutoModel.from_pretrained("SI2M-Lab/DarijaBERT")
```

Checkpoint for the  Pytorch framework  is  available for downloading in the link below:

[https://huggingface.co/SI2M-Lab/DarijaBERT](https://huggingface.co/SI2M-Lab/DarijaBERT)

This  checkpoint is  destined exclusively for research, any commercial use should be done with author's permission, please contact via email  at dbert@aiox-labs.com

# Acknowledgments
We gratefully acknowledge Google’s TensorFlow Research Cloud (TRC) program for providing us with free Cloud TPUs.
 
