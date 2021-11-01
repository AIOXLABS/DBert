#pip install pyarabic

import string
import re
import strip_tatweel,strip_tashkeel

def repted(text):
    text=re.sub(r'(.)\1+', r'\1', text)# Replace with only one (remove repetitions)  
    return text

###For Pandas dataframe 
def pre_processing(df,source,field):
    df[field] = df[source]
    df[field] = df[field].replace(r'http\S+', 'URL', regex=True).replace(r'www\S+', 'URL', regex=True) # Replace URLs with URL string
    df[field] = df[field].replace(r'@[^\s]+', 'USER', regex=True) # Replace user mentions with USER string
    df[field] = df[field].replace(r'#[^\s]+', 'HASHTAG', regex=True) # Replace Hashtags with HASHTAG string
    df=df[df[field].apply(lambda x:len(re.findall(r'[\u0600-\u06FF]+', x)))>1] #Keep sequences with at least 2 arabic words
    df[field] = df[field].apply(strip_tatweel) #Remove Tatweel string 
    df[field] = df[field].apply(strip_tashkeel) # Remove Diacritics
    df[field] = df[field].apply(repted)
    return df
