import os
import json
from tokenizers import BertWordPieceTokenizer



# Initialize a tokenizer
tokenizer = BertWordPieceTokenizer()

TOKENS_TO_ADD = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '<S>', '<T>']

Vocab_size=80000

# Customize training
tokenizer.train(files=["path/to/text/data/file.txt"],   
                vocab_size=Vocab_size, 
                min_frequency=2, 
                special_tokens=TOKENS_TO_ADD)

tokenizer.enable_truncation(max_length=128)

#Save the tokenizer vocabulary :Vocab.txt
tokenizer.save_model("path/to/config/files")

# Save tokenizer config: config.json
fw = open(os.path.join("path/to/config/files", 'config.json'), 'w')
json.dump({"do_lower_case": True, 
            "unk_token": "[UNK]", 
            "sep_token": "[SEP]", 
            "pad_token": "[PAD]", 
            "cls_token": "[CLS]", 
            "mask_token": "[MASK]", 
            "model_max_length": 128, 
            "max_len": 128,
            "model_type": "bert",
            "vocab_size":Vocab_size}, fw)
fw.close()