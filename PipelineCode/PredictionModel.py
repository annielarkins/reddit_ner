from transformers import AutoTokenizer
import transformers
import reddit_ner_tokens as get_tokens
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import numpy as np
import pandas as pd
import torch

class PredictionModel: 
    def __init__(self, verbose = False): 
        self.label_list = [
            'NA',       # not highlighted by labels
            'thing',  # noun
            'description',  # adjective
            'action'   # verb
        ]
        self.verbose = verbose
        # self.createTokenizerAndModel(model_folder_path)


    def createTokenizerAndModel(self, model_folder_path):
        if self.verbose: 
            print("Torch Cuda Available: %s" % torch.cuda.is_available())
        try: 
            self.tokenizer = AutoTokenizer.from_pretrained(model_folder_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_folder_path, num_labels=len(self.label_list))
            if self.verbose: 
                print("Model Trained")
        except: 
            print("Model Not Found: %s\n" % model_folder_path)
    
    def setTokenizerAndModel(self, tokenizer, model): 
        self.tokenizer = tokenizer 
        self.model = model 

    def predict(self, sentence, output_csv = None): 
        if not self.model: 
            print("ERROR: MODEL NOT FOUND")
            return
        tokens = self.tokenizer(sentence)
        print("Tokens Created")
        preds = self.model.forward(input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0), attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0))
        preds = torch.argmax(preds.logits.squeeze(), axis=1)
        print("Prediction made")
        words = self.tokenizer.batch_decode(tokens['input_ids'])
        print("Words done" )
        value_preds = [self.label_list[i] for i in preds]
        if output_csv: 
            pd.DataFrame({'ner': value_preds, 'words': words}).to_csv(output_csv)
            print("Values Printed to %s" % output_csv)
        if self.verbose: 
            print(pd.DataFrame({'ner': value_preds, 'words': words}))
            print("\nDone")
        return pd.DataFrame({'ner': value_preds, 'words': words})

