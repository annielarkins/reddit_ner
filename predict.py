from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
import transformers
import reddit_ner_tokens as get_tokens
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import numpy as np
import pandas as pd
import torch

print(torch.cuda.is_available())

label_list = [
    'NA',       # not highlighted by labels
    'thing',  # noun
    'description',  # adjective
    'action'   # verb
]

tokenizer = AutoTokenizer.from_pretrained('./reddit-ner.model/')

test_string1 = '''Hey everyone. I'm trying to get a sense as to whether my frustration is reasonable. In my campaign, there was recently a battle in Candlekeep. Everytime my wizard did anything, the heavily armored guards who apparently also knew 3rd level plus spells (6 of them) counterspelled everything I did, effectively causing my character to sit out the battle. That's fine. It makes sense. It's CandleKeep.'''

# '''Members will recall that, at its 2nd plenary meeting, on 20 September 2019, the assembly decided to include this item in the agenda of the seventy-fourth session. In connection with this item, I have received a letter dated 27 august 2020 from the deputy permanent representative of Brazil to the United Nations requesting that the item be included in the draft agenda of the seventy-fifth session of the assembly. I give the floor to the representative of Armenia. Members will recall that at its 2nd plenary meeting, on 20 September 2019, the assembly decided to include this item in the agenda of the seventy-fourth session. In connection with the item, a letter dated 31 august 2020 from the permanent representative of the Russian federation to the United Nations addressed to the president of the assembly has been issued as document a/74/1002, in which it is requested that the item be included in the agenda of the seventy-fifth session of the assembly. Members will recall that at its 2nd plenary meeting, on 20 September 2019, the assembly decided to include this item in the agenda of the seventy - fourth session. My delegation would like to disassociate itself from the decision to include agenda item 37 on the draft agenda of the seventy-fifth session of the general assembly. The assembly has before it five draft resolutions recommended by the third committee in paragraph 47 of its report. Before proceeding further, I should like to inform members that action on draft resolution iv, entitled situation of human rights of Rohingya Muslims and other minorities in Myanmar is postponed to a later date to allow time for the review of its programme budget implications by the fifth committee. The assembly will take action on draft resolution iv as soon as the report of the fifth committee on the programme budget implications is available. I now give the floor to delegations wishing to deliver explanations of vote or position before voting or adoption.'''

tokens = tokenizer(test_string1)
torch.tensor(tokens['input_ids']).unsqueeze(0).size()

model = AutoModelForTokenClassification.from_pretrained('./reddit-ner.model/', num_labels=len(label_list))

preds = model.forward(input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0), attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0))

preds = torch.argmax(preds.logits.squeeze(), axis=1)

words = tokenizer.batch_decode(tokens['input_ids'])

value_preds = [label_list[i] for i in preds]

pd.DataFrame({'ner': value_preds, 'words': words}).to_csv('un_ner.csv')