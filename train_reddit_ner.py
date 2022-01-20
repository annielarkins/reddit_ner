from datasets import load_metric
from transformers import AutoTokenizer
import reddit_ner_tokens as get_tokens
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import numpy as np

from sklearn.metrics import roc_curve,confusion_matrix,auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

import torch
print(torch.cuda.is_available())

label_list = [
    'NA',       # not highlighted by labels
    'thing',  # noun
    'description',  # adjective
    'action'   # verb
]

label_encoding_dict = {'NA': 0, 'thing': 1, 'description': 2, 'action': 3}

task = "ner" 
model_checkpoint = "distilbert-base-uncased"
batch_size = 32
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def plot_cm(y_true, y_pred, title):
    ''''
    input y_true-Ground Truth Labels
          y_pred-Predicted Value of Model
          title-What Title to give to the confusion matrix
    
    Draws a Confusion Matrix for better understanding of how the model is working
    
    return None
    
    '''
    y_true = MultiLabelBinarizer().fit_transform(y_true)
    y_pred = MultiLabelBinarizer().fit_transform(y_pred)
    
    figsize=(10,10)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
    plt.savefig('conf_mat.png', bbox_inches='tight')

def roc_curve_plot(fpr,tpr,roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' %roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def tokenize_and_align_labels(examples):
    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

train_dataset, test_dataset = get_tokens.get_un_token_dataset('/Users/annielarkins/Desktop/reddit_ner/data/train/', '/Users/annielarkins/Desktop/reddit_ner/data/test/')

train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

args = TrainingArguments(
    f"test-{task}",
    evaluation_strategy = "epoch",
    learning_rate=1e-1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    weight_decay=0.00001,
)

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric("seqeval")

precision_scores = []
recall_scores = []
f1_scores = []
accuracy_scores = []


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # plot_cm(true_predictions, true_labels, "Confusion Matrix")

    results = metric.compute(predictions=true_predictions, references=true_labels)
    precision_scores.append(results["overall_precision"])
    recall_scores.append(results["overall_recall"])
    f1_scores.append(results["overall_f1"])
    accuracy_scores.append(results["overall_accuracy"])
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
    
trainer = Trainer(
    model,
    args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=test_tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.evaluate()

# trainer.save_model('reddit-ner.model')

# TODO
# create graphs - test vs train? 
# roc curve

num_points = len(f1_scores)
plt.plot(range(num_points), precision_scores, label = "Precision")
plt.plot(range(num_points), recall_scores, label = "Recall")
plt.plot(range(num_points), accuracy_scores, label = "Accuracy")
plt.plot(range(num_points), f1_scores, label = "F1 Score")
plt.title("Performance Metrics")
plt.xlabel("Epoch")
plt.legend()
plt.show()
plt.savefig('metrics.png', bbox_inches='tight')