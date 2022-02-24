import numpy as np
import torch
import transformers
import pandas as pd
import numpy as np
import pandas as pd
from transformers import pipeline

def zero_shot_classification(text, possible_labels=None):
    classifier = pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli")

    if possible_labels is None:
        possible_labels = [

                            'Tech/Invention',
                            'History',
                            'Entertainment',
                            'Sports/Games',
                            'Geography',
                            'Transportation',
                            'Around the House'
                            'Food/Drink',
                            'Plants',
                            'Animals',
                            'Family'

        ]
    results = classifier(text, possible_labels)
    max_index = np.argmax(results['scores'])
    return results['labels'][max_index]