from PredictionModel import PredictionModel
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer

from InternalQuestionAnswering import in_question_answering
from ExternalQuestionAnswering import ex_question_answering
from ZeroShotClassification import zero_shot_classification
from PartOfSpeech import part_of_speech
from DictionaryComparison import dictionary_comparison

# Set up prediction model
model_folder_path = "jasminejwebb/KeywordIdentifier"

load_model = AutoModelForTokenClassification.from_pretrained(model_folder_path, num_labels=4)
load_tokenizer = AutoTokenizer.from_pretrained(model_folder_path)

model = PredictionModel(verbose = False)
model.createTokenizerAndModel(model_folder_path)

# Evaluate on abstract
model_folder_path = "jasminejwebb/KeywordIdentifier"

load_model = AutoModelForTokenClassification.from_pretrained(model_folder_path, num_labels=4)
load_tokenizer = AutoTokenizer.from_pretrained(model_folder_path)

test_model = PredictionModel(verbose = False)
test_model.createTokenizerAndModel(model_folder_path)

abstract1 = """Many algorithms have been recently developed for reducing dimensionality by projecting data onto an intrinsic non-linear manifold. Unfortunately, existing algorithms often lose significant precision in this transformation. Manifold Sculpting is a new algorithm that iteratively reduces dimensionality by simulating surface tension in local neighborhoods. We present several experiments that show Manifold Sculpting yields more accurate results than existing algorithms with both generated and natural data-sets. Manifold Sculpting is also able to benefit from both prior dimensionality reduction efforts."""
ab1_kw = test_model.predict(sentence = abstract1, output_csv='abstract_test1.csv')

# Word Level
# ex_df = ex_question_answering(abstract1, ab1_kw)
in_df = in_question_answering(abstract1, ab1_kw)
pos_df = part_of_speech(abstract1)

# Text Level
category_label = zero_shot_classification(abstract1)
dictionary_kw = dictionary_comparison(abstract1)

final_df = pos_df.set_index('word').join(in_df.set_index('word'))
final_df.to_csv('test_results.csv')

# app = FastAPI()

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

# @app.get("/keywords")
# async def identify_keywords(input_text: str):
#     kw = model.predict(sentence = input_text)
#     return {"dataframe": kw}