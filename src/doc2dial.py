# %%
 %load_ext autoreload

%autoreload 2

import os
from tqdm import tqdm
import json
import argparse
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from datasets import load_dataset
from datasets import load_metric

from importlib import reload

import spacy
from components import TorchQADataset, HFTTraining, ModelGenerator, QueryProcessor, GoldenRetriever #, DocumentRetriever, PassageRetrieval, AnswerExtractor

from metric import sharedtask1_metrics


# "BM25" | "d2v" | "hft"
model_type = "hft"
big_model = False

# %% LOAD DATASETS
cache_dir = './data_cache'

train_dataset = load_dataset(
        "doc2dial",
        name="doc2dial_rc",
        split='train',
        ignore_verifications=True,
        cache_dir=cache_dir,
    )

train_dataset_sample = load_dataset(
        "doc2dial",
        name="doc2dial_rc",
        split='train[:10%]',
        ignore_verifications=True,
        cache_dir=cache_dir,
    )

dev_dataset =  load_dataset(
        "doc2dial",
        name="doc2dial_rc",
        split='validation',
        ignore_verifications=True,
        cache_dir=cache_dir,
    )

doc_dataset = load_dataset(
        "doc2dial",
        name="document_domain",
        split="train",
        ignore_verifications=True,
        cache_dir=cache_dir
    )

dia_dataset = load_dataset(
    "doc2dial",
    name = "dialogue_domain",
    split = "train",
    ignore_verifications=True,
    cache_dir=cache_dir
    )

rc_dataset = train_dataset_sample    

print(f"Train dataset with {len(train_dataset)} instances loaded")
print(f"Train sample dataset with {len(train_dataset_sample)} instances loaded")
print(f"Dev dataset with {len(dev_dataset)} instances loaded")
print(f"Doc dataset with {len(doc_dataset)} instances loaded")
print(f"Dialogue dataset with {len(dia_dataset)} instances loaded")

# %%
SPACY_MODEL = os.environ.get('SPACY_MODEL', 'en_core_web_sm')
QA_MODEL = os.environ.get('QA_MODEL', 'distilbert-base-cased-distilled-squad')
nlp = spacy.load(SPACY_MODEL, disable=['ner', 'parser', 'textcat'])

model_generator = ModelGenerator(nlp,big_model)
query_processor = QueryProcessor(nlp)
golden_retriever = GoldenRetriever(nlp,10,big_model)

# document_retriever = DocumentRetriever()
# passage_retriever = PassageRetrieval(nlp)
# answer_extractor = AnswerExtractor(QA_MODEL, QA_MODEL)

# %% Train each document on BM25/d2v
doc_data = model_generator.train_models(doc_dataset,rc_dataset,model_type)

# %%
golden_retriever.load_data(doc_data)

# %%
# Test predictions
test_predictions = []
span_store = []
partial = []
average_rank = []
positions = []
norm_positions = []
no_answer = []

for i in tqdm(range(0,len(rc_dataset))):
    ex =rc_dataset[i]
    query = ex["question"]
    q_id  = ex["title"]

    query = query_processor.rem_user(query)
    query = query_processor.simplify(query)

    # %% Retrieve best results for current document and question
    spans = golden_retriever.retrieve_spans(query,q_id)
  
    span = spans["span_text"][0]
    span_store = np.append(span_store,spans)

    if span in ex["answers"]["text"][0]:
        partial = partial + [1]

    # pos = []
    for sp in spans["span_text"]:
        correctTag = False
        if sp in ex["answers"]["text"][0]:
            # print(pos)
            # pos = np.append(pos,1+spans["span_text"].index(sp))
            pos = spans["span_text"].index(sp)
            positions = positions + [pos]
            norm_positions = norm_positions + [pos/len(spans["span_text"])]
            correctTag = True
            break

    if correctTag == False:
        no_answer = no_answer + [1]
        


    test_predictions.append(
        {
            "id": ex["id"],
            # "prediction_text": span,
            "prediction_text": span,
            "no_answer_probability": 0.0
        })

# test_predictions_json = json.dumps(test_predictions)

# %%

predictions_json = model_type + 'predictions.json'

with open(predictions_json, 'w', encoding='utf-8') as f:
    json.dump(test_predictions, f, ensure_ascii=False, indent=4)

predictions = json.load(open(predictions_json, "r"))

final_score_test = sharedtask1_metrics(predictions_json,'train',cache_dir)

print(final_score_test)
partial_perc = (sum(partial))/len(rc_dataset)*100
print("Partial: ", partial_perc)
print("Average Normalised Rank: ", np.mean(norm_positions))

xs = range(0,len(rc_dataset)-sum(no_answer))
plt.scatter(xs,norm_positions,marker='.')
# %%
