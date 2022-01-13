# %%
import os
from tqdm import tqdm
import json
import argparse
import numpy as np

from datasets import load_dataset
from datasets import load_metric

import spacy

from components import ModelGenerator,QueryProcessor, DocumentRetrieval, PassageRetrieval, AnswerExtractor
from metric import sharedtask1_metrics

# %% LOAD DATASETS

cache_dir = './data_cache'
train_dataset = load_dataset(
        "doc2dial",
        name="doc2dial_rc",
        split='train',
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

print(f"Train dataset with {len(train_dataset)} instances loaded")
print(f"Dev dataset with {len(dev_dataset)} instances loaded")
print(f"Doc dataset with {len(doc_dataset)} instances loaded")
print(f"Dialogue dataset with {len(dia_dataset)} instances loaded")

# %%
# Put the data into lists ready for the next steps...
# train_answers = []
# train_questions = []
# train_ids = []

# for i in tqdm(range(len(train_dataset))):
#     train_answers.append(train_dataset[i]['answers']['text'])
#     train_questions.append(train_dataset[i]['question'])
#     train_ids.append(train_dataset[i]['id'])

SPACY_MODEL = os.environ.get('SPACY_MODEL', 'en_core_web_sm')
QA_MODEL = os.environ.get('QA_MODEL', 'distilbert-base-cased-distilled-squad')
nlp = spacy.load(SPACY_MODEL, disable=['ner', 'parser', 'textcat'])

model_generator = ModelGenerator(doc_dataset)
query_processor = QueryProcessor(nlp)
document_retriever = DocumentRetrieval()
passage_retriever = PassageRetrieval(nlp)
answer_extractor = AnswerExtractor(QA_MODEL, QA_MODEL)

# %%
# import concurrent.futures
# import itertools
# import operator
# import re

# import requests
# import wikipedia
# from gensim.summarization.bm25 import BM25
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering, QuestionAnsweringPipeline

# from operator import itemgetter

# %%

# document = train_dataset[0]["context"]
# passages = [p for p in document.split('\n') if p and not p.startswith('=')]
# doc_ids = doc_dataset["doc_id"]
# doc_spans = doc_dataset["spans"]
# doc_models = []
# span_idx = []
# max_idx_length = 0 
# for i in range(0,len(doc_spans)):
#     spans = doc_spans[i]
#     corpus_name = 'bm25_' + str(i)
#     passages = []
#     idx_lst = []
#     for j in range(0,len(spans)):
#         if spans[j]["tag"]=='u':
#             passages.append(spans[j]["text_sp"])
#             # span_idx.append(passages)
#             idx_lst.append(j)
#             if len(idx_lst) > max_idx_length:
#                 max_idx_length = len(idx_lst)
#     span_idx.append(idx_lst)
#     corpus = [tokenize(p) for p in passages]
#     doc_models.append(BM25(corpus))

    
# %%

question = train_dataset[0]["question"]
q_doc_id = train_dataset[0]["title"]

tokens = tokenize(question)

text_spans = []

for i in range(0,len(doc_models)):
    if doc_ids[i] == q_doc_id:
        scores = doc_models[i].get_scores(tokens)
        res_spans = list(itemgetter(*span_idx[i])(doc_spans[i]))
        break

pairs = [(s, i) for i, s in enumerate(scores)]
pairs.sort(reverse=True)
text_spans = [res_spans[i]["text_sp"] for _, i in pairs[:10]]

# %%
queries = []
for i in tqdm(range(0,5)):
    question = train_dataset[i]["question"]
    document = train_dataset[i]["context"]
    query = query_processor.generate_query(question)
    queries = np.append(queries,query)
    passages = passage_retriever.fit(document)
    # passage_rank = passage_retriever.most_similar(passages)

# queries = []
# for i in tqdm(range(0,3)):
#     query = query_processor.generate_query(train_questions[i])
#     print('Question: ',train_questions[i])
#     print('\n Query: ',query)
#     queries = np.append(queries,query)

# %%
docs = document_retriever.search(query)
# passage_retriever.fit(docs)
# passages = passage_retriever.most_similar(question)
# answers = answer_extractor.extract(question, passages)
# jsonify(answers)

# %%
# Test predictions
test_predictions = []
for ex in train_dataset:
    test_predictions.append(
        {
            "id": ex["id"],
            "prediction_text": ex["answers"]["text"][0],
            "no_answer_probability": 0.0
        })

test_predictions_json = json.dumps(test_predictions)

# %%
# with open('predictions.json', "w") as writer:
#     writer.write(json.dumps(test_predictions, indent=4) + "\n")

with open('predictions.json', 'w', encoding='utf-8') as f:
    json.dump(test_predictions, f, ensure_ascii=False, indent=4)

# %%
predictions_json = 'predictions.json'
predictions = json.load(open(predictions_json, "r"))
# %%
final_score_test = sharedtask1_metrics(predictions_json,'train',cache_dir)
# %%

