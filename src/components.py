import concurrent.futures
import itertools
import operator
import re
from tkinter import E
from datasets.load import load_dataset
import pickle
import numpy as np
from gensim.summarization.bm25 import BM25
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, QuestionAnsweringPipeline
from operator import itemgetter
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed
)
from dataclasses import dataclass, field
from transformers import DistilBertTokenizerFast
import torch
from transformers import DistilBertForQuestionAnswering
from transformers import BertConfig
import os
import logging

logger = logging.getLogger(__name__)

class TorchQADataset(torch.utils.data.Dataset):
    def __init__(self, embeddings):
        self.encodings = embeddings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

class HFTTraining():
    def __init__(self):
        self.pre_trained_model = "distilbert-base-uncased"
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def embed(self,questions,contexts):
        self.embeddings = self.tokenizer(contexts, questions, truncation=True, padding=True)

    def add_token_positions(self, answers):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(self.embeddings.char_to_token(i, answers[i]['answer_start']))
            end_positions.append(self.embeddings.char_to_token(i, answers[i]['answer_end'] - 1))
            # if None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = self.tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = self.tokenizer.model_max_length
        self.embeddings.update({'start_positions': start_positions, 'end_positions': end_positions})

    def convert_to_torch(self,torch_dataset_converter):
        self.train_dataset = torch_dataset_converter(self.embeddings)

    def model_training(self,i):
        config = BertConfig.from_pretrained( self.pre_trained_model, output_hidden_states=True)
        model = DistilBertForQuestionAnswering.from_pretrained(self.pre_trained_model)
        EPOCHS=1
        MODEL_PATH = f'./hft_models/{self.pre_trained_model}'
        training_args = TrainingArguments(
            output_dir=MODEL_PATH,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            per_gpu_train_batch_size=8,
            per_gpu_eval_batch_size=8,
            num_train_epochs=EPOCHS,
            logging_first_step=True,
            save_steps=5000,
            fp16=False
        )
        set_seed(42)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset
        )
        
        trainer.train()
        path = "./hft_models/doc" + str(i) + "_model.pt"
        trainer.save_model(path)

class ModelGenerator:
    def __init__(self,nlp,big=False):
        self.nlp = nlp
        self.big = big
        self.query_processor = QueryProcessor(nlp)
        self.tokenize = lambda text: [token.lemma_ for token in nlp(text)]
    
    def train_models(self,dataset,rc_data,model_type):
        doc_ids = dataset["doc_id"]
        doc_spans = dataset["spans"]
        doc_models = []
        max_idx_length = 0 
        span_idx = []
        big_corpus = []
        all_spans = []

        # process rc data for HFT model
        questions = [question[5:] for question in rc_data["question"]]
        contexts = [context for context in rc_data["context"]]
        answer_start = [answer["answer_start"][0] for answer in rc_data["answers"]]
        answer_end = [answer["answer_start"][0] + len(answer["text"][0]) for answer in rc_data["answers"]]
        answers = [{"answer_start":a_start,"answer_end":a_end} for a_start,a_end in list(zip(answer_start,answer_end))]
        titles = [title for title in rc_data["title"]]

        for i in range(0,len(doc_spans)):

            spans = doc_spans[i]
            passages = []
            idx_lst = []

            doc_id = doc_ids[i]

            # Checking to make sure at least one of these spans is in the answer set
            count = 0
            for title in titles:
                if doc_id == title:
                    count = count + 1 

            for j in range(0,len(spans)):
                if spans[j]["tag"]=='u':
                    all_spans = all_spans + [spans[j]["text_sp"]]
                    passages.append(spans[j]["text_sp"])
                    # passages.append(self.query_processor.simplify(spans[j]["text_sp"]))
                    idx_lst.append(j)

                    if len(idx_lst) > max_idx_length:
                        max_idx_length = len(idx_lst)

            span_idx.append(idx_lst)

            if model_type == "BM25": 
                corpus = [self.tokenize(p) for p in passages]
                model = BM25(corpus)
                big_corpus = big_corpus + corpus
                
            elif model_type == "d2v":
                documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(passages)]
                model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
                fname = 'doc2vec_models/' + str(i) + ".pickle"
                with open(fname,'wb') as f:
                    pickle.dump(all, f)

            elif model_type == 'hft' and count>0:

                PATH = "./hft_models/doc" + str(i) + "_model.pt"
                
                if os.path.isfile(PATH):
                    model = i
                    continue
                else:
                    train_index     = [i for i, x in enumerate(titles) if x == doc_id]
                    train_questions = [questions[i] for i in train_index]
                    train_answers   = [answers[i] for i in train_index]
                    train_contexts  = [contexts[i] for i in train_index]
    
                    hft_training = HFTTraining()
                    hft_training.embed(train_questions,train_contexts)
                    hft_training.add_token_positions(train_answers)
                    hft_training.convert_to_torch(TorchQADataset)
                    print("Training model ", str(i), " of ",str(len(doc_spans)))
                    hft_training.model_training(i)
                    model = i
                    #torch.save(model, path)
            elif model_type == 'hft' and count == 0:
                model =[]
                continue
            else:
                print("No Training: Incorrect model selection")

            doc_models.append(model)

        if self.big == True:
            if model_type == "BM25":
                b_model = BM25(big_corpus)

            elif model_type == "d2v":
                documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_spans)]
                b_model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
                fname = 'doc2vec_models/' + "big" + ".pickle"
                with open(fname,'wb') as f:
                    pickle.dump(b_model, f)

            elif model_type == 'hft':
                path = "./hft_models/doc_big_model.pt"
                try:
                    model = torch.load(path)
                    model.eval()
                except:
                    hft_training = HFTTraining()
                    hft_training.embed(questions,contexts)
                    hft_training.add_token_positions(answers)
                    hft_training.convert_to_torch
                    b_model = hft_training.model_training
                    torch.save(b_model, path)
            else:
                print("No Training: Incorrect model selection")
        else:
            b_model = []

        doc_data = {"doc_models":doc_models,"big_model":b_model,"doc_ids":doc_ids,"span_idx":span_idx,"doc_spans":doc_spans,"model_type":model_type}

        return doc_data

class QueryProcessor:
#   Convert question into basic speech tokens for better document retrieval


    def __init__(self, nlp, keep=None):
        self.nlp = nlp
        self.keep = keep or {'PROPN', 'NUM', 'VERB', 'NOUN', 'ADJ'}
    
    def rem_user(self,query):
        query = query.replace("user:","")
        return query

    def simplify(self, text):
        doc = self.nlp(text)
        query = ' '.join(token.text for token in doc if token.pos_ in self.keep)
        return query

class GoldenRetriever:

    def similarity(self,x, y):
        dot_prod = np.dot(x, y)
        return dot_prod / (np.linalg.norm(x)*np.linalg.norm(y))

    def __init__(self,nlp,n=10,big=False):
        self.n         = n
        self.big       = big
        self.tokenize  = lambda text: [token.lemma_ for token in nlp(text)]

    def load_data(self,data):
        self.models    = data["doc_models"]
        self.big_model = data["big_model"]
        self.doc_ids   = data["doc_ids"]
        self.span_idx  = data["span_idx"]
        self.doc_spans = data["doc_spans"]
        self.model_type = data["model_type"]
        self.all_spans = []

        if self.big == True:
            for i in range(0,len(self.models)):
                self.all_spans = self.all_spans + list(itemgetter(*self.span_idx[i])(self.doc_spans[i]))

    def retrieve_spans(self,query,q_id):

        failFlag = True

        tokens = self.tokenize(query)

        for i in range(0,len(self.models)):
            if self.big == True:
                model = self.big_model
            else:
                model = self.models[i]

            if self.doc_ids[i] == q_id:
                failFlag = False
                res_spans = list(itemgetter(*self.span_idx[i])(self.doc_spans[i]))
                break

        if failFlag == True:
            print("Error: Failed to find matching document")

        if self.model_type == "BM25":
            scores = model.get_scores(tokens)
        elif self.model_type == "d2v":
            scores = []
            for i in range(0,len(res_spans)):
            # for span in span_set):
                query_vector = model.infer_vector(tokens)
                span_vector = model.infer_vector(res_spans[i])
                scores.append(self.similarity(query_vector,span_vector))
                            

        pairs = [(s, i) for i, s in enumerate(scores)]
        pairs.sort(reverse=True)
        
        #lim=self.n
        lim = len(pairs)
        spans = {"span_text":[],"score":[]}
        spans["score"]=[s for s,_ in pairs[:lim]]

        # if self.big == True: 
        #     spans["span_text"] = [self.all_spans[i]["text_sp"] for _, i in pairs[:lim]]
        # else:
        #     spans["span_text"]=[res_spans[i]["text_sp"] for _, i in pairs[:lim]]
        spans["span_text"]=[res_spans[i]["text_sp"] for _, i in pairs[:lim]]

        return spans

# class DocumentRetriever:
#     "explanation"
#     def search(self, query):
#         pages = self.search_pages(query)
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             process_list = [executor.submit(self.search_page, page['pageid']) for page in pages['query']['search']]
#             docs = [self.post_process(p.result()) for p in process_list]
#         return docs

#     def post_process(self, doc):
#         pattern = '|'.join([
#             '== References ==',
#             '== Further reading ==',
#             '== External links',
#             '== See also ==',
#             '== Sources ==',
#             '== Notes ==',
#             '== Further references ==',
#             '== Footnotes ==',
#             '=== Notes ===',
#             '=== Sources ===',
#             '=== Citations ===',
#         ])
#         p = re.compile(pattern)
#         indices = [m.start() for m in p.finditer(doc)]
#         min_idx = min(*indices, len(doc))
#         return doc[:min_idx]


# class PassageRetrieval:

#     def __init__(self, nlp):
#         self.tokenize = lambda text: [token.lemma_ for token in nlp(text)]
#         self.bm25 = None
#         self.passages = None

#     def preprocess(self, doc):
#         passages = [p for p in doc.split('\n') if p and not p.startswith('=')]
#         return passages

#     def fit(self, docs):
#         passages = list(itertools.chain(*map(self.preprocess, docs)))

#         corpus = [self.tokenize(p) for p in passages]
#         self.bm25 = BM25(corpus)
#         self.passages = passages

#     def most_similar(self, question, topn=10):
#         tokens = self.tokenize(question)
#         scores = self.bm25.get_scores(tokens)
#         pairs = [(s, i) for i, s in enumerate(scores)]
#         pairs.sort(reverse=True)
#         passages = [self.passages[i] for _, i in pairs[:topn]]
#         return passages


# class AnswerExtractor:

#     def __init__(self, tokenizer, model):
#         tokenizer = AutoTokenizer.from_pretrained(tokenizer)
#         model = AutoModelForQuestionAnswering.from_pretrained(model)
#         self.nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer)

#     def extract(self, question, passages):
#         answers = []
#         for passage in passages:
#             try:
#                 answer = self.nlp(question=question, context=passage)
#                 answer['text'] = passage
#                 answers.append(answer)
#             except KeyError:
#                 pass
#         answers.sort(key=operator.itemgetter('score'), reverse=True)
#         return answers
