import concurrent.futures
from dataclasses import dataclass
import itertools
import operator
import re
from datasets.load import load_dataset
import pickle
import numpy as np
from gensim.summarization.bm25 import BM25
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, QuestionAnsweringPipeline
from operator import itemgetter

class ModelGenerator:
    def __init__(self,nlp):
        self.nlp = nlp
        self.tokenize = lambda text: [token.lemma_ for token in nlp(text)]
    
    def train_models(self,dataset,model_type):
        doc_ids = dataset["doc_id"]
        doc_spans = dataset["spans"]
        doc_models = []
        max_idx_length = 0 
        span_idx = []

        for i in range(0,len(doc_spans)):
            
            spans = doc_spans[i]
            passages = []
            idx_lst = []

            for j in range(0,len(spans)):
                if spans[j]["tag"]=='u':
                    passages.append(spans[j]["text_sp"])
                    idx_lst.append(j)

                    if len(idx_lst) > max_idx_length:
                        max_idx_length = len(idx_lst)

            span_idx.append(idx_lst)

            if model_type == "BM25": 
                corpus = [self.tokenize(p) for p in passages]
                model = BM25(corpus)

            elif model_type == "d2v":
                documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(passages)]
                model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
                fname = 'doc2vec_models/' + str(i) + ".pickle"
                with open(fname,'wb') as f:
                    pickle.dump(all, f)
            else:
                print("No Training: Incorrect model selection")

            doc_models.append(model)

        doc_data = {"doc_models":doc_models,"doc_ids":doc_ids,"span_idx":span_idx,"doc_spans":doc_spans,"model_type":model_type}

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

    def __init__(self,nlp,n=10):
        self.n         = n
        self.tokenize  = lambda text: [token.lemma_ for token in nlp(text)]

    def load_data(self,data):
        self.models    = data["doc_models"]
        self.doc_ids   = data["doc_ids"]
        self.span_idx  = data["span_idx"]
        self.doc_spans = data["doc_spans"]
        self.model_type = data["model_type"]

    def retrieve_spans(self,query,q_id):

        for i in range(0,len(self.models)):
            if self.doc_ids[i] == q_id:
                tokens = self.tokenize(query)
                res_spans = list(itemgetter(*self.span_idx[i])(self.doc_spans[i]))
                
                if self.model_type == "BM25":
                    scores = self.models[i].get_scores(tokens)
                elif self.model_type == "d2v":
                    scores = []
                    for span in res_spans:
                        query_vector = self.models[i].infer_vector(tokens)
                        span_vector = self.models[i].infer_vector(span)
                        scores.append(self.similarity(query_vector,span_vector))
                break

        pairs = [(s, i) for i, s in enumerate(scores)]
        pairs.sort(reverse=True)
        lim=self.n
        spans = {"span_text":[],"score":[]}
        spans["score"]=[s for s,_ in pairs[:lim]]
        spans["span_text"]=[res_spans[i]["text_sp"] for _, i in pairs[:lim]]
        return spans

class DocumentRetriever:
    "explanation"
    def search(self, query):
        pages = self.search_pages(query)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            process_list = [executor.submit(self.search_page, page['pageid']) for page in pages['query']['search']]
            docs = [self.post_process(p.result()) for p in process_list]
        return docs

    def post_process(self, doc):
        pattern = '|'.join([
            '== References ==',
            '== Further reading ==',
            '== External links',
            '== See also ==',
            '== Sources ==',
            '== Notes ==',
            '== Further references ==',
            '== Footnotes ==',
            '=== Notes ===',
            '=== Sources ===',
            '=== Citations ===',
        ])
        p = re.compile(pattern)
        indices = [m.start() for m in p.finditer(doc)]
        min_idx = min(*indices, len(doc))
        return doc[:min_idx]


class PassageRetrieval:

    def __init__(self, nlp):
        self.tokenize = lambda text: [token.lemma_ for token in nlp(text)]
        self.bm25 = None
        self.passages = None

    def preprocess(self, doc):
        passages = [p for p in doc.split('\n') if p and not p.startswith('=')]
        return passages

    def fit(self, docs):
        passages = list(itertools.chain(*map(self.preprocess, docs)))

        corpus = [self.tokenize(p) for p in passages]
        self.bm25 = BM25(corpus)
        self.passages = passages

    def most_similar(self, question, topn=10):
        tokens = self.tokenize(question)
        scores = self.bm25.get_scores(tokens)
        pairs = [(s, i) for i, s in enumerate(scores)]
        pairs.sort(reverse=True)
        passages = [self.passages[i] for _, i in pairs[:topn]]
        return passages


class AnswerExtractor:

    def __init__(self, tokenizer, model):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        model = AutoModelForQuestionAnswering.from_pretrained(model)
        self.nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer)

    def extract(self, question, passages):
        answers = []
        for passage in passages:
            try:
                answer = self.nlp(question=question, context=passage)
                answer['text'] = passage
                answers.append(answer)
            except KeyError:
                pass
        answers.sort(key=operator.itemgetter('score'), reverse=True)
        return answers
