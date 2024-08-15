import bm25s
import os
import torch
import json
from sentence_transformers import SentenceTransformer, util

valid_bm25_retrieval_models = ["bm25"]

valid_st_retrieval_models = [
    "msmarco-distilbert-base-v3",
    "gtr-t5-large",
]


class Retriever:
    def __init__(
        self,
        wikipedia_path: str,
        verbose: bool,
    ) -> None:
        """
        for each query, the top-k documents are stored in the cache, the cache is a dictionary with keys as the query and values as the top-k documents
        the key is a string of the form "entity##query##top-k##model_name"
        the value is a list of strings, each string is a corpus paragraph
        """
        self.wikipedia_path = wikipedia_path
        self.verbose = verbose
        self.cache_path = wikipedia_path + "/.cache/retriever_cache.json"
        self.cache = self._load_cache()

        if not os.path.exists(wikipedia_path + "/.cache"):
            os.mkdir(wikipedia_path + "/.cache")

    def _load_documents(self, entity: str) -> list[str]:
        file_path = os.path.join(self.wikipedia_path, entity + ".txt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        with open(file_path, "r") as f:
            paragraphs = f.read().split("\n")

        return paragraphs

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                cache = json.load(f)
        else:
            cache = {}

        return cache

    def _save_cache(self):
        old_cache = self._load_cache()
        self.cache.update(old_cache)

        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f)

    def retrieve(
        self, entity: str, query: str, top_k: int = 5, verbose: bool = True
    ) -> list[str]:
        raise NotImplementedError("Subclasses should implement this method")


class BM25Retriever(Retriever):
    def __init__(
        self,
        wikipedia_path: str,
        verbose: bool = True,
    ) -> None:
        super().__init__(wikipedia_path, verbose)
        self.bm25 = bm25s.BM25()

    def _bm25_fit(self, corpus: list[str]) -> list[str]:
        tokenized_corpus = bm25s.tokenize(corpus)
        self.bm25.index(tokenized_corpus)

    def _bm25_retrieve(self, query: str, corpus: list[str], top_k: int) -> list[str]:
        tokenized_query = bm25s.tokenize(query)
        results, scores = self.bm25.retrieve(tokenized_query, corpus=corpus, k=top_k)
        return results, scores

    def retrieve(
        self,
        entity: str,
        query: str,
        top_k: int = 5,
    ) -> list[str]:

        cache_key = entity + "#" + query[:4] + "#bm25"

        if cache_key in self.cache.keys():
            return self.cache.get(cache_key)

        else:
            corpus = self._load_documents(entity)
            self._bm25_fit(corpus=self._load_documents(entity))
            results, scores = self._bm25_retrieve(query, corpus, top_k)

            if self.verbose:
                for i in range(results.shape[1]):
                    doc, score = results[0, i], scores[0, i]
                    print(f"Rank {i+1} (score: {score:.2f}): {doc}")

            res = [d for d in results[0]]

            cache_key = entity + "#" + query[:4] + "#bm25"
            self.cache[cache_key] = res
            self._save_cache()

            return res


class SentenceTransformerRetriever(Retriever):
    def __init__(
        self,
        wikipedia_path: str,
        verbose: bool = True,
        model_name: str = "gtr-t5-large",
    ) -> None:
        super().__init__(wikipedia_path, verbose)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def _sentence_transformer_fit(self, corpus: list[str]) -> list[str]:
        return self.model.encode(corpus)

    def _sentence_transformer_retrieve(
        self, query: str, corpus: list[str], top_k: int
    ) -> list[str]:
        query_embedding = self.model.encode(query)
        corpus_embeddings = self._sentence_transformer_fit(corpus)
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        return top_results

    def retrieve(self, entity: str, query: str, top_k: int = 5) -> list[str]:

        cache_key = entity + "#" + query[:4] + "#" + self.model_name

        if cache_key in self.cache.keys():
            return self.cache.get(cache_key)

        else:
            corpus = self._load_documents(entity)
            results = self._sentence_transformer_retrieve(query, corpus, top_k)

            if self.verbose:
                for i in range(top_k):
                    doc, score = corpus[results.indices[i]], results.values[i]
                    print(f"Rank {i+1} (score: {score:.2f}): {doc}")

            res = [corpus[d] for d in results.indices]

            self.cache[cache_key] = res
            self._save_cache()

            return res


def retrieve(
    model_name: str,
    entity: str,
    query: str,
    top_k: int = 5,
    verbose: bool = True,
) -> list[str]:
    if model_name in valid_bm25_retrieval_models:
        retriever = BM25Retriever(
            wikipedia_path="OKG/wikipedia",
            verbose=verbose,
        )
    elif model_name in valid_st_retrieval_models:
        retriever = SentenceTransformerRetriever(
            wikipedia_path="OKG/wikipedia",
            verbose=verbose,
            model_name=model_name,
        )
    else:
        raise ValueError(f"Retriever model `{model_name}' is not supported")

    return retriever.retrieve(entity=entity, query=query, top_k=top_k)


if __name__ == "__main__":
    res = retrieve(
        model_name="bm25",
        entity="1930",
        query="what events occur in 1930?",
        top_k=3,
        verbose=False,
    )
    res = retrieve(
        model_name="msmarco-distilbert-base-v3",
        entity="1930",
        query="what events occur in 1930?",
        top_k=3,
        verbose=False,
    )
