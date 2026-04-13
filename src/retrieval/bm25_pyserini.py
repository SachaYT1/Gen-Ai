"""BM25 retrieval over a Pyserini Lucene index."""

import json
from typing import List, Dict

from tqdm import tqdm


class BM25Retriever:
    def __init__(self, index_name: str = "wikipedia-dpr"):
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher("/root/.cache/pyserini/indexes/lucene-index.wikipedia-dpr-100w.20210120.d1b9e6")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        hits = self.searcher.search(query, k=top_k)
        results = []
        for hit in hits:
            doc = self.searcher.doc(hit.docid)
            content = json.loads(doc.raw())
            results.append(
                {
                    "docid": hit.docid,
                    "score": hit.score,
                    "title": content.get("title", ""),
                    "text": content.get("contents", content.get("text", "")),
                }
            )
        return results

    def batch_retrieve(
        self, queries: List[str], top_k: int = 5
    ) -> List[List[Dict]]:
        return [
            self.retrieve(q, top_k)
            for q in tqdm(queries, desc="BM25 retrieval")
        ]

    def close(self):
        self.searcher.close()
