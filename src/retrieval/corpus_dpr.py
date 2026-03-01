"""Wikipedia DPR corpus management via Pyserini pre-built indexes.

The pre-built Lucene index (``wikipedia-dpr``) is downloaded automatically
by Pyserini on first use (~7 GB).  Java 11+ must be installed.
"""

INDEX_NAME = "wikipedia-dpr"


def verify_index() -> bool:
    """Smoke-test: try to open the pre-built index and print its size."""
    try:
        from pyserini.search.lucene import LuceneSearcher

        searcher = LuceneSearcher.from_prebuilt_index(INDEX_NAME)
        print(f"Index loaded successfully: {searcher.num_docs} documents")
        searcher.close()
        return True
    except Exception as e:
        print(f"Failed to load index: {e}")
        print(
            "Make sure Java 11+ is installed and pyserini is properly configured.\n"
            "  macOS:  brew install openjdk@21\n"
            "  Ubuntu: sudo apt install openjdk-21-jdk"
        )
        return False
