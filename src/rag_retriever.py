"""
TF-IDF based RAG retriever for plant disease knowledge.
"""

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.rag_loader import KnowledgeBaseLoader


class DiseaseRetriever:
    """Retrieve relevant disease knowledge using TF-IDF."""

    def __init__(self):
        self.loader = KnowledgeBaseLoader()

        self.documents = {}
        self.document_keys = []
        self.document_texts = []

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
        )

        self.document_vectors = None

    def _document_to_text(self, class_name, document):
        """Convert a knowledge document into searchable text."""

        parts = [
            class_name,
            document.get("plant", ""),
            document.get("disease", ""),
            document.get("causes", ""),
        ]

        for key in [
            "symptoms",
            "management",
            "prevention",
        ]:
            values = document.get(key, [])

            if isinstance(values, list):
                parts.extend(values)
            else:
                parts.append(str(values))

        return " ".join(parts)

    def build(self):
        """Build the TF-IDF index."""

        self.documents = self.loader.load()

        self.document_keys = list(
            self.documents.keys()
        )

        self.document_texts = [
            self._document_to_text(
                key,
                self.documents[key],
            )
            for key in self.document_keys
        ]

        self.document_vectors = (
            self.vectorizer.fit_transform(
                self.document_texts
            )
        )

        print(
            f"RAG index built successfully with "
            f"{len(self.document_keys)} documents."
        )

        return self

    def retrieve(self, query, top_k=3):
        """
        Retrieve the most relevant disease documents.

        Args:
            query: User question.
            top_k: Number of results.

        Returns:
            List of dictionaries containing scores
            and retrieved knowledge.
        """

        if not query or not query.strip():
            return []

        if self.document_vectors is None:
            self.build()

        query_vector = (
            self.vectorizer.transform(
                [query]
            )
        )

        similarities = cosine_similarity(
            query_vector,
            self.document_vectors,
        )[0]

        top_indices = similarities.argsort()[::-1][
            :top_k
        ]

        results = []

        for index in top_indices:

            class_name = self.document_keys[index]

            results.append(
                {
                    "class_name": class_name,
                    "score": float(
                        similarities[index]
                    ),
                    "knowledge": self.documents[
                        class_name
                    ],
                }
            )

        return results

    def retrieve_for_prediction(
        self,
        predicted_class,
    ):
        """Retrieve exact knowledge for a model prediction."""

        if self.document_vectors is None:
            self.build()

        document = self.documents.get(
            predicted_class
        )

        if document is None:
            return None

        return {
            "class_name": predicted_class,
            "score": 1.0,
            "knowledge": document,
        }

    @staticmethod
    def format_result(result):
        """Convert retrieved knowledge into readable text."""

        if not result:
            return "No relevant information found."

        knowledge = result["knowledge"]

        lines = [
            f"Plant: {knowledge.get('plant', 'Unknown')}",
            f"Disease: {knowledge.get('disease', 'Unknown')}",
            "",
            "Symptoms:",
        ]

        for item in knowledge.get(
            "symptoms",
            [],
        ):
            lines.append(f"- {item}")

        lines.extend(
            [
                "",
                f"Causes: {knowledge.get('causes', 'Not available.')}",
                "",
                "Management:",
            ]
        )

        for item in knowledge.get(
            "management",
            [],
        ):
            lines.append(f"- {item}")

        lines.append("")
        lines.append("Prevention:")

        for item in knowledge.get(
            "prevention",
            [],
        ):
            lines.append(f"- {item}")

        return "\n".join(lines)


if __name__ == "__main__":

    retriever = DiseaseRetriever()

    retriever.build()

    print("\nTesting RAG retrieval...")
    print("=" * 60)

    query = (
        "What are the symptoms and management "
        "of tomato early blight?"
    )

    results = retriever.retrieve(
        query,
        top_k=3,
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\nResult {rank}"
        )
        print(
            f"Class: {result['class_name']}"
        )
        print(
            f"Similarity: {result['score']:.4f}"
        )

        print(
            DiseaseRetriever.format_result(
                result
            )
        )