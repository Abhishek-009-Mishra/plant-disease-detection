"""
RAG knowledge-base loader.

Loads the plant disease knowledge from JSON and prepares
documents for retrieval.
"""

import json
from pathlib import Path

from src.config import config


class KnowledgeBaseLoader:
    """Load and manage plant disease knowledge."""

    def __init__(self, knowledge_path=None):
        self.knowledge_path = Path(
            knowledge_path
            if knowledge_path
            else config.KNOWLEDGE_DIR
        )

        if self.knowledge_path.is_dir():
            self.knowledge_path = (
                self.knowledge_path / "plant_diseases.json"
            )

        self.knowledge = {}

    def load(self):
        """Load the JSON knowledge base."""

        if not self.knowledge_path.exists():
            raise FileNotFoundError(
                f"Knowledge base not found:\n"
                f"{self.knowledge_path}"
            )

        with open(
            self.knowledge_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.knowledge = json.load(file)

        if not self.knowledge:
            raise ValueError(
                "Knowledge base is empty."
            )

        print(
            f"Loaded {len(self.knowledge)} "
            f"knowledge entries."
        )

        return self.knowledge

    def get_document(self, class_name):
        """Return knowledge for a specific model class."""

        if not self.knowledge:
            self.load()

        return self.knowledge.get(class_name)

    def get_all_documents(self):
        """Return all knowledge documents."""

        if not self.knowledge:
            self.load()

        return self.knowledge


def load_knowledge_base():
    """Convenience function."""

    loader = KnowledgeBaseLoader()

    return loader.load()


if __name__ == "__main__":
    knowledge = load_knowledge_base()

    print("\nKnowledge Base")
    print("=" * 60)

    for class_name in knowledge:
        print(f"- {class_name}")