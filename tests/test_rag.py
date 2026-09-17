from src.rag_loader import load_knowledge_base
from src.rag_retriever import DiseaseRetriever


def test_knowledge_base_loading():
    knowledge = load_knowledge_base()

    assert knowledge is not None
    assert len(knowledge) == 38


def test_rag_retrieval():
    retriever = DiseaseRetriever()

    results = retriever.retrieve(
        "Tomato Early blight symptoms",
        top_k=3
    )

    assert len(results) > 0
    assert len(results) <= 3