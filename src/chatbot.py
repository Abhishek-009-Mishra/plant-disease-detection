"""
Groq-powered RAG chatbot for plant disease advisory.
"""

from groq import Groq

from src.config import config
from src.rag_retriever import DiseaseRetriever


class PlantDiseaseChatbot:
    """Generate grounded plant-disease answers using RAG + Groq."""

    def __init__(self):
        if not config.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not configured.\n"
                "Add your Groq API key to the .env file."
            )

        self.client = Groq(
            api_key=config.GROQ_API_KEY
        )

        self.retriever = DiseaseRetriever()

    def _build_context(self, results):
        """Convert retrieved knowledge into LLM context."""

        if not results:
            return "No relevant plant-disease information was found."

        context_parts = []

        for result in results:
            knowledge = result["knowledge"]

            symptoms = "\n".join(
                f"- {item}"
                for item in knowledge.get("symptoms", [])
            )

            management = "\n".join(
                f"- {item}"
                for item in knowledge.get("management", [])
            )

            prevention = "\n".join(
                f"- {item}"
                for item in knowledge.get("prevention", [])
            )

            context_parts.append(
                f"""
Disease Class: {result["class_name"]}
Plant: {knowledge.get("plant", "Unknown")}
Disease: {knowledge.get("disease", "Unknown")}

Symptoms:
{symptoms}

Causes:
{knowledge.get("causes", "Not available.")}

Management:
{management}

Prevention:
{prevention}
"""
            )

        return "\n---\n".join(context_parts)

    def ask(self, question, predicted_class=None):
        """
        Answer a user question using retrieved knowledge.

        If predicted_class is supplied, the corresponding disease
        knowledge is prioritized.
        """

        if not question or not question.strip():
            return "Please enter a question."

        # Retrieve knowledge
        if predicted_class:
            prediction_result = (
                self.retriever.retrieve_for_prediction(
                    predicted_class
                )
            )

            results = []

            if prediction_result:
                results.append(prediction_result)

            # Also retrieve semantically related information
            results.extend(
                self.retriever.retrieve(
                    question,
                    top_k=config.RETRIEVAL_TOP_K
                )
            )

        else:
            results = self.retriever.retrieve(
                question,
                top_k=config.RETRIEVAL_TOP_K
            )

        # Remove duplicate classes
        unique_results = []
        seen = set()

        for result in results:
            class_name = result["class_name"]

            if class_name not in seen:
                seen.add(class_name)
                unique_results.append(result)

        context = self._build_context(unique_results)

        system_prompt = """
You are a plant disease advisory assistant.

Your job is to answer questions about plant diseases using
the provided retrieved knowledge.

IMPORTANT RULES:
1. Use the retrieved knowledge as your primary source.
2. Do not invent disease symptoms, causes, treatments, or prevention methods.
3. If the retrieved knowledge does not contain enough information,
   clearly say that the available knowledge is insufficient.
4. Give practical and easy-to-understand explanations.
5. Do not claim that an image-based prediction is a guaranteed diagnosis.
6. If a disease prediction is provided, refer to it as a model prediction.
7. Recommend consulting a qualified agricultural professional
   when the situation requires expert diagnosis or treatment.
"""

        user_prompt = f"""
Retrieved plant-disease knowledge:

{context}

User question:
{question}
"""

        response = self.client.chat.completions.create(
            model=config.GROQ_MODEL,
            temperature=config.GROQ_TEMPERATURE,
            max_tokens=config.GROQ_MAX_TOKENS,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        return response.choices[0].message.content


def start_chat():
    """Start an interactive chatbot session."""

    chatbot = PlantDiseaseChatbot()

    print("\n" + "=" * 60)
    print("PLANT DISEASE RAG CHATBOT")
    print("=" * 60)
    print("Ask questions about plant diseases.")
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("\nGoodbye!")
            break

        if not question:
            continue

        try:
            answer = chatbot.ask(question)

            print("\nAssistant:")
            print(answer)

        except Exception as e:
            print("\nERROR:")
            print(str(e))


if __name__ == "__main__":
    start_chat()