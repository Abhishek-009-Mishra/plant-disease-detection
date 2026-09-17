"""
End-to-end plant disease diagnosis.

Combines:
    1. Computer Vision prediction
    2. RAG knowledge retrieval
    3. Groq-powered advisory
"""

from src.predictor import PlantDiseasePredictor
from src.chatbot import PlantDiseaseChatbot


class PlantDiseaseDiagnoser:
    """Run complete plant disease diagnosis."""

    def __init__(self):
        self.predictor = PlantDiseasePredictor()
        self.chatbot = PlantDiseaseChatbot()

    def diagnose(self, image_path):
        """
        Predict the disease and generate an AI advisory.
        """

        # -----------------------------------------------------
        # STEP 1 - COMPUTER VISION PREDICTION
        # -----------------------------------------------------

        result = self.predictor.predict(
            image_path,
            save_history=True
        )

        # -----------------------------------------------------
        # STEP 2 - DISPLAY PREDICTION
        # -----------------------------------------------------

        self.predictor.display_prediction(result)

        # -----------------------------------------------------
        # STEP 3 - BUILD QUESTION FOR RAG + GROQ
        # -----------------------------------------------------

        question = (
            f"The computer vision model predicted "
            f"{result['plant']} - {result['disease']} "
            f"with {result['confidence'] * 100:.2f}% confidence. "
            f"Provide an advisory explaining the likely symptoms, "
            f"causes, management and prevention for this prediction."
        )

        # -----------------------------------------------------
        # STEP 4 - RAG + GROQ
        # -----------------------------------------------------

        print("\n" + "=" * 60)
        print("GENERATING AI ADVISORY")
        print("=" * 60)

        answer = self.chatbot.ask(
            question=question,
            predicted_class=result["top_predictions"][0]["class_name"]
        )

        # -----------------------------------------------------
        # STEP 5 - DISPLAY ADVISORY
        # -----------------------------------------------------

        print("\n" + "-" * 60)
        print("AI PLANT DISEASE ADVISORY")
        print("-" * 60)

        print(answer)

        print("\n" + "-" * 60)
        print(
            "NOTE: This is an AI-based prediction and advisory. "
            "It should not be treated as a guaranteed diagnosis."
        )
        print("-" * 60)

        return {
            "prediction": result,
            "advisory": answer
        }


def diagnose_image(image_path):
    """Convenience function for main.py."""

    diagnoser = PlantDiseaseDiagnoser()

    return diagnoser.diagnose(
        image_path
    )


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print(
            'Usage: python -m src.diagnose "path/to/leaf.jpg"'
        )
        raise SystemExit(1)

    diagnose_image(
        sys.argv[1]
    )