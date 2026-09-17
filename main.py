"""
main.py
-------
Single CLI entry point for the whole project.

Usage:

Direct commands:
    python main.py analyze
    python main.py analyze --representation grayscale
    python main.py train
    python main.py evaluate
    python main.py predict "path/to/leaf.jpg"
    python main.py history
    python main.py build-rag
    python main.py chat
    python main.py diagnose "path/to/leaf.jpg"

Interactive menu:
    python main.py
"""

import argparse
import sys

from src.config import config
from src.dataset_analysis import DatasetAnalyzer
from src.utils import print_banner


MENU_TEXT = """
1. Dataset Analysis
2. Train Model
3. Evaluate Model
4. Predict Disease
5. Prediction History
6. Build RAG Knowledge Base
7. Chatbot
8. Diagnose + Chat
9. Exit
"""


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "AI-Powered Plant Disease Detection "
            "and Intelligent Advisory System"
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    # =========================================================
    # DATASET ANALYSIS
    # =========================================================

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run dataset analysis / EDA",
    )

    analyze_parser.add_argument(
        "--representation",
        choices=config.SUPPORTED_REPRESENTATIONS,
        default=config.DEFAULT_REPRESENTATION,
        help=(
            "Which dataset representation to analyze "
            "(default: %(default)s)"
        ),
    )

    analyze_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots (faster, stats only)",
    )

    # =========================================================
    # TRAIN
    # =========================================================

    subparsers.add_parser(
        "train",
        help="Train the Computer Vision model",
    )

    # =========================================================
    # EVALUATE
    # =========================================================

    subparsers.add_parser(
        "evaluate",
        help="Evaluate the trained model",
    )

    # =========================================================
    # PREDICT
    # =========================================================

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict plant disease from a single image",
    )

    predict_parser.add_argument(
        "image",
        help="Path to the leaf image",
    )

    # =========================================================
    # HISTORY
    # =========================================================

    subparsers.add_parser(
        "history",
        help="Show prediction history",
    )

    # =========================================================
    # BUILD RAG
    # =========================================================

    subparsers.add_parser(
        "build-rag",
        help="Build the RAG knowledge base",
    )

    # =========================================================
    # CHATBOT
    # =========================================================

    subparsers.add_parser(
        "chat",
        help="Start the RAG chatbot",
    )

    # =========================================================
    # DIAGNOSE
    # =========================================================

    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Predict disease and provide AI advisory",
    )

    diagnose_parser.add_argument(
        "image",
        help="Path to the leaf image",
    )

    return parser


def run_analyze(
    representation: str,
    generate_plots: bool,
) -> None:
    """Run dataset analysis."""

    analyzer = DatasetAnalyzer(
        representation=representation
    )

    analyzer.analyze(
        generate_plots=generate_plots
    )


def show_prediction_history() -> None:
    """Display saved prediction history."""

    from src.database import PredictionDatabase

    database = PredictionDatabase()

    history = database.get_history()

    print("\n" + "=" * 95)
    print("PREDICTION HISTORY")
    print("=" * 95)

    if not history:
        print("\nNo prediction history found.")
        print("Run a prediction first using:")
        print(
            'python main.py predict "path/to/leaf.jpg"'
        )
        print("=" * 95)
        return

    print(
        f"\n{'ID':<5}"
        f"{'DATE/TIME':<20}"
        f"{'PLANT':<18}"
        f"{'DISEASE':<35}"
        f"{'CONF.':<10}"
    )

    print("-" * 95)

    for row in history:

        (
            prediction_id,
            image_path,
            timestamp,
            plant,
            disease,
            confidence,
            top_predictions,
        ) = row

        print(
            f"{prediction_id:<5}"
            f"{timestamp:<20}"
            f"{plant[:17]:<18}"
            f"{disease[:34]:<35}"
            f"{confidence * 100:.2f}%"
        )

    print("-" * 95)

    print(
        f"Total predictions: "
        f"{database.get_count()}"
    )

    print("=" * 95)


def run_interactive_menu() -> None:
    """Run the interactive CLI menu."""

    print_banner(
        "PLANT DISEASE DETECTION SYSTEM"
    )

    print(MENU_TEXT)

    choice = input(
        "Select an option: "
    ).strip()

    # =========================================================
    # OPTION 1 - DATASET ANALYSIS
    # =========================================================

    if choice == "1":

        rep = input(
            f"Representation "
            f"[{'/'.join(config.SUPPORTED_REPRESENTATIONS)}] "
            f"(default: {config.DEFAULT_REPRESENTATION}): "
        ).strip()

        rep = (
            rep
            if rep
            else config.DEFAULT_REPRESENTATION
        )

        run_analyze(
            rep,
            generate_plots=True,
        )

    # =========================================================
    # OPTION 2 - TRAIN
    # =========================================================

    elif choice == "2":

        from src.train import train_model

        train_model()

    # =========================================================
    # OPTION 3 - EVALUATE
    # =========================================================

    elif choice == "3":

        from src.evaluate import evaluate_model

        evaluate_model()

    # =========================================================
    # OPTION 4 - PREDICT
    # =========================================================

    elif choice == "4":

        image_path = input(
            "\nEnter the path to the leaf image:\n"
        ).strip()

        if not image_path:

            print(
                "\nERROR: Image path cannot be empty."
            )

            return

        from src.predictor import predict_image

        predict_image(
            image_path
        )

    # =========================================================
    # OPTION 5 - HISTORY
    # =========================================================

    elif choice == "5":

        show_prediction_history()

    # =========================================================
    # OPTION 6 - BUILD RAG
    # =========================================================

    elif choice == "6":

        from src.rag_retriever import DiseaseRetriever

        print(
            "\n" + "=" * 60
        )

        print(
            "BUILDING RAG KNOWLEDGE BASE"
        )

        print(
            "=" * 60
        )

        retriever = DiseaseRetriever()

        retriever.build()

        print(
            "\nRAG knowledge base built successfully."
        )

    # =========================================================
    # OPTION 7 - CHATBOT
    # =========================================================

    elif choice == "7":

        from src.chatbot import start_chat

        start_chat()

    # =========================================================
    # OPTION 8 - DIAGNOSE + CHAT
    # =========================================================

    elif choice == "8":

        image_path = input(
            "\nEnter the path to the leaf image:\n"
        ).strip()

        if not image_path:

            print(
                "\nERROR: Image path cannot be empty."
            )

            return

        from src.diagnose import diagnose_image

        diagnose_image(
            image_path
        )

    # =========================================================
    # OPTION 9 - EXIT
    # =========================================================

    elif choice == "9":

        print("Goodbye.")

        sys.exit(0)

    else:

        print(
            "Invalid choice."
        )


def main() -> None:
    """Main entry point."""

    parser = build_parser()

    args = parser.parse_args()

    try:

        # =====================================================
        # NO COMMAND -> INTERACTIVE MENU
        # =====================================================

        if args.command is None:

            run_interactive_menu()

            return

        # =====================================================
        # ANALYZE
        # =====================================================

        if args.command == "analyze":

            run_analyze(
                args.representation,
                generate_plots=not args.no_plots,
            )

        # =====================================================
        # TRAIN
        # =====================================================

        elif args.command == "train":

            from src.train import train_model

            train_model()

        # =====================================================
        # EVALUATE
        # =====================================================

        elif args.command == "evaluate":

            from src.evaluate import evaluate_model

            evaluate_model()

        # =====================================================
        # PREDICT
        # =====================================================

        elif args.command == "predict":

            from src.predictor import predict_image

            predict_image(
                args.image
            )

        # =====================================================
        # HISTORY
        # =====================================================

        elif args.command == "history":

            show_prediction_history()

        # =====================================================
        # BUILD RAG
        # =====================================================

        elif args.command == "build-rag":

            from src.rag_retriever import DiseaseRetriever

            print(
                "\n" + "=" * 60
            )

            print(
                "BUILDING RAG KNOWLEDGE BASE"
            )

            print(
                "=" * 60
            )

            retriever = DiseaseRetriever()

            retriever.build()

            print(
                "\nRAG knowledge base built successfully."
            )

        # =====================================================
        # CHATBOT
        # =====================================================

        elif args.command == "chat":

            from src.chatbot import start_chat

            start_chat()

        # =====================================================
        # DIAGNOSE + CHAT
        # =====================================================

        elif args.command == "diagnose":

            from src.diagnose import diagnose_image

            diagnose_image(
                args.image
            )

        # =====================================================
        # UNKNOWN COMMAND
        # =====================================================

        else:

            print(
                f"The '{args.command}' command "
                "is not implemented yet."
            )

    except FileNotFoundError as e:

        print("\nERROR:")

        print(str(e))

        sys.exit(1)

    except ValueError as e:

        print("\nERROR:")

        print(str(e))

        sys.exit(1)

    except KeyboardInterrupt:

        print(
            "\nCancelled by user."
        )

        sys.exit(1)

    except Exception as e:

        print("\nERROR:")

        print(str(e))

        sys.exit(1)


if __name__ == "__main__":
    main()