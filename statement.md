# Problem Statement

> Draft version - written in Phase 1, will be reviewed/finalized in Phase 13
> once training results and the RAG system exist to reference.

## Problem Statement

Farmers and home growers often struggle to quickly and accurately identify plant
diseases from visual symptoms on leaves, leading to delayed or incorrect treatment
and avoidable crop loss. Expert agronomists are not always available, and generic
web searches do not provide a structured, evidence-based diagnostic and advisory
workflow.

## Project Scope

- Automatically classify a leaf image into a specific plant + disease/healthy
  category using a Computer Vision model trained with transfer learning on the
  PlantVillage dataset.
- Provide a confidence score and top-k alternative predictions rather than a
  single unqualified answer, since image classification is probabilistic, not
  a certified diagnosis.
- Maintain a history of past predictions for reference.
- Provide a Retrieval-Augmented Generation (RAG) chatbot, backed by a curated
  local knowledge base and the Grok LLM, that can answer follow-up questions
  about a detected disease (symptoms, causes, prevention, management) using
  retrieved, source-grounded context rather than the LLM's unaided memory.
- Out of scope for this version: web/GUI interface (CLI only), real-time video
  analysis, and treatment recommendations that require licensed agronomic advice.

## Target Users

- Students/researchers studying applied Computer Vision.
- Small-scale farmers or hobbyist gardeners seeking a first-pass diagnosis.
- Agricultural extension workers wanting a quick triage tool.

## High-Level Features

1. Automatic dataset discovery and EDA (plants, diseases, class balance, image
   integrity) without assuming a fixed number of classes.
2. Configurable Computer Vision preprocessing pipeline (validation, resize,
   normalization, augmentation) supporting color/grayscale/segmented inputs.
3. Transfer-learning-based multi-class leaf classifier with full evaluation
   (accuracy, precision, recall, F1, confusion matrix).
4. CLI-based single-image prediction with confidence and top-k output.
5. Persistent prediction history (SQLite).
6. RAG-based chatbot over a local, editable knowledge base, using the Grok API,
   clearly distinguishing model predictions from general advisory information.
7. End-to-end "diagnose" workflow: predict a disease, then optionally chat
   about that specific result.
