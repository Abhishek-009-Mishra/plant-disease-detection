# 🌿 AI-Powered Plant Disease Detection and Intelligent Advisory System
 
An end-to-end **Computer Vision + Retrieval-Augmented Generation (RAG)** system that detects plant diseases from leaf images and provides AI-powered information about symptoms, causes, management, and prevention.
 
The project uses **MobileNetV2 transfer learning** for image classification, **TF-IDF + Cosine Similarity** for knowledge retrieval, **Groq LLM** for natural-language advisory, and **SQLite** for prediction history.
 
---
 
## 📌 Overview
 
Plant diseases can significantly affect crop health and agricultural productivity. Identifying diseases from leaf symptoms manually can be difficult, particularly when multiple diseases have visually similar symptoms.
 
This project provides an AI-based decision-support system that analyzes plant leaf images and predicts the most probable disease class.
 
After classification, the predicted disease is used to retrieve relevant information from a structured disease knowledge base. A Groq-powered Large Language Model then generates an understandable advisory containing information such as symptoms, causes, management, and prevention.
 
### Complete Pipeline
 
```text
                    Leaf Image
                        │
                        ▼
              Image Validation
                        │
                        ▼
             Image Preprocessing
                        │
                        ▼
               MobileNetV2 Model
                        │
                        ▼
          Plant + Disease + Confidence
                        │
                        ▼
              Disease Knowledge Base
                        │
                        ▼
            TF-IDF + Cosine Similarity
                        │
                        ▼
                 Retrieved Context
                        │
                        ▼
                    Groq LLM
                        │
                        ▼
              AI Disease Advisory
                        │
                        ▼
                SQLite Prediction
                     History
```
 
> **Disclaimer:** The system provides an AI-based classification and informational advisory. It should not be considered a guaranteed agricultural diagnosis or a replacement for professional agricultural advice.
 
---
 
## 🎯 Problem Statement
 
Farmers and plant growers may face difficulties identifying plant diseases quickly from visual symptoms. Different diseases can produce similar patterns on leaves, making manual identification challenging.
 
The goal of this project is to develop a Computer Vision-based system capable of analyzing plant leaf images and identifying the most probable disease class. The system is further extended with a RAG-based advisory component that retrieves disease information and generates an understandable response using a Large Language Model.
 
---
 
## 🎯 Objectives
 
The major objectives of this project are:
 
1. Detect plant diseases from leaf images using Computer Vision.
2. Classify images into 38 plant/disease classes.
3. Perform image preprocessing and augmentation.
4. Use transfer learning with MobileNetV2.
5. Evaluate the model using standard classification metrics.
6. Create a structured plant-disease knowledge base.
7. Implement TF-IDF and cosine-similarity-based retrieval.
8. Integrate retrieved information with a Groq-powered LLM.
9. Provide disease symptoms, causes, management, and prevention information.
10. Store prediction results in SQLite.
11. Provide prediction history.
12. Validate important system components using automated testing.
13. Maintain a modular and organized project structure.
---
 
## ✨ Features
 
### 🌱 Plant Disease Detection
 
The system accepts a plant leaf image and predicts:
 
- Plant type
- Disease/health condition
- Confidence score
- Top 3 predictions
### 🖼️ Image Preprocessing
 
The preprocessing pipeline performs:
 
- Image validation
- Image readability checking
- Minimum-size validation
- Image resizing
- BGR to RGB conversion
- Pixel normalization
- Optional denoising
### 🧠 Deep Learning Classification
 
The project uses MobileNetV2 with ImageNet pretrained weights and a custom classification layer for the 38 PlantVillage classes.
 
### 🔎 RAG-Based Knowledge Retrieval
 
Relevant disease information is retrieved using:
 
- TF-IDF vectorization
- Cosine similarity
### 🤖 AI Advisory
 
The retrieved information is provided to the Groq LLM to generate a natural-language response covering:
 
- Symptoms
- Causes
- Management
- Prevention
### 🗃️ Prediction History
 
Prediction results are stored in a SQLite database for later retrieval.
 
### 🧪 Automated Testing
 
The project contains pytest tests covering:
 
- Configuration
- Database
- Model
- Prediction
- Preprocessing
- RAG retrieval
Final result:
 
```text
16 passed in 5.61s
```
 
---
 
## 📊 Dataset
 
The project uses the PlantVillage dataset obtained through the Hugging Face dataset repository.
 
Dataset identifier:
 
```text
mohanty/PlantVillage
```
 
The downloaded dataset contains three image representations:
 
```text
raw/
├── color/
├── grayscale/
└── segmented/
```
 
The current model uses the **color** representation.
 
### Dataset Statistics
 
| Property                  | Value        |
|----------------------------|-------------|
| Total Images                | 54,305      |
| Classes                     | 38          |
| Plant Types                 | 14          |
| Healthy Classes             | 12          |
| Diseased Classes            | 26          |
| Minimum Images per Class    | 152         |
| Maximum Images per Class    | 5,507       |
| Average Images per Class    | 1,429.1     |
| Class Imbalance Ratio       | 36.23×      |
| Common Image Dimension      | 256 × 256   |
| Sampled Corrupt Images      | 0           |
 
### 🌿 Plant Types
 
The dataset contains the following plant categories:
 
- Apple
- Blueberry
- Cherry
- Corn (Maize)
- Grape
- Orange
- Peach
- Pepper (Bell)
- Potato
- Raspberry
- Soybean
- Squash
- Strawberry
- Tomato
### 📂 Dataset Split
 
The dataset was divided into training, validation, and testing sets.
 
| Split       | Images     | Percentage |
|-------------|------------|------------|
| Training    | 38,013     | 70%        |
| Validation  | 8,146      | 15%        |
| Testing     | 8,146      | 15%        |
| **Total**   | **54,305** | **100%**   |
 
The split configuration is stored in:
 
```text
data/splits/splits_color.json
```
 
No corrupt images were skipped during the final dataset split.
 
---
 
## 🔍 Exploratory Data Analysis
 
Before model training, the dataset was analyzed to understand its structure and quality.
 
The EDA process includes:
 
- Total image count
- Number of classes
- Plant categories
- Healthy vs diseased classes
- Class distribution
- Minimum and maximum class sizes
- Dataset imbalance
- Image dimensions
- Basic image integrity validation
EDA outputs are stored in:
 
```text
reports/eda/
```
 
The analysis identified a significant class imbalance, with an imbalance ratio of approximately **36.23×** between the largest and smallest classes.
 
---
 
## 🧠 Model
 
### MobileNetV2
 
The project uses MobileNetV2 transfer learning for plant disease classification.
 
MobileNetV2 was selected as a relatively lightweight CNN architecture suitable for image classification while providing strong feature extraction capabilities through ImageNet pretrained weights.
 
### Model Architecture
 
```text
Input Image
    │
    ▼
224 × 224 × 3
    │
    ▼
Rescaling
    │
    ▼
MobileNetV2
    │
    ▼
Global Average Pooling
    │
    ▼
Dropout (0.30)
    │
    ▼
Dense Layer
    │
    ▼
Softmax
    │
    ▼
38 Classes
```
 
### Model Configuration
 
| Parameter          | Value                            |
|---------------------|-----------------------------------|
| Model                | MobileNetV2                       |
| Input Size           | 224 × 224 × 3                     |
| Number of Classes    | 38                                 |
| Batch Size           | 32                                 |
| Epochs               | 20                                  |
| Learning Rate        | 0.0001                             |
| Optimizer            | Adam                                |
| Loss Function        | Sparse Categorical Crossentropy    |
| Dropout              | 0.30                                |
| Random Seed          | 42                                   |
 
### Model Parameters
 
```text
Total Parameters       : 2,306,662
Trainable Parameters   : 48,678
Non-Trainable          : 2,257,984
```
 
---
 
## 📈 Training Results
 
The model was trained for 20 epochs.
 
Early stopping was used to restore the best-performing model weights.
 
**Best Validation Accuracy: 93.36%**
 
The best validation performance occurred at epoch 18.
 
The trained model is saved as:
 
```text
models/plant_disease_model.keras
```
 
---
 
## 🧪 Model Evaluation
 
The final model was evaluated on the independent test dataset containing **8,146 images**.
 
### Overall Performance
 
| Metric               | Result |
|-----------------------|--------|
| Test Accuracy          | 93.47% |
| Macro Precision        | 0.91   |
| Macro Recall           | 0.93   |
| Macro F1-Score         | 0.92   |
| Weighted Precision     | 0.94   |
| Weighted Recall        | 0.93   |
| Weighted F1-Score      | 0.93   |
 
### Class-Level Performance
 
Some classes achieved strong performance:
 
| Class                  | F1-Score |
|--------------------------|----------|
| Apple Healthy             | 0.97     |
| Blueberry Healthy         | 0.99     |
| Orange Huanglongbing      | 0.99     |
| Soybean Healthy           | 0.98     |
 
Some challenging classes included:
 
| Class                        | F1-Score |
|--------------------------------|----------|
| Tomato Early Blight             | 0.68     |
| Potato Healthy                  | 0.72     |
| Tomato Target Spot              | 0.75     |
| Corn Cercospora Leaf Spot       | 0.79     |
 
These results indicate that visually similar disease categories remain more challenging for the classifier.
 
Evaluation outputs are stored in:
 
```text
reports/evaluation/
```
 
---
 
## 🔎 RAG System
 
The project extends the Computer Vision classifier with a lightweight Retrieval-Augmented Generation system.
 
The disease knowledge base contains information for all 38 classes.
 
Location:
 
```text
knowledge/plant_diseases.json
```
 
Each knowledge entry contains:
 
- Plant
- Disease
- Symptoms
- Causes
- Management
- Prevention
### RAG Architecture
 
```text
User Question
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Cosine Similarity
      │
      ▼
Relevant Disease Information
      │
      ▼
Retrieved Context
      │
      ▼
Groq LLM
      │
      ▼
Natural Language Advisory
```
 
The current implementation uses in-memory TF-IDF retrieval rather than a persistent vector database.
 
---
 
## 🤖 Groq AI Advisory
 
The chatbot uses the Groq API to generate natural-language responses based on retrieved disease information.
 
The system can answer questions related to:
 
- Disease symptoms
- Possible causes
- Disease management
- Prevention
- General information about the predicted condition
The predicted class can be supplied as additional context so that the advisory is focused on the detected disease.
 
The chatbot is designed to use retrieved knowledge rather than relying only on unrestricted model generation.
 
---
 
## 🗃️ SQLite Prediction History
 
The project stores prediction results using SQLite.
 
Database:
 
```text
data/predictions.db
```
 
The prediction table stores:
 
- `id`
- `image_path`
- `timestamp`
- `plant`
- `disease`
- `confidence`
- `top_predictions`
The database functionality is implemented in:
 
```text
src/database.py
```
 
This allows the system to maintain a history of analyzed images and their predictions.
 
---
 
## 🔄 End-to-End Diagnosis
 
The project provides a complete diagnosis workflow through:
 
```bash
python main.py diagnose "path\to\leaf.jpg"
```
 
The workflow performs:
 
1. Load image
2. Validate image
3. Preprocess image
4. Load trained model
5. Predict disease
6. Generate top-3 predictions
7. Save prediction to SQLite
8. Retrieve disease information
9. Generate Groq advisory
10. Display final result
---
 
## 🖥️ Command Line Interface
 
The application provides the following commands:
 
- `analyze`
- `train`
- `evaluate`
- `predict`
- `history`
- `build-rag`
- `chat`
- `diagnose`
### Analyze Dataset
 
```bash
python main.py analyze
```
 
Analyzes the dataset and generates EDA information.
 
### Train Model
 
```bash
python main.py train
```
 
Trains the MobileNetV2 classifier.
 
### Evaluate Model
 
```bash
python main.py evaluate
```
 
Evaluates the trained model using the test dataset.
 
### Predict Disease
 
```bash
python main.py predict "path\to\leaf.jpg"
```
 
Generates the top-3 disease predictions.
 
### Build RAG Knowledge Base
 
```bash
python main.py build-rag
```
 
Loads and prepares the disease knowledge base for retrieval.
 
### Start Chatbot
 
```bash
python main.py chat
```
 
Starts the RAG-based AI chatbot.
 
### Complete Diagnosis
 
```bash
python main.py diagnose "path\to\leaf.jpg"
```
 
Runs the complete **Prediction → History → RAG → AI Advisory** workflow.
 
### View Prediction History
 
```bash
python main.py history
```
 
Displays previously stored predictions.
 
---
 
## 🏗️ Project Structure
 
```text
plant-disease-detection/
│
├── raw/
│   ├── color/
│   ├── grayscale/
│   └── segmented/
│
├── knowledge/
│   └── plant_diseases.json
│
├── models/
│   ├── plant_disease_model.keras
│   └── class_names.json
│
├── vector_db/
│
├── data/
│   └── splits/
│       └── splits_color.json
│
├── reports/
│   ├── eda/
│   └── evaluation/
│
├── notebooks/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset_loader.py
│   ├── dataset_analysis.py
│   ├── preprocessing.py
│   ├── augmentation.py
│   ├── data_pipeline.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predictor.py
│   ├── database.py
│   ├── rag_loader.py
│   ├── rag_retriever.py
│   ├── chatbot.py
│   ├── utils.py
│   └── diagnose.py
│
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_database.py
│   ├── test_model.py
│   ├── test_predictor.py
│   ├── test_preprocessing.py
│   └── test_rag.py
│
├── main.py
├── requirements.txt
├── README.md
├── statement.md
├── .env.example
├── .gitignore
└── LICENSE
```
 
---
 
## 🧩 System Modules
 
### Dataset Module
 
Responsible for dataset loading, analysis, splitting, and pipeline creation.
 
- `dataset_loader.py`
- `dataset_analysis.py`
- `data_pipeline.py`
### Computer Vision Module
 
Responsible for preprocessing, augmentation, training, evaluation, and prediction.
 
- `preprocessing.py`
- `augmentation.py`
- `model.py`
- `train.py`
- `evaluate.py`
- `predictor.py`
### RAG Module
 
Responsible for loading the disease knowledge base and retrieving relevant information.
 
- `rag_loader.py`
- `rag_retriever.py`
### Chatbot Module
 
Responsible for connecting retrieved knowledge with the Groq LLM.
 
- `chatbot.py`
### Database Module
 
Responsible for storing and retrieving prediction history.
 
- `database.py`
### Diagnosis Module
 
Combines prediction, database storage, RAG retrieval, and AI advisory.
 
- `diagnose.py`
---
 
## 🛠️ Technologies Used
 
| Category              | Technology                  |
|------------------------|------------------------------|
| Programming Language    | Python                      |
| Computer Vision         | OpenCV                      |
| Deep Learning           | TensorFlow / Keras          |
| CNN Architecture        | MobileNetV2                 |
| Data Processing         | NumPy / Pandas              |
| Machine Learning        | Scikit-learn                |
| Retrieval               | TF-IDF + Cosine Similarity  |
| LLM                     | Groq API                    |
| Database                | SQLite                      |
| Testing                 | Pytest                      |
| Version Control         | Git / GitHub                |
| Dataset                 | PlantVillage                |
 
---
 
## ⚙️ Installation
 
### 1. Clone the Repository
 
```bash
git clone https://github.com/Abhishek-009-Mishra/plant-disease-detection.git
```
 
Move into the project directory:
 
```bash
cd plant-disease-detection
```
 
### 2. Create a Virtual Environment
 
For Windows:
 
```bash
python -m venv venv
```
 
Activate it:
 
```bash
venv\Scripts\activate
```
 
### 3. Install Dependencies
 
```bash
pip install -r requirements.txt
```
 
### 4. Configure Groq API
 
Create a `.env` file in the project root:
 
```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.3
GROQ_MAX_TOKENS=600
```
 
The `.env` file should never be committed to GitHub.
 
---
 
## 🧪 Testing
 
The project uses pytest for automated testing.
 
Run:
 
```bash
python -m pytest tests -v
```
 
### Final Test Result
 
```text
16 passed in 5.61s
```
 
All automated tests passed successfully.
 
### Tested Components
 
- Configuration
- Dataset split configuration
- Training configuration
- SQLite database creation
- Prediction storage
- Prediction history retrieval
- Model architecture
- Disease prediction
- Invalid image handling
- Image preprocessing
- Image validation
- Knowledge-base loading
- RAG retrieval
---
 
## 🔐 Security
 
The project uses environment variables for sensitive API credentials.
 
The following should not be committed:
 
- `.env`
- `venv/`
- `__pycache__/`
- `*.pyc`
The API key should always be kept private.
 
---
 
## ⚠️ Limitations
 
The current implementation has several limitations:
 
- The model is trained using the PlantVillage dataset and may not perform equally well on real-world field images.
- Complex backgrounds, poor lighting, blur, and unusual image angles may affect prediction performance.
- Some visually similar disease classes have lower F1-scores.
- The current RAG implementation uses TF-IDF and cosine similarity rather than a persistent embedding vector database.
- The chatbot requires a valid Groq API key.
- The current interface is command-line based.
- Model predictions should not be treated as guaranteed agricultural diagnoses.
---
 
## 🚀 Future Enhancements
 
The following improvements can be added in future versions:
 
- Web-based user interface
- Mobile application
- Persistent vector database
- Semantic embedding models
- Model fine-tuning
- Grad-CAM explainability
- Leaf segmentation
- Real-world field datasets
- Multilingual advisory
- Disease severity estimation
- Weather-based recommendations
- Cloud deployment
- User-specific prediction history
- Confidence-based uncertainty detection
---
 
## 📚 References
 
- PlantVillage Dataset — Hugging Face Dataset Repository: [`mohanty/PlantVillage`](https://huggingface.co/datasets/mohanty/PlantVillage)
- TensorFlow / Keras documentation — MobileNetV2
- OpenCV documentation — Image Processing
- Scikit-learn documentation — TF-IDF and Cosine Similarity
- Groq API documentation
- Python SQLite documentation
- Pytest documentation
---
 
## 👨‍💻 Author
 
**Abhishek Mishra**
 
B.Tech Computer Science and Engineering — Artificial Intelligence & Machine Learning
 
Vellore Institute of Technology, Bhopal
 
GitHub: [Abhishek-009-Mishra](https://github.com/Abhishek-009-Mishra)
 
---
 
## 📜 Disclaimer
 
This project has been developed for educational and academic purposes as a Computer Vision project.
 
The disease predictions and AI-generated recommendations are intended to provide informational assistance. They should not be considered a substitute for professional agricultural diagnosis, laboratory testing, or expert agricultural advice.
 