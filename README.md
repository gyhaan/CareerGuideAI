# Career Recommendation Chatbot

Link to the notebook: https://colab.research.google.com/drive/1THhTexXj6-VNMQbzQZnAjHp3GhfSRFRS?usp=sharing

## Overview
The **Career Recommendation Chatbot** is a domain-specific conversational AI designed to provide personalized career guidance within the education sector. Built for an NLP course assignment, it assists students and young professionals in exploring career paths by answering questions about skills, qualifications, job prospects, and challenges in fields like technology, healthcare, finance, and more. The chatbot leverages a fine-tuned T5 (Text-to-Text Transfer Transformer) model from Hugging Face, implemented with TensorFlow, and is deployed via a user-friendly Streamlit web interface.

This repository contains all code, data, and documentation for the project, including a Jupyter Notebook for data preprocessing and model training, a Streamlit app for user interaction, and evaluation results.

## Project Structure
```
Career-Recommendation-Chatbot/
├── app/
│   └── app.py                  # Streamlit UI implementation
├── notebook/
│   └── Career_Recommendation_chatbot.ipynb  # Jupyter Notebook for data preprocessing, training, and evaluation
├── model_evaluation_results.csv  # Evaluation metrics for T5-base and Flan-T5-base
├── model_evaluation_results-t5-small.csv  # Evaluation metrics for T5-small
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Dataset
The dataset consists of approximately 1,000 question-answer pairs curated from public sources such as career advice forums, educational websites, and Q&A platforms (e.g., Stack Exchange, Reddit). It covers diverse user intents, including:
- https://huggingface.co/datasets/Pradeep016/career-guidance-qa-dataset
- Career path inquiries (e.g., "What career if I like coding?").
- Skill and qualification requirements (e.g., "What skills for software engineering?").
- Salary expectations and work-life balance.
- Emerging fields like AI and sustainability.

### Preprocessing
- **Cleaning**: Removed HTML tags, special characters, duplicates, and handled missing values.
- **Normalization**: Converted text to lowercase, standardized punctuation, and expanded contractions.
- **Tokenization**: Used T5’s WordPiece tokenizer with a maximum sequence length of 512 tokens, prefixing questions with "question:" and answers with "answer:".
- **Splitting**: Divided into 80% training, 10% validation, and 10% test sets.

The preprocessing steps are detailed in `notebook/Career_Recommendation_chatbot.ipynb`.

## Model and Training
The chatbot uses a generative question-answering approach with three pre-trained Transformer models from Hugging Face:
- T5-small (baseline).
- T5-base.
- Google/Flan-T5-base.

### Fine-Tuning
- **Framework**: TensorFlow with Hugging Face’s `transformers` library.
- **Hyperparameters**:
  - Learning rates: 5e-05, 0.0001.
  - Batch size: 8 (constrained by Google Colab GPU).
  - Epochs: 5.
  - Optimizer: AdamW with weight decay of 0.01.
- **Environment**: Fine-tuning was performed on Google Colab with GPU acceleration.

The T5-base model with a learning rate of 0.0001 achieved the best performance, improving over the T5-small baseline by 68% in BLEU score and 38% in F1 score.

### Evaluation Metrics
Performance was evaluated using BLEU and F1 scores on the test set, with qualitative testing for response relevance:

| Model Name       | BLEU Score | F1 Score | Train Loss | Improvement over Baseline (%) |
|------------------|------------|----------|------------|-------------------------------|
| T5-small        | 0.081      | 0.269    | 1.083      | Baseline                     |
| Google/Flan-T5-base | 0.128  | 0.369    | 1.387      | +58% (BLEU), +37% (F1)       |
| T5-base         | 0.136      | 0.372    | 0.612      | +68% (BLEU), +38% (F1)       |

- **BLEU (0.136)**: Indicates reasonable fluency for a generative task, though limited by varied phrasing.
- **F1 (0.372)**: Reflects decent token-level alignment with reference answers.
- **Qualitative Testing**: 85% of test queries received relevant responses, with out-of-domain queries (e.g., "What’s the weather?") correctly rejected.

See `model_evaluation_results.csv` and `model_evaluation_results-t5-small.csv` for detailed metrics.

## Deployment
The chatbot is deployed using Streamlit (`app/app.py`), providing an intuitive web interface:
- **Features**:
  - Chat-style input box for user queries.
  - Conversation history display using `st.session_state`.
  - Keyword-based filter to ensure responses are career-related, with a fallback for out-of-domain queries ("Sorry, I am not able to answer that. Try asking a career-related question.").
  - Runs the fine-tuned Google/Flan-T5 model with PyTorch, using GPU if available.
- **Response Generation**: Uses beam search (`num_beams=4`) with a maximum output length of 64 tokens for concise, coherent answers.

## Setup Instructions
To run the chatbot locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/uwituzeb/Career-Recommendation-Chatbot.git
   cd Career-Recommendation-Chatbot
   ```

2. **Install Dependencies**:
   Ensure Python 3.8+ is installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   Dependencies include: `transformers`, `torch`, `streamlit`, `datasets`, `evaluate`, `nltk`.

3. **Download Model Checkpoints**:
   - The fine-tuned Flan-T5 model is stored in `./career-chatbot-flan-t5`. Ensure it’s available or re-run the fine-tuning in `notebook/Career_Recommendation_chatbot.ipynb`.
   - Alternatively, download pre-trained models from Hugging Face (e.g., `google/flan-t5-base`).

4. **Run the Streamlit App**:
   ```bash
   streamlit run app/app.py
   ```
   Access the app at `http://localhost:8501` in your browser.

5. **Run the Notebook** (Optional):
   Open `notebook/Career_Recommendation_chatbot.ipynb` in Jupyter or Google Colab to explore data preprocessing, training, and evaluation.

## Example Conversations
Below are sample interactions with the chatbot via the Streamlit interface:

**Query 1**: "What skills do I need for software engineering?"  
**Response**: "You’ll need skills in Python, Java, or C++, plus problem-solving and teamwork. A bachelor’s in computer science is common."

**Query 2**: "Is data science a good career?"  
**Response**: "Data science is a great career with high demand. It requires skills in statistics, Python, and data visualization."

**Query 3**: "What’s the weather today?" (Out-of-domain)  
**Response**: "Sorry, I am not able to answer that. Try asking a career-related question."

These examples demonstrate the chatbot’s ability to provide relevant, concise answers and handle out-of-domain queries appropriately.

## Running the Demo
- **Streamlit App**: Launch `app.py` to interact with the chatbot live.
- **Notebook**: Use `Career_Recommendation_chatbot.ipynb` to replicate data preprocessing, fine-tuning, and evaluation.
- **Demo Video**:[ A 7-minute video showcasing the project is available in the repository, covering code structure, model implementation, and live interactions.
](https://youtu.be/XxbiV3V63Pc)
## Future Improvements
- Expand the dataset with more diverse question-answer pairs.
- Integrate multimodal inputs.
- Explore larger models (e.g., T5-large) or advanced fine-tuning techniques.
- Deploy as a mobile app for broader accessibility.
