# Career Recommendation Chatbot

## Overview

This project implements a domain-specific Career-Recommendation chatbot using a pre-trained language model fine-tuned on a career recommendation Q&A dataset. The chatbot is built to answer questions and provide accurate information about different career fields to guide students in choosing a career pathway and making informed choices.

## Dataset Used

[Link to the dataset](https://huggingface.co/datasets/Pradeep016/career-guidance-qa-dataset)

The dataset used provides career guidance information for a variety of career roles. It includes question and answer pairs related to differenct career roles, covering aspects like job responsibilities, skills, career progression, salary expectations and more.

Our dataset contains 1620 rows of question-answer pairs.

### **Columns**

**Role**: Name of career role, the dataset contains 54 career roles

**Question**: The question related to a career role

**Answer**: The answer to the asked question providing relevant information about the career role.

## Data Exploration And Preprocessing

- **Tokenization**: The text data was tokenized using flan-t5-base pre-trained model suitable for conversational chatbots.
- **Normalization**: Text was normalized by removing unnecessary symbols and noise in our data.
- **Dataset Splitting**: Dataset was split into train and test data.
- **Encoding**:The tokenized text was encoded into input IDs and attention masks, which are required by the model for training and inference.

## Model Design And Architecture

The model uses the flan-t5-base transformer architecture  `google/flan-t5-base` from Hugging Face fine-tuned for our career recommendation dataset. 

The model is trained for 10 epochs using AdamW as the optimizer, a learning rate of `5e-5`, weight decay of `0.01` and a batch size of `8`

## Evaluation

The model is evaluated using:
- BLEU score to measure similarity between predicted and reference text.
- F1-score to measure precision and recall between predicted and reference text.

### Results
- A BLEU score of 0.652 indicates that the generated responses have approximately 65.2% overlap with the reference answers in terms of matching n-grams.
- The model achieved an f1-score of 0.734 suggesting a good balance between precision and recall.

## Deployment

Our model and application are deployed separately for better modularity due to the large file sizes:

- **Model Deployment**: The trained model is hosted on [Hugging Face Hub](https://huggingface.co/Bernice-24/career-recommendation-model)
- **Application Deployment**: The chatbot is deployed as a [Hugging Face Space](https://huggingface.co/spaces/Bernice-24/Career-recommendation-chatbot), providing a UI for users to interact with the app by asking questions and getting career recommendations.

## Example Usage

![Screenshot 2025-06-22 213435](https://github.com/user-attachments/assets/7fac352a-aa3e-4d11-a3c3-9c1bf5322e33)

![Screenshot 2025-06-22 213620](https://github.com/user-attachments/assets/1fe500df-8bbb-4fe3-8ba6-0afcf3aa7788)

![Screenshot 2025-06-22 233704](https://github.com/user-attachments/assets/337bd79b-c10f-4e11-a465-fadca03dab2a)

## Project Structure

```
Career-Recommendation-Chatbot/
├── app.py                  # Contains gradio demo app.py
├── notebook             # Contains .ipynb notebook
├── saved-models/           # Fine-tuned model
├── requirements.txt      # installed libraries
├── README.md
```

## Project Setup

1. Clone the repository
   
   ```
   git clone https://github.com/uwituzeb/Career-Recommendation-Chatbot.git
   ```
   
3. Create virtual environment

   ```
   python -m venv venv
   source venv/Scripts/activate
   ```
    
4. Install requirements

   ```
   pip install requirements
   ```
   
5. Run the app

   ```
   python app.py
   ```

## Conclusion

The project shows a practical example usage for domain-specific chatbots. By fine-tuning and experimenting with different models on a career recommendation dataset, we built a Transformer-based chatbot that answers career-related questions. The final model performed significantly well with a loss of 0.438, average BLEU score of 0.090 and F1-score of 0.440. Future improvements will focus on expanding the dataset and increasing GPU capabilities.
