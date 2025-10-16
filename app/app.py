import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Load model and tokenizer
# -----------------------------
@st.cache_resource
def load_model():
    model_dir = "./career-chatbot-flan-t5"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_model()

FALLBACK_RESPONSE = "Sorry, I am not able to answer that. Try asking a career-related question."

# -----------------------------
# Keyword-based career relevance check
# -----------------------------
def is_career_related(question):
    """Simple keyword-based career relevance check."""
    career_keywords = [
        "career", "job", "work", "profession", "role", "skills", "resume",
        "cv", "interview", "salary", "promotion", "employee", "employer",
        "company", "business", "position", "occupation", "recruit", "hire",
        "employment", "training", "qualification", "experience", "industry",
        "manager", "developer", "engineer", "analyst", "designer", "data",
        "marketing", "finance", "consultant", "career path"
    ]
    q_lower = question.lower()
    return any(word in q_lower for word in career_keywords)

# -----------------------------
# Generate chatbot answer
# -----------------------------
def generate_career_answer(question, chat_history=None, max_length=64):
    if chat_history is None:
        chat_history = []

    # ✅ Check if question is career-related first
    if not is_career_related(question):
        return FALLBACK_RESPONSE

    # Build conversation context
    context = ""
    for user_q, bot_a in chat_history:
        context += f"User: {user_q}\nBot: {bot_a}\n"
    context += f"User: {question}\nBot:"

    # Chatbot prompt
    input_text = (
        "You are a helpful career advice chatbot. "
        "Answer career-related questions in a friendly and informative way.\n"
        + context
    )

    # Tokenize input
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=512
    ).to(device)

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="CareerGuide AI", page_icon="🎯", layout="centered")

st.title("🎯 CareerGuide AI")
st.markdown(
    "👋 Welcome! I'm your personal career guide.\n\n"
    "Ask me anything about **career paths, job roles, or skills**, and I'll help you out."
)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for user_q, bot_a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_q)
    with st.chat_message("assistant"):
        st.markdown(bot_a)

# User input
user_input = st.chat_input("Ask your career question...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        answer = generate_career_answer(user_input, st.session_state.chat_history)

    with st.chat_message("assistant"):
        st.markdown(answer)

    # Update chat history
    st.session_state.chat_history.append((user_input, answer))
