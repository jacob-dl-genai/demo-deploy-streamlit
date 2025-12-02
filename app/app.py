import streamlit as st
import torch
# Changed to DistilBert to match the new training script
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os

# Configuration
MODEL_PATH = "../tiny_bert_sentiment"

st.set_page_config(page_title="Custom BERT App", page_icon="🧠")

st.title("🧠 My Custom Trained BERT")
st.write("This app loads a DistilBERT model locally from your disk.")

# Check if model exists
if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
    st.error(f"❌ Could not find the model directory: `{MODEL_PATH}`")
    st.info("Please run `python train_model.py` first to generate the model files.")
    st.stop()

@st.cache_resource
def load_local_model():
    try:
        # Load from local directory
        # Using DistilBert classes explicitly
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

with st.spinner("Loading model from disk..."):
    tokenizer, model = load_local_model()

if tokenizer and model:
    st.success("✅ Model loaded successfully from local files!")

    # Input Area
    user_text = st.text_area("Enter text to classify:", "I am very happy with this result")

    if st.button("Classify"):
        if user_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            # Inference
            try:
                inputs = tokenizer(user_text, return_tensors="pt", truncation=True, max_length=64)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                logits = outputs.logits
                predicted_class_id = logits.argmax().item()
                
                # Since we trained 0=Negative, 1=Positive
                labels = ["Negative 🙁", "Positive 🙂"]
                prediction_label = labels[predicted_class_id]
                
                st.metric(label="Sentiment", value=prediction_label)
                
                with st.expander("View Raw Logits"):
                    st.write(logits.detach().numpy())
            except Exception as e:
                st.error(f"Prediction error: {e}")