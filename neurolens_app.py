import streamlit as st
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
from PIL import Image
import requests
from io import BytesIO
import logging

# Set up logging
logging.basicConfig(filename="neurolens.log", level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# Set up the Streamlit page
st.title("NeuroLens: AI Vision & Language")
st.write("Upload an image or enter a URL, see its description, ask a question, or chat with NeuroLens!")

# Load models with caching
@st.cache_resource
def load_vision_models():
    logger.info("Loading vision models")
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    vqa = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")
    return captioner, vqa

@st.cache_resource
def load_gpt_model():
    logger.info("Loading GPT model")
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    return tokenizer, model

captioner, vqa = load_vision_models()
gpt_tokenizer, gpt_model = load_gpt_model()

# Function to generate dialogue
def generate_dialogue(prompt, max_length=50):
    logger.info(f"Generating dialogue for prompt: {prompt}")
    inputs = gpt_tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = gpt_model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=gpt_tokenizer.eos_token_id,
        temperature=0.7,
        top_k=50,
        do_sample=True
    )
    response = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response or "Hmm, let's try that again!"

# Image input
option = st.radio("Choose input method:", ("Upload Image", "Enter Image URL"))
image = None

if option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        logger.info("Image uploaded")
else:
    url = st.text_input("Enter image URL (e.g., https://example.com/cat.jpg)")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            logger.info(f"Image loaded from URL: {url}")
        except:
            st.error("Invalid URL or image. Please try another.")
            logger.error(f"Failed to load image from URL: {url}")

if image:
    # Display the image
    st.image(image, caption="Selected Image", use_column_width=True)

    # Image Captioning
    st.subheader("Image Caption")
    caption = captioner(image)[0]["generated_text"]
    st.write(f"**NeuroLens says**: {caption}")
    logger.info(f"Caption generated: {caption}")
    caption_prompt = f"Act as a friendly AI commenting on this: The image shows {caption}."
    caption_dialogue = generate_dialogue(caption_prompt, max_length=50)
    st.write(f"**NeuroLens chats**: {caption_dialogue}")

    # Visual Question Answering
    st.subheader("Ask a Question")
    question = st.text_input("Enter your question (e.g., 'Is the cat black?')")
    if question:
        result = vqa(image, question, top_k=1)
        answer = result[0]["answer"]
        confidence = result[0]["score"]
        st.write(f"**NeuroLens answers**: {answer} ({confidence*100:.1f}% sure)")
        logger.info(f"VQA question: {question}, answer: {answer}")
        vqa_prompt = f"As a friendly AI, comment on this: The answer to '{question}' is '{answer}'."
        vqa_dialogue = generate_dialogue(vqa_prompt, max_length=50)
        st.write(f"**NeuroLens chats**: {vqa_dialogue}")

# Chatbot section
st.subheader("Chat with NeuroLens")
user_input = st.text_input("Say something to NeuroLens (e.g., 'Tell me about cats!')")
if user_input:
    chat_prompt = f"As a friendly AI, respond to this: {user_input}"
    chat_response = generate_dialogue(chat_prompt, max_length=50)
    st.write(f"**NeuroLens chats**: {chat_response}")
    logger.info(f"Chat input: {user_input}, response: {chat_response}")
