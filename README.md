NeuroLens: AI Vision & Language
NeuroLens is an AI that describes images and answers questions about them using Hugging Face Transformers. Built in Python, it runs on Google Colab.

Features:

Image Captioning: Describes images (e.g., “A cat on a couch”).
Visual Question Answering (VQA): Answers questions (e.g., “Is the cat black?” → “Yes”).

How to Run

Open Google Colab.
Install dependencies:
!pip install transformers pillow requests


Copy neurolens.py into a Colab cell.
Run and follow prompts: upload an image or enter a URL, then ask a question.

Example

Image: Cat (URL: https://images.unsplash.com/photo-1561948955-570b270e7c36).
Caption: “A cat sleeping.”
Question: “Is the cat black?”
Answer: “Yes (92% sure).”

Next Steps

Add a chatbot and reinforcement learning.
Deploy as a web app.

Built by Hansha
