import torch
import gradio as gr

# Use a pipeline as a high-level helper
from transformers import pipeline

#pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", torch_dtype=torch.bfloat16)

model_path = "C:/Users/ankitdwivedi/OneDrive - Adobe/Desktop/NLP Projects/Video to Text Summarization/Model/models--sshleifer--distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff"

text_summary = pipeline("summarization", model=model_path, torch_dtype=torch.bfloat16)

# text = """In 2008, Modi published a Gujarati book titled Jyotipunj, which contains profiles of RSS leaders. The longest was of M. S. Golwalkar, under whose leadership the RSS expanded and whom Modi refers to as Pujniya Shri Guruji (Guru worthy of worship).[519] According to The Economic Times, Modi's intention was to explain the workings of the RSS to his readers, and to reassure RSS members he remained ideologically aligned with them."""

# print(text_summary(text));


def summary(input):
    output = text_summary(input)
    return output[0]['summary_text']

gr.close_all()

demo = gr.Interface(fn=summary, inputs=[gr.Textbox(label="Input text to summarize",lines = 6)],outputs=[gr.Textbox(label="Summarized Text",lines = 4)],title="Project 1: Text summary",
                    description="""This is a simple text summarization model.""")
demo.launch()