import sys 
sys.path.append('../')
import numpy as np 
import gradio as gr
from call_api_process import call_sentence_embedding, call_ranking_passage
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(text1, text2):
    # result= call_sentence_embedding(text= [text1, text2])
    # result= np.asarray(result)
    # return cosine_similarity(result[0].reshape(1, -1), result[1].reshape(1, -1))[0]

    return call_ranking_passage(question= [text1], passages= [text2])
# Tạo Gradio Interface
text1, text2= "Tôi muốn ăn cơm", "Tôi không muốn ăn cơm"

iface = gr.Interface(
    fn=calculate_similarity,
    inputs=[gr.Textbox(label="Sentence A", value= text1), gr.Textbox(label="Sentence B", value= text2)],
    outputs=gr.Textbox(label="Similarity Score"),
    title="Text Similarity Calculator",
    description="Calculate similarity between two text inputs using Cosine Similarity"
)

# Khởi chạy Gradio Interface
iface.launch()
