import sys
sys.path.append('../')
import numpy as np 
import pandas as pd 
import torch 
import chromadb
import gradio as gr
from chromadb import Documents, EmbeddingFunction, Embeddings 
from pyvi import ViTokenizer

import rag_chatbot
from rag_chatbot.utils.preprocess_text import TextFormat 
from rag_chatbot.utils.responses import ResponsewithRule

from call_api_process import call_generate_answer, call_ranking_passage, call_sentence_embedding 
from rank_bm25 import BM25Plus

print(f'Gradion Version: {gr.__version__}')

### Chromadb 
class EmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        # print(input)
        return call_sentence_embedding(text= input)
    
chroma_client = chromadb.PersistentClient(path= 'D:\\qa_information_utt\\database') 


def get_utt_db(name= 'infor_utt'):
    return chroma_client.get_collection(name= name, embedding_function= EmbeddingFunction())

def get_chitchat_db(name= 'chitchat'): 
    return chroma_client.get_collection(name= name, embedding_function= EmbeddingFunction())

utt_db= get_utt_db()
chitchat_db= get_chitchat_db( )
### 
### Load db infor teacher 
teacher_df= pd.read_json('D:\\qa_information_utt\\database\\thongtin_giangvien_re.json')

###
### Generate answer
INTRODUCE_CTX= "Xin chào, tôi tên là UTT Assistant, một trí tuệ nhân tạo được phát triển nhằm giải đáp thông tin cho sinh viên ĐH CN GTVT.  \
Tôi được tạo ra bằng công nghệ học máy sâu trên nền mô hình ngôn ngữ đã được huấn luyện hằng trăm triệu điểm dữ liệu trước đó rồi được tinh chỉnh để có thể trả lời câu hỏi như hiện tại. \
Phiên bản đầu tiên của tôi cũng chính là ngày tôi mới sinh ra là vào ngày 10/10/2023. \
Tôi không có cảm xúc hay giới tính gì nhưng khả năng của tôi là luôn hỗ trợ giải đáp mọi thắc mắc cho các bạn sinh viên. \
Tôi hy vọng rằng có thể giúp bạn với các câu hỏi yêu cầu của bạn"

def gen_introduce(query): ## special case: qa instroduce
    return call_generate_answer(query, passage= INTRODUCE_CTX, top_k= 40, top_p= 0.75,
            temperature= 0.3, new_tokens= 200, type_prompt= 'introduce').replace("UTT Assistant response: ", '')

def gen_short(query, context): ## case: short context qa
    return call_generate_answer(query, context, new_tokens= 150, top_k= 40, top_p= 0.75,
            temperature= 0.3, type_prompt= 'short').replace("A: ", '')

def gen_normq(query, context): ## case: normal context qa 
    return call_generate_answer(query, context, new_tokens= 500, temperature= 0.7, top_k= 40, top_p= 0.75,
                type_prompt= 'precise').replace("Answer: ", '')
###
### retrieval and ranker
def rank_bm25(query, docs):
    query= ViTokenizer.tokenize(TextFormat.preprocess_text(query.replace('\n', ' '))).lower()
    docs = list(map(lambda x: ViTokenizer.tokenize(TextFormat.preprocess_text(x.replace('\n', ' '))).lower(), docs))
    tokenized_docs = [doc.split() for doc in docs]
    bm25 = BM25Plus(tokenized_docs)

    # Tính điểm BM25 cho một câu truy vấn
    tokenized_query = query.split()
    return bm25.get_scores(tokenized_query)

def retrieval(question, top_k= 20, threshold_sim = 0.47, threshold_rank= 0.3, threshold_bm25= 0.4, num_reserve= 5, debug= True): 
    result= utt_db.query(
        query_texts= [ViTokenizer.tokenize(TextFormat.preprocess_text(question.replace('\n', ' '))).lower()],
        n_results= top_k, 
    )

    idx_condition_search= np.where(np.array(result['distances'][0]) < threshold_sim)[0]
    doc_search= np.array(result['documents'][0])[idx_condition_search].tolist()
    
    if debug:
        print(f'Sim_score: {result["distances"][0]}')

    if threshold_rank and len(doc_search) == 0: 
        doc_search= result['documents'][0][: num_reserve]

    cross_encoder_score= call_ranking_passage(question= [question] * len(doc_search), passages= doc_search)
    if threshold_rank: 
        cross_encoder_score = np.array(cross_encoder_score)
        cross_encoder_score[cross_encoder_score < threshold_rank] = 0.

    if debug: 
        print(f'Cross_encoder_score: {cross_encoder_score}')

    score_bm25= rank_bm25(question, doc_search) / len(doc_search)
    score_bm25= torch.sigmoid(torch.tensor(score_bm25)).numpy()
    # element-wise 
    cross_bm25_score = (threshold_bm25 * score_bm25)  +  (np.array(cross_encoder_score) * (1 - threshold_bm25))

    if threshold_rank: 
        cross_bm25_score[cross_bm25_score < threshold_rank]= 0. 
        
        if np.sum(cross_bm25_score) == 0.: 
            return '', doc_search, cross_encoder_score, cross_bm25_score # Nothing 

    if debug: 
        print(f'Cross_bm25_score: {cross_bm25_score}')
        print(f'\nRetrieval: {doc_search[cross_bm25_score.argmax()]}\nScore: {cross_bm25_score.max()}')

    return doc_search[cross_bm25_score.argmax()], doc_search, cross_encoder_score, cross_bm25_score
###
### Chitchat module and cls teacher-infor question 
def cls_chitchat(query, threshold= 0.6, top_k= 10): 
    # if > threshold is chitchat
    search_result= chitchat_db.query(query_texts= [query],
                         n_results= top_k)
    result= 1- np.mean(search_result['distances'][0])

    if result > threshold: 
        return True 
    return False 

def cls_info_teacher(text): 
    # cls with keyword 
    ### defind keyword 
    keywords= ['thầy', 'cô']
    splt_txt= text.lower().split(' ')
    for word in splt_txt: 
        if word in keywords:
            return True
    return False
###
### Other

respons_rule= ResponsewithRule(rag_chatbot.RESPONSE_RULE)
def clear_output(text): 
    return text.replace('ối,', '').replace('úp mở', '').replace('ối.','')
### Chat 
def chat(message, threshold_sim: float = 0.5, threshold_rank: float = 0.3,
        threshold_bm25: float = 0.3, top_k= 20, debug= True):

    if cls_chitchat(message): 
        if debug: 
            print('Your message is Chitchat domain')
        return gen_introduce(message)
    else: 
        # This domain in this step is utt 
        # check qa infor teacher 
        if cls_info_teacher(message): 
            if debug:
                print('Your message is Teacher-infor case')

            result_bm25= rank_bm25(message, teacher_df.iloc[:, 0].values).argmax()

            if debug:
                print(f'Retrieval teacher information: {teacher_df.iloc[result_bm25, 0]}\nScore: {result_bm25}')
            # return gen_short(message, context= teacher_df.iloc[result_bm25, 0])
            return f'Thông tin của giảng viên bạn cần tìm là: {teacher_df.iloc[result_bm25, 0]}'
        # case normal 
        if debug: 
            print('Your message is QA-UTT domain')
        search_result, _, _, _= retrieval(message, top_k= top_k, threshold_bm25= threshold_bm25, 
                                        threshold_rank= threshold_rank, threshold_sim= threshold_sim, debug= debug)
        if search_result == '': 
            return respons_rule.reply_nonanswer()
        
        if len(ViTokenizer.tokenize(TextFormat.preprocess_text(search_result.replace('\n', ' '))).split()) < 15: 
            return clear_output(gen_short(message, context= search_result))
        
        return clear_output(gen_normq(message, context= search_result))
###


def respond(message, history):


    bot_message= chat(message)

    # chat_history.append((message, bot_message))
    # time.sleep(2)
    return bot_message

mychatbot= gr.Chatbot(
    value=[[None, respons_rule.reply_begin_conversation()]],
    avatar_images=["assert/user.png", "assert/utt_logo.png"],
    bubble_full_width= False, 
    show_label= False, 
    show_copy_button= True, 
    # render_markdown= True, 
    # line_breaks= True,
    height= 550, 
    likeable=True, 
)

demo = gr.ChatInterface(fn= respond, chatbot= mychatbot, title= "UTT Assistant", examples=
                        ['trường công nghệ giao thông có mấy cơ sở', 'trường công nghệ giao thông vận tải có bao nhiêu giảng viên',
                         "cho em hỏi trường công nghệ giao thông có địa chỉ ở đâu vậy ạ", "trường cao đẳng Công Chính có từ năm bao nhiêu", 
                         "bạn cho mình hỏi sứ mệnh và chiến lược đào tạo của trường là gì ạ", "trường công nghệ giao thông vận tải được thành lập vào ngày nào", 
                         "cho mình xin thông tin thầy Lương Hoàng Anh", "cho mình xin thông tin cô Bùi Thị Như ạ", "cho em xin thông tin của thầy Lê Trung Kiên ạ",
                         "cho em xin thông tin của thầy Nguyễn Văn Thắng ạ", "nội dung đánh giá điểm rèn luyện gồm những gì", "sinh viên không được phép làm những điều gì", 
                         "công tác sinh viên là gì"])

demo.launch(show_api= False)
