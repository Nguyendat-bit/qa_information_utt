from typing import List
import requests 
import json 
import torch 
import os
# os.path.dirname(__file__)

with open( os.path.join(os.path.dirname(__file__), 'config.json'), 'r') as f: 
    config= json.load(f)
api_cfg= config['api']

URL= f"http://{api_cfg['host']}:{api_cfg['port']}/"


headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}


def request_post(url, data):
    try: 
        response= requests.post(url, headers= headers, data= json.dumps(data))
        if response.status_code == 200: 
            return response.json() 
        else:
           raise f"Error: {response.status_code}"
    except requests.RequestException as e:
    # Xử lý khi có lỗi trong quá trình gửi yêu cầu
        raise f"Request Error: {e}"

def call_sentence_embedding(text: List[str]): # return list  
    data= {
        'text': text 
    }
    return json.loads(request_post(url= URL + 'sentence_embedding', data= data))



def call_ranking_passage(question: List[str], passages: List[str]):
    data= { 
        'question': question, 
        'passages': passages,
        'auto_cpu_cuda': api_cfg['auto_cpu_cuda']
    }
    return json.loads(request_post(url= URL + 'passage_ranking', data= data))


def call_generate_answer(question: str, passage: str, new_tokens: int= 1024, 
                         top_k: int= 100, top_p: float = 0.9, 
                         temperature: float= 0.56, do_sample: bool= True, type_prompt= "precise"):
    data= {
        'question': question, 
        'passage': passage,
        'new_tokens':  new_tokens, 
        'top_k': top_k, 
        'top_p': top_p,
        'do_sample': do_sample, 
        'temperature': temperature, 
        'type_prompt': type_prompt,
        'auto_cpu_cuda': api_cfg['auto_cpu_cuda']
    }
    return request_post(url= URL + 'qa_generation', data= data)


if __name__ == "__main__": 

    # print(call_ranking_passage(
    #     question= ['Bạn là ai', 'Bạn là ai', 'Bạn là ai'],
    #     passages= ['Xin chào, bạn là ai ạ', 'bạn là ai vậy', 'tôi ko biết bạn']
    # ))

    print(
        call_ranking_passage(
            question=["tầm nhìn của trường là gì", "tầm nhìn của trường là gì"],
            passages=["""+ Sứ mạng:
“Trường Đại học Công nghệ Giao thông vận tải có sứ mạng đào tạo và cung cấp nguồn nhân lực chất lượng cao theo hướng ứng dụng, đa ngành, đa lĩnh vực, nghiên cứu khoa học và chuyển giao công nghệ phục vụ sự nghiệp phát triển của ngành Giao thông vận tải và của đất nước, phù hợp với xu thế phát triển quốc tế, hội nhập với nền giáo dục đại học tiên tiến của khu vực và trên thế giới”.""",
"""+Tầm nhìn:
Đến năm 2030, có một số ngành đào tạo ngang tầm với các trường đại học có uy tín trong khu vực và trên thế giới; là trung tâm nghiên cứu khoa học ứng dụng, chuyển giao công nghệ mới và hợp tác quốc tế trong lĩnh vực Giao thông vận tải.
Đến năm 2045, chào mừng kỷ niệm 100 năm thành lập Trường, trở thành trường đại học thông minh, trung tâm nghiên cứu khoa học, chuyển giao công nghệ và hợp tác quốc tế trong lĩnh vực Giao thông vận tải"""]        )
    )