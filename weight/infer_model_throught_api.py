import json, os 
import uvicorn
from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware
from load_constant_base_model import (
    qa_generation, 
    sentence_embedd, 
    passage_rank,
    CREATIVE_PROMPT,
    SHORT_PROMPT, 
    PRECISE_PROMPT,
    INTRODUCE_PROMPT
)
from pyvi import ViTokenizer 

# import package 
import torch 
from rag_chatbot import (
    SentenceEmbedding, 
    GenAnsModelCasualLM, 
    Ranker,
)
from rag_chatbot.utils.preprocess_text import TextFormat 


os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

with open('config.json', 'r') as f: 
    config= json.load(f)

gen_cfg= config['gen_answer']
ranker_cfg= config['ranker']
embedd_cfg= config['sentence_embedd']

print('==== Load Config ====')
print(config)


# Load Model 

# generation answer 
# init
gen_answer= GenAnsModelCasualLM(
    model_name= gen_cfg['model_name'], 
    torch_dtype= torch.float16 if gen_cfg['fp16'] else torch.float32, 
    device= torch.device(gen_cfg['device']), 
    lora_r= gen_cfg['lora_r'], 
    lora_alpha= gen_cfg['lora_alpha'],
    lora_dropout= gen_cfg['lora_dropout'], 
    target_modules= gen_cfg['target_modules'], 
    # quantization_config= BitsAndBytesConfig(load_in_8bit= True, 
    #                                         llm_int8_skip_modules= ['lm_head'])
    )

### prepare inference 
gen_answer.prepare_inference(
    ckpt_dir= gen_cfg['ckpt'], 
    merge_lora= True, 
    torch_compile= False
)

### auto load cpu gpu to save memory 
if config['api']['auto_cpu_cuda']:
    gen_answer.model.cpu() 


## reranker between 
## init 
ranker= Ranker(
    model_name= embedd_cfg['model_name'], 
    type_backbone= embedd_cfg['type_backbone'], 
    using_hidden_states= embedd_cfg['using_hidden_states'], 
    num_label= embedd_cfg['num_label'],
    hidden_dim= embedd_cfg['hidden_dim'], 
    torch_dtype= torch.float16 if ranker_cfg['fp16'] else torch.float32, 
    device= torch.device(ranker_cfg['device']) , 
    )

### load ckpt 
ranker.load_ckpt(path= ranker_cfg['ckpt'])

### auto load cpu gpu to save memory 
if config['api']['auto_cpu_cuda']:
    ranker.model.model.cpu() 


##  sentence embedding 
### init 
sentence_embedding= SentenceEmbedding(
    model_name= embedd_cfg['model_name'], 
    type_backbone= embedd_cfg['type_backbone'], 
    using_hidden_states= embedd_cfg['using_hidden_states'], 
    concat_embeddings= embedd_cfg['concat_embeddings'],
    num_label= embedd_cfg['num_label'],
    hidden_dim= embedd_cfg['hidden_dim'], 
    torch_dtype= torch.float16 if embedd_cfg['fp16'] else torch.float32, 
    device= torch.device(embedd_cfg['device']), 
    )
### load ckpt 
sentence_embedding.load_ckpt(path= embedd_cfg['ckpt'])



# API
app = FastAPI(
    title="Question Answering Modules API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins= ['*'],
    allow_credentials= True, 
    allow_methods= ['*'],
    allow_headers= ['*'],
)

@app.post('/sentence_embedding', tags=['Sentence Embedding -- Input: Text -> Output: Vector X'])
async def Sentence_embedding(item: sentence_embedd):   
    text= list(map(lambda x: ViTokenizer.tokenize(TextFormat.preprocess_text(x.replace('\n', ' '))).lower(), item.text))
    embedding= sentence_embedding.encode(text)
    # embedding_normarlize= torch.nn.functional.normalize(embedding, p= 2, dim= -1)
    return json.dumps(embedding.tolist()) 


@app.post('/passage_ranking', tags=['Passage Ranking -- Input: Pair (Text, Passage) -> Output: Score'])
async def Passage_ranking(item: passage_rank): 
    if item.auto_cpu_cuda:  ### load in cuda 
        ranker.model.to(ranker_cfg['device'])

    question= list(map(lambda x: ViTokenizer.tokenize(TextFormat.preprocess_text(x.replace('\n', ' '))).lower(),item.question))
    passages= list(map(lambda x: ViTokenizer.tokenize(TextFormat.preprocess_text(x.replace('\n', ' '))).lower(),item.passages))

    pairs_text= [list(i) for i in zip(question, passages)]
    
    result= json.dumps(ranker.predict(text= pairs_text).view(-1, ).tolist())

    if item.auto_cpu_cuda: ### transfer to ram and delete cache model cuda 
        ranker.model.cpu()
        torch.cuda.empty_cache()

    return result

@app.post('/qa_generation', tags= ['Question Answering Generation -- Input: Pair(Passage, Question) -> Output: Answer'])
async def QA_generation(item: qa_generation): 
    if item.auto_cpu_cuda == True:  ### load in cuda 
        gen_answer.model.to(gen_cfg['device'])
         
    question= TextFormat.preprocess_text(item.question)  
    passage= TextFormat.preprocess_text(item.passage)

    assert item.type_prompt in ['precise', 'creative', 'short', 'introduce'] # Must one of that 

    config_gen= {'max_new_tokens': item.new_tokens,
            'top_k': item.top_k, 
            'top_p': item.top_p, 
            'temperature': item.temperature,
            'do_sample': item.do_sample
    }
    if item.type_prompt == 'precise': 
        prompt = PRECISE_PROMPT
    elif item.type_prompt == 'short': 
        prompt = SHORT_PROMPT
    elif item.type_prompt == 'creative':
        prompt = CREATIVE_PROMPT 
    elif item.type_prompt == 'introduce':
        prompt = INTRODUCE_PROMPT

    text= prompt.format(context= passage, question= question)

    # torch.cuda.empty_cache()

    result= gen_answer.gen([text], config_gen= config_gen).split('\n')[-1]

    if item.auto_cpu_cuda: ### transfer to ram and delete cache model cuda 
        gen_answer.model.cpu()
        torch.cuda.empty_cache()

    return result






