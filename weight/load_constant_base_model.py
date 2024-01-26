from typing import List 
from pydantic import BaseModel 


### qa extractive 
class qa_generation(BaseModel):
    question: str 
    passage: str  
    top_k: int = 100
    top_p: float= 0.9
    temperature: float= 0.56
    do_sample: bool= True
    new_tokens: int= 1024
    type_prompt: str= 'precise',  ### short, creative, precise
    auto_cpu_cuda: bool= True



### sentence embedding 
class sentence_embedd(BaseModel):
    text: List[str]


### passage ranking 
class passage_rank(BaseModel): 
    question: List[str] 
    passages: List[str] 
    auto_cpu_cuda: bool = True


## defind prompt and config gen 

## Defaut config
# config_gen= {'max_new_tokens': 500, 
#             'top_k': 100, 
#             'top_p': 0.9, 
#             'temperature': 0.56,
#              'do_sample': True
# }

CREATIVE_PROMPT= "### Instruction: {question}\n\n{context}\n### Answer: "

SHORT_PROMPT= "Q: {question}\nCtx: {context}\nA: " ### for shorten context top-k 30 top-p 0.9 temperature 0.5 

PRECISE_PROMPT= "### Instruction: You are an AI assistant. Provide a detailed answer so user don’t need to search outside to understand the answer.\
\n### Task:\nContext: {context}\nGiven the paragraph above, Please answer \
the following question: {question}\nAnswer: "

INTRODUCE_PROMPT= "### Instruct: You are an AI assistant named UTT Assistant, \
and this is your information: {context}\nPlease answer the user's question about yourself. User: {question}.\nUTT Assistant response: "
### config gen temperature= 0.56, top-k 70, top-p 0.9 
