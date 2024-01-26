import uvicorn
import json 
import os 

with open('config.json', 'r') as f: 
    config= json.load(f)

api_cfg= config['api']

if __name__ == "__main__":
    uvicorn.run("infer_model_throught_api:app", host= api_cfg['host'], 
                port= api_cfg['port'], reload= api_cfg['reload'])