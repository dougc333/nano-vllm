1) HF tranformers contains the pytorch code for the model, it is here: /Users/dc/nano-vllm/.venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py
It has been copied to test_progs. 
2) the weights are loaded under .cache or a user specified directory do a search on model weight downloads and the HF download will produce multiple weight files, config.json. The weights match the source code
3) implemwent teh model under nanovllm/models/

4) verify each model produces teh same output and identify any errors per layer

What are the choices of what you can contribute: add quantization strategies, 4bit to replicate teh numerical accuracy per layer. 
Work in higher layers with routers and envs which involve multiple GPUs which you dont have access to. 

 
