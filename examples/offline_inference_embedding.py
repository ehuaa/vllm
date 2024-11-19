from vllm import LLM
import json
from transformers import AutoTokenizer
import time


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'
task = 'Given a web search query, retrieve relevant passages that answer the query'
max_length = 4096

# data = [json.loads(line)["text"] for line in (open("/nas/czh/sfr/539667.jsonl", "r"))]
# data = data[9:10] * 10
data = []
with open('/nas/czh/part_1.tsv', 'r', encoding='utf-8') as file:
    for line in file:
        row = line.strip().split('|')[1].strip()
        data.append(row)
        
# with open('/nas/czh/sfr/539667.jsonl', 'r', encoding='utf-8') as file:
#     for line in file:
#         row = json.loads(line)["text"]
#         data.append(row)

model = LLM(model="/nas/czh/sfr/SFR-Embedding-Mistral", enforce_eager=True, disable_sliding_window=True)

tokenizer = AutoTokenizer.from_pretrained("/nas/czh/sfr/SFR-Embedding-Mistral")
tokenizer.add_eos_token = True
step_size = 8


def model_encode(first_ind: int):
    prompts = [get_detailed_instruct(task, it) for it in data[first_ind: first_ind + step_size]]
    batch_dict = tokenizer(prompts, max_length=max_length - 1, padding=True, truncation=True, return_tensors="pt")
    prompt_encode_id_dict = [{"prompt_token_ids": input_ids} for input_ids in batch_dict["input_ids"]]
    # for i in prompt_encode_id_dict:
    #     print(len(i["prompt_token_ids"]))
    model.encode(prompt_encode_id_dict)

# offline data to llm engine
prompts = [get_detailed_instruct(task, it) for it in data]

llm_engine_prompt_encode_id_dict = [{"prompt_token_ids": tokenizer(prompt, max_length=max_length - 1, padding=True, truncation=True, return_tensors="pt")["input_ids"][0]} for prompt in prompts]

t0 = time.perf_counter()
# Generate embedding. The output is a list of EmbeddingRequestOutputs.
# for ind in range(0, len(data), step_size):
#     outputs = model_encode(ind)
outputs = model.encode(llm_engine_prompt_encode_id_dict)
print(time.perf_counter() - t0)
# Print the outputs.
# for output in outputs:
#     print(output.outputs.embedding[:10])  # list of 4096 floats
    # print(len(output.outputs.embedding))
