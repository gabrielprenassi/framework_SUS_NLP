from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from sentiment_analysis.src.llms.token_id import get_token
import pandas as pd
import torch
import os
import random
import numpy
import unicodedata
import re

# REPRODUTIBILIDADE ---------------------
SEED=2025
random.seed(SEED); torch.manual_seed(SEED); numpy.random.seed(seed=SEED)

# MODEL ---------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
name_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
token_access = get_token()

model = AutoModelForCausalLM.from_pretrained(name_model, 
                                             torch_dtype="auto", 
                                             device_map=device,
                                             offload_buffers=True,
                                             token=token_access,
                                             trust_remote_code=True,
                                             use_cache=False,
                                             load_in_4bit=True,
                                             use_safetensors=True)

tokenizer = AutoTokenizer.from_pretrained(name_model, 
                                          torch_dtype="auto", 
                                          device_map=device, 
                                          offload_buffers=True, 
                                          token=token_access, 
                                          use_safetensors=True,
                                          trust_remote_code=True) 

tokenizer.pad_token = tokenizer.eos_token        
terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

# FUNÇÕES DE VERIFICAÇÃO DA SAÍDA DO MODELO
def normalize_text(text):
    """Normaliza a string removendo acentos, convertendo para minúsculas e eliminando espaços extras."""
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    return text

def remove_duplicates(responses):
    """Remove respostas repetidas com base no texto normalizado, mantendo a ordem original."""
    seen = set()
    unique_responses = []
    
    for response in responses:
        norm = normalize_text(response)
        if norm not in seen:
            seen.add(norm)
            unique_responses.append(response)  # Mantém a forma original da string
    
    return unique_responses

def extract_section(text, section_name):
    """Extrai uma seção do texto baseada no título passado."""
    pattern = rf"\*\*{section_name}\*\*:\n\n(.+?)(?=\n\n\*\*|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def process_feedback(text):
    """Processa o texto para remover repetições nas seções negative points e suggestions e retorna None se faltar alguma seção."""
    # Extrair seções do texto
    positive_points = extract_section(text, "Positive Points")
    negative_points = extract_section(text, "Negative Points")
    suggestions = extract_section(text, "Suggestions")

    # Converter listas numeradas em listas simples
    def extract_list(section_text):
        return [line.split(". ", 1)[-1].strip() for line in section_text.split("\n") if line.strip()]

    pos_items = extract_list(positive_points)
    neg_items = extract_list(negative_points)
    sugg_items = extract_list(suggestions)

    # Remover duplicatas
    unique_pos_items = remove_duplicates(pos_items)
    unique_neg_items = remove_duplicates(neg_items)
    unique_sugg_items = remove_duplicates(sugg_items)

    # Reformatar as listas numeradas
    def format_list(items):
        return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))

    formatted_pos_points = format_list(unique_pos_items)
    formatted_neg_points = format_list(unique_neg_items)
    formatted_suggestions = format_list(unique_sugg_items)

    # Construir o texto final
    result = ""
    if positive_points:
        result += f"**Positive Points**:\n\n{formatted_pos_points}"
    if negative_points:
        result += f"\n\n**Negative Points**:\n\n{formatted_neg_points}"
    if suggestions:
        result += f"\n\n**Suggestions**:\n\n{formatted_suggestions}"
    
    return result

# SUMMARIZATION -------------------------

def load_data(topic, total_topics):
    df = pd.read_csv('./data/dataFrame.csv')
    
    td = pd.read_csv(f'./topicmodeling/data_num_topics/{total_topics}/Resumo_Topicos_Dominantes.csv')
    papers_with_topic = td[td['dominant_topic'] == topic]['papers'].tolist()
    
    comments = df[df['ID'].isin(papers_with_topic)]
    comments = comments.reset_index(drop=True)
    comments = list(comments['comments'])
    
    print(len(comments))

    return comments

def get_standard_prompt(comments):

    prompt = [
        {'role': 'system', 'content': '''You will receive a set of user comments, and your task is to summarize them by extracting the positive points, negative points, and suggestions mentioned.\n\n
[Instructions]\n
Step 1: Determine the main points highlighted by users in a general manner.\n
- Do not highlight points that have been little commented on, return only those with the most relevance in the set.\n
- Generalize the points highlighted in the comments as much as possible without losing the context.\n\n
Step 2: Return the positive, negative and suggestion points according to the response pattern specified below.\n
    - Example of response:\n\n
    **Positive Points**:\n
    1. Positive Point 1\n
    2. Positive Point 2\n
    **Negative Points**:\n\n
    1. Negative Point 1\n
    2. Negative Point 2\n
    **Suggestions**\n\n
    1. Suggestion 1\n
    2. Suggestion 2\n
- Do not generate anything other than what is requested in the response pattern.\n
- Summarize the content without directly quoting user comments.\n
- Do not create new points; only summarize the existing ones.\n
- Do not use the first person in the generated text.'''}]
    
    for i, comment in enumerate(comments):
        prompt.append({'role': f'user_{i}', 'content': f'{comment}'})
    
    return prompt
    
def get_additional_prompt(partial_summary, removed_comments):


    prompt = [
        {'role': 'system', 'content': '''You will receive a partial summary of user comments. This summary has already integrated some feedback, but additional comments will be provided for you to enrich it further. Your task is to update and rewrite the summary, ensuring that all relevant points from the new comments are integrated while maintaining the original structure.
[Instructions]\n
Step 1: Analyze the partial summary and the new comments.\n
- Identify the main points in a general manner, considering both the existing summary and the new feedback.\n
- Do not include points that were mentioned only once or are not significant in the overall set.\n
- Maintain the context while ensuring a neutral, structured, and concise synthesis.\n\n
Step 2: Rewrite the entire summary following the response pattern below.\n
- The final response must fully integrate both the partial summary and the new comments.\n
- Structure the response into three categories: Positive Points, Negative Points, and Suggestions.\n
    - Example of response:\n\n
    **Positive Points**:\n
    1. Positive Point 1\n
    2. Positive Point 2\n
    **Negative Points**:\n\n
    1. Negative Point 1\n
    2. Negative Point 2\n
    **Suggestions**\n\n
    1. Suggestion 1\n
    2. Suggestion 2\n 
- Do not generate anything beyond the requested structure.\n
- Summarize the content without directly quoting user comments.\n
- Do not create new points; only summarize the existing ones.\n
- Do not use the first person in the generated text.\n
- Ensure that the updated summary remains well-structured, cohesive, and neutral.'''},
    
    {'role': 'user_0', 'content': f'{partial_summary}'}
]
    for i, comment in enumerate(removed_comments):
        prompt.append({'role': f'user_{i+1}', 'content': f'{comment}'})

    return prompt


def remove_tokens_for_classification(comments, total_number_of_tokens, target_number_of_tokens=12000, batch_size=10):
        tokens_removed = 0
        current_number_of_tokens = total_number_of_tokens
        removed_comments = []
                
        while total_number_of_tokens - tokens_removed > target_number_of_tokens:    
            if not comments:
                break

            batch_size = min(batch_size, len(comments))
            removed_batch = [comments.pop() for _ in range(batch_size)]
            removed_comments.extend(removed_batch)
            
            prompt = get_standard_prompt(comments=comments)

            inputs = tokenizer.apply_chat_template(prompt,
                                           add_generation_prompt=True,
                                           return_dict=True,
                                           return_tensors="pt").to(device)


            current_number_of_tokens = inputs['input_ids'][0].numel()
            tokens_removed = total_number_of_tokens - current_number_of_tokens
        
        return inputs.to("cuda"), removed_comments

def get_summary(comments):
    removed_comments = []
    
    prompt = get_standard_prompt(comments=comments)

    inputs = tokenizer.apply_chat_template(prompt,
                                           add_generation_prompt=True,
                                           return_dict=True,
                                           return_tensors="pt").to(device)
        
    while inputs['input_ids'].numel() > 12000:
        print('Retirada de comentários', inputs['input_ids'].numel(), end=' ')
        inputs, removed = remove_tokens_for_classification(comments=comments, 
                                                            total_number_of_tokens=inputs['input_ids'].numel()) 
        removed_comments = removed
        print(inputs['input_ids'].numel())

    set_seed(SEED)
    outputs = model.generate(inputs['input_ids'],
                             attention_mask=inputs['attention_mask'],
                             max_new_tokens=1500,
                             eos_token_id=terminators,
                             pad_token_id=tokenizer.eos_token_id,
                             do_sample=True,
                             temperature=0.05,
                             top_p=0.9,
                             use_cache=False)
    
    summary = outputs[0][inputs['input_ids'].shape[-1]:]
    summary = tokenizer.decode(summary, skip_special_tokens=True)
    print(summary)
    summary = process_feedback(summary)

    torch.cuda.empty_cache()
    
    while removed_comments:  
        print(summary)

        prompt = get_additional_prompt(summary, removed_comments)
         
        inputs = tokenizer.apply_chat_template(prompt,
                                               add_generation_prompt=True,
                                               return_dict=True,
                                               return_tensors="pt").to(device)
        
        if inputs['input_ids'].numel() > 12000:
            print('Retirada adicional de comentários', inputs['input_ids'].numel(), end=' ')
            inputs, removed = remove_tokens_for_classification(comments=removed_comments, 
                                                                total_number_of_tokens=inputs['input_ids'].numel())
            removed_comments = removed
            print(inputs['input_ids'].numel())
        else:
            removed_comments = []  # Se estiver dentro do limite, pode continuar
        
        set_seed(SEED)
        outputs = model.generate(inputs['input_ids'],
                                 attention_mask=inputs['attention_mask'],
                                 max_new_tokens=1500,
                                 eos_token_id=terminators,
                                 pad_token_id=tokenizer.eos_token_id,
                                 do_sample=True,
                                 temperature=0.05,
                                 top_p=0.9,
                                 use_cache=False)
        
        summary = outputs[0][inputs['input_ids'].shape[-1]:]
        summary = tokenizer.decode(summary, skip_special_tokens=True)
        summary = process_feedback(summary)

        torch.cuda.empty_cache()    
                
    return summary  


def save_summary(text, total_number_of_topics, topic):
    if not os.path.exists(f'./summarization/outLLM/detailed_summarization/{total_number_of_topics}'): os.makedirs(f'./summarization/outLLM/detailed_summarization/{total_number_of_topics}')
    with open(f'./summarization/outLLM/detailed_summarization/{total_number_of_topics}/summary_topic_{topic}.txt', 'w') as file:
        file.write(text)
        
# MAIN ---------------------------------

#for total_t in [5, 10, 15]:
def run_detailed():
    for total_t in [5, 10, 15]:
        for t in range(total_t):
            print(f'topic {t}')
            comments = load_data(t, total_t)
            summary = get_summary(comments)
            save_summary(summary, total_t, t)
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()