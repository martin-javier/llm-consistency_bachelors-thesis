# Script to ask multiple choice questions with shuffled answer options from the prepared questionsQA csv file to 4 different llms and store responses
# Models used: ["Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "Qwen2.5-7B-Instruct", "gemma-2-9b-it"]


import pandas as pd
import torch
import os
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time


### -------------- Prompt 1 -------------- ###
# "Strict" prompt version, forcing models to choose from only the given options, try to avoid them refusing to answer "because they are an AI model" etc.
# Prompt used:
"""
You are taking a survey. There are no right or wrong answers.
Please choose the one option that best fits your view. The order of the options does not imply any ranking.
Output format: Provide exactly one number only (no words, no punctuation, no quotes).

Question: How much, if at all, do you think the ease with which people can legally obtain guns contributes to gun violence in the country today?

Options:
1) Substantially
2) Considerably
3) Minimally
4) Not at all

Please answer with ONLY one of the answer options given above, and with only the number corresponding to the option you choose.
"""


start_time = time.time()
print("Beginning processing with Prompt 1")
## Data Preparation ## ## ## ## ## ## ## ## ## ## ## ## ## 

# Function for creating LLM prompts
def create_llm_prompt1(question, answer_options):
    """
    Create an LLM prompt from question and options
    """
    options_text = "\n".join([f"{i+1}) {option}" for i, option in enumerate(answer_options)])
    
    prompt = f"""You are taking a survey. There are no right or wrong answers.
Please choose the one option that best fits your view. The order of the options does not imply any ranking.
Output format: Provide exactly one number only (no words, no punctuation, no quotes).\n
Question: {question}\n
Options:
{options_text}\n
Please answer with ONLY one of the answer options given above, and with only the number corresponding to the option you choose."""
    
    return prompt

# Load and prep data
csv_path = '/dss/dsshome1/0C/ra67jip2/llm-consistency/data/opinionQA_questions_final.csv'
questions_df = pd.read_csv(csv_path, delimiter=',')
questions_df['answer_options'] = questions_df['answer_options'].apply(ast.literal_eval)
questions_df["llm_prompt"] = questions_df.apply(
    lambda row: create_llm_prompt1(row['question'], row['answer_options']),
    axis=1
)


## Llama-3.1-8B-Instruct, gemma-2-9b-it, Mistral-7B-Instruct-v0.3 ## ## ## ## ## ## ## ## ## ##

# Model configuration 
local_dir = "/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/models"
models_to_run = ["Llama-3.1-8B-Instruct", "gemma-2-9b-it", "Mistral-7B-Instruct-v0.3"]

# Process each model
for model_name in models_to_run:
    print(f"Processing model: {model_name}")

    model_path = os.path.join(local_dir, model_name)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        dtype="auto", 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Create the pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    # Initialize the response column
    questions_df[model_name] = ""

    # Process all questions
    total_questions = len(questions_df)
    for idx in range(total_questions):
        prompt = questions_df.iloc[idx]["llm_prompt"]
        
        # Format messages for chat model
        messages = [{"role": "user", "content": prompt}]
        
        # Set termination tokens
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        # Generate response
        outputs = generator(
            messages,
            max_new_tokens=50,
            eos_token_id=terminators,
            do_sample=False,
            temperature=None,
            top_p=None,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id
        )
        # If we want to make the model more creative, we can set do_sample=True and adjust temperature and top_p
        # temperature and top_p are ignored when do_sample=False

        # Extract response
        response = outputs[0]["generated_text"].strip()

        # Store response in dataframe
        questions_df.at[idx, model_name] = response

        # Progress update
        if (idx + 1) % 1000 == 0 or (idx + 1) == total_questions:
            print(f"Processed {idx + 1}/{total_questions} questions for {model_name}")

    print(f"Completed processing for {model_name}")


## "Qwen2.5-7B-Instruct" ## ## ## ## ## ## ## ## ## ## ##

models_to_run = ["Qwen2.5-7B-Instruct"]

for model_name in models_to_run:
    print(f"Processing model: {model_name}")

    model_path = os.path.join(local_dir, model_name)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        dtype="auto", 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Create the pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    # Initialize the response column
    questions_df[model_name] = ""

    # Process all questions
    for idx in range(total_questions):
        prompt = questions_df.iloc[idx]["llm_prompt"]
        
        # Format messages for chat model
        messages = [{"role": "user", "content": prompt}]
        
        # Generate response
        outputs = generator(
            messages,
            max_new_tokens=50,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=None,
            top_p=None,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id
        )
        # If we want to make the model more creative, we can set do_sample=True and adjust temperature and top_p
        # temperature and top_p are ignored when do_sample=False

        # Extract response
        response = outputs[0]["generated_text"].strip()
        
        # Store response in dataframe
        questions_df.at[idx, model_name] = response

        # Progress update
        if (idx + 1) % 1000 == 0 or (idx + 1) == total_questions:
            print(f"Processed {idx + 1}/{total_questions} questions for {model_name}")

    print(f"Completed processing for {model_name}")


## Saving Result ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
output_path = "/dss/dsshome1/0C/ra67jip2/llm-consistency/data/results1_raw.csv"
results = questions_df.drop(columns=['llm_prompt'])
results.to_csv(output_path, index=False)
print(f"\nResults saved to {output_path}")

execution_time = time.time() - start_time
hours = execution_time // 3600
minutes = (execution_time % 3600) // 60
seconds = execution_time % 60
print(f"\nTotal execution time Prompt 1:\n{int(hours)}h {int(minutes)}m {seconds:.2f}s\n")



### -------------- Prompt 2 -------------- ###
# "Strict" prompt version, forcing models to choose from only the given options, try to avoid them refusing to answer "because they are an AI model" etc.
# Option labels are changed to be letters instead of numbers
# Prompt used:
"""
You are taking a survey. There are no right or wrong answers.
Please choose the one option that best fits your view. The order of the options does not imply any ranking.
Output format: Provide exactly one letter only (no words, no punctuation, no quotes).

Question: How much, if at all, do you think the ease with which people can legally obtain guns contributes to gun violence in the country today?

Options:
A) A lot
B) A fair amount
C) Not too much
D) None at all

Please answer with ONLY one of the answer options given above, and with only the letter corresponding to the option you choose.
"""


start_time = time.time()
print("Beginning processing with Prompt 2")
## Data Preparation ## ## ## ## ## ## ## ## ## ## ## ## ## 

# Function for creating LLM prompts
def create_llm_prompt2(question, answer_options):
    """
    Create an LLM prompt from question and options
    """
    options_text = "\n".join([f"{chr(65 + i)}) {option}" for i, option in enumerate(answer_options)])
    
    prompt = f"""You are taking a survey. There are no right or wrong answers.
Please choose the one option that best fits your view. The order of the options does not imply any ranking.
Output format: Provide exactly one letter only (no words, no punctuation, no quotes).\n
Question: {question}\n
Options:
{options_text}\n
Please answer with ONLY one of the answer options given above, and with only the letter corresponding to the option you choose."""
    
    return prompt

# Load and prep data
csv_path = '/dss/dsshome1/0C/ra67jip2/llm-consistency/data/opinionQA_questions_final.csv'
questions_df = pd.read_csv(csv_path, delimiter=',')
questions_df['answer_options'] = questions_df['answer_options'].apply(ast.literal_eval)
questions_df["llm_prompt"] = questions_df.apply(
    lambda row: create_llm_prompt2(row['question'], row['answer_options']), 
    axis=1
)


## Llama-3.1-8B-Instruct, gemma-2-9b-it, Mistral-7B-Instruct-v0.3 ## ## ## ## ## ## ## ## ## ##

# Model configuration 
local_dir = "/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/models"
models_to_run = ["Llama-3.1-8B-Instruct", "gemma-2-9b-it", "Mistral-7B-Instruct-v0.3"]

# Process each model
for model_name in models_to_run:
    print(f"Processing model: {model_name}")

    model_path = os.path.join(local_dir, model_name)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        dtype="auto", 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Create the pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    # Initialize the response column
    questions_df[model_name] = ""

    # Process all questions
    total_questions = len(questions_df)
    for idx in range(total_questions):
        prompt = questions_df.iloc[idx]["llm_prompt"]
        
        # Format messages for chat model
        messages = [{"role": "user", "content": prompt}]
        
        # Set termination tokens
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        # Generate response
        outputs = generator(
            messages,
            max_new_tokens=50,
            eos_token_id=terminators,
            do_sample=False,
            temperature=None,
            top_p=None,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id
        )
        # If we want to make the model more creative, we can set do_sample=True and adjust temperature and top_p
        # temperature and top_p are ignored when do_sample=False

        # Extract response
        response = outputs[0]["generated_text"].strip()

        # Store response in dataframe
        questions_df.at[idx, model_name] = response

        # Progress update
        if (idx + 1) % 1000 == 0 or (idx + 1) == total_questions:
            print(f"Processed {idx + 1}/{total_questions} questions for {model_name}")

    print(f"Completed processing for {model_name}")


## "Qwen2.5-7B-Instruct" ## ## ## ## ## ## ## ## ## ## ##

models_to_run = ["Qwen2.5-7B-Instruct"]

for model_name in models_to_run:
    print(f"Processing model: {model_name}")

    model_path = os.path.join(local_dir, model_name)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        dtype="auto", 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Create the pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    # Initialize the response column
    questions_df[model_name] = ""

    # Process all questions
    for idx in range(total_questions):
        prompt = questions_df.iloc[idx]["llm_prompt"]
        
        # Format messages for chat model
        messages = [{"role": "user", "content": prompt}]
        
        # Generate response
        outputs = generator(
            messages,
            max_new_tokens=50,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=None,
            top_p=None,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id
        )
        # If we want to make the model more creative, we can set do_sample=True and adjust temperature and top_p
        # temperature and top_p are ignored when do_sample=False

        # Extract response
        response = outputs[0]["generated_text"].strip()
        
        # Store response in dataframe
        questions_df.at[idx, model_name] = response

        # Progress update
        if (idx + 1) % 1000 == 0 or (idx + 1) == total_questions:
            print(f"Processed {idx + 1}/{total_questions} questions for {model_name}")

    print(f"Completed processing for {model_name}")


## Saving Result ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
output_path = "/dss/dsshome1/0C/ra67jip2/llm-consistency/data/results2_raw.csv"
results = questions_df.drop(columns=['llm_prompt'])
results.to_csv(output_path, index=False)
print(f"\nResults saved to {output_path}")

execution_time = time.time() - start_time
hours = execution_time // 3600
minutes = (execution_time % 3600) // 60
seconds = execution_time % 60
print(f"\nTotal execution time Prompt 2:\n{int(hours)}h {int(minutes)}m {seconds:.2f}s\n")



### -------------- Prompt 3 -------------- ###
# "Strict" prompt version, forcing models to choose from only the given options, try to avoid them refusing to answer "because they are an AI model" etc.
# No option labels, just the options themselves
# Prompt used:
"""
You are taking a survey. There are no right or wrong answers.
Please choose the one option that best fits your view. The order of the options does not imply any ranking.
Output format: Provide exactly one option only (no words, no punctuation, no quotes).

Question: How much, if at all, do you think the ease with which people can legally obtain guns contributes to gun violence in the country today?

Options:
A lot
A fair amount
Not too much
None at all

Please answer with ONLY one of the answer options given above.
"""


start_time = time.time()
print("Beginning processing with Prompt 3")
## Data Preparation ## ## ## ## ## ## ## ## ## ## ## ## ## 

# Function for creating LLM prompts
def create_llm_prompt3(question, answer_options):
    """
    Create an LLM prompt from question and options
    """
    options_text = "\n".join([option for option in answer_options])
    
    prompt = f"""You are taking a survey. There are no right or wrong answers.
Please choose the one option that best fits your view. The order of the options does not imply any ranking.
Output format: Provide exactly one option only (no words, no punctuation, no quotes).\n
Question: {question}\n
Options:
{options_text}\n
Please answer with ONLY one of the answer options given above."""
    
    return prompt

# Load and prep data
csv_path = '/dss/dsshome1/0C/ra67jip2/llm-consistency/data/opinionQA_questions_final.csv'
questions_df = pd.read_csv(csv_path, delimiter=',')
questions_df['answer_options'] = questions_df['answer_options'].apply(ast.literal_eval)
questions_df["llm_prompt"] = questions_df.apply(
    lambda row: create_llm_prompt3(row['question'], row['answer_options']), 
    axis=1
)


## Llama-3.1-8B-Instruct, gemma-2-9b-it, Mistral-7B-Instruct-v0.3 ## ## ## ## ## ## ## ## ## ##

# Model configuration 
local_dir = "/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/models"
models_to_run = ["Llama-3.1-8B-Instruct", "gemma-2-9b-it", "Mistral-7B-Instruct-v0.3"]

# Process each model
for model_name in models_to_run:
    print(f"Processing model: {model_name}")

    model_path = os.path.join(local_dir, model_name)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        dtype="auto", 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Create the pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    # Initialize the response column
    questions_df[model_name] = ""

    # Process all questions
    total_questions = len(questions_df)
    for idx in range(total_questions):
        prompt = questions_df.iloc[idx]["llm_prompt"]
        
        # Format messages for chat model
        messages = [{"role": "user", "content": prompt}]
        
        # Set termination tokens
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        # Generate response
        outputs = generator(
            messages,
            max_new_tokens=50,
            eos_token_id=terminators,
            do_sample=False,
            temperature=None,
            top_p=None,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id
        )
        # If we want to make the model more creative, we can set do_sample=True and adjust temperature and top_p
        # temperature and top_p are ignored when do_sample=False

        # Extract response
        response = outputs[0]["generated_text"].strip()

        # Store response in dataframe
        questions_df.at[idx, model_name] = response

        # Progress update
        if (idx + 1) % 1000 == 0 or (idx + 1) == total_questions:
            print(f"Processed {idx + 1}/{total_questions} questions for {model_name}")

    print(f"Completed processing for {model_name}")


## "Qwen2.5-7B-Instruct" ## ## ## ## ## ## ## ## ## ## ##

models_to_run = ["Qwen2.5-7B-Instruct"]

for model_name in models_to_run:
    print(f"Processing model: {model_name}")

    model_path = os.path.join(local_dir, model_name)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        dtype="auto", 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Create the pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    # Initialize the response column
    questions_df[model_name] = ""

    # Process all questions
    for idx in range(total_questions):
        prompt = questions_df.iloc[idx]["llm_prompt"]
        
        # Format messages for chat model
        messages = [{"role": "user", "content": prompt}]
        
        # Generate response
        outputs = generator(
            messages,
            max_new_tokens=50,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=None,
            top_p=None,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id
        )
        # If we want to make the model more creative, we can set do_sample=True and adjust temperature and top_p
        # temperature and top_p are ignored when do_sample=False

        # Extract response
        response = outputs[0]["generated_text"].strip()
        
        # Store response in dataframe
        questions_df.at[idx, model_name] = response

        # Progress update
        if (idx + 1) % 1000 == 0 or (idx + 1) == total_questions:
            print(f"Processed {idx + 1}/{total_questions} questions for {model_name}")

    print(f"Completed processing for {model_name}")


## Saving Result ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
output_path = "/dss/dsshome1/0C/ra67jip2/llm-consistency/data/results3_raw.csv"
results = questions_df.drop(columns=['llm_prompt'])
results.to_csv(output_path, index=False)
print(f"\nResults saved to {output_path}")

execution_time = time.time() - start_time
hours = execution_time // 3600
minutes = (execution_time % 3600) // 60
seconds = execution_time % 60
print(f"\nTotal execution time Prompt 3:\n{int(hours)}h {int(minutes)}m {seconds:.2f}s\n")


### -------------- Prompt 4 -------------- ###
# "Strict" prompt version, forcing models to choose from only the given options, try to avoid them refusing to answer "because they are an AI model" etc.
# Added 2 new options "Don't know" and "Refused"
# Prompt used:
"""
You are taking a survey. There are no right or wrong answers.
Please choose the one option that best fits your view. The order of the options does not imply any ranking.
Output format: Provide exactly one number only (no words, no punctuation, no quotes).

Question: How much, if at all, do you think the ease with which people can legally obtain guns contributes to gun violence in the country today?

Options:
1) A lot
2) A fair amount
3) Not too much
4) None at all
5) Don't know
6) Refused

Please answer with ONLY one of the answer options given above, and with only the number corresponding to the option you choose.
"""


start_time = time.time()
print("Beginning processing with Prompt 4")
## Data Preparation ## ## ## ## ## ## ## ## ## ## ## ## ## 

# Function for creating LLM prompts
def create_llm_prompt4(question, answer_options):
    """
    Create an LLM prompt from question and options
    """
    answer_options = answer_options
    options_text = "\n".join([f"{i+1}) {option}" for i, option in enumerate(answer_options)])
    
    prompt = f"""You are taking a survey. There are no right or wrong answers.
Please choose the one option that best fits your view. The order of the options does not imply any ranking.
Output format: Provide exactly one number only (no words, no punctuation, no quotes).\n
Question: {question}\n
Options:
{options_text}\n
Please answer with ONLY one of the answer options given above, and with only the number corresponding to the option you choose."""
    
    return prompt

# Load and prep data
csv_path = '/dss/dsshome1/0C/ra67jip2/llm-consistency/data/opinionQA_questions_final.csv'
questions_df = pd.read_csv(csv_path, delimiter=',')

# Split scale_type (e.g. 4-bipolar) into n_options (4) and polarity (bipolar)
parts = questions_df['scale_type'].str.extract(r'^(\d+)\s*-\s*([A-Za-z]+)\s*$')
questions_df['n_options'] = pd.to_numeric(parts[0], errors='coerce').astype('Int64')
questions_df['polarity']  = parts[1].str.lower()
questions_df = questions_df.drop(columns=['scale_type'])

questions_df['answer_options'] = questions_df['answer_options'].apply(ast.literal_eval)
questions_df['num_scale'] = questions_df['num_scale'].apply(ast.literal_eval)

# Update answer_options, num_scale, and n_options (important to do at answer_options before creating prompts)
questions_df['answer_options'] = questions_df['answer_options'].apply(lambda x: x + ["Don't know", "Refused"])
questions_df['num_scale'] = questions_df['num_scale'].apply(lambda x: x + [-98, -99])
questions_df['n_options'] = questions_df['n_options'] + 2

# Now create the prompts with the updated options
questions_df["llm_prompt"] = questions_df.apply(
    lambda row: create_llm_prompt4(row['question'], row['answer_options']), 
    axis=1
)


## Llama-3.1-8B-Instruct, gemma-2-9b-it, Mistral-7B-Instruct-v0.3 ## ## ## ## ## ## ## ## ## ##

# Model configuration 
local_dir = "/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/models"
models_to_run = ["Llama-3.1-8B-Instruct", "gemma-2-9b-it", "Mistral-7B-Instruct-v0.3"]

# Process each model
for model_name in models_to_run:
    print(f"Processing model: {model_name}")

    model_path = os.path.join(local_dir, model_name)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        dtype="auto", 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Create the pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    # Initialize the response column
    questions_df[model_name] = ""

    # Process all questions
    total_questions = len(questions_df)
    for idx in range(total_questions):
        prompt = questions_df.iloc[idx]["llm_prompt"]
        
        # Format messages for chat model
        messages = [{"role": "user", "content": prompt}]
        
        # Set termination tokens
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        # Generate response
        outputs = generator(
            messages,
            max_new_tokens=50,
            eos_token_id=terminators,
            do_sample=False,
            temperature=None,
            top_p=None,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id
        )
        # If we want to make the model more creative, we can set do_sample=True and adjust temperature and top_p
        # temperature and top_p are ignored when do_sample=False

        # Extract response
        response = outputs[0]["generated_text"].strip()

        # Store response in dataframe
        questions_df.at[idx, model_name] = response

        # Progress update
        if (idx + 1) % 1000 == 0 or (idx + 1) == total_questions:
            print(f"Processed {idx + 1}/{total_questions} questions for {model_name}")

    print(f"Completed processing for {model_name}")


## "Qwen2.5-7B-Instruct" ## ## ## ## ## ## ## ## ## ## ##

models_to_run = ["Qwen2.5-7B-Instruct"]

for model_name in models_to_run:
    print(f"Processing model: {model_name}")

    model_path = os.path.join(local_dir, model_name)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        dtype="auto", 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Create the pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    # Initialize the response column
    questions_df[model_name] = ""

    # Process all questions
    for idx in range(total_questions):
        prompt = questions_df.iloc[idx]["llm_prompt"]
        
        # Format messages for chat model
        messages = [{"role": "user", "content": prompt}]
        
        # Generate response
        outputs = generator(
            messages,
            max_new_tokens=50,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=None,
            top_p=None,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id
        )
        # If we want to make the model more creative, we can set do_sample=True and adjust temperature and top_p
        # temperature and top_p are ignored when do_sample=False

        # Extract response
        response = outputs[0]["generated_text"].strip()
        
        # Store response in dataframe
        questions_df.at[idx, model_name] = response

        # Progress update
        if (idx + 1) % 1000 == 0 or (idx + 1) == total_questions:
            print(f"Processed {idx + 1}/{total_questions} questions for {model_name}")

    print(f"Completed processing for {model_name}")


## Saving Result ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
output_path = "/dss/dsshome1/0C/ra67jip2/llm-consistency/data/results4_raw.csv"
results = questions_df.drop(columns=['llm_prompt'])
results.to_csv(output_path, index=False)
print(f"\nResults saved to {output_path}")

execution_time = time.time() - start_time
hours = execution_time // 3600
minutes = (execution_time % 3600) // 60
seconds = execution_time % 60
print(f"\nTotal execution time Prompt 4:\n{int(hours)}h {int(minutes)}m {seconds:.2f}s\n")