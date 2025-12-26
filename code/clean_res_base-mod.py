# Analysis of the results from run_quest_base-models.py and run_quest_shuff_base-mod.py
# Models used (base variations): ['Llama-3.1-8B', 'Mistral-7B-v0.3', 'Qwen2.5-7B', 'gemma-2-9b']

import pandas as pd
import ast
import os
import re
from pathlib import Path

# set working directory
# 1. Determine the location of this script/process
try:
    # If running as a script
    current_loc = Path(__file__).parent.resolve()
except NameError:
    # If running interactively
    current_loc = Path(os.getcwd())

# 2. Check if we are inside the 'code' folder and need to move up
if current_loc.name == 'code':
    script_dir = current_loc.parent
else:
    script_dir = current_loc

# 3. Set Working Directory
os.chdir(script_dir)

# helper function: extract correct numeric value of response given
def create_numerical_value(df):
    """
    Create a numerical value column by mapping clean_response to num_scale values.
    Works whether clean_response entries are ints or digit-strings; leaves 'unusable' unchanged.
    """
    df = df.copy()
    df['num_value'] = 'unusable'  # Initialize with 'unusable'
    
    for idx, row in df.iterrows():
        clean_response = row['clean_response']

        # Handle numeric types (int/float)
        if isinstance(clean_response, (int, float)) and not pd.isna(clean_response):
            option_index = int(clean_response) - 1
        # Handle string digits (e.g. "1")
        elif isinstance(clean_response, str) and clean_response.isdigit():
            option_index = int(clean_response) - 1
        else:
            # not a usable numeric response, leave as 'unusable'
            continue
        
        # Parse the num_scale list (keep original behaviour)
        try:
            num_scale_list = ast.literal_eval(row['num_scale'])
        except Exception:
            # fallback if parsing fails
            num_scale_list = []
        
        # Get the numerical value from num_scale
        if 0 <= option_index < len(num_scale_list):
            df.at[idx, 'num_value'] = num_scale_list[option_index]
        else:
            df.at[idx, 'num_value'] = 'unusable'
    
    return df


### -------------- Prompt 1 (Reworded Answer Options) -------------- ###
# Prompt used:
"""
Question: How much, if at all, do you think the ease with which people can legally obtain guns contributes to gun violence in the country today?

Options:
1) Substantially
2) Considerably
3) Minimally
4) Not at all

The best answer is option number:
"""

results = pd.read_csv(os.path.join(script_dir, "data", "raw", "res1_raw_base-models.csv"), delimiter=',')

# Split scale_type into n_options and polarity (Extract digits before '-' and letters after '-')
parts = results['scale_type'].str.extract(r'^(\d+)\s*-\s*([A-Za-z]+)\s*$')
results['n_options'] = pd.to_numeric(parts[0], errors='coerce').astype('Int64')
results['polarity']  = parts[1].str.lower()
results = results.drop(columns=['scale_type'])

# Convert to long format
model_columns = ['Llama-3.1-8B', 'Mistral-7B-v0.3', 'Qwen2.5-7B', 'gemma-2-9b']
results_long = results.melt(
    id_vars=[col for col in results.columns if col not in model_columns],
    value_vars=model_columns,
    var_name='model',
    value_name='response'
)

# Replace \n and \r (including multiples) with a single space
results_long['response'] = (results_long['response']
                  .astype('string')
                  .str.replace(r'[\r\n]+', ' ', regex=True)
                  .str.replace(r'\s+', ' ', regex=True)
                  .str.strip())

def process_responses_bm1(df):
    """
    Cleans response column to extract a single numeric answer.
    
    Methodology:
    1. Standard Priorities (1-4).
    2. Fallback (Unique Number Check).
    3. Edge cases: # TODO: If results are regenerated all cases (maybe apart from 5) need to be checked thorougly
       - EC 1: Model Specific Leading Digit.
       - EC 2: Dot Notation (No %).
       - EC 3: Letter "A)" or "A."
       - EC 4: Leading Digit is 3 or 4.
       - EC 5: Word Match (Start of string).
    """
    df = df.copy()
    edge_case_models = {'Qwen2.5-7B', 'gemma-2-9b'}

    def _clean_row(row):
        text = str(row['response'])
        n_opts = row['n_options']
        
        # --- 1. Standard Priorities ---
        # Priority 1: "Answer: <digit>"
        match_ans = re.search(r'Answer:\s*(\d+)', text, re.IGNORECASE)
        if match_ans:
            val = int(match_ans.group(1))
            if 1 <= val <= n_opts: return str(val)
        
        # Priority 2: "<digit> Explanation"
        match_expl = re.search(r'(\d+)\s+Explan', text, re.IGNORECASE)
        if match_expl:
            val = int(match_expl.group(1))
            if 1 <= val <= n_opts: return str(val)

        # Priority 3: "<digit> Question"
        match_quest = re.search(r'(\d+)\s+Question', text, re.IGNORECASE)
        if match_quest:
            val = int(match_quest.group(1))
            if 1 <= val <= n_opts: return str(val)

        # Priority 4: "X)" at start
        match_paren = re.search(r'^(\d+)\)', text)
        if match_paren:
            val = int(match_paren.group(1))
            if 1 <= val <= n_opts: return str(val)

        # --- 2. Fallback (Unique Number Check) ---
        all_numbers = re.findall(r'\d+', text)
        valid_candidates = {int(x) for x in all_numbers if 1 <= int(x) <= n_opts}
        
        # If exactly one unique valid number was found, return it.
        if len(valid_candidates) == 1:
            return str(list(valid_candidates)[0])

        # --- 3. Edge cases (only if previous steps failed -> clean_response == 'unusable') ---
        # Edge case 1: Qwen & gemma leading digit
        if row.get('model') in edge_case_models:
            match_leading = re.search(r'^(\d+)', text)
            if match_leading:
                val = int(match_leading.group(1))
                if 1 <= val <= n_opts: return str(val)

        # Edge case 2: Dot Notation "X." (and no percentage signs in whole string)
        if '%' not in text:
            match_dot = re.search(r'(\d+)\.', text)
            if match_dot:
                val = int(match_dot.group(1))
                if 1 <= val <= n_opts: return str(val)

        # Edge case 3: Letter "A)" or "A." -> 1 (Case Sensitive)
        if re.search(r'\sA[).]', text):
            return "1"

        # Edge Case 4: Leading Digit is 3 or 4
        match_final_leading = re.search(r'^(\d+)', text)
        if match_final_leading:
            val = int(match_final_leading.group(1))
            # Strict check: Must be 3 or 4, and within bounds
            if val in [3, 4] and 1 <= val <= n_opts:
                return str(val)
            
        # Helper to process Letter or Digit
        def parse_val(candidate):
            if candidate.isdigit():
                return int(candidate)
            elif len(candidate) == 1 and candidate.isalpha():
                return ord(candidate.upper()) - 64
            return -1
        
        # Edge Case 5: "answer is..."
        match_ec2 = re.search(r'answer is[:\s]*([A-Z]|\d+)', text, re.IGNORECASE)
        if match_ec2:
            val = parse_val(match_ec2.group(1))
            if 1 <= val <= n_opts:
                return str(val)

        # Edge case 6: Text matching answer options at start of response
        # Parse options
        options = ast.literal_eval(row['answer_options'])
        
        # Compress response
        clean_resp_text = text.lower().replace(" ", "")
        
        # 1. Create a list of (clean_text, original_index)
        check_list = []
        for idx, opt_text in enumerate(options):
            c_opt = str(opt_text).lower().replace(" ", "")
            if c_opt:
                check_list.append((c_opt, idx + 1))
        
        # 2. Sort options by length descending (match longest)
        check_list.sort(key=lambda x: len(x[0]), reverse=True)

        # 3. Check for match at start only
        for clean_opt, original_idx in check_list:
            if clean_resp_text.startswith(clean_opt):
                return str(original_idx)

        return 'unusable'

    # Apply the logic row-wise
    df['clean_response'] = df.apply(_clean_row, axis=1)
    
    return df

results_long = process_responses_bm1(results_long)
results_long = create_numerical_value(results_long)

results_long.to_csv(os.path.join(script_dir, "data", "clean", "res1_clean_bm.csv"), index=False)


### -------------- Prompt 2 (Reworded Answer Options) -------------- ###
# Prompt used:
"""
Question: How much, if at all, do you think the ease with which people can legally obtain guns contributes to gun violence in the country today?

Options:
A) A lot
B) A fair amount
C) Not too much
D) None at all

The best answer is option letter:
"""

results = pd.read_csv(os.path.join(script_dir, "data", "raw", "res2_raw_base-models.csv"), delimiter=',')

# Split scale_type into n_options and polarity (Extract digits before '-' and letters after '-')
parts = results['scale_type'].str.extract(r'^(\d+)\s*-\s*([A-Za-z]+)\s*$')
results['n_options'] = pd.to_numeric(parts[0], errors='coerce').astype('Int64')
results['polarity']  = parts[1].str.lower()
results = results.drop(columns=['scale_type'])

# Convert to long format
model_columns = ['Llama-3.1-8B', 'Mistral-7B-v0.3', 'Qwen2.5-7B', 'gemma-2-9b']
results_long = results.melt(
    id_vars=[col for col in results.columns if col not in model_columns],
    value_vars=model_columns,
    var_name='model',
    value_name='response'
)

# Replace \n and \r (including multiples) with a single space
results_long['response'] = (results_long['response']
                  .astype('string')
                  .str.replace(r'[\r\n]+', ' ', regex=True)
                  .str.replace(r'\s+', ' ', regex=True)
                  .str.strip())

def process_responses_bm2(df):
    """
    Extracts a single uppercase letter answer (A, B, C...), direct letter extraction is prioritized over text-based matching.

    Methodology:
    1. Regex Extraction: Finds standalone uppercase letters (excluding 'I'), returns the first valid match.
    2. Fallback A (Start): Checks if response starts with an answer option wording (longest match wins).
    3. Fallback B (Anywhere): Checks if an answer option wording appears anywhere in the response (first match wins).
    4. Returns the valid letter or 'unusable'.
    """
    df = df.copy()

    def _clean_row(row):
        text = str(row['response'])
        n_opts = row['n_options']
        
        # 1. Regex for standalone letters (A, B, C...)
        matches = re.findall(r'(?:^|\s|[\(*])([A-HJ-Z])(?=[).:\s*]|$)', text)
        if matches:
            first_letter = matches[0]
            if (ord(first_letter) - 64) <= n_opts:
                return first_letter

        # 2. Fallback: Text matching (options -> letter)
        options = ast.literal_eval(row['answer_options'])

        clean_resp = text.lower().replace(" ", "")
        
        # Prepare list: (clean_text, letter)
        check_list = []
        for idx, opt in enumerate(options):
            c_opt = str(opt).lower().replace(" ", "")
            if c_opt:
                # chr(65) is 'A', 66 is 'B', etc.
                check_list.append((c_opt, chr(65 + idx)))

        # Priority A: Check start (longest match first)
        check_list.sort(key=lambda x: len(x[0]), reverse=True)
        for c_opt, letter in check_list:
            if clean_resp.startswith(c_opt):
                if (ord(letter) - 64) <= n_opts: return letter

        # Priority B: Check first occurrence in wholde response
        found_matches = []
        for c_opt, letter in check_list:
            pos = clean_resp.find(c_opt)
            if pos != -1:
                found_matches.append((pos, letter))
        
        if found_matches:
            found_matches.sort(key=lambda x: x[0]) # Sort by position ascending
            letter = found_matches[0][1]
            if (ord(letter) - 64) <= n_opts: return letter

        return 'unusable'

    df['clean_response'] = df.apply(_clean_row, axis=1)

    return df


results_long = process_responses_bm2(results_long)

# Convert letters to numbers (A=1, B=2, etc.) to match the other clean results
results_long['clean_response'] = results_long['clean_response'].apply(
    lambda x: str(ord(x) - 64) if x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' else x
)

results_long = create_numerical_value(results_long)

results_long.to_csv(os.path.join(script_dir, "data", "clean", "res2_clean_bm.csv"), index=False)


### -------------- Prompt 3 (Reworded Answer Options) -------------- ###
# Prompt used:
"""
Question: How much, if at all, do you think the ease with which people can legally obtain guns contributes to gun violence in the country today?

Options:
A lot
A fair amount
Not too much
None at all

The best answer is option:
"""

results = pd.read_csv(os.path.join(script_dir, "data", "raw", "res3_raw_base-models.csv"), delimiter=',')

# Split scale_type into n_options and polarity (Extract digits before '-' and letters after '-')
parts = results['scale_type'].str.extract(r'^(\d+)\s*-\s*([A-Za-z]+)\s*$')
results['n_options'] = pd.to_numeric(parts[0], errors='coerce').astype('Int64')
results['polarity']  = parts[1].str.lower()
results = results.drop(columns=['scale_type'])

# Convert to long format
model_columns = ['Llama-3.1-8B', 'Mistral-7B-v0.3', 'Qwen2.5-7B', 'gemma-2-9b']
results_long = results.melt(
    id_vars=[col for col in results.columns if col not in model_columns],
    value_vars=model_columns,
    var_name='model',
    value_name='response'
)

# Replace \n and \r (including multiples) with a single space
results_long['response'] = (results_long['response']
                  .astype('string')
                  .str.replace(r'[\r\n]+', ' ', regex=True)
                  .str.replace(r'\s+', ' ', regex=True)
                  .str.strip())

def process_responses_bm3(df):
    """
    Cleans responses based on exact wording matches (Left-to-Right priority),
    with multiple Fallbacks.
    
    Methodology:
    1. Text Match (Priority 1): Earliest wording match wins.
    2. Fallback A: Numeric Search (First valid digit).
    3. Fallback B: Letter Search (A-Z excluding I).
    4. Fallback C: Typo Rescue for Qwen2.5-7B (Fixes "Most", "mosty", "mostlly" -> "mostly")
    """
    df = df.copy()

    def _clean_row(row):
        # Strip whitespace to ensure we catch "###" even if there is a leading space
        text = str(row['response']).strip()
        n_opts = int(row['n_options'])
        options = ast.literal_eval(row['answer_options'])

        # --- 1. Text Matching Strategy ---
        clean_resp = text.lower().replace(" ", "")
        found_matches = []
        
        for idx, opt in enumerate(options):
            c_opt = str(opt).lower().replace(" ", "")
            if not c_opt: continue
            
            pos = clean_resp.find(c_opt)
            if pos != -1:
                # Store (Position, Length, Option_Index_String)
                found_matches.append((pos, len(c_opt), str(idx + 1)))

        if found_matches:
            # Sort: Earliest Position first, Longest Length second
            found_matches.sort(key=lambda x: (x[0], -x[1]))
            return found_matches[0][2]
        
        # Early Exit: If response starts with "###" it's always unusable
        if text.startswith("###"):
            return 'unusable'

        # --- 3. Fallback: Numeric Search ---
        all_numbers = re.findall(r'\d+', text)
        
        for num in all_numbers:
            # Check if number is within valid range (1 to n_options)
            if 1 <= int(num) <= n_opts:
                # Return the FIRST valid number found
                return num
            
        # --- 4. Fallback B: Letter Search (Uppercase excluding I) ---
        letter_matches = re.findall(r'(?:^|\s|[\(*>])([A-HJ-Z])(?=[).:\s*]|$)', text)
        
        for letter in letter_matches:
            val = ord(letter) - 64
            if 1 <= val <= n_opts:
                return str(val)
            
        # --- 5. Fallback C: Qwen Typo Rescue ---
        # Fix the specific typos
        clean_resp_q = clean_resp.replace("mosty", "mostly").replace("mostlly", "mostly")
        clean_resp_q = re.sub(r'most(?!ly)', 'mostly', clean_resp_q)
        
        found_matches_q = []
        for idx, opt in enumerate(options):
            c_opt = str(opt).lower().replace(" ", "")
            if not c_opt: continue
            
            # Check fixed string
            pos = clean_resp_q.find(c_opt)
            if pos != -1:
                found_matches_q.append((pos, len(c_opt), str(idx + 1)))

        if found_matches_q:
            # Same sorting logic: Earliest and Longest
            found_matches_q.sort(key=lambda x: (x[0], -x[1]))
            return found_matches_q[0][2]

        return 'unusable'

    df['clean_response'] = df.apply(_clean_row, axis=1)

    return df

results_long = process_responses_bm3(results_long)
results_long = create_numerical_value(results_long)

results_long.to_csv(os.path.join(script_dir, "data", "clean", "res3_clean_bm.csv"), index=False)


### -------------- Prompt 4 (Reworded Answer Options) -------------- ###
# Prompt used:
"""
Question: How much, if at all, do you think the ease with which people can legally obtain guns contributes to gun violence in the country today?

Options:
1) A lot
2) A fair amount
3) Not too much
4) None at all
5) Don't know
6) Refused

The best answer is option number:
"""

results = pd.read_csv(os.path.join(script_dir, "data", "raw", "res4_raw_base-models.csv"), delimiter=',')

# Convert to long format
model_columns = ['Llama-3.1-8B', 'Mistral-7B-v0.3', 'Qwen2.5-7B', 'gemma-2-9b']
results_long = results.melt(
    id_vars=[col for col in results.columns if col not in model_columns],
    value_vars=model_columns,
    var_name='model',
    value_name='response'
)

# Replace \n and \r (including multiples) with a single space
results_long['response'] = (results_long['response']
                  .astype('string')
                  .str.replace(r'[\r\n]+', ' ', regex=True)
                  .str.replace(r'\s+', ' ', regex=True)
                  .str.strip())

def process_responses_bm4(df):
    """
    Cleans response column to extract a single numeric answer.
    
    Methodology:
    1. Standard Priorities (1-5).
    2. Fallback (Unique Number Check).
    3. Edge Case 1: Text Matching (Start of string only).
    4. Edge Case 2: "correct/best answer is option number..."
    5. Edge Case 3: "correct/best answer is..."
    6. Edge Case 4: Leading Digit (Starts with digit, but only after all previous cleaning to not match incorrect ones).
    """
    df = df.copy()

    def _clean_row(row):
        text = str(row['response'])
        n_opts = row['n_options']
        
        # --- 1. Standard Priorities ---
        # Priority 1: "Answer: <digit>"
        match_ans = re.search(r'Answer:\s*(\d+)', text, re.IGNORECASE)
        if match_ans:
            val = int(match_ans.group(1))
            if 1 <= val <= n_opts: return str(val)

        # Priority 2: "<digit> Answer:" (Only for Qwen2.5-7B)
        if row.get('model') == 'Qwen2.5-7B':
            match_ans2 = re.search(r'(\d+)\sAnswer:', text, re.IGNORECASE)
            if match_ans2:
                val = int(match_ans2.group(1))
                if 1 <= val <= n_opts: return str(val)
        
        # Priority 3: "<digit> Explanation"
        match_expl = re.search(r'(\d+)\.?\s+Ex', text, re.IGNORECASE)
        if match_expl:
            val = int(match_expl.group(1))
            if 1 <= val <= n_opts: return str(val)

        # Priority 4: "<digit> Question"
        match_quest = re.search(r'(\d+)\s+Question', text, re.IGNORECASE)
        if match_quest:
            val = int(match_quest.group(1))
            if 1 <= val <= n_opts: return str(val)

        # Priority 5: "X)" at start
        match_paren = re.search(r'^(\d+)\)', text)
        if match_paren:
            val = int(match_paren.group(1))
            if 1 <= val <= n_opts: return str(val)

        # --- 2. Fallback (Unique Number Check) ---
        all_numbers = re.findall(r'\d+', text)
        valid_candidates = {int(x) for x in all_numbers if 1 <= int(x) <= n_opts}
        
        # If exactly one unique valid number was found, return it.
        if len(valid_candidates) == 1:
            return str(list(valid_candidates)[0])
        
        # --- 3. Edge Cases ---
        # Edge Case 1: Text matching (Start of response ONLY)
        try:
            options = ast.literal_eval(row['answer_options'])
            # Compress response
            clean_resp = text.lower().replace(" ", "")
            
            # Prepare list: (clean_text, number_string)
            check_list = []
            for idx, opt in enumerate(options):
                c_opt = str(opt).lower().replace(" ", "")
                if c_opt:
                    check_list.append((c_opt, str(idx + 1)))

            # Sort by Length Descending (Match longest option first)
            check_list.sort(key=lambda x: len(x[0]), reverse=True)

            # Check for match at beginning
            for c_opt, num_str in check_list:
                if clean_resp.startswith(c_opt):
                    if 1 <= int(num_str) <= n_opts:
                        return num_str
        except:
            pass 

        # Helper to process Letter or Digit
        def parse_val(candidate):
            if candidate.isdigit():
                return int(candidate)
            elif len(candidate) == 1 and candidate.isalpha():
                return ord(candidate.upper()) - 64
            return -1

        # Edge Case 2: "correct/best answer is option number..."
        match_ec1 = re.search(r'(?:correct|best)\s+answer is option number[:\s]*([A-Z]|\d+)', text, re.IGNORECASE)
        if match_ec1:
            val = parse_val(match_ec1.group(1))
            if 1 <= val <= n_opts:
                return str(val)

        # Edge Case 3: "answer is..."
        match_ec2 = re.search(r'answer is[:\s]*([A-Z]|\d+)', text, re.IGNORECASE)
        if match_ec2:
            val = parse_val(match_ec2.group(1))
            if 1 <= val <= n_opts:
                return str(val)
            
        # Edge Case 4 (LD): Leading Digit (Matches a digit at the very start)
        match_lead = re.search(r'^\s*(\d+)', text)
        if match_lead:
            val = int(match_lead.group(1))
            if 1 <= val <= n_opts:
                return str(val)

        return 'unusable'

    # Apply the logic row-wise
    df['clean_response'] = df.apply(_clean_row, axis=1)
    
    return df

results_long = process_responses_bm4(results_long)
results_long = create_numerical_value(results_long)

results_long.to_csv(os.path.join(script_dir, "data", "clean", "res4_clean_bm.csv"), index=False)



###############################################################
###        PART 2: PROCESSING SHUFFLED RESULTS              ###
###############################################################

### -------------- Prompt 1 (Shuffled Answer Options) -------------- ###
# Same Prompt structure as Prompt 1 for Reworded Answer Options
results_shuff = pd.read_csv(os.path.join(script_dir, "data", "raw", "res1_shuff_raw_bm.csv"), delimiter=',')

# Split scale_type into n_options and polarity
parts_shuff = results_shuff['scale_type'].str.extract(r'^(\d+)\s*-\s*([A-Za-z]+)\s*$')
results_shuff['n_options'] = pd.to_numeric(parts_shuff[0], errors='coerce').astype('Int64')
results_shuff['polarity']  = parts_shuff[1].str.lower()
results_shuff = results_shuff.drop(columns=['scale_type'])

# Convert to long format
model_columns_shuff = ['Llama-3.1-8B', 'Mistral-7B-v0.3', 'Qwen2.5-7B', 'gemma-2-9b']
results_long_shuff = results_shuff.melt(
    id_vars=[col for col in results_shuff.columns if col not in model_columns_shuff],
    value_vars=model_columns_shuff,
    var_name='model',
    value_name='response'
)

# Cleanup strings
results_long_shuff['response'] = (results_long_shuff['response']
                  .astype('string')
                  .str.replace(r'[\r\n]+', ' ', regex=True)
                  .str.replace(r'\s+', ' ', regex=True)
                  .str.strip())

# Reuse process_responses1 from Part with Reworded Answer Options
results_long_shuff = process_responses_bm1(results_long_shuff)
results_long_shuff = create_numerical_value(results_long_shuff)

results_long_shuff.to_csv(os.path.join(script_dir, "data", "clean", "res1_shuff_clean_bm.csv"), index=False)


### -------------- Prompt 2 (Shuffled Answer Options) -------------- ###
# Same Prompt structure as Prompt 2 for Reworded Answer Options
results_shuff = pd.read_csv(os.path.join(script_dir, "data", "raw", "res2_shuff_raw_bm.csv"), delimiter=',')

# Split scale_type into n_options and polarity
parts_shuff = results_shuff['scale_type'].str.extract(r'^(\d+)\s*-\s*([A-Za-z]+)\s*$')
results_shuff['n_options'] = pd.to_numeric(parts_shuff[0], errors='coerce').astype('Int64')
results_shuff['polarity']  = parts_shuff[1].str.lower()
results_shuff = results_shuff.drop(columns=['scale_type'])

# Convert to long format
model_columns_shuff = ['Llama-3.1-8B', 'Mistral-7B-v0.3', 'Qwen2.5-7B', 'gemma-2-9b']
results_long_shuff = results_shuff.melt(
    id_vars=[col for col in results_shuff.columns if col not in model_columns_shuff],
    value_vars=model_columns_shuff,
    var_name='model',
    value_name='response'
)

# Cleanup strings
results_long_shuff['response'] = (results_long_shuff['response']
                  .astype('string')
                  .str.replace(r'[\r\n]+', ' ', regex=True)
                  .str.replace(r'\s+', ' ', regex=True)
                  .str.strip())

# Reuse process_responses1 from Part with Reworded Answer Options
results_long_shuff = process_responses_bm2(results_long_shuff)

# Convert letters to numbers (A=1, B=2, etc.) to match the other clean results
results_long_shuff['clean_response'] = results_long_shuff['clean_response'].apply(
    lambda x: str(ord(x) - 64) if x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' else x
)

results_long_shuff = create_numerical_value(results_long_shuff)

results_long_shuff.to_csv(os.path.join(script_dir, "data", "clean", "res2_shuff_clean_bm.csv"), index=False)


### -------------- Prompt 3 (Shuffled Answer Options) -------------- ###
# Same Prompt structure as Prompt 3 for Reworded Answer Options
results_shuff = pd.read_csv(os.path.join(script_dir, "data", "raw", "res3_shuff_raw_bm.csv"), delimiter=',')

# Split scale_type into n_options and polarity
parts_shuff = results_shuff['scale_type'].str.extract(r'^(\d+)\s*-\s*([A-Za-z]+)\s*$')
results_shuff['n_options'] = pd.to_numeric(parts_shuff[0], errors='coerce').astype('Int64')
results_shuff['polarity']  = parts_shuff[1].str.lower()
results_shuff = results_shuff.drop(columns=['scale_type'])

# Convert to long format
model_columns_shuff = ['Llama-3.1-8B', 'Mistral-7B-v0.3', 'Qwen2.5-7B', 'gemma-2-9b']
results_long_shuff = results_shuff.melt(
    id_vars=[col for col in results_shuff.columns if col not in model_columns_shuff],
    value_vars=model_columns_shuff,
    var_name='model',
    value_name='response'
)

# Cleanup strings
results_long_shuff['response'] = (results_long_shuff['response']
                  .astype('string')
                  .str.replace(r'[\r\n]+', ' ', regex=True)
                  .str.replace(r'\s+', ' ', regex=True)
                  .str.strip())

# Reuse process_responses1 from Part with Reworded Answer Options
results_long_shuff = process_responses_bm3(results_long_shuff)
results_long_shuff = create_numerical_value(results_long_shuff)

results_long_shuff.to_csv(os.path.join(script_dir, "data", "clean", "res3_shuff_clean_bm.csv"), index=False)


### -------------- Prompt 4 (Shuffled Answer Options) -------------- ###
# Same Prompt structure as Prompt 4 for Reworded Answer Options
results_shuff = pd.read_csv(os.path.join(script_dir, "data", "raw", "res4_shuff_raw_bm.csv"), delimiter=',')

# Convert to long format
model_columns_shuff = ['Llama-3.1-8B', 'Mistral-7B-v0.3', 'Qwen2.5-7B', 'gemma-2-9b']
results_long_shuff = results_shuff.melt(
    id_vars=[col for col in results_shuff.columns if col not in model_columns_shuff],
    value_vars=model_columns_shuff,
    var_name='model',
    value_name='response'
)

# Cleanup strings
results_long_shuff['response'] = (results_long_shuff['response']
                  .astype('string')
                  .str.replace(r'[\r\n]+', ' ', regex=True)
                  .str.replace(r'\s+', ' ', regex=True)
                  .str.strip())

# Reuse process_responses1 from Part with Reworded Answer Options
results_long_shuff = process_responses_bm4(results_long_shuff)
results_long_shuff = create_numerical_value(results_long_shuff)

results_long_shuff.to_csv(os.path.join(script_dir, "data", "clean", "res4_shuff_clean_bm.csv"), index=False)
