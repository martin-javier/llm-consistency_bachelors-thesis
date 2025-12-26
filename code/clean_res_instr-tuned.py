# Clean results from run_questions.py.
# Multiple choice questions with 5 variations of answer options (that mean the same) asked to 4 instruction tuned llms with 4 different promts
# AND
# Clean results from run_quest_shuffled.py.
# Multiple choice questions with 6 variations of shuffled answer options asked to 4 instruction tuned llms with 4 different promts

# Models used (instruction tuned): ['Llama-3.1-8B-Instruct', 'gemma-2-9b-it', 'Mistral-7B-Instruct-v0.3', 'Qwen2.5-7B-Instruct']


import pandas as pd
import ast
import os
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
You are taking a survey. There are no right or wrong answers.
Please choose the one option that best fits your view. The order of the options does not imply any ranking.
Output format: Provide exactly one number only (no words, no punctuation, no quotes).

Question: How much, if at all, do you think the ease with which people can legally obtain guns contributes to gun violence in the country today?

Options:
1) A lot
2) A fair amount
3) Not too much
4) None at all

Please answer with ONLY one of the answer options given above, and with only the number corresponding to the option you choose.
"""

raw_res_dir = os.path.join(script_dir, "data", "raw", "results1_raw.csv")
results = pd.read_csv(raw_res_dir, delimiter=',')

# Split scale_type into n_options and polarity (Extract digits before '-' and letters after '-')
parts = results['scale_type'].str.extract(r'^(\d+)\s*-\s*([A-Za-z]+)\s*$')
results['n_options'] = pd.to_numeric(parts[0], errors='coerce').astype('Int64')
results['polarity']  = parts[1].str.lower()
results = results.drop(columns=['scale_type'])

# Convert to long format
model_columns = ['Llama-3.1-8B-Instruct', 'gemma-2-9b-it', 'Mistral-7B-Instruct-v0.3', 'Qwen2.5-7B-Instruct']
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

def process_responses1(df):
    """
    Cleans response column to extract a single numeric answer.
    
    Methodology:
    1. Extraction: Grabs the leading integer from the response.
    2. Logic Check A (Range): Is the number > n_options?
    3. Logic Check B (Ambiguity): Are there multiple unique numbers (e.g., "1 and 2")?
    4. Returns: The valid string number or 'unusable'.
    """
    df = df.copy()

    # Extract the leading number
    extracted = df['response'].astype(str).str.extract(r'^(\d+)', expand=False)

    # Convert to numeric for comparison
    numeric_vals = pd.to_numeric(extracted, errors='coerce')

    # Logic Check A: Value is strictly larger than allowed options
    out_of_bounds = numeric_vals > df['n_options']
    
    # Logic Check B: Response contains more than 1 unique number (e.g. "1 2")
    ambiguous = df['response'].astype(str).str.findall(r'\d+').apply(
        lambda x: len(set(x)) > 1 if isinstance(x, list) else False
    )

    # Apply Filters & Fill Missing
    extracted.loc[out_of_bounds | ambiguous] = 'unusable'
    df['clean_response'] = extracted.fillna('unusable')

    return df

results_long = process_responses1(results_long)
results_long = create_numerical_value(results_long)

clean_res_dir = os.path.join(script_dir, "data", "clean", "results1_clean.csv")
results_long.to_csv(clean_res_dir, index=False)


### -------------- Prompt 2 (Reworded Answer Options) -------------- ###
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

raw_res_dir = os.path.join(script_dir, "data", "raw", "results2_raw.csv")
results = pd.read_csv(raw_res_dir, delimiter=',')

# Split scale_type into n_options and polarity (Extract digits before '-' and letters after '-')
parts = results['scale_type'].str.extract(r'^(\d+)\s*-\s*([A-Za-z]+)\s*$')
results['n_options'] = pd.to_numeric(parts[0], errors='coerce').astype('Int64')
results['polarity']  = parts[1].str.lower()
results = results.drop(columns=['scale_type'])

# Convert to long format
model_columns = ['Llama-3.1-8B-Instruct', 'gemma-2-9b-it', 'Mistral-7B-Instruct-v0.3', 'Qwen2.5-7B-Instruct']
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

def process_responses2(df):
    """
    Cleans response column to extract a single letter answer.
    
    Methodology:
    1. Primary Pass: Checks start of string for a letter.
    2. Validation: Checks if letter is within 'n_options' and not ambiguous (multiple unique letters).
    3. Rescue Pass: If unusable, checks end of string for a standalone letter (A-F).
    """
    df = df.copy()

    # Extract leading letter (e.g., "A) text" -> "A")
    extracted = df['response'].astype(str).str.extract(r'^([A-Za-z])', expand=False).str.upper()

    # Logic Check 1: Range (Is A (1) > n_options?)
    numeric_vals = extracted.apply(lambda x: ord(x) - 64 if pd.notnull(x) else 0)
    
    # Logic Check 2: Ambiguity (Are there different letters B-H & J-Z involved?)
    ambiguous_mask = df['response'].astype(str).str.findall(r'\b([B-HJ-Z])\b').apply(
        lambda x: len(set(x)) > 1 if isinstance(x, list) else False
    )

    # Apply 'unusable' to invalid entries and fill missing matches
    extracted.loc[(numeric_vals > df['n_options']) | ambiguous_mask] = 'unusable'
    df['clean_response'] = extracted.fillna('unusable')

    unusable_mask = df['clean_response'] == 'unusable'
    
    if unusable_mask.any():
        # Regex: Word boundary + A-F + optional punctuation + End of string ($)
        rescue = df.loc[unusable_mask, 'response'].astype(str).str.extract(r'\b([A-F])\W*$', expand=False).str.upper()
        
        # Validation: Check if rescue letter is within range
        rescue_numeric = rescue.apply(lambda x: ord(x) - 64 if pd.notnull(x) else 0)
        valid_rescue = (rescue.notnull()) & (rescue_numeric <= df.loc[unusable_mask, 'n_options'])
        
        # Update valid rescues in main dataframe
        df.loc[rescue[valid_rescue].index, 'clean_response'] = rescue[valid_rescue]

    return df


results_long = process_responses2(results_long)

# Convert letters to numbers (A=1, B=2, etc.) to match the other clean results
results_long['clean_response'] = results_long['clean_response'].apply(
    lambda x: str(ord(x) - 64) if x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' else x
)

results_long = create_numerical_value(results_long)

clean_res_dir = os.path.join(script_dir, "data", "clean", "results2_clean.csv")
results_long.to_csv(clean_res_dir, index=False)


### -------------- Prompt 3 (Reworded Answer Options) -------------- ###
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


raw_res_dir = os.path.join(script_dir, "data", "raw", "results3_raw.csv")
results = pd.read_csv(raw_res_dir, delimiter=',')

# Split scale_type into n_options and polarity (Extract digits before '-' and letters after '-')
parts = results['scale_type'].str.extract(r'^(\d+)\s*-\s*([A-Za-z]+)\s*$')
results['n_options'] = pd.to_numeric(parts[0], errors='coerce').astype('Int64')
results['polarity']  = parts[1].str.lower()
results = results.drop(columns=['scale_type'])

# Convert to long format
model_columns = ['Llama-3.1-8B-Instruct', 'gemma-2-9b-it', 'Mistral-7B-Instruct-v0.3', 'Qwen2.5-7B-Instruct']
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

def process_responses3(df):
    """
    Matches response text against 'answer_options' using a multi-stage rescue strategy.
    
    Methodology:
    1. Normalization: Compresses text (removes spaces/underscores) to handle "Afair amount".
    2. Heuristic: Sorts options by length (longest first) to prevent partial matches.
    3. Stage 1: Checks original response.
    4. Stage 2 (Rescue): If 0 matches, prepends "a", "an", then "with" to handle missing stopwords.
    5. Returns: Number (Index+1) or 'unusable'.
    """
    df = df.copy()

    def solve_row(row):
        # Setup
        raw_resp = str(row['response']).lower()
        # Compress response: "small amount" -> "smallamount"
        base_clean_resp = raw_resp.replace(" ", "").replace("_", "")
        
        try:
            options = ast.literal_eval(row['answer_options'])
        except:
            return 'unusable'

        # Sort options by length of compressed string (descending)
        options_sorted = sorted(enumerate(options), key=lambda x: -len(str(x[1] or "").replace(" ", "")))

        # helper: Core Matching Logic
        def find_match_in_string(target_string):
            """Returns: Int (Match), 'ambiguous' (Too many matches), or None (No match)"""
            found_matches = []      
            matched_number = None   

            for idx, opt_text in options_sorted:
                original_text = str(opt_text).lower()
                if not original_text: continue

                # Compress the option: "A small amount" -> "asmallamount"
                comp_opt = original_text.replace(" ", "").replace("_", "")

                if comp_opt in target_string:
                    # Substring check: Ignore "Bad" if we already found "Very Bad"
                    is_substring = any(comp_opt in prev for prev in found_matches)
                    
                    if not is_substring:
                        found_matches.append(comp_opt)
                        matched_number = idx + 1
                        
                        # Stop immediately if we find conflicting answers
                        if len(found_matches) > 1:
                            return 'ambiguous'
            
            if len(found_matches) == 1:
                return matched_number
            return None
        
        # 1. Check Original
        res = find_match_in_string(base_clean_resp)
        if res == 'ambiguous': return 'unusable'
        if res is not None:    return res

        # 2. Rescue Attempts/Prefixes (only if 1. returned None, meaning 0 matches)
        for prefix in ['a', 'an', 'with']:
            res = find_match_in_string(prefix + base_clean_resp)
            
            if res == 'ambiguous': return 'unusable'
            if res is not None:    return res

        # If everything fails has to be unusable
        return 'unusable'

    # Apply to df
    df['clean_response'] = df.apply(solve_row, axis=1)

    return df

results_long = process_responses3(results_long)
results_long = create_numerical_value(results_long)

clean_res_dir = os.path.join(script_dir, "data", "clean", "results3_clean.csv")
results_long.to_csv(clean_res_dir, index=False)


### -------------- Prompt 4 (Reworded Answer Options) -------------- ###
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

raw_res_dir = os.path.join(script_dir, "data", "raw", "results4_raw.csv")
results = pd.read_csv(raw_res_dir, delimiter=',')

# Convert to long format
model_columns = ['Llama-3.1-8B-Instruct', 'gemma-2-9b-it', 'Mistral-7B-Instruct-v0.3', 'Qwen2.5-7B-Instruct']
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

results_long = process_responses1(results_long)
results_long = create_numerical_value(results_long)

clean_res_dir = os.path.join(script_dir, "data", "clean", "results4_clean.csv")
results_long.to_csv(clean_res_dir, index=False)



###############################################################
###        PART 2: PROCESSING SHUFFLED RESULTS              ###
###############################################################

### -------------- Prompt 1 (Shuffled Answer Options) -------------- ###
# Same Prompt structure as Prompt 1 for Reworded Answer Options
raw_res_dir_shuff = os.path.join(script_dir, "data", "raw", "res1_shuffled_raw.csv")
results_shuff = pd.read_csv(raw_res_dir_shuff, delimiter=',')

# Split scale_type into n_options and polarity
parts_shuff = results_shuff['scale_type'].str.extract(r'^(\d+)\s*-\s*([A-Za-z]+)\s*$')
results_shuff['n_options'] = pd.to_numeric(parts_shuff[0], errors='coerce').astype('Int64')
results_shuff['polarity']  = parts_shuff[1].str.lower()
results_shuff = results_shuff.drop(columns=['scale_type'])

# Convert to long format
model_columns_shuff = ['Llama-3.1-8B-Instruct', 'gemma-2-9b-it', 'Mistral-7B-Instruct-v0.3', 'Qwen2.5-7B-Instruct']
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
results_long_shuff = process_responses1(results_long_shuff)
results_long_shuff = create_numerical_value(results_long_shuff)

clean_res_dir_shuff = os.path.join(script_dir, "data", "clean", "res1_shuffled_clean.csv")
results_long_shuff.to_csv(clean_res_dir_shuff, index=False)


### -------------- Prompt 2 (Shuffled Answer Options) -------------- ###
# Same Prompt structure as Prompt 2 for Reworded Answer Options
raw_res_dir_shuff = os.path.join(script_dir, "data", "raw", "res2_shuffled_raw.csv")
results_shuff = pd.read_csv(raw_res_dir_shuff, delimiter=',')

parts_shuff = results_shuff['scale_type'].str.extract(r'^(\d+)\s*-\s*([A-Za-z]+)\s*$')
results_shuff['n_options'] = pd.to_numeric(parts_shuff[0], errors='coerce').astype('Int64')
results_shuff['polarity']  = parts_shuff[1].str.lower()
results_shuff = results_shuff.drop(columns=['scale_type'])

results_long_shuff = results_shuff.melt(
    id_vars=[col for col in results_shuff.columns if col not in model_columns_shuff],
    value_vars=model_columns_shuff,
    var_name='model',
    value_name='response'
)

results_long_shuff['response'] = (results_long_shuff['response']
                  .astype('string')
                  .str.replace(r'[\r\n]+', ' ', regex=True)
                  .str.replace(r'\s+', ' ', regex=True)
                  .str.strip())

# Reuse process_responses2 from Part with Reworded Answer Options
results_long_shuff = process_responses2(results_long_shuff)

# Convert letters to numbers
results_long_shuff['clean_response'] = results_long_shuff['clean_response'].apply(
    lambda x: str(ord(x) - 64) if x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' else x
)

results_long_shuff = create_numerical_value(results_long_shuff)

clean_res_dir_shuff = os.path.join(script_dir, "data", "clean", "res2_shuffled_clean.csv")
results_long_shuff.to_csv(clean_res_dir_shuff, index=False)


### -------------- Prompt 3 (Shuffled Answer Options) -------------- ###
# Same Prompt structure as Prompt 3 for Reworded Answer Options
raw_res_dir_shuff = os.path.join(script_dir, "data", "raw", "res3_shuffled_raw.csv")
results_shuff = pd.read_csv(raw_res_dir_shuff, delimiter=',')

parts_shuff = results_shuff['scale_type'].str.extract(r'^(\d+)\s*-\s*([A-Za-z]+)\s*$')
results_shuff['n_options'] = pd.to_numeric(parts_shuff[0], errors='coerce').astype('Int64')
results_shuff['polarity']  = parts_shuff[1].str.lower()
results_shuff = results_shuff.drop(columns=['scale_type'])

results_long_shuff = results_shuff.melt(
    id_vars=[col for col in results_shuff.columns if col not in model_columns_shuff],
    value_vars=model_columns_shuff,
    var_name='model',
    value_name='response'
)

results_long_shuff['response'] = (results_long_shuff['response']
                  .astype('string')
                  .str.replace(r'[\r\n]+', ' ', regex=True)
                  .str.replace(r'\s+', ' ', regex=True)
                  .str.strip())

# Reuse process_responses3 from Part with Reworded Answer Options
results_long_shuff = process_responses3(results_long_shuff)
results_long_shuff = create_numerical_value(results_long_shuff)

clean_res_dir_shuff = os.path.join(script_dir, "data", "clean", "res3_shuffled_clean.csv")
results_long_shuff.to_csv(clean_res_dir_shuff, index=False)


### -------------- Prompt 4 (Shuffled Answer Options) -------------- ###
# Same Prompt structure as Prompt 4 for Reworded Answer Options
raw_res_dir_shuff = os.path.join(script_dir, "data", "raw", "res4_shuffled_raw.csv")
results_shuff = pd.read_csv(raw_res_dir_shuff, delimiter=',')

results_long_shuff = results_shuff.melt(
    id_vars=[col for col in results_shuff.columns if col not in model_columns_shuff],
    value_vars=model_columns_shuff,
    var_name='model',
    value_name='response'
)

results_long_shuff['response'] = (results_long_shuff['response']
                  .astype('string')
                  .str.replace(r'[\r\n]+', ' ', regex=True)
                  .str.replace(r'\s+', ' ', regex=True)
                  .str.strip())

# Reuse process_responses1 from Part with Reworded Answer Options
results_long_shuff = process_responses1(results_long_shuff)
results_long_shuff = create_numerical_value(results_long_shuff)

clean_res_dir_shuff = os.path.join(script_dir, "data", "clean", "res4_shuffled_clean.csv")
results_long_shuff.to_csv(clean_res_dir_shuff, index=False)
