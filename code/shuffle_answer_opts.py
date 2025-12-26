# Creates a dataset of questions with shuffled answer options
# Takes the first variations of answer options (the original one or closest to original if it was reworded) and creates 5 randomly shuffled variations
# Result is 6 rows per question shuffeling indicated by answer_var_id col. In first row the original options in original order followed by 5 different shuffled variations 

import pandas as pd
import ast
import random
from copy import deepcopy
from typing import List
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


# ---------- helper functions ----------
def _permute_list_by_seed(lst: List, seed: int):
    """Return a new list which is a permutation of lst determined by seed.
       Also returns the permutation indices used so you can apply same perm to other lists."""
    if lst is None:
        return None, None
    n = len(lst)
    indices = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(indices)
    permuted = [lst[i] for i in indices]
    return permuted, indices

def permute_parallel(lists: List[List], seed: int):
    """
    Permute multiple lists in the same way using `seed`.
    lists: list of lists with same length. Returns list of permuted lists.
    """
    if not lists:
        return []
    base = lists[0]
    _, indices = _permute_list_by_seed(base, seed)
    if indices is None:
        return [None for _ in lists]
    out = []
    for lst in lists:
        out.append([lst[i] for i in indices])
    return out


# ---------- data prep ----------
df = pd.read_csv(os.path.join(script_dir, "data", "clean", "opinionQA_questions_final.csv"), delimiter=',')

SEEDS = [2, 7, 10, 67, 101]

def parse_list_column(x):
    if pd.isna(x):
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return ast.literal_eval(x)

df['answer_options_list'] = df['answer_options'].apply(parse_list_column)
df['num_scale_list'] = df['num_scale'].apply(parse_list_column)

# ---------- Build shuffled variants ----------
rows = []
# iterate over question_id groups but only use the row with answer_var_id == 1 as base
for qid, group in df.groupby('question_id', dropna=False):
    # use the row with answer_var_id == 1 as base
    base_rows = group[group['answer_var_id'] == 1]

    base_row = base_rows.iloc[0]

    # Extract required fields
    question = base_row['question']
    answer_options = deepcopy(base_row['answer_options_list'])
    num_scale = deepcopy(base_row['num_scale_list'])

    # Build the original (variant 1) row
    r = base_row.to_dict()
    r['answer_options'] = answer_options  # keep as list (will be serialized when saving)
    r['num_scale'] = num_scale
    r['answer_var_id'] = 1
    r['shuffle_seed'] = pd.NA
    rows.append(r)

    # Create 5 shuffled variants
    for vi, seed in enumerate(SEEDS, start=2):
        permuted_lists = permute_parallel([answer_options, num_scale], seed)
        
        permuted_answer_options = permuted_lists[0]
        permuted_num_scale = permuted_lists[1]
        
        r2 = base_row.to_dict()
        r2['answer_options'] = permuted_answer_options
        r2['num_scale'] = permuted_num_scale
        r2['shuffle_seed'] = seed
        r2['answer_var_id'] = vi 
        rows.append(r2)

# assemble DataFrame
shuffled_df = pd.DataFrame(rows)

# Reorder columns: keep original columns and add seed
cols_out = list(df.columns.drop(['answer_options_list','num_scale_list'])) + ['shuffle_seed']

# Avoid duplicates and keep columns present
cols_out = [c for c in cols_out if c in shuffled_df.columns]
shuffled_df = shuffled_df[cols_out]

shuffled_df.to_csv(os.path.join(script_dir, "data", "clean", "opQA_shuffled_ans-opt.csv"), index=False)
