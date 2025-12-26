# Analysis of the results from run_questions.py and run_quest_shuffled.py.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import ast
from collections import Counter
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
from itertools import combinations
import matplotlib.patches as mpatches
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


### ---------------------------------------------------------------------------
### Data Loading and Merging ### ### ###
### ---------------------------------------------------------------------------

# Load data from reworded results (5 diff worded answer options that mean the same)
# Instruction tuned models
res1_reworded_it = pd.read_csv(os.path.join(script_dir, "data", "clean", "results1_clean.csv"), delimiter=',')
res2_reworded_it = pd.read_csv(os.path.join(script_dir, "data", "clean", "results2_clean.csv"), delimiter=',')
res3_reworded_it = pd.read_csv(os.path.join(script_dir, "data", "clean", "results3_clean.csv"), delimiter=',')
res4_reworded_it = pd.read_csv(os.path.join(script_dir, "data", "clean", "results4_clean.csv"), delimiter=',')
# Base models
res1_reworded_bm = pd.read_csv(os.path.join(script_dir, "data", "clean", "res1_clean_bm.csv"), delimiter=',')
res2_reworded_bm = pd.read_csv(os.path.join(script_dir, "data", "clean", "res2_clean_bm.csv"), delimiter=',')
res3_reworded_bm = pd.read_csv(os.path.join(script_dir, "data", "clean", "res3_clean_bm.csv"), delimiter=',')
res4_reworded_bm = pd.read_csv(os.path.join(script_dir, "data", "clean", "res4_clean_bm.csv"), delimiter=',')

# Load data from shuffled results (6 diff shuffled answer options taken from the 1st answer variation -> original or closest to original)
# Instruction tuned models
res1_shuffled_it = pd.read_csv(os.path.join(script_dir, "data", "clean", "res1_shuffled_clean.csv"), delimiter=',')
res2_shuffled_it = pd.read_csv(os.path.join(script_dir, "data", "clean", "res2_shuffled_clean.csv"), delimiter=',')
res3_shuffled_it = pd.read_csv(os.path.join(script_dir, "data", "clean", "res3_shuffled_clean.csv"), delimiter=',')
res4_shuffled_it = pd.read_csv(os.path.join(script_dir, "data", "clean", "res4_shuffled_clean.csv"), delimiter=',')
# Base models
res1_shuffled_bm = pd.read_csv(os.path.join(script_dir, "data", "clean", "res1_shuff_clean_bm.csv"), delimiter=',')
res2_shuffled_bm = pd.read_csv(os.path.join(script_dir, "data", "clean", "res2_shuff_clean_bm.csv"), delimiter=',')
res3_shuffled_bm = pd.read_csv(os.path.join(script_dir, "data", "clean", "res3_shuff_clean_bm.csv"), delimiter=',')
res4_shuffled_bm = pd.read_csv(os.path.join(script_dir, "data", "clean", "res4_shuff_clean_bm.csv"), delimiter=',')

res1_shuffled_it = res1_shuffled_it.drop(columns=['shuffle_seed'])
res2_shuffled_it = res2_shuffled_it.drop(columns=['shuffle_seed'])
res3_shuffled_it = res3_shuffled_it.drop(columns=['shuffle_seed'])
res4_shuffled_it = res4_shuffled_it.drop(columns=['shuffle_seed'])
res1_shuffled_bm = res1_shuffled_bm.drop(columns=['shuffle_seed'])
res2_shuffled_bm = res2_shuffled_bm.drop(columns=['shuffle_seed'])
res3_shuffled_bm = res3_shuffled_bm.drop(columns=['shuffle_seed'])
res4_shuffled_bm = res4_shuffled_bm.drop(columns=['shuffle_seed'])

def process_and_merge(r1, r2, r3, r4):
    """
    Renames, merges, and cleans a set of 4 result dataframes.
    """
    # 1. Rename columns
    r1_renamed = r1.rename(columns={
        'response': 'response1',
        'clean_response': 'clean_response1', 
        'change': 'change1',
        'num_value': 'num_value1'
    })

    r2_renamed = r2.rename(columns={
        'response': 'response2',
        'clean_response': 'clean_response2',
        'change': 'change2', 
        'num_value': 'num_value2'
    })

    r3_renamed = r3.rename(columns={
        'response': 'response3',
        'clean_response': 'clean_response3',
        'change': 'change3', 
        'num_value': 'num_value3'
    })

    # Note: Prompt 4 has specific metadata columns we need to suffix to avoid collisions
    r4_renamed = r4.rename(columns={
        'answer_options': 'answer_options4',
        'num_scale': 'num_scale4', 
        'n_options': 'n_options4',
        'response': 'response4',
        'clean_response': 'clean_response4',
        'change': 'change4', 
        'num_value': 'num_value4'
    })

    # 2. Merge Datasets
    # Common columns for Prompts 1, 2, 3
    common_cols = ['question_id', 'question_var_id', 'answer_var_id', 'question', 
                   'answer_options', 'num_scale', 'question_type', 'subject', 
                   'source', 'n_options', 'polarity', 'model']

    # Merge 1 -> 2 -> 3
    merged = pd.merge(r1_renamed, r2_renamed, on=common_cols, how='inner', validate='one_to_one')
    merged = pd.merge(merged, r3_renamed, on=common_cols, how='inner', validate='one_to_one')

    # Merge 4
    # Result 4 has different answer_options, num_scale, n_options -> exclude them from join keys
    common_cols_result4 = [col for col in common_cols if col not in ['answer_options', 'num_scale', 'n_options']]
    merged = pd.merge(merged, r4_renamed, on=common_cols_result4, how='inner', validate='one_to_one')

    # 3. Clean 'clean_response' columns (Standardize to strings)
    def standardize_clean_response(series):
        # Convert all numeric responses to strings, keep 'unusable' as string
        return series.apply(lambda x: str(int(x)) if isinstance(x, (int, float)) and x != 'unusable' else str(x))
    
    for i in range(1, 5):
        merged[f'clean_response{i}'] = standardize_clean_response(merged[f'clean_response{i}'])

    # 4. Convert 'num_value' columns to numeric
    def convert_num_value(series):
        # Convert num_value column to float, handling 'unusable' as NaN
        return pd.to_numeric(series, errors='coerce')

    for i in range(1, 5):
        merged[f'num_value{i}'] = convert_num_value(merged[f'num_value{i}'])
        
    return merged

# Apply to reworded Results
merged_res_reworded_it = process_and_merge(res1_reworded_it, res2_reworded_it, res3_reworded_it, res4_reworded_it)
print(len(merged_res_reworded_it)) #### 24700
merged_res_reworded_it_bm = process_and_merge(res1_reworded_bm, res2_reworded_bm, res3_reworded_bm, res4_reworded_bm)
print(len(merged_res_reworded_it_bm)) #### 24700

# Apply to Shuffled Results
merged_res_shuffled_it = process_and_merge(res1_shuffled_it, res2_shuffled_it, res3_shuffled_it, res4_shuffled_it)
print(len(merged_res_shuffled_it)) #### 29640
merged_res_shuffled_bm = process_and_merge(res1_shuffled_bm, res2_shuffled_bm, res3_shuffled_bm, res4_shuffled_bm)
print(len(merged_res_shuffled_bm)) #### 29640

# Need to realign responses for shuffled datasets (Since shuffled option 1 might be different from the original option 1, then some plots would be incorrect)
def realign_shuffled_responses(df):
    """
    Adjusts clean_response columns for shuffled data to match the Standard scale.
    
    Logic:
    1. Rename 'clean_responseX' -> 'clean_response_selectedX' (position chosen in shuffled list).
    2. Parse 'num_scale' and sort it descending (Standard Order: High -> Low) e.g. [1.0, 0.33, -0.33, -1.0]
    3. Find where 'num_valueX' fits in that sorted list (e.g. -0.33 -> 3rd place)
    4. The index + 1 is the canonical Standard Option Number (in example new clean_response = "3").
    """
    df = df.copy()
    
    for i in range(1, 5):
        clean_col = f'clean_response{i}'
        selected_col = f'clean_response_selected{i}'
        num_val_col = f'num_value{i}'
        
        # Determine which scale column to use (Prompts 1, 2, 3 share 'num_scale'. Prompt 4 has 'num_scale4')
        if i == 4:
            scale_col = 'num_scale4'
        else:
            scale_col = 'num_scale'
        
        # 1. Rename: Save the original selection (position)
        if clean_col in df.columns:
            df = df.rename(columns={clean_col: selected_col})
        
        # 2. Re-Map: Logic function
        def get_canonical_id(row):
            # Get the value we want to find
            val = row[num_val_col]
            
            # If val is NaN (unusable), we can't map it
            if pd.isna(val):
                return 'unusable'
            
            # Parse the scale list from string "[1.0, -0.3, ...]"
            scale_list = ast.literal_eval(row[scale_col])
                
            # Sort descending (Highest -> Lowest) to match Standard Order
            sorted_scale = sorted(scale_list, reverse=True)
                
            # Find the index of our value in this sorted list
            idx = sorted_scale.index(val)
            return str(idx + 1)
        
        # Apply row-wise
        df[clean_col] = df.apply(get_canonical_id, axis=1)
        
    return df

merged_res_shuffled_it = realign_shuffled_responses(merged_res_shuffled_it)
merged_res_shuffled_bm = realign_shuffled_responses(merged_res_shuffled_bm)


### ---------------------------------------------------------------------------
### Analysis and Visualization ### ### ### ### ###
### ---------------------------------------------------------------------------

## -----------------------------------------------------------
## Distribution of clean_responses across all prompts ## ## ##
## -----------------------------------------------------------

def plot_distributions(df, dataset_label, model_type, filename_suffix, mode='absolute', include_title=True):
    """
    Creates a 2x2 grid of barplots (counts or percentages) for the 4 prompts.
    
    Parameters:
    - df: The dataframe containing results.
    - dataset_label: "Reworded" or "Shuffled"
    - model_type: "Instruction-Tuned" or "Base Models"
    - filename_suffix: e.g. "it_rew", "bm_shuff"
    - mode: 'absolute' (counts) or 'percentage' (relative frequency).
    - include_title: Boolean, if False, suppresses the main figure title for paper-ready plots.
    """
    # Desired order: Llama -> Mistral -> Qwen -> Gemma
    it_order = [
        'Llama-3.1-8B-Instruct', 
        'Mistral-7B-Instruct-v0.3', 
        'Qwen2.5-7B-Instruct', 
        'gemma-2-9b-it'
    ]
    
    base_order = [
        'Llama-3.1-8B', 
        'Mistral-7B-v0.3', 
        'Qwen2.5-7B', 
        'gemma-2-9b'
    ]
    
    # Select the correct order based on the model_type string
    if "Base" in model_type:
        model_order = base_order
    else:
        model_order = it_order

    # Font sizes
    FS_MAIN_TITLE = 22
    FS_SUB_TITLE  = 20
    FS_AXIS_LABEL = 18
    FS_TICKS      = 15
    FS_LEG_TITLE  = 18
    FS_LEG_TEXT   = 16
    
    # Construct Dynamic Title
    metric_label = "(Count)" if mode == 'absolute' else "(%)"
    plot_title = f'Distribution of {dataset_label} Responses {metric_label} - {model_type}'
    
    # Construct Filename
    mode_prefix = "abs" if mode == 'absolute' else "perc"
    if include_title:
        filename = f"answer_{mode_prefix}_distr_{filename_suffix}.pdf"
        save_path = os.path.join(script_dir, "plots", "answer_distributions", filename)
    else:
        filename = f"answer_{mode_prefix}_distr_{filename_suffix}_notitle.pdf"
        notitle_dir = os.path.join(script_dir, "plots", "notitle", "answer_distributions")
        # ensure directory exists
        if not os.path.exists(notitle_dir):
            os.makedirs(notitle_dir)
        save_path = os.path.join(notitle_dir, filename)

    prompts = [
        ('clean_response1', 'Prompt 1'),
        ('clean_response2', 'Prompt 2'), 
        ('clean_response3', 'Prompt 3'),
        ('clean_response4', 'Prompt 4')
    ]

    all_possible_responses = ['1', '2', '3', '4', '5', '6', '7', '8', 'unusable']
    
    custom_colors = [
        '#2ca02c',  # response = 1
        '#ff7f0e',  # response = 2
        '#1f77b4',  # response = 3
        '#d62728',  # response = 4
        '#9467bd',  # response = 5
        '#8c564b',  # response = 6
        '#e377c2',  # response = 7
        '#7f7f7f',  # response = 8
        '#bcbd22',  # response = unusable
    ]
    color_mapping = {resp: custom_colors[i] for i, resp in enumerate(all_possible_responses)}

    # --- STEP 1: Set fixed ylims for comparability ---
    if mode == 'absolute':
        y_limit = 5200
    elif mode == 'percentage':
        y_limit = 70

    # --- STEP 2: Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(19, 10))
    
    # Conditionally add title
    if include_title:
        fig.suptitle(plot_title, fontsize=FS_MAIN_TITLE, fontweight='bold', y=0.98)
    
    axes_flat = axes.flatten()

    for idx, (col, title) in enumerate(prompts):
        ax = axes_flat[idx]
        
        # Always use the full list for hue_order to ensure consistent bar slots/colors across all plots
        hue_order = all_possible_responses
        
        if mode == 'absolute':
            sns.countplot(
                data=df,
                x='model',
                hue=col,
                ax=ax,
                hue_order=hue_order, # Forces consistent bar width
                palette=color_mapping,
                order=model_order # forces model order
            )
            ylabel = 'Number of Responses'

        elif mode == 'percentage':
            counts = df.groupby(['model', col]).size().reset_index(name='count')
            totals = df.groupby('model').size().reset_index(name='total')
            props = pd.merge(counts, totals, on='model')
            props['percentage'] = (props['count'] / props['total']) * 100
            
            sns.barplot(
                data=props,
                x='model',
                y='percentage',
                hue=col,
                ax=ax,
                hue_order=hue_order, # Forces consistent bar width
                palette=color_mapping,
                order=model_order # forces model order
            )
            ylabel = 'Percentage of Responses'

        # Apply Fixed Limit
        ax.set_ylim(0, y_limit)

        # Styling
        ax.set_title(title, fontsize=FS_SUB_TITLE)
        ax.set_xlabel('') 
        ax.tick_params(axis='x', labelsize=FS_TICKS, rotation=10)
        ax.tick_params(axis='y', labelsize=FS_TICKS)
        
        if idx % 2 == 0:
            ax.set_ylabel(ylabel, fontsize=FS_AXIS_LABEL)
        else:
            ax.set_ylabel('')
            
        if ax.get_legend():
            ax.get_legend().remove()

    # --- STEP 3: Legend & Save ---
    # Adjust layout dynamically based on title presence
    top_margin = 0.91 if include_title else 0.99
    
    plt.subplots_adjust(left=0.05, right=0.85, top=top_margin, bottom=0.1, hspace=0.3, wspace=0.15)
    
    # Determine legend items based on dataset type
    if dataset_label == "Shuffled":
        legend_responses = ['1', '2', '3', '4', '5', '6', '7', 'unusable']
    else:
        # Reworded
        legend_responses = ['1', '2', '3', '4', '5', '6', '7', '8', 'unusable']

    handles = []
    labels = []
    for response in legend_responses:
        handle = mpatches.Patch(facecolor=color_mapping[response], edgecolor='black', linewidth=0.5, label=response)
        handles.append(handle)
        labels.append(response)
        
    fig.legend(
        handles=handles,
        labels=labels,
        title='Response Options',
        title_fontsize=FS_LEG_TITLE,
        loc='center right',
        bbox_to_anchor=(0.99, 0.5),
        fontsize=FS_LEG_TEXT,
        frameon=True,
        shadow=True
    )

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot: {save_path}")
    plt.close()

## With title (for slides)
# 1. Instruction Tuned Models
plot_distributions(merged_res_reworded_it, "Reworded", "Instruction-Tuned", "it_rew", mode='absolute', include_title=True)
plot_distributions(merged_res_reworded_it, "Reworded", "Instruction-Tuned", "it_rew", mode='percentage', include_title=True)
plot_distributions(merged_res_shuffled_it, "Shuffled", "Instruction-Tuned", "it_shuff", mode='absolute', include_title=True)
plot_distributions(merged_res_shuffled_it, "Shuffled", "Instruction-Tuned", "it_shuff", mode='percentage', include_title=True)

# 2. Base Models
plot_distributions(merged_res_reworded_it_bm, "Reworded", "Base Models", "bm_rew", mode='absolute', include_title=True)
plot_distributions(merged_res_reworded_it_bm, "Reworded", "Base Models", "bm_rew", mode='percentage', include_title=True)
plot_distributions(merged_res_shuffled_bm, "Shuffled", "Base Models", "bm_shuff", mode='absolute', include_title=True)
plot_distributions(merged_res_shuffled_bm, "Shuffled", "Base Models", "bm_shuff", mode='percentage', include_title=True)

## Without title (for paper)
# 3. Instruction Tuned Models
plot_distributions(merged_res_reworded_it, "Reworded", "Instruction-Tuned", "it_rew", mode='absolute', include_title=False)
plot_distributions(merged_res_reworded_it, "Reworded", "Instruction-Tuned", "it_rew", mode='percentage', include_title=False)
plot_distributions(merged_res_shuffled_it, "Shuffled", "Instruction-Tuned", "it_shuff", mode='absolute', include_title=False)
plot_distributions(merged_res_shuffled_it, "Shuffled", "Instruction-Tuned", "it_shuff", mode='percentage', include_title=False)

# 4. Base Models
plot_distributions(merged_res_reworded_it_bm, "Reworded", "Base Models", "bm_rew", mode='absolute', include_title=False)
plot_distributions(merged_res_reworded_it_bm, "Reworded", "Base Models", "bm_rew", mode='percentage', include_title=False)
plot_distributions(merged_res_shuffled_bm, "Shuffled", "Base Models", "bm_shuff", mode='absolute', include_title=False)
plot_distributions(merged_res_shuffled_bm, "Shuffled", "Base Models", "bm_shuff", mode='percentage', include_title=False)


## -------------------------------------------------
## Response Comparison Across All 4 Prompts ## ## ##
## -------------------------------------------------
# Are the models influenced by the minor variations in prompt (diff option labels and adding Don't know and Refused)?
# This analysis might be especially interesting for prompt 1-3 (numeric option labels, letters as labels and no labels, everything else is the same)
# 'All Same' means the model answered the same question with same answer options 4 times with the same answer
# 'All Different' means the model gave 4 different answers to the same question with same answer options across the minor variations in prompts

def plot_consistency(df, dataset_label, model_type, filename_suffix, mode='absolute', include_title=True):
    """
    Analyzes consistency across clean_response1 to clean_response4.
    - Excludes rows where ANY response is 'unusable'.
    - Categorizes into: '4 Same Answers', '3 Same Answers', '2 Same Answers', '4 Different Answers'.
    - Plots either Absolute Counts or Percentages.
    - Includes fixed y-axis limits (4600 for absolute counts, 62.5% for percentage).
    - Enforces specific model ordering on the x-axis.
    """
    df = df.copy()

    # Define model order
    it_order = [
        'Llama-3.1-8B-Instruct', 
        'Mistral-7B-Instruct-v0.3', 
        'Qwen2.5-7B-Instruct', 
        'gemma-2-9b-it'
    ]
    
    base_order = [
        'Llama-3.1-8B', 
        'Mistral-7B-v0.3', 
        'Qwen2.5-7B', 
        'gemma-2-9b'
    ]
    
    # Select order
    if "Base" in model_type:
        model_order = base_order
    else:
        model_order = it_order

    # Font Sizes
    FS_MAIN_TITLE = 22
    FS_AXIS_LABEL = 18
    FS_TICKS      = 15
    FS_LEG_TITLE  = 18
    FS_LEG_TEXT   = 16

    # Exclude rows with ANY 'unusable' response
    response_cols = [f'clean_response{i}' for i in range(1, 5)]
    
    missing_cols = [c for c in response_cols if c not in df.columns]
    if missing_cols:
        print(f"Skipping plot for {dataset_label}-{model_type}: Missing columns {missing_cols}")
        return

    mask_unusable = df[response_cols].apply(lambda row: 'unusable' in row.values, axis=1)
    df_clean = df[~mask_unusable].copy()
    
    if len(df_clean) == 0:
        print(f"No valid data for {dataset_label}-{model_type} after filtering unusable.")
        return

    # 1. Define consistency logic
    def get_consistency_label(row):
        responses = [row[col] for col in response_cols]
        counts = Counter(responses).values()
        max_freq = max(counts) if counts else 0
        
        if max_freq == 4:
            return '4 Same Answers'
        elif max_freq == 3:
            return '3 Same Answers'
        elif max_freq == 2:
            return '2 Same Answers'
        else:
            return '4 Different Answers'

    df_clean['consistency_cat'] = df_clean.apply(get_consistency_label, axis=1)
    
    # Define Order
    cat_order = ['4 Same Answers', '3 Same Answers', '2 Same Answers', '4 Different Answers']
    
    # 2. Plotting setup
    plt.figure(figsize=(19, 10))
    
    # Fixed ylims & Title
    if mode == 'absolute':
        y_limit = 4600
        metric_str = "(Absolute Count)"
        mode_prefix = "abs"
    elif mode == 'percentage':
        y_limit = 62.5
        metric_str = "(Percentage)"
        mode_prefix = "perc"

    full_title = f'Consistency Across 4 Prompts: {dataset_label} Answer Options {metric_str} - {model_type}'

    # Filename Logic
    if include_title:
        filename = f"consistency_{mode_prefix}_{filename_suffix}.pdf"
        save_path = os.path.join(script_dir, "plots", filename)
    else:
        filename = f"consistency_{mode_prefix}_{filename_suffix}_notitle.pdf"
        notitle_dir = os.path.join(script_dir, "plots", "notitle")
        if not os.path.exists(notitle_dir):
            os.makedirs(notitle_dir)
        save_path = os.path.join(notitle_dir, filename)
    
    # Plotting
    if mode == 'absolute':
        sns.countplot(
            data=df_clean,
            x='model',
            hue='consistency_cat',
            hue_order=cat_order,
            palette='viridis',
            order=model_order
        )
        plt.ylabel('Number of Questions', fontsize=FS_AXIS_LABEL)

    elif mode == 'percentage':
        counts = df_clean.groupby(['model', 'consistency_cat'], observed=False).size().reset_index(name='count')
        totals = df_clean.groupby('model', observed=False).size().reset_index(name='total')
        
        props = pd.merge(counts, totals, on='model')
        props['percentage'] = (props['count'] / props['total']) * 100
        
        sns.barplot(
            data=props,
            x='model',
            y='percentage',
            hue='consistency_cat',
            hue_order=cat_order,
            palette='viridis',
            order=model_order
        )
        plt.ylabel('Percentage of Questions', fontsize=FS_AXIS_LABEL)

    # Apply Fixed Limit
    plt.ylim(0, y_limit)

    # Conditional Title
    if include_title:
        plt.title(full_title, fontsize=FS_MAIN_TITLE, fontweight='bold', y=1.02)
    else:
        # If no title, we don't call plt.title()
        pass

    # 3. Styling
    plt.xlabel('')
    plt.tick_params(axis='x', labelsize=FS_TICKS)
    plt.tick_params(axis='y', labelsize=FS_TICKS)
    
    # Legend sizing
    plt.legend(
        title='Consistency Level', 
        bbox_to_anchor=(1.01, 1), 
        loc='upper left',
        title_fontsize=FS_LEG_TITLE,
        fontsize=FS_LEG_TEXT
    )
    
    plt.tight_layout()
    
    # Ensure dir exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot: {save_path}")
    plt.close()

## With title (for slides)
# 1. Instruction Tuned Models
plot_consistency(merged_res_reworded_it, "Reworded", "Instruction-Tuned Models", "it_rew", mode='absolute', include_title=True)
plot_consistency(merged_res_shuffled_it, "Shuffled", "Instruction-Tuned Models", "it_shuff", mode='absolute', include_title=True)
plot_consistency(merged_res_reworded_it, "Reworded", "Instruction-Tuned Models", "it_rew", mode='percentage', include_title=True)
plot_consistency(merged_res_shuffled_it, "Shuffled", "Instruction-Tuned Models", "it_shuff", mode='percentage', include_title=True)

# 2. Base Models
plot_consistency(merged_res_reworded_it_bm, "Reworded", "Base Models", "bm_rew", mode='absolute', include_title=True)
plot_consistency(merged_res_shuffled_bm, "Shuffled", "Base Models", "bm_shuff", mode='absolute', include_title=True)
plot_consistency(merged_res_reworded_it_bm, "Reworded", "Base Models", "bm_rew", mode='percentage', include_title=True)
plot_consistency(merged_res_shuffled_bm, "Shuffled", "Base Models", "bm_shuff", mode='percentage', include_title=True)

## Without title (for paper)
# 3. Instruction Tuned Models
plot_consistency(merged_res_reworded_it, "Reworded", "Instruction-Tuned Models", "it_rew", mode='absolute', include_title=False)
plot_consistency(merged_res_shuffled_it, "Shuffled", "Instruction-Tuned Models", "it_shuff", mode='absolute', include_title=False)
plot_consistency(merged_res_reworded_it, "Reworded", "Instruction-Tuned Models", "it_rew", mode='percentage', include_title=False)
plot_consistency(merged_res_shuffled_it, "Shuffled", "Instruction-Tuned Models", "it_shuff", mode='percentage', include_title=False)

# 4. Base Models
plot_consistency(merged_res_reworded_it_bm, "Reworded", "Base Models", "bm_rew", mode='absolute', include_title=False)
plot_consistency(merged_res_shuffled_bm, "Shuffled", "Base Models", "bm_shuff", mode='absolute', include_title=False)
plot_consistency(merged_res_reworded_it_bm, "Reworded", "Base Models", "bm_rew", mode='percentage', include_title=False)
plot_consistency(merged_res_shuffled_bm, "Shuffled", "Base Models", "bm_shuff", mode='percentage', include_title=False)


## ---------------------------------------------------------------------------
## Consistency Analysis - Average Distance Between Answer Variations ## ## ##
## ---------------------------------------------------------------------------
# Do the models give the same or different answers to the same question and same prompt when only the answer options change?
# Since there are varying scales I can't just compare clean_response
# So take the avg of all pairwise distances between the num_value of all answers
# NOTE: There are 'unusable' and -98 and -99 values in num_value, these values are flagged invalid + only questions with 5 valid num_value values are used for computation

# helper: robust parsing & filtering
INVALID_MARKERS = {-98, -99}  # treat these as missing (num_value in prompt 4 for 'Don't know' and 'Refused')

def _to_valid_floats(values):
    """Filter out None, NaN, and sentinel markers."""
    out = []
    for v in values:
        if v is None: continue
        if (isinstance(v, float) and np.isnan(v)): continue
        try:
            fv = float(v)
        except Exception:
            continue
        if int(fv) in INVALID_MARKERS: continue
        out.append(fv)
    return out

# helper: average pairwise absolute distance (ignoring special missing markers)
def avg_pairwise_abs_distance(values, require_full=False, expected_n=5):
    """Compute mean of |xi - xj| over all pairs."""
    vals = _to_valid_floats(values)
    n = len(vals)
    if require_full and n != expected_n:
        return np.nan
    if n < 2:
        return np.nan
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            total += abs(vals[i] - vals[j])
            count += 1
    return total / count if count > 0 else np.nan

# Consistency Analysis1 Plotting Function
def generate_avg_distance_plots(df, dataset_label, model_type, filename_prefix, include_title=True):
    """
    1. Calculates pairwise distances for Prompts 1-4.
    2. Generates one 2x2 figure per prompt (showing 4 models).
    """
    df = df.copy()
    prompts = [1, 2, 3, 4]
    
    # Define model order
    it_order = [
        'Llama-3.1-8B-Instruct', 
        'Mistral-7B-Instruct-v0.3', 
        'Qwen2.5-7B-Instruct', 
        'gemma-2-9b-it'
    ]
    
    base_order = [
        'Llama-3.1-8B', 
        'Mistral-7B-v0.3', 
        'Qwen2.5-7B', 
        'gemma-2-9b'
    ]
    
    # Select order
    if "Base" in model_type:
        model_order = base_order
    else:
        model_order = it_order

    # Font Sizes
    FS_MAIN_TITLE = 22
    FS_SUB_TITLE  = 20
    FS_AXIS_LABEL = 18
    FS_TICKS      = 15
    FS_LEG_TEXT   = 16

    # --- STEP 1: Calc distances ---
    avgdist_data = {}
    
    for p in prompts:
        col = f'num_value{p}'
        if col not in df.columns: continue
        
        grouped = df.groupby(['model', 'question_id'])[col].apply(list).reset_index(name='values_list')

        # Determine expected_n dynamically
        if grouped.empty:
            print(f"[{dataset_label}] Prompt {p}: No data found.")
            avgdist_data[p] = grouped
            continue

        typical_len = int(grouped['values_list'].apply(len).median())
        # print(f"[{dataset_label}-{model_type}] Prompt {p}: Expecting {typical_len} valid answers per question.")

        grouped['avg_pairwise_dist'] = grouped['values_list'].apply(
            lambda lst: avg_pairwise_abs_distance(lst, require_full=True, expected_n=typical_len)
        )
        avgdist_data[p] = grouped

    # --- STEP 2: Plotting Loop ---
    for p in prompts:
        if p not in avgdist_data: continue
        dfp = avgdist_data[p]
        if dfp.empty: continue
        
        # 2a. Fixed Y-Limit
        y_limit = 12.8 
        
        # 2b. Plotting
        fig, axes = plt.subplots(2, 2, figsize=(19, 10))
        
        # Conditional Title
        if include_title:
            fig.suptitle(f'Prompt {p}: Distribution of Average Pairwise Distance ({dataset_label}) - {model_type}', 
                         fontsize=FS_MAIN_TITLE, fontweight='bold', y=0.98)
        
        axes_flat = axes.flatten()
        
        # Iterate over fixed model order
        for idx, model in enumerate(model_order):
            if idx >= len(axes_flat): break
            ax = axes_flat[idx]
            
            # Extract data for specific model
            model_vals = dfp[dfp['model'] == model]['avg_pairwise_dist'].dropna()
            
            if len(model_vals) > 0:
                # Histogram
                fixed_bins = np.linspace(0, 2.0, 41) # 41 edges = 40 bins
                sns.histplot(model_vals, bins=fixed_bins, stat='density', alpha=0.6, ax=ax, edgecolor=None, color='#1f77b4')
                
                # KDE (Check variance first)
                if model_vals.nunique() > 1:
                    sns.kdeplot(model_vals, ax=ax, linewidth=2)
                
                # Metrics
                med = float(model_vals.median())
                mean = float(model_vals.mean())
                
                # Lines
                ax.axvline(med, color='black', linestyle='--', linewidth=1.5, label=f'Median: {med:.3f}')
                ax.axvline(mean, color='black', linestyle='-', linewidth=1.5, label=f'Mean: {mean:.3f}')
                
                # Styling
                ax.set_xlim(0, 2.0)
                ax.set_ylim(0, y_limit) 
                
                # Legend
                ax.legend(loc='upper right', fontsize=FS_LEG_TEXT)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_xlim(0, 2.0)
                ax.set_ylim(0, y_limit)

            ax.set_title(model, fontsize=FS_SUB_TITLE)
            ax.set_xlabel('Avg Pairwise Distance', fontsize=FS_AXIS_LABEL)
            
            # Y-label only on left column
            if idx % 2 == 0:
                ax.set_ylabel('Density', fontsize=FS_AXIS_LABEL)
            else:
                ax.set_ylabel('')
                
            # Tick Sizes
            ax.tick_params(axis='x', labelsize=FS_TICKS)
            ax.tick_params(axis='y', labelsize=FS_TICKS)

        # Adjust Layout
        top_margin = 0.90 if include_title else 0.99
        plt.subplots_adjust(top=top_margin, bottom=0.08, left=0.05, right=0.95, hspace=0.3, wspace=0.2)
        
        # Construct Filename & Path
        if include_title:
            filename = f"{filename_prefix}_prompt{p}.pdf"
            save_path = os.path.join(script_dir, "plots", "distance_distributions", filename)
        else:
            filename = f"{filename_prefix}_prompt{p}_notitle.pdf"
            notitle_dir = os.path.join(script_dir, "plots", "notitle", "distance_distributions")
            if not os.path.exists(notitle_dir):
                os.makedirs(notitle_dir)
            save_path = os.path.join(notitle_dir, filename)

        # Ensure dir exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved plot: {save_path}")
        plt.close()

## With title
# 1. Instruction Tuned
generate_avg_distance_plots(merged_res_reworded_it, "Reworded", "Instruction-Tuned Models", "apd_distr_it_rew", include_title=True)
generate_avg_distance_plots(merged_res_shuffled_it, "Shuffled", "Instruction-Tuned Models", "apd_distr_it_shuff", include_title=True)

# 2. Base Models
generate_avg_distance_plots(merged_res_reworded_it_bm, "Reworded", "Base Models", "apd_distr_bm_rew", include_title=True)
generate_avg_distance_plots(merged_res_shuffled_bm, "Shuffled", "Base Models", "apd_distr_bm_shuff", include_title=True)

## Without title
# 3. Instruction Tuned
generate_avg_distance_plots(merged_res_reworded_it, "Reworded", "Instruction-Tuned Models", "apd_distr_it_rew", include_title=False)
generate_avg_distance_plots(merged_res_shuffled_it, "Shuffled", "Instruction-Tuned Models", "apd_distr_it_shuff", include_title=False)

# 4. Base Models
generate_avg_distance_plots(merged_res_reworded_it_bm, "Reworded", "Base Models", "apd_distr_bm_rew", include_title=False)
generate_avg_distance_plots(merged_res_shuffled_bm, "Shuffled", "Base Models", "apd_distr_bm_shuff", include_title=False)


## ---------------------------------------------------------------------------
## Consistency Analysis2 - Maximum Distance Between Answer Variations ## ## ##
## ---------------------------------------------------------------------------
# Do the models give the same or different answers to the same question and same prompt when only the answer options change?
# Since there are varying scales I can't just compare clean_response
# So take the max distance of all pairwise distances between the num_value of all answers
# NOTE: There are 'unusable' and -98 and -99 values in num_value, these values are flagged invalid + only questions with 5 valid num_value values are used for computation

# helper: robust parsing & filtering -> exact same one as in Consistency Analysis1

# helper: maximum pairwise absolute distance (ignoring NaN) 
def max_pairwise_abs_distance(values, require_full=False, expected_n=5):
    """Compute the maximum absolute pairwise distance among values."""
    vals = _to_valid_floats(values)
    n = len(vals)
    
    if require_full and n != expected_n:
        return np.nan
    if n < 2:
        return np.nan
        
    maxd = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(vals[i] - vals[j])
            if d > maxd:
                maxd = d
    return maxd

# Consistency Analysis2 Plotting Function
def generate_max_distance_plots(df, dataset_label, model_type, filename_prefix, include_title=True):
    """
    1. Calculates maximum pairwise distances for Prompts 1-4.
    2. Generates one 2x2 figure per prompt (showing 4 models).
    """
    df = df.copy()
    prompts = [1, 2, 3, 4] 
    
    # Define model order
    it_order = [
        'Llama-3.1-8B-Instruct', 
        'Mistral-7B-Instruct-v0.3', 
        'Qwen2.5-7B-Instruct', 
        'gemma-2-9b-it'
    ]
    
    base_order = [
        'Llama-3.1-8B', 
        'Mistral-7B-v0.3', 
        'Qwen2.5-7B', 
        'gemma-2-9b'
    ]
    
    # Select order
    if "Base" in model_type:
        model_order = base_order
    else:
        model_order = it_order

    # Font Sizes
    FS_MAIN_TITLE = 22
    FS_SUB_TITLE  = 20
    FS_AXIS_LABEL = 18
    FS_TICKS      = 15
    FS_LEG_TEXT   = 16

    # --- STEP 1: Calc distances ---
    maxdist_data = {}
    
    for p in prompts:
        col = f'num_value{p}'
        if col not in df.columns: continue

        grouped = df.groupby(['model', 'question_id'])[col].apply(list).reset_index(name='values_list')
        
        if grouped.empty:
             print(f"[{dataset_label}-{model_type}] Prompt {p}: No data.")
             continue

        # Determine expected_n dynamically
        typical_len = int(grouped['values_list'].apply(len).median())
        # print(f"[{dataset_label}-{model_type}] Prompt {p}: Expecting {typical_len} valid answers per question.")

        grouped['max_pairwise_dist'] = grouped['values_list'].apply(
            lambda lst: max_pairwise_abs_distance(lst, require_full=True, expected_n=typical_len)
        )
        maxdist_data[p] = grouped

    # --- STEP 2: Plotting Loop ---
    for p in prompts:
        if p not in maxdist_data: continue
        dfp = maxdist_data[p]
        if dfp.empty: continue
        
        # 2a. Fixed ylim for all plots
        y_limit = 14
        
        # 2b. Plotting
        fig, axes = plt.subplots(2, 2, figsize=(19, 10))
        
        # Conditional Title
        if include_title:
            fig.suptitle(f'Prompt {p}: Distribution of Maximum Pairwise Distance ({dataset_label}) - {model_type}', 
                         fontsize=FS_MAIN_TITLE, fontweight='bold', y=0.98)
        
        axes_flat = axes.flatten()
        
        # Iterate over fixed model order
        for idx, model in enumerate(model_order):
            if idx >= len(axes_flat): break
            ax = axes_flat[idx]
            
            # Extract data for specific model
            model_vals = dfp[dfp['model'] == model]['max_pairwise_dist'].dropna()
            
            if len(model_vals) > 0:
                # Histogram
                fixed_bins = np.linspace(0, 2.0, 41) # 41 edges = 40 bins
                sns.histplot(model_vals, bins=fixed_bins, stat='density', alpha=0.6, ax=ax, edgecolor=None, color='#1f77b4')
                
                # KDE (Check variance first)
                if model_vals.nunique() > 1:
                    sns.kdeplot(model_vals, ax=ax, linewidth=2)
                
                # Metrics
                med = float(model_vals.median())
                mean = float(model_vals.mean())
                
                # Lines
                ax.axvline(med, color='black', linestyle='--', linewidth=1.5, label=f'Median: {med:.3f}')
                ax.axvline(mean, color='black', linestyle='-', linewidth=1.5, label=f'Mean: {mean:.3f}')
                
                # Styling
                ax.set_xlim(0, 2.0)
                ax.set_ylim(0, y_limit) 
                
                # Legend
                ax.legend(loc='upper right', fontsize=FS_LEG_TEXT)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_xlim(0, 2.0)
                ax.set_ylim(0, y_limit)

            ax.set_title(model, fontsize=FS_SUB_TITLE)
            ax.set_xlabel('Max Pairwise Distance', fontsize=FS_AXIS_LABEL)
            
            # Y-label only on left column
            if idx % 2 == 0:
                ax.set_ylabel('Density', fontsize=FS_AXIS_LABEL)
            else:
                ax.set_ylabel('')

            # Tick Sizes
            ax.tick_params(axis='x', labelsize=FS_TICKS)
            ax.tick_params(axis='y', labelsize=FS_TICKS)

        # Adjust Layout
        top_margin = 0.90 if include_title else 0.99
        plt.subplots_adjust(top=top_margin, bottom=0.08, left=0.05, right=0.95, hspace=0.3, wspace=0.2)
        
        # Construct Filename & Path
        if include_title:
            filename = f"{filename_prefix}_prompt{p}.pdf"
            save_path = os.path.join(script_dir, "plots", "distance_distributions", filename)
        else:
            filename = f"{filename_prefix}_prompt{p}_notitle.pdf"
            notitle_dir = os.path.join(script_dir, "plots", "notitle", "distance_distributions")
            if not os.path.exists(notitle_dir):
                os.makedirs(notitle_dir)
            save_path = os.path.join(notitle_dir, filename)

        # Ensure dir exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved plot: {save_path}")
        plt.close()

## With title
# 1. Instruction Tuned
generate_max_distance_plots(merged_res_reworded_it, "Reworded", "Instruction-Tuned Models", "mpd_distr_it_rew", include_title=True)
generate_max_distance_plots(merged_res_shuffled_it, "Shuffled", "Instruction-Tuned Models", "mpd_distr_it_shuff", include_title=True)

# 2. Base Models
generate_max_distance_plots(merged_res_reworded_it_bm, "Reworded", "Base Models", "mpd_distr_bm_rew", include_title=True)
generate_max_distance_plots(merged_res_shuffled_bm, "Shuffled", "Base Models", "mpd_distr_bm_shuff", include_title=True)

## Without title
# 3. Instruction Tuned
generate_max_distance_plots(merged_res_reworded_it, "Reworded", "Instruction-Tuned Models", "mpd_distr_it_rew", include_title=False)
generate_max_distance_plots(merged_res_shuffled_it, "Shuffled", "Instruction-Tuned Models", "mpd_distr_it_shuff", include_title=False)

# 4. Base Models
generate_max_distance_plots(merged_res_reworded_it_bm, "Reworded", "Base Models", "mpd_distr_bm_rew", include_title=False)
generate_max_distance_plots(merged_res_shuffled_bm, "Shuffled", "Base Models", "mpd_distr_bm_shuff", include_title=False)


## ----------------------------------------------
## Bias Analysis ## ## ##
## ----------------------------------------------
# Do variations in prompt or changing the answer options make the models answer more positively or negatively?
# We can compare reworded to shuffled (even tho shuffled has 6 answers per questions and reworded 5) because stat='density' not 'count' in sns.histplot()
# This argument ensures that the total area of the histogram sums up to exactly 1
#### Could also do overlapping for base and instruction tuned models to compare the model types

# Bias Analysis Helper: Clean Numeric Column
INVALID_MARKERS = {-98, -99}

def clean_numeric_series(series):
    """Takes a pandas Series, drops NaNs, and filters out -98/-99."""
    s = pd.to_numeric(series, errors='coerce')
    s = s.dropna()
    mask = ~s.isin(INVALID_MARKERS)
    return s[mask]

# Bias Analysis Plotting Function (Overlapping Histograms)
def generate_bias_overlap_plots(df1, label1, df2, label2, filename_prefix, include_title=True):
    """
    Generates 2x2 figures (one per model) showing overlapping histograms.
    """
    df1 = df1.copy()
    df2 = df2.copy()
    
    models = sorted(df1['model'].unique())
    prompts = [1, 2, 3, 4] 
    
    # Colors
    c1 = '#1f77b4' # Blue
    c2 = '#ff7f0e' # Orange
    alpha_val = 0.5
    
    # Font Sizes
    FS_MAIN_TITLE = 22
    FS_SUB_TITLE  = 20
    FS_AXIS_LABEL = 18
    FS_TICKS      = 15
    FS_LEG_TEXT   = 16
    
    # Fixed Y-Limit (highest spike is Gemma-2-9b Prompt 2 ~10)
    y_limit = 10.5

    for model in models:
        # --- Create Figure ---
        fig, axes = plt.subplots(2, 2, figsize=(19, 10))
        
        # Conditional Title
        if include_title:
            formatted_model_name = model[0].upper() + model[1:] if len(model) > 0 else model
            fig.suptitle(f'{formatted_model_name}: Distribution of Answers Across Prompts ({label1} vs {label2})', 
                         fontsize=FS_MAIN_TITLE, fontweight='bold', y=0.98)
        
        axes_flat = axes.flatten()
        
        for idx, p in enumerate(prompts):
            ax = axes_flat[idx]
            col = f'num_value{p}'

            if col not in df1.columns or col not in df2.columns:
                ax.text(0.5, 0.5, 'Missing Column', ha='center', va='center')
                continue

            if model not in df2['model'].values:
                raw_vals2 = pd.Series(dtype='float64')
            else:
                raw_vals2 = df2[df2['model'] == model][col]
            raw_vals1 = df1[df1['model'] == model][col]
            
            vals1 = clean_numeric_series(raw_vals1)
            vals2 = clean_numeric_series(raw_vals2)
            
            if len(vals1) > 0 or len(vals2) > 0:
                # Plot histograms
                if len(vals1) > 0:
                    sns.histplot(vals1, bins=30, stat='density', alpha=alpha_val, 
                                 color=c1, label=label1, ax=ax, edgecolor=None)
                if len(vals2) > 0:
                    sns.histplot(vals2, bins=30, stat='density', alpha=alpha_val, 
                                 color=c2, label=label2, ax=ax, edgecolor=None)
                
                # Plot KDE
                if len(vals1) > 1 and vals1.nunique() > 1:
                    sns.kdeplot(vals1, color=c1, ax=ax, linewidth=2, warn_singular=False)
                if len(vals2) > 1 and vals2.nunique() > 1:
                    sns.kdeplot(vals2, color=c2, ax=ax, linewidth=2, warn_singular=False)
                
                ax.set_ylim(0, y_limit)
                ax.set_xlim(-1.05, 1.05)

                if idx == 0:
                    ax.legend(loc='upper right', fontsize=FS_LEG_TEXT)

            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_xlim(-1.05, 1.05)
                ax.set_ylim(0, y_limit)

            ax.set_title(f'Prompt {p}', fontsize=FS_SUB_TITLE)
            ax.set_xlabel('Numeric Answer Value', fontsize=FS_AXIS_LABEL)
            
            if idx % 2 == 0:
                ax.set_ylabel('Density', fontsize=FS_AXIS_LABEL)
            else:
                ax.set_ylabel('')

            # Tick Sizes
            ax.tick_params(axis='x', labelsize=FS_TICKS)
            ax.tick_params(axis='y', labelsize=FS_TICKS)

        # Adjust Layout
        rect_val = [0, 0, 1, 0.96] if include_title else [0, 0, 1, 1]
        plt.tight_layout(rect=rect_val)
        
        # Filename & Path
        safe_model_name = model.lower().replace(" ", "_").replace("/", "-")
        
        if include_title:
             filename = f"{filename_prefix}_{safe_model_name}.pdf"
             save_dir = os.path.join(script_dir, "plots", "answer_distributions")
        else:
             filename = f"{filename_prefix}_{safe_model_name}_notitle.pdf"
             save_dir = os.path.join(script_dir, "plots", "notitle", "answer_distributions")

        if not os.path.exists(save_dir):
             os.makedirs(save_dir)

        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved bias plot for {model} to {save_path}")
        plt.close()

# 1. Instruction Tuned Models
# With Title
generate_bias_overlap_plots(merged_res_reworded_it, "Reworded", merged_res_shuffled_it, "Shuffled", "ans_distr_it", include_title=True)
# Without Title
generate_bias_overlap_plots(merged_res_reworded_it, "Reworded", merged_res_shuffled_it, "Shuffled", "ans_distr_it", include_title=False)

# 2. Base Models
# With Title
generate_bias_overlap_plots(merged_res_reworded_it_bm, "Reworded", merged_res_shuffled_bm, "Shuffled", "ans_distr_bm", include_title=True)
# Without Title
generate_bias_overlap_plots(merged_res_reworded_it_bm, "Reworded", merged_res_shuffled_bm, "Shuffled", "ans_distr_bm", include_title=False)


## ----------------------------------------------
## Test-Retest / Stability-Type Measures ## ## ##
## ----------------------------------------------
# How robust is the model's internal logic against variations in prompt structure? 
# If a model rates Question A higher than Question B in Prompt 1, does it maintain that relationship in Prompt 2, or does the format change alter its logic?
# Compute correlation between all pairs of num_value columns for each model. High Pearson’s r suggests high stability/consistency across prompts.
# 1. Pairwise Comparison: We compare every combination of prompts (e.g., Prompt 1 vs. Prompt 2, 1 vs. 3, etc.).
# 2. Data Cleaning: Invalid responses (-98, -99) are treated as NaN and excluded pairwise to ensure clean statistical comparison.
# 3. Compute Pearson's r: Calculate the linear correlation coefficient between the numeric answer values of the two prompts.
# Interpretation:
# - High Correlation (r > 0.8): High Stability. The model understands the underlying task regardless of option labels or new options.
# - Low Correlation (r < 0.5): Low Stability. The phrasing and/or option labels change the model's perception of the question leading to different responses.

# Function to compute correlation between two columns
def get_clean_correlation(df, col1, col2):
    """
    Computes Pearson correlation between two columns, 
    treating -98/-99 as NaN and dropping missing values pairwise.
    """
    # Create a subset
    subset = df[[col1, col2]].copy()
    
    # Replace invalid markers with NaN and drop rows where either col contains NaN
    subset.replace({-98: np.nan, -99: np.nan}, inplace=True)
    subset = subset.dropna()
    
    if len(subset) < 2:
        return np.nan
        
    return subset[col1].corr(subset[col2])

# Main analysis Function
def generate_stability_analysis(df, dataset_label, model_type, filename_prefix, include_title=True):
    """
    1. Computes pairwise Pearson correlations between Prompts 1-4 for each model.
    2. Prints a matrix of results.
    3. Generates a Violin Plot of the correlation distributions.
    """
    df = df.copy()
    prompts = [1, 2, 3, 4]
    
    # Define model order
    it_order = [
        'Llama-3.1-8B-Instruct', 
        'Mistral-7B-Instruct-v0.3', 
        'Qwen2.5-7B-Instruct', 
        'gemma-2-9b-it'
    ]
    
    base_order = [
        'Llama-3.1-8B', 
        'Mistral-7B-v0.3', 
        'Qwen2.5-7B', 
        'gemma-2-9b'
    ]
    
    # Select order
    if "Base" in model_type:
        model_order = base_order
    else:
        model_order = it_order

    # Font Sizes
    FS_MAIN_TITLE = 22
    FS_AXIS_LABEL = 18
    FS_TICKS      = 15

    correlation_results = []
    
    # Use existing models in DF, but sorted by our fixed order
    models_present = [m for m in model_order if m in df['model'].unique()]

    # --- 1. Calc Correlations ---
    for model in models_present:
        model_data = df[df['model'] == model]
        
        for p1, p2 in combinations(prompts, 2):
            col1 = f'num_value{p1}'
            col2 = f'num_value{p2}'
            
            if col1 not in df.columns or col2 not in df.columns:
                continue

            corr = get_clean_correlation(model_data, col1, col2)
            
            if not np.isnan(corr):
                correlation_results.append({
                    'model': model,
                    'prompt_pair': f'{p1}-{p2}',
                    'correlation': corr
                })

    if not correlation_results:
        print(f"[{dataset_label}-{model_type}] No valid correlation data found.")
        return

    corr_df = pd.DataFrame(correlation_results)

    # --- 2. Print Matrix (Optional Log) ---
    print(f"\n--- {dataset_label} ({model_type}): Correlation Matrix (Model x Prompt Pair) ---")
    corr_matrix = corr_df.pivot(index='model', columns='prompt_pair', values='correlation')
    print(corr_matrix.round(3))
    print("-" * 60)

    # --- 3. Plotting ---
    plt.figure(figsize=(19, 10))
    
    # Violin Plot with Enforced Order
    sns.violinplot(
        data=corr_df, 
        x='model', 
        y='correlation', 
        inner='quartile', 
        color='#1f77b4', 
        linewidth=1.5, 
        order=model_order
    )

    # Styling
    if include_title:
        plt.title(f'Stability Analysis: Correlation of Answer Values Across Prompts ({dataset_label}) - {model_type}', 
                  fontsize=FS_MAIN_TITLE, fontweight='bold', y=1.02)
    
    plt.xlabel('Model', fontsize=FS_AXIS_LABEL)
    plt.ylabel('Pearson Correlation (r)', fontsize=FS_AXIS_LABEL)
    
    # Ticks & Grid
    plt.yticks(np.arange(0, 1.05, 0.1), fontsize=FS_TICKS)
    plt.xticks(fontsize=FS_TICKS)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)

    # Adjust Layout
    top_margin = 0.96 if include_title else 0.99
    plt.tight_layout(rect=[0, 0, 1, top_margin])

    # Save
    if include_title:
        filename = f"{filename_prefix}_corr_violin.pdf"
        save_dir = os.path.join(script_dir, "plots")
    else:
        filename = f"{filename_prefix}_corr_violin_notitle.pdf"
        save_dir = os.path.join(script_dir, "plots", "notitle")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved correlation plot to {save_path}")
    plt.close()

# 1. Instruction Tuned Models
# With Title
generate_stability_analysis(merged_res_reworded_it, "Reworded", "Instruction-Tuned Models", "stability_it_rew", include_title=True)
generate_stability_analysis(merged_res_shuffled_it, "Shuffled", "Instruction-Tuned Models", "stability_it_shuff", include_title=True)
# Without Title
generate_stability_analysis(merged_res_reworded_it, "Reworded", "Instruction-Tuned Models", "stability_it_rew", include_title=False)
generate_stability_analysis(merged_res_shuffled_it, "Shuffled", "Instruction-Tuned Models", "stability_it_shuff", include_title=False)

# 2. Base Models
# With Title
generate_stability_analysis(merged_res_reworded_it_bm, "Reworded", "Base Models", "stability_bm_rew", include_title=True)
generate_stability_analysis(merged_res_shuffled_bm, "Shuffled", "Base Models", "stability_bm_shuff", include_title=True)
# Without Title
generate_stability_analysis(merged_res_reworded_it_bm, "Reworded", "Base Models", "stability_bm_rew", include_title=False)
generate_stability_analysis(merged_res_shuffled_bm, "Shuffled", "Base Models", "stability_bm_shuff", include_title=False)

#### Notes on Interpretation of Matrices:

## Instruction Tuned models:
""" 
Reworded (Instruction-Tuned Models): Correlation Matrix (Model x Prompt Pair)
prompt_pair                 1-2    1-3    1-4    2-3    2-4    3-4
model
Llama-3.1-8B-Instruct     0.820  0.701  0.894  0.712  0.797  0.683
Mistral-7B-Instruct-v0.3  0.707  0.722  0.913  0.646  0.709  0.703
Qwen2.5-7B-Instruct       0.818  0.743  0.883  0.737  0.799  0.726
gemma-2-9b-it             0.860  0.778  0.897  0.820  0.841  0.749

Shuffled (Instruction-Tuned Models): Correlation Matrix (Model x Prompt Pair)
prompt_pair                 1-2    1-3    1-4    2-3    2-4    3-4
model
Llama-3.1-8B-Instruct     0.797  0.636  0.796  0.686  0.778  0.644
Mistral-7B-Instruct-v0.3  0.730  0.709  0.901  0.627  0.728  0.718
Qwen2.5-7B-Instruct       0.886  0.783  0.921  0.777  0.882  0.780
gemma-2-9b-it             0.849  0.733  0.856  0.784  0.824  0.717
"""
# Table Reworded Responses test Language Stability (Do they understand what the options mean even if the words are changed?)
# Table Shuffled Responses test Order Stability (Do they pick the answer because it's true, or because it's Option A?)

# - columns involving Prompt 3 (1-3, 2-3, 3-4). These are consistently the lowest numbers for almost every model:
# => Prompt 3 (no option labels) is the outlier/Odd one out, the phrasing causes the models to shift their answers compared to other prompts.

# - the pair 1-4 is consistently the highest (often ~0.90):
# => Prompt 1 and 4 are very similar in structure or difficulty, the models treat them nearly identically.

# - Gemma shows the best results in Reworded Responses Matrix (only 2nd in 1-4):
# => Gemma is the best at Lanugage Understanding, it understands true meaning of answer options.

# - Qwen shows best results in Shuffled Responses Matrix (only 2nd in 2-3). Corr dropped for other models but Qwen's actually increased:
# => Qwen is the most robust model against answer ordering, i.e. its best at tracking the content of the answer options and ignoring their order.

# - Llama is a solid middle-ground performer but suffers the most from Shuffling (notice drops from Reworded to Shuffled are highest):
# => Llama has a higher positional bias than other models (especially Gemma and Qwen), so when the options get shuffled the stability drops

# - Mistral is the most volatile (highest highs: 0.913 in 1-4 Reworded & the lowest lows: 0.627 in 2-3 Shuffled)
# => Might be that when the prompt structure matches its training data (Prompts 1 & 4) its solid but when the prompt vaires from it Mistral becomes inconsistent

## Base models:
"""
Reworded (Base Models): Correlation Matrix (Model x Prompt Pair)
prompt_pair        1-2    1-3    1-4    2-3    2-4    3-4
model
Llama-3.1-8B     0.580  0.398  0.644  0.464  0.536  0.461
Mistral-7B-v0.3  0.485  0.408  0.454  0.354  0.336  0.278
Qwen2.5-7B       0.608  0.575  0.750  0.467  0.560  0.481
gemma-2-9b       0.534  0.374  0.564  0.473  0.627  0.504

Shuffled (Base Models): Correlation Matrix (Model x Prompt Pair)
prompt_pair        1-2    1-3    1-4    2-3    2-4    3-4
model
Llama-3.1-8B     0.545  0.354  0.594  0.441  0.511  0.413
Mistral-7B-v0.3  0.309  0.183  0.496  0.191  0.269  0.244
Qwen2.5-7B       0.608  0.428  0.849  0.431  0.572  0.400
gemma-2-9b       0.468  0.155  0.499  0.312  0.557  0.378
"""
# Table Reworded Responses test Language Stability (Do they understand what the options mean even if the words are changed?)
# Table Shuffled Responses test Order Stability (Do they pick the answer because it's true, or because it's Option A?)

# Overall Correlation levels are significantly lower (mostly 0.30 - 0.60) compared to IT models (0.70 - 0.90):
# => Base models lack the "alignment" to strictly follow the multiple-choice format. They are likely treating the prompts as text completion tasks
# rather than distinct logic problems, leading to much higher volatility.

# - Prompt 3 (1-3, 2-3, 3-4) causes drastic drops, even lower than in IT models (e.g., Gemma drops to 0.155 in Shuffled 1-3):
# => Without option labels (A, B, C), Base models struggle significantly to map the answer back to the question.
# They rely heavily on the structural "scaffolding" (like "1." or "A)") to maintain consistency.

# - The pair 1-4 remains the "anchor" of stability (consistently highest correlations, e.g., Qwen Shuffled 1-4 is 0.849):
# => Even without instruction tuning, the models recognize that Prompt 1 and Prompt 4 are structurally identical.
# This confirms that structural similarity is the strongest predictor of consistency for Base models.

# - Qwen2.5-7B is the clear "Champion" of the Base models (Highest correlations in almost every cell, e.g., 0.849 in 1-4 Shuffled):
# => Qwen's base pre-training likely included more high-quality multiple-choice data or logic puzzles.
# It is the only Base model that approaches the stability levels of an Instruction-Tuned model.

# - Gemma-2-9b shows a massive "Fragility" to Shuffling (Correlation 1-3 drops from 0.374 Reworded -> 0.155 Shuffled):
# => Unlike its IT counterpart (which was the NLU leader), Gemma Base collapses when order changes. 
# This suggests Gemma's robust understanding of language is heavily dependent on the Instruction Tuning / RLHF stage; the Base model is very sensitive to pattern disruption.

# - Mistral-7B-v0.3 performs poorly across the board (Multiple correlations < 0.20 in Shuffled):
# => Mistral Base is effectively "guessing" or completing text randomly when the order is shuffled.
# It has almost no inherent logical stability for this specific multiple-choice task without finetuning.

# - Llama-3.1-8B is the "Stable Average":
# => It doesn't reach Qwen's highs, but it avoids Gemma and Mistral's catastrophic lows (0.15-0.20).
# It maintains a moderate, consistent baseline of ~0.40-0.50 regardless of the transformation.


## ---------------------------------------------------------------------------
## Volatility Analysis - Standard Deviation of Num Values of Responses ## ## ##
## ---------------------------------------------------------------------------
# How much do the model's answers fluctuate for a single question due to rewording or shuffling? 
# Does the model consistently settle on a specific numeric value, or does it oscillate between high and low scores?
# Lower std indicates higher consistency -> responses are similar
# Metric needs to be calculated on "per-question" basis -> score for each question
# To make this presentable I need to create an "overall metric" -> Mean Std deviation for each model and summarize in a table and boxplot
# 1. For every unique question_id, collect the valid `num_value` list (5 in reworded or 6 in shuffled dataset).
# 2. Calculate the standard deviation of these values for each question individually.
# 3. Compute mean of these per-question standard deviations for each Model and Prompt to create an "overall score" for Volatility.

# Helper: robust std calculation
INVALID_MARKERS = {-98, -99}

def get_valid_std(series):
    """
    Cleans series of invalid markers and calculates standard deviation.
    Returns NaN if fewer than 2 valid points.
    """
    # Coerce to numeric
    s = pd.to_numeric(series, errors='coerce')
    
    # Remove NaNs and Invalid Markers
    valid_s = s[~s.isna() & ~s.isin(INVALID_MARKERS)]
    
    if len(valid_s) < 2:
        return np.nan
        
    return valid_s.std()

# Main Function
def generate_std_analysis(df, dataset_label, model_type, filename_prefix, include_title=True):
    """
    1. Calculates Standard Deviation of answers per Question.
    2. Prints a summary table of the Mean Std Dev per model.
    3. Generates 2x2 Boxplots showing the distribution of Std Devs.
    """
    df = df.copy()
    prompts = [1, 2, 3, 4]
    
    # Define model order
    it_order = [
        'Llama-3.1-8B-Instruct', 
        'Mistral-7B-Instruct-v0.3', 
        'Qwen2.5-7B-Instruct', 
        'gemma-2-9b-it'
    ]
    
    base_order = [
        'Llama-3.1-8B', 
        'Mistral-7B-v0.3', 
        'Qwen2.5-7B', 
        'gemma-2-9b'
    ]
    
    # Select order
    if "Base" in model_type:
        model_order = base_order
    else:
        model_order = it_order

    # Font Sizes
    FS_MAIN_TITLE = 22
    FS_SUB_TITLE  = 20
    FS_AXIS_LABEL = 18
    FS_TICKS      = 15
    FS_LEG_TEXT   = 16

    plot_data = []
    summary_stats = []

    # --- 1. Calculate Std Dev ---
    for p in prompts:
        col = f'num_value{p}'
        if col not in df.columns: continue
        
        # Calculate Std Dev per question
        std_per_question = df.groupby(['model', 'question_id'])[col].apply(get_valid_std).reset_index(name='std_dev')
        
        # Add metadata for plotting
        std_per_question['prompt'] = p
        plot_data.append(std_per_question)
        
        # Calculate Summary Metric for Table
        mean_stds = std_per_question.groupby('model')['std_dev'].mean().reset_index()
        for _, row in mean_stds.iterrows():
            summary_stats.append({
                'model': row['model'],
                'prompt': p,
                'mean_std_dev': row['std_dev']
            })

    if not plot_data:
        print(f"[{dataset_label}-{model_type}] No data found.")
        return

    full_plot_df = pd.concat(plot_data, ignore_index=True)
    summary_df = pd.DataFrame(summary_stats)

    # --- 2. Print Summary Table (Optional Log) ---
    print(f"\n--- {dataset_label} ({model_type}): Mean Standard Deviation (Lower = More Consistent) ---")
    pivot_table = summary_df.pivot(index='model', columns='prompt', values='mean_std_dev')
    print(pivot_table.round(3))
    print("-" * 60)

    # --- 3. Plotting (2x2 Grid of Boxplots) ---
    fig, axes = plt.subplots(2, 2, figsize=(19, 10))
    
    # Conditional Title
    if include_title:
        fig.suptitle(f'Volatility Analysis: Standard Deviation of Answers ({dataset_label}) - {model_type}', 
                     fontsize=FS_MAIN_TITLE, fontweight='bold', y=0.98)
    
    axes_flat = axes.flatten()

    for idx, p in enumerate(prompts):
        ax = axes_flat[idx]
        
        # Filter data for this prompt
        prompt_data = full_plot_df[full_plot_df['prompt'] == p]
        
        if not prompt_data.empty:
            sns.boxplot(
                data=prompt_data,
                x='model',
                y='std_dev',
                color='#1f77b4',
                width=0.5,
                ax=ax,
                order=model_order,
                fliersize=3, 
                linewidth=1.5,
                showmeans=True,
                meanline=True,
                meanprops={'color': '#ff7f0e', 'linewidth': 2, 'linestyle': '--'}
            )

        # Styling
        ax.set_title(f'Prompt {p}', fontsize=FS_SUB_TITLE)
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelsize=FS_TICKS, rotation=10)
        ax.tick_params(axis='y', labelsize=FS_TICKS)
        ax.set_ylim(0, 1.45)
        
        if idx % 2 == 0:
            ax.set_ylabel('Standard Deviation (Per Question)', fontsize=FS_AXIS_LABEL)
        else:
            ax.set_ylabel('')
            
        # Legend (Only on first plot)
        if idx == 0:
            legend_elements = [
                Line2D([0], [0], color='#1f77b4', lw=4, label='IQR (Median inside)'),
                Line2D([0], [0], color='#ff7f0e', linewidth=2, linestyle='--', label='Mean Std Dev'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=5, label='Outliers')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=FS_LEG_TEXT)

    # Layout Adjustment
    top_margin = 0.91 if include_title else 0.99
    plt.subplots_adjust(top=top_margin, bottom=0.08, left=0.05, right=0.95, hspace=0.3, wspace=0.15)
    
    # Save Logic
    if include_title:
        filename = f"{filename_prefix}_std_boxplot.pdf"
        save_dir = os.path.join(script_dir, "plots")
    else:
        filename = f"{filename_prefix}_std_boxplot_notitle.pdf"
        save_dir = os.path.join(script_dir, "plots", "notitle")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved std analysis to {save_path}")
    plt.close()

# 1. Instruction Tuned
# With Title
generate_std_analysis(merged_res_reworded_it, "Reworded", "Instruction-Tuned Models", "volatility_it_rew", include_title=True)
generate_std_analysis(merged_res_shuffled_it, "Shuffled", "Instruction-Tuned Models", "volatility_it_shuff", include_title=True)
# Without Title
generate_std_analysis(merged_res_reworded_it, "Reworded", "Instruction-Tuned Models", "volatility_it_rew", include_title=False)
generate_std_analysis(merged_res_shuffled_it, "Shuffled", "Instruction-Tuned Models", "volatility_it_shuff", include_title=False)

# 2. Base Models
# With Title
generate_std_analysis(merged_res_reworded_it_bm, "Reworded", "Base Models", "volatility_bm_rew", include_title=True)
generate_std_analysis(merged_res_shuffled_bm, "Shuffled", "Base Models", "volatility_bm_shuff", include_title=True)
# Without Title
generate_std_analysis(merged_res_reworded_it_bm, "Reworded", "Base Models", "volatility_bm_rew", include_title=False)
generate_std_analysis(merged_res_shuffled_bm, "Shuffled", "Base Models", "volatility_bm_shuff", include_title=False)

#### Notes on Interpretation of Matrices:
# Analysis provides a view of internal consistency -> How much did the numeric score wobble across variations of answer options?

## Instruction Tuned models:
"""
Reworded (Instruction-Tuned Models): Mean Standard Deviation (Lower = More Consistent)
prompt                        1      2      3      4
model
Llama-3.1-8B-Instruct     0.207  0.230  0.248  0.203
Mistral-7B-Instruct-v0.3  0.221  0.227  0.287  0.199
Qwen2.5-7B-Instruct       0.232  0.249  0.244  0.235
gemma-2-9b-it             0.174  0.171  0.179  0.148

Shuffled (Instruction-Tuned Models): Mean Standard Deviation (Lower = More Consistent)
prompt                        1      2      3      4
model
Llama-3.1-8B-Instruct     0.277  0.232  0.146  0.273
Mistral-7B-Instruct-v0.3  0.263  0.320  0.242  0.250
Qwen2.5-7B-Instruct       0.310  0.277  0.207  0.279
gemma-2-9b-it             0.233  0.187  0.122  0.217
"""
# - Gemma-2-9b-it is the "Stability Queen" (Consistently lowest Std Dev across all Reworded prompts, ~0.17):
# => This confirms Gemma has the tightest internal logic. When you ask the same question 5 times with different words,
# Gemma's numeric answers cluster much closer together than any other model.

# - The "Prompt 3 Paradox" in Shuffled Data (Prompt 3 has the LOWEST volatility for Llama, Gemma, and Qwen, e.g. Gemma 0.122):
# => This is a fascinating finding. Prompt 3 has no answer labels (no A, B, C).
# => In Prompts 1, 2, & 4, models likely suffer from "Label vs. Content" conflict
# (e.g., "I want to pick 'A' because I like the letter A, but I want to pick 'Agree' because of the text"). Shuffling breaks this link, causing high volatility.
# => In Prompt 3, there are no labels. The model is forced to rely only on the text content. Paradoxically,
# this makes it more consistent under shuffling because there is no positional/label bias to confuse it.

# - Shuffling introduces more noise than Rewording (Values generally jump from ~0.20 to ~0.27):
# => Changing the order of answers is harder for models to handle than changing the wording of answers.

# - Prompt 4 (Standard 1-8 scale with labels) is the most stable format for Rewording (Values ~0.20):
# => The combination of explicit numeric labels (1-8) and standard wording provides the strongest "anchor" for the models, reducing volatility.

## Base models:
"""
Reworded (Base Models): Mean Standard Deviation (Lower = More Consistent)
prompt               1      2      3      4
model
Llama-3.1-8B     0.357  0.315  0.411  0.331
Mistral-7B-v0.3  0.299  0.259  0.454  0.275
Qwen2.5-7B       0.254  0.304  0.416  0.221
gemma-2-9b       0.344  0.325  0.421  0.229

Shuffled (Base Models): Mean Standard Deviation (Lower = More Consistent)
prompt               1      2      3      4
model
Llama-3.1-8B     0.367  0.403  0.445  0.328
Mistral-7B-v0.3  0.503  0.400  0.365  0.482
Qwen2.5-7B       0.323  0.361  0.363  0.330
gemma-2-9b       0.553  0.346  0.415  0.378
"""
# - Overall Volatility is roughly 2x higher than Instruction Tuned models (Values often > 0.35 or 0.40):
# => Base models are much "noisier." Even when they give the "same" answer text, the underlying numeric probability/value fluctuates wildly.
# They lack the fine-tuned conviction of IT models.

# - Prompt 3 (Reworded) is a "Collapse point" for Base Models (Highest volatility, ~0.41 - 0.45):
# => Unlike IT models, Base models need the labels (A, B, C) to structure their output.
# When you remove labels (Prompt 3), Base models struggle to map the question to a consistent score, resulting in high volatility.

# - Gemma-2-9b Base crashes under Shuffling (Std Dev spikes to 0.553 in Prompt 1):
# => While Gemma IT was the most stable, Gemma Base is extremely sensitive to answer order.
# This highlights that Gemma's stability is almost entirely a result of its Instruction Tuning/RLHF, not its base architecture.

# - Qwen2.5-7B is the most robust Base Model (Consistently lower Std Devs, e.g., ~0.25 Reworded):
# => Qwen Base behaves more like an Instruction Tuned model than the others.
# It maintains respectable consistency even without the "Instruction" training, suggesting a very high-quality pre-training dataset involving logic or multiple-choice tasks.

# - Mistral-7B-v0.3 Base is highly unpredictable (High volatility across the board, peaking at 0.503):
# => This confirms Mistral Base treats these prompts more as creative writing generation than logical evaluation,
# leading to answers that drift significantly based on minor input changes.


### ---------------------------------------------------------------------------
### Descriptive Statisitcs ### ### ### ### ###
### ---------------------------------------------------------------------------

reworded_data = pd.read_csv(os.path.join(script_dir, "data", "clean", "opinionQA_questions_final.csv"), delimiter=',')
shuffled_data = pd.read_csv(os.path.join(script_dir, "data", "clean", "opQA_shuffled_ans-opt.csv"), delimiter=',')

def print_simple_stats(df, name):
    # Filter for unique questions only
    unique_df = df.drop_duplicates(subset='question_id')
    
    print(f"\n--- {name} ---")
    print(f"Total Unique Questions: {len(unique_df)}")
    
    # --- Helper to create Count + % table ---
    def get_stats_table(series):
        counts = series.value_counts()
        percs = series.value_counts(normalize=True) * 100
        stats = pd.DataFrame({'Count': counts, 'Percentage': percs})
        stats['Percentage'] = stats['Percentage'].map('{:.2f}%'.format)
        return stats

    print("\n[Question Types]")
    print(get_stats_table(unique_df['question_type']))
    
    print("\n[Subjects]")
    print(get_stats_table(unique_df['subject']))

# Same results for both datasets, they share same questions
print_simple_stats(reworded_data, "Reworded Data")
print_simple_stats(shuffled_data, "Shuffled Data")

def print_scale_stats_with_percentage(df, name):
    print(f"\n--- {name} ---")
    print(f"Total Questions: {len(df)}")
    # Calculate Counts and Percentages
    counts = df['scale_type'].value_counts()
    percs = df['scale_type'].value_counts(normalize=True) * 100
        
    # Combine into a DataFrame for a clean view
    stats = pd.DataFrame({'Count': counts, 'Percentage': percs})
        
    # Format the percentage column
    stats['Percentage'] = stats['Percentage'].map('{:.2f}%'.format)
        
    print(stats)

# Scales are different (all rows to capture different scale variations)
print_scale_stats_with_percentage(reworded_data, "Reworded Data")
print_scale_stats_with_percentage(shuffled_data, "Shuffled Data")

"""
--- Reworded Data & Shuffled Data (same for both, they share same questions) ---
Total Unique Questions: 1235
[Question Types]
                  Count Percentage
question_type
Agreement           404     32.71%
Quantity            214     17.33%
BetterOrWorse       101      8.18%
Importance           91      7.37%
Likelihood           85      6.88%
PositiveNegative     67      5.43%
GoodOrBad            50      4.05%
Problem              48      3.89%
Frequency            38      3.08%
HowWell              38      3.08%
Priority             35      2.83%
Acceptance           31      2.51%
Reason               15      1.21%
Concern              11      0.89%
IncreaseDecrease      7      0.57%
[Subjects]
                       Count Percentage
subject
Politics & Government    297     24.05%
General Opinion          272     22.02%
Economy & Work           226     18.30%
Science                  110      8.91%
Race & Ethnicity          83      6.72%
Media & News              74      5.99%
Trust & Fear              56      4.53%
Guns                      27      2.19%
Social                    22      1.78%
Crime                     20      1.62%
Education                 19      1.54%
Media                     15      1.21%
Ethics & Values           14      1.13%


--- Reworded Data ---
Total Questions: 6175
            Count Percentage
scale_type
4-unipolar   1858     30.09%
4-bipolar    1760     28.50%
5-bipolar    1375     22.27%
5-unipolar    635     10.28%
6-bipolar     512      8.29%
6-unipolar     35      0.57%


--- Shuffled Data ---
Total Questions: 7410
            Count Percentage
scale_type
4-unipolar   3408     45.99%
4-bipolar    2424     32.71%
5-bipolar    1350     18.22%
5-unipolar    228      3.08%
"""
