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
merged_res_reworded_bm = process_and_merge(res1_reworded_bm, res2_reworded_bm, res3_reworded_bm, res4_reworded_bm)
print(len(merged_res_reworded_bm)) #### 24700

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
plot_distributions(merged_res_reworded_bm, "Reworded", "Base Models", "bm_rew", mode='absolute', include_title=True)
plot_distributions(merged_res_reworded_bm, "Reworded", "Base Models", "bm_rew", mode='percentage', include_title=True)
plot_distributions(merged_res_shuffled_bm, "Shuffled", "Base Models", "bm_shuff", mode='absolute', include_title=True)
plot_distributions(merged_res_shuffled_bm, "Shuffled", "Base Models", "bm_shuff", mode='percentage', include_title=True)

## Without title (for paper)
# 3. Instruction Tuned Models
plot_distributions(merged_res_reworded_it, "Reworded", "Instruction-Tuned", "it_rew", mode='absolute', include_title=False)
plot_distributions(merged_res_reworded_it, "Reworded", "Instruction-Tuned", "it_rew", mode='percentage', include_title=False)
plot_distributions(merged_res_shuffled_it, "Shuffled", "Instruction-Tuned", "it_shuff", mode='absolute', include_title=False)
plot_distributions(merged_res_shuffled_it, "Shuffled", "Instruction-Tuned", "it_shuff", mode='percentage', include_title=False)

# 4. Base Models
plot_distributions(merged_res_reworded_bm, "Reworded", "Base Models", "bm_rew", mode='absolute', include_title=False)
plot_distributions(merged_res_reworded_bm, "Reworded", "Base Models", "bm_rew", mode='percentage', include_title=False)
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
plot_consistency(merged_res_reworded_bm, "Reworded", "Base Models", "bm_rew", mode='absolute', include_title=True)
plot_consistency(merged_res_shuffled_bm, "Shuffled", "Base Models", "bm_shuff", mode='absolute', include_title=True)
plot_consistency(merged_res_reworded_bm, "Reworded", "Base Models", "bm_rew", mode='percentage', include_title=True)
plot_consistency(merged_res_shuffled_bm, "Shuffled", "Base Models", "bm_shuff", mode='percentage', include_title=True)

## Without title (for paper)
# 3. Instruction Tuned Models
plot_consistency(merged_res_reworded_it, "Reworded", "Instruction-Tuned Models", "it_rew", mode='absolute', include_title=False)
plot_consistency(merged_res_shuffled_it, "Shuffled", "Instruction-Tuned Models", "it_shuff", mode='absolute', include_title=False)
plot_consistency(merged_res_reworded_it, "Reworded", "Instruction-Tuned Models", "it_rew", mode='percentage', include_title=False)
plot_consistency(merged_res_shuffled_it, "Shuffled", "Instruction-Tuned Models", "it_shuff", mode='percentage', include_title=False)

# 4. Base Models
plot_consistency(merged_res_reworded_bm, "Reworded", "Base Models", "bm_rew", mode='absolute', include_title=False)
plot_consistency(merged_res_shuffled_bm, "Shuffled", "Base Models", "bm_shuff", mode='absolute', include_title=False)
plot_consistency(merged_res_reworded_bm, "Reworded", "Base Models", "bm_rew", mode='percentage', include_title=False)
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
generate_avg_distance_plots(merged_res_reworded_bm, "Reworded", "Base Models", "apd_distr_bm_rew", include_title=True)
generate_avg_distance_plots(merged_res_shuffled_bm, "Shuffled", "Base Models", "apd_distr_bm_shuff", include_title=True)

## Without title
# 3. Instruction Tuned
generate_avg_distance_plots(merged_res_reworded_it, "Reworded", "Instruction-Tuned Models", "apd_distr_it_rew", include_title=False)
generate_avg_distance_plots(merged_res_shuffled_it, "Shuffled", "Instruction-Tuned Models", "apd_distr_it_shuff", include_title=False)

# 4. Base Models
generate_avg_distance_plots(merged_res_reworded_bm, "Reworded", "Base Models", "apd_distr_bm_rew", include_title=False)
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
generate_max_distance_plots(merged_res_reworded_bm, "Reworded", "Base Models", "mpd_distr_bm_rew", include_title=True)
generate_max_distance_plots(merged_res_shuffled_bm, "Shuffled", "Base Models", "mpd_distr_bm_shuff", include_title=True)

## Without title
# 3. Instruction Tuned
generate_max_distance_plots(merged_res_reworded_it, "Reworded", "Instruction-Tuned Models", "mpd_distr_it_rew", include_title=False)
generate_max_distance_plots(merged_res_shuffled_it, "Shuffled", "Instruction-Tuned Models", "mpd_distr_it_shuff", include_title=False)

# 4. Base Models
generate_max_distance_plots(merged_res_reworded_bm, "Reworded", "Base Models", "mpd_distr_bm_rew", include_title=False)
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
generate_bias_overlap_plots(merged_res_reworded_bm, "Reworded", merged_res_shuffled_bm, "Shuffled", "ans_distr_bm", include_title=True)
# Without Title
generate_bias_overlap_plots(merged_res_reworded_bm, "Reworded", merged_res_shuffled_bm, "Shuffled", "ans_distr_bm", include_title=False)


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
generate_stability_analysis(merged_res_reworded_bm, "Reworded", "Base Models", "stability_bm_rew", include_title=True)
generate_stability_analysis(merged_res_shuffled_bm, "Shuffled", "Base Models", "stability_bm_shuff", include_title=True)
# Without Title
generate_stability_analysis(merged_res_reworded_bm, "Reworded", "Base Models", "stability_bm_rew", include_title=False)
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

def get_valid_std(values, expected_n):
    """
    Calculates std dev ONLY if the number of valid values matches expected_n.
    Otherwise returns NaN (excluding the question).
    """
    # Convert to numeric Series
    s = pd.Series(pd.to_numeric(values, errors='coerce'))
    
    # Remove NaNs and Invalid Markers
    valid_s = s[~s.isna() & ~s.isin(INVALID_MARKERS)]
    
    # Check: Exclude if we don't have the full set of answers
    if len(valid_s) != expected_n:
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
    
    # --- 0. Define Sorting Order ---
    it_order = ['Llama-3.1-8B-Instruct', 'Mistral-7B-Instruct-v0.3', 'Qwen2.5-7B-Instruct', 'gemma-2-9b-it']
    base_order = ['Llama-3.1-8B', 'Mistral-7B-v0.3', 'Qwen2.5-7B', 'gemma-2-9b']
    
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
        
        # Group by Question to get the list of 5 (or 6) answers
        grouped = df.groupby(['model', 'question_id'])[col].apply(list).reset_index(name='values_list')
        
        if grouped.empty: continue

        # Dynamically determine expected N (e.g., 5 for Reworded, 6 for Shuffled)
        typical_len = int(grouped['values_list'].apply(len).median())

        # Apply Strict Calculation
        grouped['std_dev'] = grouped['values_list'].apply(lambda x: get_valid_std(x, typical_len))
        
        # Drop excluded questions (NaNs)
        std_per_question = grouped.dropna(subset=['std_dev']).copy()
        
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

    # --- 2. Print Summary Table ---
    print(f"\n--- {dataset_label} ({model_type}): Mean Standard Deviation (Lower = More Consistent) ---")
    pivot_table = summary_df.pivot(index='model', columns='prompt', values='mean_std_dev')
    print(pivot_table.round(3))
    print("-" * 60)

    # --- 3. Plotting (2x2 Grid of Boxplots) ---
    fig, axes = plt.subplots(2, 2, figsize=(19, 10))
    
    if include_title:
        fig.suptitle(f'Volatility Analysis: Standard Deviation of Answers ({dataset_label}) - {model_type}', 
                     fontsize=FS_MAIN_TITLE, fontweight='bold', y=0.98)
    
    axes_flat = axes.flatten()

    for idx, p in enumerate(prompts):
        ax = axes_flat[idx]
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
generate_std_analysis(merged_res_reworded_bm, "Reworded", "Base Models", "volatility_bm_rew", include_title=True)
generate_std_analysis(merged_res_shuffled_bm, "Shuffled", "Base Models", "volatility_bm_shuff", include_title=True)
# Without Title
generate_std_analysis(merged_res_reworded_bm, "Reworded", "Base Models", "volatility_bm_rew", include_title=False)
generate_std_analysis(merged_res_shuffled_bm, "Shuffled", "Base Models", "volatility_bm_shuff", include_title=False)

#### Notes on Interpretation of Matrices:
# Analysis provides a view of internal consistency -> How much did the numeric score wobble across variations of answer options?

## Instruction Tuned models:
"""
Reworded (Instruction-Tuned Models): Mean Standard Deviation (Lower = More Consistent)
prompt                        1      2      3      4
model
Llama-3.1-8B-Instruct     0.207  0.229  0.248  0.203
Mistral-7B-Instruct-v0.3  0.221  0.227  0.287  0.206
Qwen2.5-7B-Instruct       0.232  0.249  0.244  0.241
gemma-2-9b-it             0.174  0.172  0.178  0.164

Shuffled (Instruction-Tuned Models): Mean Standard Deviation (Lower = More Consistent)
prompt                        1      2      3      4
model
Llama-3.1-8B-Instruct     0.277  0.233  0.146  0.274
Mistral-7B-Instruct-v0.3  0.263  0.320  0.240  0.249
Qwen2.5-7B-Instruct       0.310  0.277  0.206  0.273
gemma-2-9b-it             0.232  0.187  0.121  0.220
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
Llama-3.1-8B     0.357  0.315  0.411  0.340
Mistral-7B-v0.3  0.302  0.254  0.453  0.294
Qwen2.5-7B       0.254  0.304  0.415  0.224
gemma-2-9b       0.344  0.325  0.421  0.229

Shuffled (Base Models): Mean Standard Deviation (Lower = More Consistent)
prompt               1      2      3      4
model
Llama-3.1-8B     0.367  0.403  0.445  0.328
Mistral-7B-v0.3  0.508  0.403  0.365  0.510
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

## --- How many questions are exluded by ignoring all with one or more unusable, "don't know" or "refused" answers? ---
## So the percentage of questions excluded when there's min. one answer that doesn't fit on [-1, 1] scale:

def analyze_exclusion_by_model(df, dataset_name):
    """
    Generates an exclusion table for EACH MODEL separately.
    Calculates the percentage of questions excluded per prompt for that specific model.
    """
    models = sorted(df['model'].unique())
    
    print(f"\n{'#'*60}")
    print(f" DETAILED EXCLUSION ANALYSIS BY MODEL: {dataset_name}")
    print(f"{'#'*60}")
    
    for model in models:
        # Filter data for this specific model
        df_model = df[df['model'] == model]
        
        results = []
        total_questions = df_model['question_id'].nunique()
        
        for p in range(1, 5):
            col = f'num_value{p}'
            if col not in df_model.columns: continue
                
            # --- Identify Invalid Answers ---
            # 1. NaN (Unusable)
            mask_nan = df_model[col].isna()
            
            # 2. Extended Options (-98, -99)
            mask_ext = df_model[col].isin([-98, -99])
            
            # 3. Total Invalid (Strictly outside [-1, 1], including NaN)
            numeric_vals = df_model[col].fillna(0) 
            mask_out_of_bounds = (numeric_vals < -1) | (numeric_vals > 1)
            mask_total_invalid = mask_nan | mask_out_of_bounds

            # --- Counts ---
            count_nan = df_model.loc[mask_nan, 'question_id'].nunique()
            count_ext = df_model.loc[mask_ext, 'question_id'].nunique()
            count_total = df_model.loc[mask_total_invalid, 'question_id'].nunique()
            
            pct_nan = (count_nan / total_questions) * 100
            pct_ext = (count_ext / total_questions) * 100
            pct_total = (count_total / total_questions) * 100
            
            results.append({
                'Prompt': f'P{p}',
                'Questions': total_questions,
                'Unusable (NaN)': f"{count_nan} ({pct_nan:.1f}%)",
                'Extended (-98/-99)': f"{count_ext} ({pct_ext:.1f}%)",
                'Total Excluded': f"{count_total} ({pct_total:.1f}%)"
            })
            
        res_df = pd.DataFrame(results)
        print(f"\n>> Model: {model}")
        print(res_df.to_string(index=False, col_space=15))
        print("-" * 60)

# 1. Instruction Tuned
analyze_exclusion_by_model(merged_res_reworded_it, "Reworded - IT")
analyze_exclusion_by_model(merged_res_shuffled_it, "Shuffled - IT")

# 2. Base Models
analyze_exclusion_by_model(merged_res_reworded_bm, "Reworded - Base")
analyze_exclusion_by_model(merged_res_shuffled_bm, "Shuffled - Base")

"""
############################################################
 DETAILED EXCLUSION ANALYSIS BY MODEL: Reworded - IT
############################################################
>> Model: Llama-3.1-8B-Instruct
         Prompt       Questions  Unusable (NaN) Extended (-98/-99)  Total Excluded
             P1            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P2            1235        5 (0.4%)           0 (0.0%)        5 (0.4%)
             P3            1235        1 (0.1%)           0 (0.0%)        1 (0.1%)
             P4            1235        0 (0.0%)        159 (12.9%)     159 (12.9%)

>> Model: Mistral-7B-Instruct-v0.3
         Prompt       Questions  Unusable (NaN) Extended (-98/-99)  Total Excluded
             P1            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P2            1235        2 (0.2%)           0 (0.0%)        2 (0.2%)
             P3            1235       14 (1.1%)           0 (0.0%)       14 (1.1%)
             P4            1235        0 (0.0%)        311 (25.2%)     311 (25.2%)

>> Model: Qwen2.5-7B-Instruct
         Prompt       Questions  Unusable (NaN) Extended (-98/-99)  Total Excluded
             P1            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P2            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P3            1235        1 (0.1%)           0 (0.0%)        1 (0.1%)
             P4            1235        0 (0.0%)        234 (18.9%)     234 (18.9%)

>> Model: gemma-2-9b-it
         Prompt       Questions  Unusable (NaN) Extended (-98/-99)  Total Excluded
             P1            1235        3 (0.2%)           0 (0.0%)        3 (0.2%)
             P2            1235        2 (0.2%)           0 (0.0%)        2 (0.2%)
             P3            1235        1 (0.1%)           0 (0.0%)        1 (0.1%)
             P4            1235        0 (0.0%)        514 (41.6%)     514 (41.6%)

############################################################
 DETAILED EXCLUSION ANALYSIS BY MODEL: Shuffled - IT
############################################################
>> Model: Llama-3.1-8B-Instruct
         Prompt       Questions  Unusable (NaN) Extended (-98/-99)  Total Excluded
             P1            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P2            1235        2 (0.2%)           0 (0.0%)        2 (0.2%)
             P3            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P4            1235        0 (0.0%)          79 (6.4%)       79 (6.4%)

>> Model: Mistral-7B-Instruct-v0.3
         Prompt       Questions  Unusable (NaN) Extended (-98/-99)  Total Excluded
             P1            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P2            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P3            1235       17 (1.4%)           0 (0.0%)       17 (1.4%)
             P4            1235        0 (0.0%)        228 (18.5%)     228 (18.5%)

>> Model: Qwen2.5-7B-Instruct
         Prompt       Questions  Unusable (NaN) Extended (-98/-99)  Total Excluded
             P1            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P2            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P3            1235        2 (0.2%)           0 (0.0%)        2 (0.2%)
             P4            1235        0 (0.0%)         122 (9.9%)      122 (9.9%)

>> Model: gemma-2-9b-it
         Prompt       Questions  Unusable (NaN) Extended (-98/-99)  Total Excluded
             P1            1235        2 (0.2%)           0 (0.0%)        2 (0.2%)
             P2            1235        1 (0.1%)           0 (0.0%)        1 (0.1%)
             P3            1235        3 (0.2%)           0 (0.0%)        3 (0.2%)
             P4            1235        0 (0.0%)        558 (45.2%)     558 (45.2%)

             
############################################################
 DETAILED EXCLUSION ANALYSIS BY MODEL: Reworded - Base
############################################################
>> Model: Llama-3.1-8B
         Prompt       Questions  Unusable (NaN) Extended (-98/-99)  Total Excluded
             P1            1235        3 (0.2%)           0 (0.0%)        3 (0.2%)
             P2            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P3            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P4            1235        3 (0.2%)          68 (5.5%)       71 (5.7%)

>> Model: Mistral-7B-v0.3
         Prompt       Questions  Unusable (NaN) Extended (-98/-99)  Total Excluded
             P1            1235     192 (15.5%)           0 (0.0%)     192 (15.5%)
             P2            1235     173 (14.0%)           0 (0.0%)     173 (14.0%)
             P3            1235       19 (1.5%)           0 (0.0%)       19 (1.5%)
             P4            1235     607 (49.1%)          43 (3.5%)     641 (51.9%)

>> Model: Qwen2.5-7B
         Prompt       Questions  Unusable (NaN) Extended (-98/-99)  Total Excluded
             P1            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P2            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P3            1235        6 (0.5%)           0 (0.0%)        6 (0.5%)
             P4            1235        0 (0.0%)          92 (7.4%)       92 (7.4%)

>> Model: gemma-2-9b
         Prompt       Questions  Unusable (NaN) Extended (-98/-99)  Total Excluded
             P1            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P2            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P3            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P4            1235        0 (0.0%)           3 (0.2%)        3 (0.2%)

############################################################
 DETAILED EXCLUSION ANALYSIS BY MODEL: Shuffled - Base
############################################################
>> Model: Llama-3.1-8B
         Prompt       Questions  Unusable (NaN) Extended (-98/-99)  Total Excluded
             P1            1235        1 (0.1%)           0 (0.0%)        1 (0.1%)
             P2            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P3            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P4            1235        0 (0.0%)          19 (1.5%)       19 (1.5%)

>> Model: Mistral-7B-v0.3
         Prompt       Questions  Unusable (NaN) Extended (-98/-99)  Total Excluded
             P1            1235     146 (11.8%)           0 (0.0%)     146 (11.8%)
             P2            1235      116 (9.4%)           0 (0.0%)      116 (9.4%)
             P3            1235       10 (0.8%)           0 (0.0%)       10 (0.8%)
             P4            1235     499 (40.4%)           6 (0.5%)     505 (40.9%)

>> Model: Qwen2.5-7B
         Prompt       Questions  Unusable (NaN) Extended (-98/-99)  Total Excluded
             P1            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P2            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P3            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P4            1235        0 (0.0%)          14 (1.1%)       14 (1.1%)

>> Model: gemma-2-9b
         Prompt       Questions  Unusable (NaN) Extended (-98/-99)  Total Excluded
             P1            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P2            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P3            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
             P4            1235        0 (0.0%)           0 (0.0%)        0 (0.0%)
"""


### ---------------------------------------------------------------------------
### "Mixed Plots" for slides ### ### ### ### ###
### ---------------------------------------------------------------------------

## ----------------------------------------------------------------------------
## Distribution of clean_response across 2 prompts for both model types ## ## ##
## ----------------------------------------------------------------------------

def plot_distributions_mixed(df_base, df_it, dataset_label, prompts, filename_suffix, include_title=True):
    """
    Creates a 2x2 grid comparing Base Models (Left) vs Instruction-Tuned Models (Right)
    for two specific prompts. (Percentage Mode Only).
    
    Parameters:
    - df_base: Dataframe for Base Models.
    - df_it: Dataframe for Instruction-Tuned Models.
    - dataset_label: "Reworded" or "Shuffled".
    - prompts: Tuple of two integers, e.g., (1, 2).
    - filename_suffix: e.g., "compare_p1_p2".
    - include_title: Boolean, if False, suppresses the main figure title.
    """
    
    # --- 0. Define Sorting Order ---
    it_order = [
        'Llama-3.1-8B-Instruct', 'Mistral-7B-Instruct-v0.3', 
        'Qwen2.5-7B-Instruct', 'gemma-2-9b-it'
    ]
    base_order = [
        'Llama-3.1-8B', 'Mistral-7B-v0.3', 
        'Qwen2.5-7B', 'gemma-2-9b'
    ]

    # Font Sizes
    FS_MAIN_TITLE = 22
    FS_SUB_TITLE  = 20
    FS_AXIS_LABEL = 18
    FS_TICKS      = 15
    FS_LEG_TITLE  = 18
    FS_LEG_TEXT   = 16

    # Construct Dynamic Title
    p1, p2 = prompts
    plot_title = f'Distribution of Responses on {dataset_label} Dataset'
    
    # Construct Filename
    filename = f"ans_distr_comp_{filename_suffix}.pdf"
    
    # Ensure save directory exists
    slides_dir = os.path.join(script_dir, "plots", "slides")
    if not os.path.exists(slides_dir):
        os.makedirs(slides_dir)
    save_path = os.path.join(slides_dir, filename)

    # Define Column Mapping
    cols_config = [
        (df_base, "Base Models", base_order), 
        (df_it, "IT Models", it_order)
    ]
    
    # Define Rows (Prompts)
    rows_config = [
        (f'clean_response{p1}', f'Prompt {p1}'),
        (f'clean_response{p2}', f'Prompt {p2}')
    ]

    all_possible_responses = ['1', '2', '3', '4', '5', '6', '7', '8', 'unusable']
    
    custom_colors = [
        '#2ca02c', '#ff7f0e', '#1f77b4', '#d62728', 
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'
    ]
    color_mapping = {resp: custom_colors[i] for i, resp in enumerate(all_possible_responses)}

    # --- Plotting ---
    y_limit = 70
    fig, axes = plt.subplots(2, 2, figsize=(19, 10))
    
    if include_title:
        fig.suptitle(plot_title, fontsize=FS_MAIN_TITLE, fontweight='bold', y=0.98)
    
    # Loop over Rows (Prompts)
    for row_idx, (col_name, prompt_title) in enumerate(rows_config):
        # Loop over Columns (Base vs IT)
        for col_idx, (df, type_label, order) in enumerate(cols_config):
            ax = axes[row_idx, col_idx]
            
            hue_order = all_possible_responses
            
            # Percentage Calculation
            counts = df.groupby(['model', col_name]).size().reset_index(name='count')
            totals = df.groupby('model').size().reset_index(name='total')
            props = pd.merge(counts, totals, on='model')
            props['percentage'] = (props['count'] / props['total']) * 100
            
            sns.barplot(
                data=props, x='model', y='percentage', hue=col_name, ax=ax,
                hue_order=hue_order, palette=color_mapping, order=order
            )

            # Styling
            ax.set_ylim(0, y_limit)
            
            # Subplot Title: "Base - Prompt 1"
            ax.set_title(f"{type_label} - {prompt_title}", fontsize=FS_SUB_TITLE)
            
            ax.set_xlabel('') 
            ax.tick_params(axis='x', labelsize=FS_TICKS, rotation=10)
            ax.tick_params(axis='y', labelsize=FS_TICKS)
            
            # Y-Label only on left column
            if col_idx == 0:
                ax.set_ylabel('Percentage of Responses', fontsize=FS_AXIS_LABEL)
            else:
                ax.set_ylabel('')
            
            if ax.get_legend():
                ax.get_legend().remove()

    # --- Legend & Save ---
    top_margin = 0.91 if include_title else 0.99
    plt.subplots_adjust(left=0.05, right=0.85, top=top_margin, bottom=0.1, hspace=0.35, wspace=0.15)
    
    if dataset_label == "Shuffled":
        legend_responses = ['1', '2', '3', '4', '5', '6', '7', 'unusable']
    else:
        legend_responses = ['1', '2', '3', '4', '5', '6', '7', '8', 'unusable']

    handles = [mpatches.Patch(facecolor=color_mapping[r], edgecolor='black', linewidth=0.5, label=r) for r in legend_responses]
        
    fig.legend(
        handles=handles, labels=legend_responses, title='Response Options',
        title_fontsize=FS_LEG_TITLE, loc='center right', bbox_to_anchor=(0.99, 0.5),
        fontsize=FS_LEG_TEXT, frameon=True, shadow=True
    )
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved slide plot: {save_path}")
    plt.close()

# 1. Reworded: Prompt 1 vs 2 & Prompt 3 vs 4
plot_distributions_mixed(merged_res_reworded_bm, merged_res_reworded_it, "Reworded", (1, 2), "rew_p1v2", include_title=True)
plot_distributions_mixed(merged_res_reworded_bm, merged_res_reworded_it, "Reworded", (3, 4), "rew_p3v4", include_title=True)

# 2. Shuffled: Prompt 1 vs 2 & Prompt 3 vs 4
plot_distributions_mixed(merged_res_shuffled_bm, merged_res_shuffled_it, "Shuffled", (1, 2), "shuff_p1v2", include_title=True)
plot_distributions_mixed(merged_res_shuffled_bm, merged_res_shuffled_it, "Shuffled", (3, 4), "shuff_p3v4", include_title=True)


## ----------------------------------------------------------------------------
## APD & MPD "mixed" plots for slides ## ## ##
## ----------------------------------------------------------------------------

# helpers (reused)
INVALID_MARKERS = {-98, -99}

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

def avg_pairwise_abs_distance(values, require_full=False, expected_n=5):
    """Compute mean of |xi - xj| over all pairs."""
    vals = _to_valid_floats(values)
    n = len(vals)
    if require_full and n != expected_n: return np.nan
    if n < 2: return np.nan
    total = sum(abs(vals[i] - vals[j]) for i in range(n) for j in range(i+1, n))
    count = n * (n - 1) // 2
    return total / count if count > 0 else np.nan

def max_pairwise_abs_distance(values, require_full=False, expected_n=5):
    """Compute max |xi - xj| over all pairs."""
    vals = _to_valid_floats(values)
    n = len(vals)
    if require_full and n != expected_n: return np.nan
    if n < 2: return np.nan
    maxd = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = abs(vals[i] - vals[j])
            if d > maxd: maxd = d
    return maxd

# ==========================
# 1. APD Mixed Plot Function
# ==========================
def generate_avg_distance_plots_mixed(df_base, df_it, base_models, it_models, dataset_label, prompt, filename_suffix):
    """
    Generates a 2x2 APD plot mixing Base and IT models for a specific Prompt.
    Layout: Left Col = Base Models, Right Col = IT Models.
    """
    # Font Sizes
    FS_MAIN_TITLE = 22
    FS_SUB_TITLE  = 20
    FS_AXIS_LABEL = 18
    FS_TICKS      = 15
    FS_LEG_TEXT   = 16

    # Data Preparation
    def get_prompt_data(df, p):
        col = f'num_value{p}'
        if col not in df.columns: return pd.DataFrame()
        grouped = df.groupby(['model', 'question_id'])[col].apply(list).reset_index(name='values_list')
        if grouped.empty: return pd.DataFrame()
        
        typical_len = int(grouped['values_list'].apply(len).median())
        grouped['avg_pairwise_dist'] = grouped['values_list'].apply(
            lambda lst: avg_pairwise_abs_distance(lst, require_full=True, expected_n=typical_len)
        )
        return grouped

    dfp_base = get_prompt_data(df_base, prompt)
    dfp_it = get_prompt_data(df_it, prompt)
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(19, 10))
    plot_title = f'APD Distribution on {dataset_label} Dataset (Prompt {prompt})'
    fig.suptitle(plot_title, fontsize=FS_MAIN_TITLE, fontweight='bold', y=0.98)
    
    # Configuration: (Row, Col, Dataframe, ModelName)
    # Row 0: Model 1 (Base vs IT)
    # Row 1: Model 2 (Base vs IT)
    plot_config = [
        (0, 0, dfp_base, base_models[0]), # Top-Left: Base Model 1
        (0, 1, dfp_it,   it_models[0]),   # Top-Right: IT Model 1
        (1, 0, dfp_base, base_models[1]), # Bottom-Left: Base Model 2
        (1, 1, dfp_it,   it_models[1])    # Bottom-Right: IT Model 2
    ]

    y_limit = 12.8

    for r, c, df, model_name in plot_config:
        ax = axes[r, c]
        
        if not df.empty:
            model_vals = df[df['model'] == model_name]['avg_pairwise_dist'].dropna()
        else:
            model_vals = pd.Series()

        if len(model_vals) > 0:
            # Histogram
            fixed_bins = np.linspace(0, 2.0, 41) 
            sns.histplot(model_vals, bins=fixed_bins, stat='density', alpha=0.6, ax=ax, edgecolor=None, color='#1f77b4')
            
            # KDE
            if model_vals.nunique() > 1:
                sns.kdeplot(model_vals, ax=ax, linewidth=2)
            
            # Metrics
            med = float(model_vals.median())
            mean = float(model_vals.mean())
            
            ax.axvline(med, color='black', linestyle='--', linewidth=1.5, label=f'Median: {med:.3f}')
            ax.axvline(mean, color='black', linestyle='-', linewidth=1.5, label=f'Mean: {mean:.3f}')
            
            ax.legend(loc='upper right', fontsize=FS_LEG_TEXT)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')

        # Styling
        ax.set_xlim(0, 2.0)
        ax.set_ylim(0, y_limit) 
        ax.set_title(model_name, fontsize=FS_SUB_TITLE)
        ax.set_xlabel('Avg Pairwise Distance', fontsize=FS_AXIS_LABEL)
        
        # Y-Label only on left column
        if c == 0:
            ax.set_ylabel('Density', fontsize=FS_AXIS_LABEL)
        else:
            ax.set_ylabel('')
            
        ax.tick_params(axis='x', labelsize=FS_TICKS)
        ax.tick_params(axis='y', labelsize=FS_TICKS)

    # Layout & Save
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.05, right=0.95, hspace=0.35, wspace=0.15)
    
    filename = f"apd_mixed_p{prompt}_{filename_suffix}.pdf"
    save_dir = os.path.join(script_dir, "plots", "slides")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, filename)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved APD slide plot: {save_path}")
    plt.close()


# ==========================
# 2. MPD Mixed Plot Function
# ==========================
def generate_max_distance_plots_mixed(df_base, df_it, base_models, it_models, dataset_label, prompt, filename_suffix):
    """
    Generates a 2x2 MPD plot mixing Base and IT models for a specific Prompt.
    Layout: Left Col = Base Models, Right Col = IT Models.
    """
    # Font Sizes
    FS_MAIN_TITLE = 22
    FS_SUB_TITLE  = 20
    FS_AXIS_LABEL = 18
    FS_TICKS      = 15
    FS_LEG_TEXT   = 16

    # Data Preparation
    def get_prompt_data(df, p):
        col = f'num_value{p}'
        if col not in df.columns: return pd.DataFrame()
        grouped = df.groupby(['model', 'question_id'])[col].apply(list).reset_index(name='values_list')
        if grouped.empty: return pd.DataFrame()
        
        typical_len = int(grouped['values_list'].apply(len).median())
        grouped['max_pairwise_dist'] = grouped['values_list'].apply(
            lambda lst: max_pairwise_abs_distance(lst, require_full=True, expected_n=typical_len)
        )
        return grouped

    dfp_base = get_prompt_data(df_base, prompt)
    dfp_it = get_prompt_data(df_it, prompt)
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(19, 10))
    plot_title = f'MPD Distribution on {dataset_label} Dataset (Prompt {prompt})'
    fig.suptitle(plot_title, fontsize=FS_MAIN_TITLE, fontweight='bold', y=0.98)
    
    # Configuration
    plot_config = [
        (0, 0, dfp_base, base_models[0]), 
        (0, 1, dfp_it,   it_models[0]),   
        (1, 0, dfp_base, base_models[1]), 
        (1, 1, dfp_it,   it_models[1])    
    ]

    y_limit = 14.0

    for r, c, df, model_name in plot_config:
        ax = axes[r, c]
        
        if not df.empty:
            model_vals = df[df['model'] == model_name]['max_pairwise_dist'].dropna()
        else:
            model_vals = pd.Series()

        if len(model_vals) > 0:
            # Histogram
            fixed_bins = np.linspace(0, 2.0, 41) 
            sns.histplot(model_vals, bins=fixed_bins, stat='density', alpha=0.6, ax=ax, edgecolor=None, color='#1f77b4')
            
            # KDE
            if model_vals.nunique() > 1:
                sns.kdeplot(model_vals, ax=ax, linewidth=2)
            
            # Metrics
            med = float(model_vals.median())
            mean = float(model_vals.mean())
            
            ax.axvline(med, color='black', linestyle='--', linewidth=1.5, label=f'Median: {med:.3f}')
            ax.axvline(mean, color='black', linestyle='-', linewidth=1.5, label=f'Mean: {mean:.3f}')
            
            ax.legend(loc='upper right', fontsize=FS_LEG_TEXT)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')

        # Styling
        ax.set_xlim(0, 2.0)
        ax.set_ylim(0, y_limit) 
        ax.set_title(model_name, fontsize=FS_SUB_TITLE)
        ax.set_xlabel('Max Pairwise Distance', fontsize=FS_AXIS_LABEL)
        
        if c == 0:
            ax.set_ylabel('Density', fontsize=FS_AXIS_LABEL)
        else:
            ax.set_ylabel('')
            
        ax.tick_params(axis='x', labelsize=FS_TICKS)
        ax.tick_params(axis='y', labelsize=FS_TICKS)

    # Layout & Save
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.05, right=0.95, hspace=0.35, wspace=0.15)
    
    filename = f"mpd_mixed_p{prompt}_{filename_suffix}.pdf"
    save_dir = os.path.join(script_dir, "plots", "slides")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, filename)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved MPD slide plot: {save_path}")
    plt.close()


# Base models: ['Llama-3.1-8B', 'Mistral-7B-v0.3', 'Qwen2.5-7B', 'gemma-2-9b']
# IT models: ['Llama-3.1-8B-Instruct', 'Mistral-7B-Instruct-v0.3', 'Qwen2.5-7B-Instruct', 'gemma-2-9b-it']

# Scenario 1: APD - Llama & Gemma (Base vs IT) - Shuffled - Prompt 1
base_models_apd = ['Llama-3.1-8B', 'gemma-2-9b']
it_models_apd = ['Llama-3.1-8B-Instruct', 'gemma-2-9b-it']
generate_avg_distance_plots_mixed(
    merged_res_shuffled_bm, 
    merged_res_shuffled_it, 
    base_models_apd, 
    it_models_apd, 
    "Shuffled", 
    prompt=1, 
    filename_suffix="llama_gemma_shuff"
)

# Scenario 1: APD - Qwen & Mistral (Base vs IT) - Shuffled - Prompt 1
base_models_apd = ['Qwen2.5-7B', 'Mistral-7B-v0.3']
it_models_apd = ['Qwen2.5-7B-Instruct', 'Mistral-7B-Instruct-v0.3']
generate_avg_distance_plots_mixed(
    merged_res_shuffled_bm, 
    merged_res_shuffled_it, 
    base_models_apd, 
    it_models_apd, 
    "Shuffled", 
    prompt=1, 
    filename_suffix="qwen_mistral_shuff"
)

# Scenario 3: MPD - Mistral & Gemma (Base vs IT) - Reworded - Prompt 2
base_models_mpd = ['Mistral-7B-v0.3', 'gemma-2-9b']
it_models_mpd = ['Mistral-7B-Instruct-v0.3', 'gemma-2-9b-it']
generate_max_distance_plots_mixed(
    merged_res_reworded_bm, 
    merged_res_reworded_it, 
    base_models_mpd, 
    it_models_mpd, 
    "Reworded", 
    prompt=2, 
    filename_suffix="mistral_gemma_rew"
)


## -------------------------------------------------
## Cross-Prompt Consistency side by side (Base vs IT) ## ## ##
## -------------------------------------------------

def plot_consistency_mixed(df_base, df_it, dataset_label, filename_suffix, mode='percentage', include_title=True):
    """
    Creates a side-by-side consistency plot (Base vs IT) for slides.
    Features:
    - High contrast color palette (Dark to Light).
    - Single Legend.
    - Side-by-side layout (1 row, 2 columns).
    """
    
    it_order = [
        'Llama-3.1-8B-Instruct', 'Mistral-7B-Instruct-v0.3', 
        'Qwen2.5-7B-Instruct', 'gemma-2-9b-it'
    ]
    base_order = [
        'Llama-3.1-8B', 'Mistral-7B-v0.3', 
        'Qwen2.5-7B', 'gemma-2-9b'
    ]
    
    # Font Sizes
    FS_MAIN_TITLE = 22
    FS_SUB_TITLE  = 20
    FS_AXIS_LABEL = 18
    FS_TICKS      = 15
    FS_LEG_TITLE  = 18
    FS_LEG_TEXT   = 16

    # Derived from Viridis to be more distinguishable
    custom_palette = [
        '#440154',  # 4 Same: Dark Purple
        '#3b528b',  # 3 Same: Medium Blue
        '#21918c',  # 2 Same: Teal
        '#5ec962'   # 4 Diff: Light Green
    ]
    
    cat_order = ['4 Same Answers', '3 Same Answers', '2 Same Answers', '4 Different Answers']

    # --- helper: Process Dataframe ---
    def process_consistency(df, order):
        response_cols = [f'clean_response{i}' for i in range(1, 5)]
        
        # Filter unusable
        mask = df[response_cols].apply(lambda row: 'unusable' in row.values, axis=1)
        df_clean = df[~mask].copy()
        
        # Calculate Consistency
        def get_label(row):
            counts = Counter([row[c] for c in response_cols]).values()
            max_freq = max(counts) if counts else 0
            if max_freq == 4: return '4 Same Answers'
            elif max_freq == 3: return '3 Same Answers'
            elif max_freq == 2: return '2 Same Answers'
            else: return '4 Different Answers'

        df_clean['consistency_cat'] = df_clean.apply(get_label, axis=1)
        
        # Aggregate
        if mode == 'percentage':
            counts = df_clean.groupby(['model', 'consistency_cat'], observed=False).size().reset_index(name='count')
            totals = df_clean.groupby('model', observed=False).size().reset_index(name='total')
            props = pd.merge(counts, totals, on='model')
            props['percentage'] = (props['count'] / props['total']) * 100
            return props
        else:
            return df_clean

    # Process both datasets
    data_base = process_consistency(df_base, base_order)
    data_it = process_consistency(df_it, it_order)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(19, 10))
    
    plot_title = f'Cross-Prompt Consistency Comparison on {dataset_label} Dataset'
    
    if include_title:
        fig.suptitle(plot_title, fontsize=FS_MAIN_TITLE, fontweight='bold', y=0.98)

    # Configuration Loop
    configs = [
        (axes[0], data_base, "Base Models", base_order),
        (axes[1], data_it, "IT Models", it_order)
    ]

    y_limit = 70 if mode == 'percentage' else 4600

    for idx, (ax, data, title, order) in enumerate(configs):
        if mode == 'percentage':
            sns.barplot(
                data=data, x='model', y='percentage', hue='consistency_cat',
                hue_order=cat_order, palette=custom_palette, order=order, ax=ax
            )
            ylabel = 'Percentage of Questions'
        else:
            sns.countplot(
                data=data, x='model', hue='consistency_cat',
                hue_order=cat_order, palette=custom_palette, order=order, ax=ax
            )
            ylabel = 'Number of Questions'

        # Styling
        ax.set_ylim(0, y_limit)
        ax.set_title(title, fontsize=FS_SUB_TITLE)
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelsize=FS_TICKS, rotation=15)
        ax.tick_params(axis='y', labelsize=FS_TICKS)
        
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=FS_AXIS_LABEL)
        else:
            ax.set_ylabel('')
        
        if ax.get_legend():
            ax.get_legend().remove()

    # --- Legend & Save ---
    top_margin = 0.90 if include_title else 0.98
    plt.subplots_adjust(left=0.05, right=0.85, top=top_margin, bottom=0.1, wspace=0.15)
    
    # Legend
    handles = [mpatches.Patch(facecolor=custom_palette[i], edgecolor='black', label=cat) for i, cat in enumerate(cat_order)]
    
    fig.legend(
        handles=handles, title='Consistency Level',
        title_fontsize=FS_LEG_TITLE, loc='center right', bbox_to_anchor=(0.99, 0.8),
        fontsize=FS_LEG_TEXT, frameon=True, shadow=True
    )

    # Filename
    mode_prefix = "perc" if mode == 'percentage' else "abs"
    filename = f"consistency_comp_{mode_prefix}_{filename_suffix}.pdf"
    save_dir = os.path.join(script_dir, "plots", "slides")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, filename)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved slide consistency plot: {save_path}")
    plt.close()

# 1. Reworded Comparison (Percentage)
plot_consistency_mixed(merged_res_reworded_bm, merged_res_reworded_it, "Reworded", "rew", mode='percentage', include_title=True)

# 2. Shuffled Comparison (Percentage)
plot_consistency_mixed(merged_res_shuffled_bm, merged_res_shuffled_it, "Shuffled", "shuff", mode='percentage', include_title=True)


## -------------------------------------------------
## Overlapping Histograms side by side (Base vs IT) ## ## ##
## -------------------------------------------------

# --- helper (reused from above) ---
INVALID_MARKERS = {-98, -99}

def clean_numeric_series(series):
    """Takes a pandas Series, drops NaNs, and filters out -98/-99."""
    s = pd.to_numeric(series, errors='coerce')
    s = s.dropna()
    mask = ~s.isin(INVALID_MARKERS)
    return s[mask]

def generate_bias_mixed_plots(
    df_rew_base, df_shuff_base, 
    df_rew_it, df_shuff_it, 
    base_model_name, it_model_name, 
    prompts, filename_suffix, 
    include_title=True
):
    """
    Generates a 2x2 grid comparing Base vs IT models for two specific prompts.
    Each subplot compares Reworded vs Shuffled distributions.
    
    Layout:
    Row 0: Prompt P1 (Base Left, IT Right)
    Row 1: Prompt P2 (Base Left, IT Right)
    """
    
    # Font Sizes
    FS_MAIN_TITLE = 22
    FS_SUB_TITLE  = 20
    FS_AXIS_LABEL = 18
    FS_TICKS      = 15
    FS_LEG_TEXT   = 16
    
    # Config
    c1 = '#1f77b4' # Blue (Reworded)
    c2 = '#ff7f0e' # Orange (Shuffled)
    label1 = "Reworded"
    label2 = "Shuffled"
    alpha_val = 0.5
    y_limit = 10.5

    p1, p2 = prompts
    
    # Create Figure
    fig, axes = plt.subplots(2, 2, figsize=(19, 10))
    
    if include_title:
        base_short = base_model_name.split("-")[0] # e.g. "Llama"
        fig.suptitle(f'{base_short}: Answer Distribution Base vs IT (Prompts {p1} & {p2})', 
                     fontsize=FS_MAIN_TITLE, fontweight='bold', y=0.98)
        
    # Define the 4 subplots configuration
    # (Row, Col, RewordedDF, ShuffledDF, ModelName, PromptNum)
    plot_config = [
        (0, 0, df_rew_base, df_shuff_base, f"Base: {base_model_name}", p1),
        (0, 1, df_rew_it,   df_shuff_it,   f"IT: {it_model_name}",   p1),
        (1, 0, df_rew_base, df_shuff_base, f"Base: {base_model_name}", p2),
        (1, 1, df_rew_it,   df_shuff_it,   f"IT: {it_model_name}",   p2)
    ]

    for r, c, df_rew, df_shuff, title_prefix, p in plot_config:
        ax = axes[r, c]
        col = f'num_value{p}'
        
        # Extract Data
        # Base Model Name might be in df_rew/df_shuff, we need to filter by the specific model name passed in args
        # The dfs passed in likely contain ALL models, so we filter by the specific name
        # We need to strip the prefix "Base: " or "IT: " for filtering if it's just for display
        real_model_name = title_prefix.split(": ")[1]

        if real_model_name not in df_rew['model'].values:
             raw1 = pd.Series(dtype='float64')
        else:
             raw1 = df_rew[df_rew['model'] == real_model_name][col]
             
        if real_model_name not in df_shuff['model'].values:
             raw2 = pd.Series(dtype='float64')
        else:
             raw2 = df_shuff[df_shuff['model'] == real_model_name][col]

        vals1 = clean_numeric_series(raw1)
        vals2 = clean_numeric_series(raw2)

        if len(vals1) > 0 or len(vals2) > 0:
            # Histograms (bins=30)
            if len(vals1) > 0:
                sns.histplot(vals1, bins=30, stat='density', alpha=alpha_val, 
                             color=c1, label=label1, ax=ax, edgecolor=None)
            if len(vals2) > 0:
                sns.histplot(vals2, bins=30, stat='density', alpha=alpha_val, 
                             color=c2, label=label2, ax=ax, edgecolor=None)
            
            # KDE
            if len(vals1) > 1 and vals1.nunique() > 1:
                sns.kdeplot(vals1, color=c1, ax=ax, linewidth=2, warn_singular=False)
            if len(vals2) > 1 and vals2.nunique() > 1:
                sns.kdeplot(vals2, color=c2, ax=ax, linewidth=2, warn_singular=False)
            
            # Limits
            ax.set_ylim(0, y_limit)
            ax.set_xlim(-1.05, 1.05)
            
            # Legend (Only top-right plot to avoid clutter, or top-left)
            if r == 0 and c == 1:
                ax.legend(loc='upper right', fontsize=FS_LEG_TEXT)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(0, y_limit)

        # Titles & Labels
        # Title format: "Base - Prompt 1"
        display_title = "Base" if c == 0 else "IT"
        ax.set_title(f"{display_title} - Prompt {p}", fontsize=FS_SUB_TITLE)
        
        ax.set_xlabel('Numeric Answer Value', fontsize=FS_AXIS_LABEL)
        
        # Y-Label only on left column
        if c == 0:
            ax.set_ylabel('Density', fontsize=FS_AXIS_LABEL)
        else:
            ax.set_ylabel('')

        # Ticks
        ax.tick_params(axis='x', labelsize=FS_TICKS)
        ax.tick_params(axis='y', labelsize=FS_TICKS)

    # Layout
    rect_val = [0, 0, 1, 0.96] if include_title else [0, 0, 1, 1]
    plt.tight_layout(rect=rect_val)
    
    # Save
    filename = f"ans_distr_mixed_{filename_suffix}.pdf"
    save_dir = os.path.join(script_dir, "plots", "slides")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, filename)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved mixed bias plot: {save_path}")
    plt.close()


# Base models: ['Llama-3.1-8B', 'Mistral-7B-v0.3', 'Qwen2.5-7B', 'gemma-2-9b']
# IT models: ['Llama-3.1-8B-Instruct', 'Mistral-7B-Instruct-v0.3', 'Qwen2.5-7B-Instruct', 'gemma-2-9b-it']

# 1. Llama Comparison (Prompt 1 & 3)
generate_bias_mixed_plots(
    df_rew_base=merged_res_reworded_bm, 
    df_shuff_base=merged_res_shuffled_bm, 
    df_rew_it=merged_res_reworded_it, 
    df_shuff_it=merged_res_shuffled_it, 
    base_model_name='Llama-3.1-8B', 
    it_model_name='Llama-3.1-8B-Instruct', 
    prompts=(1, 3), 
    filename_suffix="llama_p1_p3",
    include_title=True
)

# 2. Mistral Comparison (Prompt 1 & 3)
generate_bias_mixed_plots(
    df_rew_base=merged_res_reworded_bm, 
    df_shuff_base=merged_res_shuffled_bm, 
    df_rew_it=merged_res_reworded_it, 
    df_shuff_it=merged_res_shuffled_it, 
    base_model_name='Mistral-7B-v0.3', 
    it_model_name='Mistral-7B-Instruct-v0.3', 
    prompts=(1, 3), 
    filename_suffix="mistral_p1_p3",
    include_title=True
)



### ---------------------------------------------------------------------------
### Plots for paper (bigger fontsizes, etc. for improved readability) ### ### ### ### ###
### ---------------------------------------------------------------------------

## Bias overview qwen (ans_distr_it_qwen2.5-7b-instruct_notitle.pdf, ans_distr_bm_qwen2.5-7b_notitle.pdf)

def generate_paper_bias_overlap_plots(df1, label1, df2, label2, filename_prefix, target_models=None):
    """
    Generates 2x2 figures showing overlapping histograms, optimized for ACL/EMNLP papers.
    Saves exclusively 'notitle' versions to the plots/paper/ directory.
    """
    df1 = df1.copy()
    df2 = df2.copy()
    
    # Filter for targeted models to save generation time
    models = sorted(df1['model'].unique())
    if target_models:
        models = [m for m in models if m in target_models]
        
    prompts = [1, 2, 3, 4] 
    
    # Colors
    c1 = '#1f77b4' # Blue
    c2 = '#ff7f0e' # Orange
    alpha_val = 0.5
    
    # Font Sizes for Paper Readability
    FS_SUB_TITLE  = 36
    FS_AXIS_LABEL = 32
    FS_TICKS      = 26
    FS_LEG_TEXT   = 28
    LINE_WIDTH    = 4
    
    # Fixed Y-Limit
    y_limit = 10.5

    for model in models:
        fig, axes = plt.subplots(2, 2, figsize=(19, 10))
        axes_flat = axes.flatten()
        
        for idx, p in enumerate(prompts):
            ax = axes_flat[idx]
            col = f'num_value{p}'

            if col not in df1.columns or col not in df2.columns:
                ax.text(0.5, 0.5, 'Missing Column', ha='center', va='center', fontsize=FS_TICKS)
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
                
                # Plot KDE with thicker lines
                if len(vals1) > 1 and vals1.nunique() > 1:
                    sns.kdeplot(vals1, color=c1, ax=ax, linewidth=LINE_WIDTH, warn_singular=False)
                if len(vals2) > 1 and vals2.nunique() > 1:
                    sns.kdeplot(vals2, color=c2, ax=ax, linewidth=LINE_WIDTH, warn_singular=False)
                
                ax.set_ylim(0, y_limit)
                ax.set_xlim(-1.05, 1.05)

                if idx == 1:
                    ax.legend(loc='upper left', fontsize=FS_LEG_TEXT, framealpha=0.9)

            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=FS_TICKS)
                ax.set_xlim(-1.05, 1.05)
                ax.set_ylim(0, y_limit)

            ax.set_title(f'Prompt {p}', fontsize=FS_SUB_TITLE, pad=15)
            
            # Axis Labels & Ticks
            if idx < 2:
                # Top row: hide the main axis label, BUT keep the numeric tick labels
                ax.set_xlabel('')
                ax.tick_params(axis='x', labelsize=FS_TICKS)
            else:
                # Bottom row: show everything normally
                ax.set_xlabel('Numeric Answer Value', fontsize=FS_AXIS_LABEL, labelpad=10)
                ax.tick_params(axis='x', labelsize=FS_TICKS)
            
            if idx % 2 == 0:
                ax.set_ylabel('Density', fontsize=FS_AXIS_LABEL, labelpad=10)
            else:
                ax.set_ylabel('')

            ax.tick_params(axis='y', labelsize=FS_TICKS)

        # Adjust Layout
        plt.tight_layout()
        
        # Filename & Path
        safe_model_name = model.lower().replace(" ", "_").replace("/", "-")
        filename = f"{filename_prefix}_{safe_model_name}_notitle.pdf"
        
        save_dir = os.path.join("plots", "paper")
        if not os.path.exists(save_dir):
             os.makedirs(save_dir)

        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved paper bias plot for {model} to {save_path}")
        plt.close()

# 1. Instruction Tuned Model: Qwen2.5-7B-Instruct
generate_paper_bias_overlap_plots(
    df1=merged_res_reworded_it, 
    label1="Reworded", 
    df2=merged_res_shuffled_it, 
    label2="Shuffled", 
    filename_prefix="ans_distr_it",
    target_models=["Qwen2.5-7B-Instruct"] 
)

# 2. Base Model: Qwen2.5-7B
generate_paper_bias_overlap_plots(
    df1=merged_res_reworded_bm, 
    label1="Reworded", 
    df2=merged_res_shuffled_bm, 
    label2="Shuffled", 
    filename_prefix="ans_distr_bm",
    target_models=["Qwen2.5-7B"] 
)


## Plain answer distribution (answer_perc_distr_bm_rew_notitle.pdf, answer_perc_distr_it_rew_notitle.pdf)

def plot_paper_distributions(df, dataset_label, model_type, filename_suffix, mode='percentage'):
    """
    Creates a 2x2 grid of barplots for the 4 prompts, optimized for ACL/EMNLP papers.
    Saves exclusively to the plots/paper/ directory.
    """
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
    
    # Define the shortened labels for the plots
    it_short_labels = ['Llama-IT', 'Mistral-IT', 'Qwen-IT', 'Gemma-IT']
    base_short_labels = ['Llama-Base', 'Mistral-Base', 'Qwen-Base', 'Gemma-Base']
    
    if "Base" in model_type:
        model_order = base_order
        display_labels = base_short_labels
    else:
        model_order = it_order
        display_labels = it_short_labels

    # Font Sizes
    FS_SUB_TITLE  = 36
    FS_AXIS_LABEL = 32
    FS_TICKS      = 26
    FS_LEG_TITLE  = 32
    FS_LEG_TEXT   = 28
    
    # Construct Filename
    mode_prefix = "abs" if mode == 'absolute' else "perc"
    filename = f"answer_{mode_prefix}_distr_{filename_suffix}_notitle.pdf"
    
    save_dir = os.path.join("plots", "paper")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, filename)

    prompts = [
        ('clean_response1', 'Prompt 1'),
        ('clean_response2', 'Prompt 2'), 
        ('clean_response3', 'Prompt 3'),
        ('clean_response4', 'Prompt 4')
    ]

    all_possible_responses = ['1', '2', '3', '4', '5', '6', '7', '8', 'unusable']
    
    custom_colors = [
        '#2ca02c', '#ff7f0e', '#1f77b4', '#d62728', 
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'
    ]
    color_mapping = {resp: custom_colors[i] for i, resp in enumerate(all_possible_responses)}

    # Fixed ylims for comparability
    if mode == 'absolute':
        y_limit = 5200
    elif mode == 'percentage':
        y_limit = 70

    fig, axes = plt.subplots(2, 2, figsize=(19, 10))
    axes_flat = axes.flatten()

    for idx, (col, title) in enumerate(prompts):
        ax = axes_flat[idx]
        hue_order = all_possible_responses
        
        if mode == 'absolute':
            sns.countplot(
                data=df, x='model', hue=col, ax=ax,
                hue_order=hue_order, palette=color_mapping, order=model_order
            )
            ylabel = 'Number of Responses'

        elif mode == 'percentage':
            counts = df.groupby(['model', col]).size().reset_index(name='count')
            totals = df.groupby('model').size().reset_index(name='total')
            props = pd.merge(counts, totals, on='model')
            props['percentage'] = (props['count'] / props['total']) * 100
            
            sns.barplot(
                data=props, x='model', y='percentage', hue=col, ax=ax,
                hue_order=hue_order, palette=color_mapping, order=model_order
            )
            ylabel = 'Responses (in %)'

        # Styling
        ax.set_title(title, fontsize=FS_SUB_TITLE, pad=15)
        ax.set_xlabel('') 
        ax.tick_params(axis='y', labelsize=FS_TICKS)
        
        # Determine X-axis label visibility and text
        if idx < 2:
            ax.set_xticklabels([])
            ax.tick_params(axis='x', length=0) 
        else:
            # Safely lock the generated ticks
            ax.set_xticks(ax.get_xticks()) 
            ax.set_xticklabels(display_labels, fontsize=FS_TICKS, rotation=15)
        
        if idx % 2 == 0:
            ax.set_ylabel(ylabel, fontsize=FS_AXIS_LABEL, labelpad=10)
        else:
            ax.set_ylabel('')
            
        if ax.get_legend():
            ax.get_legend().remove()
            
        # Hard lock the y-limits & y-ticks
        ax.set_ylim(0, y_limit)
        if mode == 'percentage':
            ax.set_yticks(range(0, y_limit + 1, 20))

    # Layout adjusted for larger text
    plt.subplots_adjust(left=0.08, right=0.75, top=0.95, bottom=0.15, hspace=0.4, wspace=0.15)
    
    if dataset_label == "Shuffled":
        legend_responses = ['1', '2', '3', '4', '5', '6', '7', 'unusable']
    else:
        legend_responses = ['1', '2', '3', '4', '5', '6', '7', '8', 'unusable']

    handles = []
    labels = []
    for response in legend_responses:
        handle = mpatches.Patch(facecolor=color_mapping[response], edgecolor='black', linewidth=1, label=response)
        handles.append(handle)
        labels.append(response)
        
    fig.legend(
        handles=handles,
        labels=labels,
        title='Options',
        title_fontsize=FS_LEG_TITLE,
        loc='center left',
        bbox_to_anchor=(0.77, 0.5),
        fontsize=FS_LEG_TEXT,
        frameon=True,
        shadow=True
    )
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved paper plot: {save_path}")
    plt.close()

# 1. Base Models - Percentage - Reworded
plot_paper_distributions(
    df=merged_res_reworded_bm, 
    dataset_label="Reworded", 
    model_type="Base Models", 
    filename_suffix="bm_rew", 
    mode='percentage'
)

# 2. Instruction-Tuned Models - Percentage - Reworded
plot_paper_distributions(
    df=merged_res_reworded_it, 
    dataset_label="Reworded", 
    model_type="Instruction-Tuned", 
    filename_suffix="it_rew", 
    mode='percentage'
)


## APD Distance Histogram Plots 
## (apd_distr_bm_rew_prompt1_notitle.pdf, apd_distr_it_rew_prompt1_notitle.pdf, apd_distr_bm_shuff_prompt1_notitle.pdf, apd_distr_it_shuff_prompt1_notitle.pdf)

def generate_paper_avg_distance_plots(df, dataset_label, model_type, filename_prefix, target_prompts=None):
    """
    1. Calculates pairwise distances for specified Prompts.
    2. Generates one 2x2 figure per prompt (showing 4 models), optimized for papers.
    Saves exclusively 'notitle' versions to the plots/paper/ directory.
    """
    df = df.copy()
    prompts = target_prompts if target_prompts else [1, 2, 3, 4]
    
    # Define model order and shortened names for paper titles
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
    
    it_short_labels = ['Llama-IT', 'Mistral-IT', 'Qwen-IT', 'Gemma-IT']
    base_short_labels = ['Llama-Base', 'Mistral-Base', 'Qwen-Base', 'Gemma-Base']
    
    # Select order
    if "Base" in model_type:
        model_order = base_order
        short_labels = base_short_labels
    else:
        model_order = it_order
        short_labels = it_short_labels

    # Font Sizes
    FS_SUB_TITLE  = 36
    FS_AXIS_LABEL = 32
    FS_TICKS      = 26
    FS_LEG_TEXT   = 28
    LINE_WIDTH    = 4  # For KDE
    METRIC_WIDTH  = 3  # For Mean/Median lines

    # --- STEP 1: Calc distances ---
    avgdist_data = {}
    
    for p in prompts:
        col = f'num_value{p}'
        if col not in df.columns: continue
        
        grouped = df.groupby(['model', 'question_id'])[col].apply(list).reset_index(name='values_list')

        if grouped.empty:
            print(f"[{dataset_label}] Prompt {p}: No data found.")
            avgdist_data[p] = grouped
            continue

        typical_len = int(grouped['values_list'].apply(len).median())

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
        axes_flat = axes.flatten()
        
        # Iterate over fixed model order
        for idx, model in enumerate(model_order):
            if idx >= len(axes_flat): break
            ax = axes_flat[idx]
            short_name = short_labels[idx]
            
            # Extract data for specific model
            model_vals = dfp[dfp['model'] == model]['avg_pairwise_dist'].dropna()
            
            if len(model_vals) > 0:
                # Histogram
                fixed_bins = np.linspace(0, 2.0, 61) # 61 edges = 60 bins
                sns.histplot(model_vals, bins=fixed_bins, stat='density', alpha=0.6, ax=ax, edgecolor=None, color='#1f77b4')
                
                # KDE (Check variance first)
                if model_vals.nunique() > 1:
                    sns.kdeplot(model_vals, ax=ax, linewidth=LINE_WIDTH)
                
                # Metrics
                med = float(model_vals.median())
                mean = float(model_vals.mean())
                
                # Lines
                ax.axvline(med, color='black', linestyle='--', linewidth=METRIC_WIDTH, label=f'Median: {med:.3f}')
                ax.axvline(mean, color='black', linestyle='-', linewidth=METRIC_WIDTH, label=f'Mean: {mean:.3f}')
                
                # Legend
                ax.legend(loc='upper right', fontsize=FS_LEG_TEXT, framealpha=0.9)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=FS_TICKS)

            # Styling
            ax.set_title(short_name, fontsize=FS_SUB_TITLE, pad=15)
            
            # X-axis formatting: Hide main label on top row but keep ticks
            if idx < 2:
                ax.set_xlabel('')
                ax.tick_params(axis='x', labelsize=FS_TICKS)
            else:
                ax.set_xlabel('Avg Pairwise Distance', fontsize=FS_AXIS_LABEL, labelpad=10)
                ax.tick_params(axis='x', labelsize=FS_TICKS)
            
            # Y-label only on left column
            if idx % 2 == 0:
                ax.set_ylabel('Density', fontsize=FS_AXIS_LABEL, labelpad=10)
            else:
                ax.set_ylabel('')
                
            ax.tick_params(axis='y', labelsize=FS_TICKS)
            
            # Hard lock limits at the end of the loop
            ax.set_xlim(0, 2.0)
            ax.set_ylim(0, y_limit) 
            
            ax.set_yticks(np.arange(2.5, y_limit, 2.5))

        # Adjust Layout
        plt.subplots_adjust(top=0.95, bottom=0.15, left=0.08, right=0.95, hspace=0.35, wspace=0.15)
        
        # Construct Filename & Path
        filename = f"{filename_prefix}_prompt{p}_notitle.pdf"
        
        save_dir = os.path.join("plots", "paper")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_path = os.path.join(save_dir, filename)

        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved paper plot: {save_path}")
        plt.close()

# 1. Base Models - Reworded (Prompt 1 only)
generate_paper_avg_distance_plots(
    df=merged_res_reworded_bm, 
    dataset_label="Reworded", 
    model_type="Base Models", 
    filename_prefix="apd_distr_bm_rew", 
    target_prompts=[1]
)

# 2. Instruction-Tuned Models - Reworded (Prompt 1 only)
generate_paper_avg_distance_plots(
    df=merged_res_reworded_it, 
    dataset_label="Reworded", 
    model_type="Instruction-Tuned Models", 
    filename_prefix="apd_distr_it_rew", 
    target_prompts=[1]
)

# 3. Base Models - Shuffled (Prompt 1 only)
generate_paper_avg_distance_plots(
    df=merged_res_shuffled_bm, 
    dataset_label="Shuffled", 
    model_type="Base Models", 
    filename_prefix="apd_distr_bm_shuff", 
    target_prompts=[1]
)

# 4. Instruction-Tuned Models - Shuffled (Prompt 1 only)
generate_paper_avg_distance_plots(
    df=merged_res_shuffled_it, 
    dataset_label="Shuffled", 
    model_type="Instruction-Tuned Models", 
    filename_prefix="apd_distr_it_shuff", 
    target_prompts=[1]
)


## Consistency Category barplots (consistency_perc_bm_rew_notitle.pdf, consistency_perc_it_rew_notitle.pdf)

def plot_paper_consistency(df, dataset_label, model_type, filename_suffix, mode='percentage', show_legend=True):
    """
    Analyzes consistency across clean_response1 to clean_response4.
    Optimized for ACL/EMNLP single-column paper format.
    Saves exclusively 'notitle' versions to the plots/paper/ directory.
    Can toggle the legend on/off for side-by-side paper placement.
    """
    df = df.copy()

    # Define model order and short labels
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
    
    it_short_labels = ['Llama-IT', 'Mistral-IT', 'Qwen-IT', 'Gemma-IT']
    base_short_labels = ['Llama-Base', 'Mistral-Base', 'Qwen-Base', 'Gemma-Base']
    
    # Select order
    if "Base" in model_type:
        model_order = base_order
        short_labels = base_short_labels
    else:
        model_order = it_order
        short_labels = it_short_labels

    # Font Sizes
    FS_AXIS_LABEL = 32
    FS_TICKS      = 26
    FS_LEG_TITLE  = 32
    FS_LEG_TEXT   = 28

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
    
    # 2. Plotting
    plt.figure(figsize=(19, 10)) 
    
    # Fixed ylims & Filename Logic
    if mode == 'absolute':
        y_limit = 4600
        mode_prefix = "abs"
    elif mode == 'percentage':
        y_limit = 62.5
        mode_prefix = "perc"

    # Append _nolegend to the filename if the legend is toggled off
    legend_str = "" if show_legend else "_nolegend"
    filename = f"consistency_{mode_prefix}_{filename_suffix}_notitle{legend_str}.pdf"
    
    save_dir = os.path.join("plots", "paper")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, filename)
    
    # Plotting
    ax = plt.gca() # Get current axis to manipulate ticks
    
    if mode == 'absolute':
        sns.countplot(
            data=df_clean,
            x='model',
            hue='consistency_cat',
            hue_order=cat_order,
            palette='viridis',
            order=model_order,
            ax=ax
        )
        plt.ylabel('Number of Questions', fontsize=FS_AXIS_LABEL, labelpad=10)

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
            order=model_order,
            ax=ax
        )
        plt.ylabel('Percentage of Questions', fontsize=FS_AXIS_LABEL, labelpad=10)

    # 3. Styling
    plt.ylim(0, y_limit)
    if mode == 'percentage':
        plt.yticks(np.arange(0, 70, 10))

    plt.xlabel('')
    
    # Apply short labels to x-axis
    ax.set_xticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, fontsize=FS_TICKS)
    
    plt.tick_params(axis='y', labelsize=FS_TICKS)
    
    # Legend Toggle Logic
    if show_legend:
        plt.legend(
            title='Consistency Level', 
            bbox_to_anchor=(1.01, 1), 
            loc='upper left',
            title_fontsize=FS_LEG_TITLE,
            fontsize=FS_LEG_TEXT
        )
    else:
        if ax.get_legend():
            ax.get_legend().remove()
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved paper plot: {save_path}")
    plt.close()

# 1. Base Models - Percentage - Reworded (Legend ON)
plot_paper_consistency(
    df=merged_res_reworded_bm, 
    dataset_label="Reworded", 
    model_type="Base Models", 
    filename_suffix="bm_rew", 
    mode='percentage',
    show_legend=True
)

# 2. Instruction-Tuned Models - Percentage - Reworded (Legend OFF)
plot_paper_consistency(
    df=merged_res_reworded_it, 
    dataset_label="Reworded", 
    model_type="Instruction-Tuned Models", 
    filename_suffix="it_rew", 
    mode='percentage',
    show_legend=False
)


## ---------------------------------------------------------------------------
## Random-Response Baseline Simulation ## ## ##
## ---------------------------------------------------------------------------
# Calculates the theoretical APD and MPD if answers were picked randomly from the available options

def simulate_random_baseline_all(df, dataset_label, expected_n, seed=42):
    """
    Simulates random guessing for each unique question to establish a baseline.
    Draws `expected_n` random answers per question from its specific `num_scale`.
    Calculates APD, MPD, and MSD.
    """
    np.random.seed(seed)
    
    df_unique = df.drop_duplicates(subset=['question_id']).copy()
    
    apds, mpds, stds = [], [], []
    
    for _, row in df_unique.iterrows():
        scale_str = row.get('num_scale', None)
        if not isinstance(scale_str, str):
            continue
            
        try:
            scale = ast.literal_eval(scale_str)
            scale = _to_valid_floats(scale) 
            if len(scale) < 2:
                continue
                
            simulated_answers = np.random.choice(scale, size=expected_n, replace=True)
            
            # Calculate metrics
            apd = avg_pairwise_abs_distance(simulated_answers, require_full=False)
            mpd = max_pairwise_abs_distance(simulated_answers, require_full=False)
            
            # For MSD, use Pandas std() to exactly match your get_valid_std behavior (ddof=1)
            std = pd.Series(simulated_answers).std()
            
            if not np.isnan(apd): apds.append(apd)
            if not np.isnan(mpd): mpds.append(mpd)
            if not pd.isna(std): stds.append(std)
            
        except Exception as e:
            continue
            
    mean_apd = np.mean(apds) if apds else np.nan
    mean_mpd = np.mean(mpds) if mpds else np.nan
    mean_msd = np.mean(stds) if stds else np.nan
    
    print(f"\n--- Random-Response Baseline: {dataset_label} (Seed: {seed}) ---")
    print(f"Simulated based on N={expected_n} variations per question.")
    print(f"Average Pairwise Distance (APD): {mean_apd:.3f}")
    print(f"Maximum Pairwise Distance (MPD): {mean_mpd:.3f}")
    print(f"Mean Standard Deviation (MSD): {mean_msd:.3f}")
    print("-" * 60)
    
    return mean_apd, mean_mpd, mean_msd

# Calculate for Reworded (5 variations per question)
simulate_random_baseline_all(merged_res_reworded_bm, "Reworded", expected_n=5)
"""
Random-Response Baseline: Reworded (Seed: 42)
Simulated based on N=5 variations per question.
Average Pairwise Distance (APD): 0.825
Maximum Pairwise Distance (MPD): 1.632
Mean Standard Deviation (MSD): 0.711
"""

# Calculate for Shuffled (6 variations per question)
simulate_random_baseline_all(merged_res_shuffled_bm, "Shuffled", expected_n=6)
"""
Random-Response Baseline: Shuffled (Seed: 42)
Simulated based on N=6 variations per question.
Average Pairwise Distance (APD): 0.833
Maximum Pairwise Distance (MPD): 1.734
Mean Standard Deviation (MSD): 0.724
"""
