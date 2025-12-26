# Preprocessing steps to get to the final dataset with 1235 multiple choice questions

import pandas as pd
import glob
import os
import re
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


# find all CSV files in opinionqa folder
folder_path = os.path.join(script_dir, "data", "raw", "opinion_qa")
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# read each file and store it in a list
all_dataframes = []
ignore_list = ['steer-bio.csv', 'steer-portray.csv']
for file in csv_files:
    filename = os.path.basename(file)
    # skip for two irrelevant files
    if filename in ignore_list:
        print(f"Skipping ignored file: {filename}")
        continue
    try:
        # delimiter='\t' because files are tab-separated (If files are comma-separated need to adjust this)
        df = pd.read_csv(file, delimiter='\t')
        df['source'] = os.path.basename(file)
        all_dataframes.append(df)
    except Exception as e:
        print(f"Could not read {os.path.basename(file)} due to error: {e}")

# combine all DataFrames into a single one
combined_df = pd.concat(all_dataframes, ignore_index=True)
combined_df.rename(columns={'options': 'answer_options'}, inplace=True)

# drop unnecessary cols
drop_cols = combined_df.columns[-3:].tolist()
drop_cols.extend(combined_df.columns[[0, 1, 6]].tolist())
combined_df.drop(columns=drop_cols, inplace=True)

# add ID columns for questions, question variations and answer variations
combined_df['question_id'] = pd.factorize(combined_df['question'])[0] + 1
combined_df['question_var_id'] = 1
combined_df['answer_var_id'] = 1

# remove duplicate rows (both question and answer options must match)
combined_df.drop_duplicates(subset=['question_id', 'answer_options'], keep='first', inplace=True)

# remove 3 problematic rows by hand
combined_df = combined_df[~combined_df['question_id'].isin([98, 99, 792])]

combined_df['question_id'] = pd.factorize(combined_df['question'])[0] + 1
combined_df = combined_df.sort_values('question_id').reset_index(drop=True)

# Now assign answer_var_id properly within each question group
combined_df['answer_var_id'] = combined_df.groupby('question_id').cumcount() + 1

# Verify the results (to uncomment select all and SHIFT + ALT + A)
""" print(f"\nFinal dataframe shape: {combined_df.shape}")
answer_var_counts = combined_df.groupby('question_id')['answer_var_id'].max().value_counts().sort_index()
for var_count, freq in answer_var_counts.items():
    print(f"  {var_count} variation(s): {freq} question(s)") """

# Remove and Reorder columns to have IDs first
combined_df = combined_df.drop(columns=['key', 'survey'])
cols = combined_df.columns.tolist()
cols = ['question_id', 'question_var_id', 'answer_var_id'] + cols[:-3]
combined_df = combined_df[cols]

combined_df.to_csv(os.path.join(script_dir, "data", "clean", "opinionQA_all-questions.csv"), index=False)


# Classifying questions (opinionQA_all-questions.csv to opinionQA_questions_classified_v1.csv) was done manually in Excel.
# 2 new columns: 'decision' and 'subject'. decision: 'keep', 'persona_only', 'drop' [for experiment later] & subject: topic of the question

# Rephrasing the questions so they can be assessed properly with Regex below (opinionQA_questions_classified_v1.csv to opinionQA_questions_classified_v2.csv) was also done manually in Excel.
# Manual rewording included e.g. rephrasing questions so question types can be assessed, fixing typos, changing "Joe Biden" to "the president" for generalization (if not possible to "Donald Trump")


# Continue working with manually cleaned data (opinionQA_questions_classified_v2.csv)
# We now assess the question types (e.g. Importance, Agreement, Quantity, etc.) so we can easily attach consistent Likert-style answer sets later
questions_classif = pd.read_csv(os.path.join(script_dir, "data", "clean", "opinionQA_questions_classified_v2.csv"), delimiter=';')
questions_kept = questions_classif.query("decision == 'keep'").copy()
questions_kept = questions_kept.drop_duplicates(subset='question_id', keep='first')

# Regex patterns for different question types
type_patterns = [
    ("Importance", [r"\bimportance\b", r"\bhow\s+important\b"]),
    ("Agreement", [r"\bhow\s+much\s+do\s+you\s+agree\s+or\s+disagree\s+with\s+the\s+following\s+statement\b", r"\binequality\s+is\s+acceptable\b",]), # strongly agree, agree, neither agree nor disagree, disagree or strongly disagree
    ("Concern", [r"\bhow\s+worried\b", r"\bhow\s+concerned\b"]),
    ("Frequency", [r"\bhow\s+often\b"]),
    ("Likelihood", [r"\bhow\s+likely\b", r"\b(?:are\s+)?likely(?:\s+or\s+not\s+likely)?\s+to\s+happen\b", r"\bit\s+is\s+likely\s+or\s+not\s+that\b", r"\bmore\s+or\s+less\s+likely\b"]),
    ("Quantity", [r"\bhow\s+does\b[\s\S]*?\baffect\b", r"\bto\s+what\s+extent\b[\s\S]*?\bdo\s+you\s+think\b", r"\bhow\s+confident\b", r"\bconfidence\s+in\b"]),
    ("GoodOrBad", [r"\bgood\s+or\s+bad\b"]),
    ("Reason", [r"\bis\s+a\s+reason\s+why\b", r"\bhow\s+big\s+a\s+reason\b"]),
    ("Priority", [r"\bwhat\s+priority\s+would\s+you\s+give\s+to\b"]),
    ("PositiveNegative", [r"\bpositive\s+or\s+[\s\S]*?\bnegative\b"]),
    ("IncreaseDecrease", [r"\bwill\s+(?:need\s+to\s+)?increase,\s+decrease,\s+or\s+stay\s+about\s+the\s+same\??\b", r"\bincrease\s+or\s+decrease\b"]),
    ("HowWell", [r"\bhow\s+well\b"]),
    ("Acceptance", [r"\bthink\s+it\s+is\b[\s\S]*?\bacceptable\b", r"\bare\s+acceptable\b", r"\bhow\s+acceptable\b"]),
    ("BetterOrWorse", [r"\bbetter,\s*worse\b", r"\bbetter\s+or\s+worse\b", r"\ba\s+better\b[\s\S]*?,\s+a\s+worse\b"]),
    ("Problem", [r"\bhow\s+big\s+of\s+a\s+problem\b"]),
]

# Classify questions and add new column question_type
compiled = [(t, [re.compile(p, flags=re.IGNORECASE) for p in pats]) for t, pats in type_patterns]
fallback = re.compile(r"\bhow\s+(?:much|many)\b", flags=re.IGNORECASE) # For Quantity: fallback, if questions with "how much" don't have a question type assigned yet 

def classify_question(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "UNKNOWN"
    for qtype, patterns in compiled:
        if any(p.search(text) for p in patterns):
            return qtype
    if fallback.search(text):
        return "Quantity"
    return "UNKNOWN"

questions_kept["question_type"] = questions_kept["question"].map(classify_question)
questions_kept.to_csv(os.path.join(script_dir, "data", "clean", "opinionQA_questions_classified_assessed.csv"), index=False)

answer_options = {
    "Importance": [
        {
            "options": ["Very important", "Somewhat important", "Not too important", "Not at all important"],
            "scale_type": "4-unipolar",
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Extremely important", "Fairly important", "Moderately important", "Slightly important", "Not important at all"],
            "scale_type": "5-unipolar", 
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["Essential", "Important", "Neutral", "Unimportant", "Completely unimportant"],
            "scale_type": "5-bipolar", 
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["Of great importance", "Of some importance", "Of little importance", "Of no importance"],
            "scale_type": "4-unipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Highly important", "Quite important", "Somewhat unimportant", "Completely unimportant"],
            "scale_type": "4-unipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
    ],
    "Agreement": [
        {
            "options": ["Fully agree", "Somewhat agree", "Somewhat disagree", "Fully disagree"],
            "scale_type": "4-bipolar",
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Strongly agree", "Moderately agree", "Moderately disagree", "Strongly disagree"],
            "scale_type": "4-bipolar",
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Completely agree", "Mostly agree", "Slightly agree", "Slightly disagree", "Mostly disagree", "Completely disagree"],
            "scale_type": "6-bipolar",
            "num_scale": [1.0, 0.6, 0.2, -0.2, -0.6, -1.0]
        },
        {
            "options": ["Totally agree", "Mostly agree", "Neither agree nor disagree", "Mostly disagree", "Totally disagree"],
            "scale_type": "5-bipolar",
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["Absolutely agree", "Partly agree", "Partly disagree", "Absolutely disagree"],
            "scale_type": "4-bipolar",
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
    ],
    "Concern": [
        {
            "options": ["Very concerned", "Somewhat concerned", "Not too concerned", "Not at all concerned"],
            "scale_type": "4-unipolar",
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Extremely concerned", "Fairly concerned", "Moderately concerned", "Slightly concerned", "Not concerned at all"],
            "scale_type": "5-unipolar", 
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["Highly concerned", "Quite concerned", "Minimally concerned", "Completely unconcerned"],
            "scale_type": "4-unipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Greatly worried", "Considerably worried", "Slightly worried", "Not worried at all"],
            "scale_type": "4-unipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Alarmed", "Concerned", "Somewhat unconcerned", "Totally unconcerned"],
            "scale_type": "4-bipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
    ],
    "Frequency": [
        {
            "options": ["Always", "Often", "Sometimes", "Rarely", "Never"],
            "scale_type": "5-unipolar",
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["Constantly", "Frequently", "Occasionally", "Infrequently", "Not once"],
            "scale_type": "5-unipolar",
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["All the time", "Regularly", "Now and then", "Seldom", "Never"],
            "scale_type": "5-unipolar",
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["Every time", "Quite often", "From time to time", "Hardly ever", "Never"],
            "scale_type": "5-unipolar",
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["Without exception", "Commonly", "Periodically", "Rarely ever", "Never"],
            "scale_type": "5-unipolar",
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
    ],
    "Likelihood": [
        {
            "options": ["Very likely", "Somewhat likely", "Not too likely", "Not at all likely"],
            "scale_type": "4-unipolar",
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Extremely likely", "Fairly likely", "Slightly likely", "Not likely at all"],
            "scale_type": "4-unipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Highly probable", "Moderately probable", "Neither probable nor improbable", "Somewhat improbable", "Highly improbable"],
            "scale_type": "5-bipolar", 
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["Almost certainly", "Probably", "Unlikely", "Very unlikely"],
            "scale_type": "4-bipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Very probable", "Quite probable", "Rather improbable", "Extremely improbable"],
            "scale_type": "4-bipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
    ],
    "Quantity": [
        {
            "options": ["A lot", "A fair amount", "Not too much", "None at all"],
            "scale_type": "4-unipolar",
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["A great deal", "A moderate amount", "A small amount", "None whatsoever"],
            "scale_type": "4-unipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Substantially", "Considerably", "Minimally", "Not at all"],
            "scale_type": "4-unipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Extensively", "Moderately", "Slightly", "Not in the least"],
            "scale_type": "4-unipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Extensively", "Considerably", "Quite a bit", "Very little", "Absolutely none"],
            "scale_type": "5-unipolar", 
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
    ],
    "GoodOrBad": [
        {
            "options": ["Very good", "Somewhat good", "Neither good nor bad", "Somewhat bad", "Very bad"],
            "scale_type": "5-bipolar",
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["Excellent", "Good", "Bad", "Terrible"],
            "scale_type": "4-bipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Extremely good", "Fairly good", "Fairly bad", "Extremely bad"],
            "scale_type": "4-bipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Entirely good", "Mainly good", "Neutral", "Mainly bad", "Entirely bad"],
            "scale_type": "5-bipolar", 
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["Outstanding", "Adequate", "Poor", "Very poor"],
            "scale_type": "4-bipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
    ],
    "Reason": [
        {
            "options": ["A major reason", "A somewhat big reason", "A minor reason", "Not a reason"],
            "scale_type": "4-unipolar",
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["A primary factor", "A secondary factor", "A minor factor", "A very weak factor", "Not a factor at all"],
            "scale_type": "5-unipolar", 
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["The main reason", "A contributing reason", "A small reason", "Not relevant"],
            "scale_type": "4-unipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["A key reason", "A partial reason", "A slight reason", "Not a reason at all"],
            "scale_type": "4-unipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["A principal reason", "A moderate reason", "A negligible reason", "Completely unrelated"],
            "scale_type": "4-unipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
    ],
    "Priority": [
        {
            "options": ["A top priority", "An important, but not a top priority", "A lower priority", "Should not be done"],
            "scale_type": "4-unipolar",
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Highest priority", "High priority", "Medium priority", "Low priority", "No priority at all"],
            "scale_type": "5-unipolar", 
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["Critical priority", "Significant priority", "Minor priority", "No priority"],
            "scale_type": "4-unipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Essential", "Important", "Mildly important", "Optional", "Not important", "Unnecessary"],
            "scale_type": "6-unipolar", 
            "num_scale": [1.0, 0.6, 0.2, -0.2, -0.6, -1.0]
        },
        {
            "options": ["Urgent", "Necessary", "Deferrable", "Not needed"],
            "scale_type": "4-unipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
    ],
    "PositiveNegative": [
        {
            "options": ["Very positive", "Mostly positive", "Neither positive nor negative", "Mostly negative", "Very negative"],
            "scale_type": "5-bipolar", 
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["Extremely positive", "Generally positive", "Neutral", "Generally negative", "Extremely negative"],
            "scale_type": "5-bipolar",
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["Very favorable", "Somewhat favorable", "Somewhat unfavorable", "Very unfavorable"],
            "scale_type": "4-bipolar",
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Strongly positive", "Positive", "Neither", "Negative", "Strongly negative"],
            "scale_type": "5-bipolar",
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["Entirely positive", "Moderately positive", "Moderately negative", "Entirely negative"],
            "scale_type": "4-bipolar",
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
    ],
    "IncreaseDecrease": [
        {
            "options": ["Increase by a lot", "Increase", "Stay about the same", "Decrease", "Decrease by a lot"],
            "scale_type": "5-bipolar", 
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["Greatly increase", "Slightly increase", "Remain the same", "Slightly decrease", "Greatly decrease"],
            "scale_type": "5-bipolar",
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["Substantially increase", "Modestly increase", "Modestly decrease", "Substantially decrease"],
            "scale_type": "4-bipolar",
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Will rise sharply", "Will rise", "Will stay constant", "Will fall", "Will fall sharply"],
            "scale_type": "5-bipolar",
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["Extremely increase", "Significantly increase", "Slightly increase", "Slightly decrease", "Significantly decrease", "Extremely decrease"],
            "scale_type": "6-bipolar",
            "num_scale": [1.0, 0.6, 0.2, -0.2, -0.6, -1.0]
        },
    ],
    "HowWell": [
        {
            "options": ["Very well", "Pretty well", "Not too well", "Not at all well"],
            "scale_type": "4-unipolar",
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Extremely well", "Quite well", "Not very well", "Poorly"],
            "scale_type": "4-unipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Exceptionally well", "Reasonably well", "Somewhat poorly", "Very poorly"],
            "scale_type": "4-bipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Superbly", "Adequately", "Inadequately", "Very inadequately"],
            "scale_type": "4-bipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Excellently", "Satisfactorily", "Neutral", "Unsatisfactorily", "Very unsatisfactorily"],
            "scale_type": "5-bipolar", 
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
    ],
    "Acceptance": [
        {
            "options": ["Always acceptable", "Sometimes acceptable", "Rarely acceptable", "Never acceptable"],
            "scale_type": "4-unipolar",
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Completely acceptable", "Occasionally acceptable", "Seldom acceptable", "Not once acceptable"],
            "scale_type": "4-unipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Totally acceptable", "Acceptable in a bunch of cases", "In few cases acceptable", "Not acceptable under any circumstances"],
            "scale_type": "4-unipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["Fully acceptable", "Partially acceptable", "Neutral", "Mostly unacceptable", "Completely unacceptable"],
            "scale_type": "5-bipolar", 
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["Without any reservation", "With some reservation", "With medium reservation", "With strong reservation", "Not at all acceptable"],
            "scale_type": "5-unipolar", 
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
    ],
    "BetterOrWorse": [
        {
            "options": ["A lot Better", "A little Better", "No difference", "A little Worse", "A lot Worse"],
            "scale_type": "5-bipolar", 
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["Much better", "Somewhat better", "The same", "Somewhat worse", "Much worse"],
            "scale_type": "5-bipolar", 
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["Greatly better", "Significantly better", "Slightly better", "Slightly worse", "Significantly worse", "Greatly worse"],
            "scale_type": "6-bipolar", 
            "num_scale": [1.0, 0.6, 0.2, -0.2, -0.6, -1.0]
        },
        {
            "options": ["Considerably better", "Marginally better", "Identical", "Marginally worse", "Considerably worse"],
            "scale_type": "5-bipolar", 
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["Vastly better", "A bit better", "Equal", "A bit worse", "Vastly worse"],
            "scale_type": "5-bipolar", 
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
    ],
    "Problem": [
        {
            "options": ["A very big problem", "A moderately big problem", "A small problem", "Not a problem at all"],
            "scale_type": "4-unipolar",
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["A severe problem", "A significant problem", "A minor problem", "No problem whatsoever"],
            "scale_type": "4-unipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["A critical problem", "A substantial problem", "A slight problem", "No problem at all"],
            "scale_type": "4-unipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
        {
            "options": ["A major problem", "A significant problem", "A medium problem", "A trivial problem", "Completely unproblematic"],
            "scale_type": "5-unipolar", 
            "num_scale": [1.0, 0.5, 0.0, -0.5, -1.0]
        },
        {
            "options": ["An enormous problem", "A considerable problem", "A negligible problem", "Absolutely no problem"],
            "scale_type": "4-unipolar", 
            "num_scale": [1.0, 0.33333333, -0.33333333, -1.0]
        },
    ],
}

# Condensed answer options per question type for overview
"""
answer_options_variations = {
    "Importance": [
        ["Very important", "Somewhat important", "Not too important", "Not at all important"],
        ["Extremely important", "Fairly important", "Moderately important", "Slightly important", "Not important at all"],
        ["Essential", "Important", "Neutral", "Unimportant", "Completely unimportant"],
        ["Of great importance", "Of some importance", "Of little importance", "Of no importance"],
        ["Highly important", "Quite important", "Somewhat unimportant", "Completely unimportant"]
    ],
    "Agreement": [
        ["Fully agree", "Somewhat agree", "Somewhat disagree", "Fully disagree"],
        ["Strongly agree", "Moderately agree", "Moderately disagree", "Strongly disagree"],
        ["Completely agree", "Mostly agree", "Slightly agree", "Slightly disagree", "Mostly disagree", "Completely disagree"],
        ["Totally agree", "Mostly agree", "Neither agree nor disagree", "Mostly disagree", "Totally disagree"],
        ["Absolutely agree", "Partly agree", "Partly disagree", "Absolutely disagree"]
    ],
    "Concern": [
        ["Very concerned", "Somewhat concerned", "Not too concerned", "Not at all concerned"],
        ["Extremely concerned", "Fairly concerned", "Moderately concerned", "Slightly concerned", "Not concerned at all"],
        ["Highly concerned", "Quite concerned", "Minimally concerned", "Completely unconcerned"],
        ["Greatly worried", "Considerably worried", "Slightly worried", "Not worried at all"], ### 4-unipolar
        ["Alarmed", "Concerned", "Somewhat unconcerned", "Totally unconcerned"] ### 4-bipolar
    ],
    "Frequency": [
        ["Always", "Often", "Sometimes", "Rarely", "Never"],
        ["Constantly", "Frequently", "Occasionally", "Infrequently", "Not once"],
        ["All the time", "Regularly", "Now and then", "Seldom", "Never"],
        ["Every time", "Quite often", "From time to time", "Hardly ever", "Never"],
        ["Without exception", "Commonly", "Periodically", "Rarely ever", "Never"]
    ],
    "Likelihood": [
        ["Very likely", "Somewhat likely", "Not too likely", "Not at all likely"],
        ["Extremely likely", "Fairly likely", "Slightly likely", "Not likely at all"],
        ["Highly probable", "Moderately probable", "Neither probable nor improbable", "Somewhat improbable", "Highly improbable"]
        ["Almost certainly", "Probably", "Unlikely", "Very unlikely"],
        ["Very probable", "Quite probable", "Rather improbable", "Extremely improbable"]
    ],
    "Quantity": [
        ["A lot", "A fair amount", "Not too much", "None at all"],
        ["A great deal", "A moderate amount", "A small amount", "None whatsoever"],
        ["Substantially", "Considerably", "Minimally", "Not at all"],
        ["Extensively", "Moderately", "Slightly", "Not in the least"],
        ["Extensively", "Considerably", "Quite a bit", "Very little", "Absolutely none"]
    ],
    "GoodOrBad": [
        ["Very good", "Somewhat good", "Neither good nor bad", "Somewhat bad", "Very bad"],
        ["Excellent", "Good", "Bad", "Terrible"],
        ["Extremely good", "Fairly good", "Fairly bad", "Extremely bad"],
        ["Entirely good", "Mainly good", "Neutral", "Mainly bad", "Entirely bad"]
        ["Outstanding", "Adequate", "Poor", "Very poor"]
    ],
    "Reason": [
        ["A major reason", "A somewhat big reason", "A minor reason", "Not a reason"],
        ["A primary factor", "A secondary factor", "A minor factor", "A very weak factor", "Not a factor at all"],
        ["The main reason", "A contributing reason", "A small reason", "Not relevant"],
        ["A key reason", "A partial reason", "A slight reason", "Not a reason at all"],
        ["A principal reason", "A moderate reason", "A negligible reason", "Completely unrelated"]
    ],
    "Priority": [
        ["A top priority", "An important, but not a top priority", "A lower priority", "Should not be done"],
        ["Highest priority", "High priority", "Medium priority", "Low priority", "No proiority at all],
        ["Critical priority", "Significant priority", "Minor priority", "No priority"],
        ["Essential", "Important", "Mildly important", "Optional", "Not important", "Unnecessary"],
        ["Urgent", "Necessary", "Deferrable", "Not needed"]
    ],
    "PositiveNegative": [
        ["Very positive", "Mostly positive", "Neither positive nor negative", "Mostly negative", "Very negative"],
        ["Extremely positive", "Generally positive", "Neutral", "Generally negative", "Extremely negative"],
        ["Very favorable", "Somewhat favorable", "Somewhat unfavorable", "Very unfavorable"],
        ["Strongly positive", "Positive", "Neither", "Negative", "Strongly negative"],
        ["Entirely positive", "Moderately positive", "Moderately negative", "Entirely negative"]
    ],
    "IncreaseDecrease": [
        ["Increase by a lot", "Increase", "Stay about the same", "Decrease", "Decrease by a lot"],
        ["Greatly increase", "Slightly increase", "Remain the same", "Slightly decrease", "Greatly decrease"],
        ["Substantially increase", "Modestly increase", "Modestly decrease", "Substantially decrease"],
        ["Will rise sharply", "Will rise", "Will stay constant", "Will fall", "Will fall sharply"],
        ["Extemely increase", "Significantly increase", "Slightly increase", "Slightly decrease", "Significantly decrease", "Extemely decrease"]
    ],
    "HowWell": [
        ["Very well", "Pretty well", "Not too well", "Not at all well"],
        ["Extremely well", "Quite well", "Not very well", "Poorly"],
        ["Exceptionally well", "Reasonably well", "Somewhat poorly", "Very poorly"],
        ["Superbly", "Adequately", "Inadequately", "Very inadequately"],
        ["Excellently", "Satisfactorily", "Neutral", "Unsatisfactorily", "Very unsatisfactorily"]
    ],
    "Acceptance": [
        ["Always acceptable", "Sometimes acceptable", "Rarely acceptable", "Never acceptable"],
        ["Completely acceptable", "Occasionally acceptable", "Seldom acceptable", "Not once acceptable"],
        ["Totally acceptable", "Acceptable in a bunch of cases", "In few cases acceptable", "Not acceptable under any circumstances"],
        ["Fully acceptable", "Partially acceptable", "Neutral", "Mostly unacceptable", "Completely unacceptable"],
        ["Without any reservation", "With some reservation", "With medium reservation", "With strong reservation", "Not at all acceptable"]
    ],
    "BetterOrWorse": [
        ["A lot Better", "A little Better", "No difference", "A little Worse", "A lot Worse"],
        ["Much better", "Somewhat better", "The same", "Somewhat worse", "Much worse"],
        ["Greatly better", "Significantly better", "Slightly better", "Slightly worse", "Significantly worse", "Greatly worse"],
        ["Considerably better", "Marginally better", "Identical", "Marginally worse", "Considerably worse"],
        ["Vastly better", "A bit better", "Equal", "A bit worse", "Vastly worse"]
    ],
    "Problem": [
        ["A very big problem", "A moderately big problem", "A small problem", "Not a problem at all"],
        ["A severe problem", "A significant problem", "A minor problem", "No problem whatsoever"],
        ["A critical problem", "A substantial problem", "A slight problem", "No problem at all"],
        ["A major problem", "A significant problem", "A medium problem", "A trivial problem", "Completely unproblematic"],
        ["An enormous problem", "A considerable problem", "A negligible problem", "Absolutely no problem"]
    ]
}
"""

def expand_dataframe_with_variations(df, variations_dict):
    """
    Expand dataframe with answer option variations, scale types, and numerical scales per question type (defined in dict answer_options above)
    """
    expanded_rows = []
    
    for _, row in df.iterrows():
        question_type = row['question_type']
        
        if question_type in variations_dict:
            variations = variations_dict[question_type]
            
            for var_id, variation in enumerate(variations, 1):
                new_row = row.copy()
                new_row['answer_options'] = variation['options']
                new_row['scale_type'] = variation['scale_type']
                new_row['num_scale'] = variation['num_scale']
                new_row['answer_var_id'] = var_id
                expanded_rows.append(new_row)
        else:
            # Fallback for question types not in dictionary
            print(f"Warning: Question type '{question_type}' not found in variations dictionary")
            new_row = row.copy()
            new_row['answer_var_id'] = 1
            expanded_rows.append(new_row)
    
    return pd.DataFrame(expanded_rows)

questions_expanded = expand_dataframe_with_variations(questions_kept, answer_options)

questions_expanded = questions_expanded[['question_id', 'question_var_id', 'answer_var_id', 'question', 'answer_options', 'num_scale',
                                         'scale_type', 'question_type', 'subject', 'source']]
questions_expanded.to_csv(os.path.join(script_dir, "data", "clean", "opinionQA_questions_final.csv"), index=False)
