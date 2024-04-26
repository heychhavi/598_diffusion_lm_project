from bert_score import score
import pandas as pd
import numpy as np

def compute_bert_scores(candidates, references):
    """
    Computes the BERT scores for a list of candidate sentences against reference sentences.

    Args:
    - candidates (list of str): Generated sentences.
    - references (list of str): Ground truth sentences.

    Returns:
    - DataFrame with Precision, Recall, and F1 scores for each candidate.
    """
    P, R, F1 = score(candidates, references, lang="en", verbose=True)
    results = pd.DataFrame({
        "Candidate": candidates,
        "Reference": references,
        "Precision": P.numpy(),
        "Recall": R.numpy(),
        "F1 Score": F1.numpy()
    })
    return results

# Example usage
references = ["250 TL account was received for 1 portion of rice + kuru fasulye + ayran + Cola. Service and waiters are very bad.",
"We had the opportunity to taste beyran for the first time in our lives and we had heard of the reputation of Metanet's beyran. At first; we had a bit of a prejudice; to be honest; about the intense aroma of meat or fat; but it's not like that at all. Beyrani is so delicious and balanced that we can eat it at any time of the day; morning; noon and evening",
"Excellent appetizers; great atmosphere; acceptable prices"]

candidates1 = ["UNK UNK UNK . . . . UNK of UNK UNK UNK UNK UNK UNK UNK UNK UNK and UNK are very UNK",
              "UNK UNK the UNK to UNK UNK for the UNK UNK in our UNK and we UNK UNK of the UNK of UNK UNK UNK UNK we UNK a UNK of a UNK to be UNK UNK . . . . UNK UNK UNK but UNK not UNK that at UNK UNK is UNK UNK and UNK that we can eat it at UNK UNK of the UNK UNK UNK and UNK",
              "UNK . . UNK UNK UNK"]

candidates2 = ["UNK UNK UNK UNK the Cocum are UNK of rice UNK UNK UNK UNK UNK UNK UNK UNK and UNK are very UNK",
"We had the UNK to taste UNK for the UNK UNK in our UNK and we had UNK of the UNK of UNK UNK At UNK we had a bit of a UNK to be UNK about Burger King . Our UNK or UNK but UNK not like that at UNK UNK is so delicious and UNK that we can eat it at UNK UNK of the UNK UNK UNK and UNK",
"UNK are in UNK UNK UNK"]

diff500 = ["UNK UNK UNK Midsummer House with an UNK of rice UNK UNK UNK UNK UNK UNK UNK UNK and UNK are very UNK END",
"We had the UNK to taste UNK for the UNK UNK in our UNK and we had UNK of the UNK of UNK UNK At UNK we had a bit of a UNK to be UNK about 3 out of 5 UNK or UNK but UNK not like that at UNK UNK is so delicious and UNK that we can eat it at UNK UNK of the UNK UNK UNK and UNK",
"UNK The Vaults UNK UNK UNK"]

diff1000 = ["",
"",
""]
# for i in candidates2:
#     print(len(i.split()))
# for i in candidates1:
#     print(len(i.split()))
# for i in references:
#     print(len(i.split()))

# Compute BERT scores
# bert_scores = compute_bert_scores(candidates1, references)
# bert_scores2 = compute_bert_scores(candidates2, references)
# bert_scores3 = compute_bert_scores(diff500, references)

# Print the results
# print(bert_scores)
# print(bert_scores2)
# print(bert_scores3)

import matplotlib.pyplot as plt

# Sample data
f1_scores = [0.766513,0.781068,0.775862] #bert_scores['F1 Score'].tolist()
f1_scores2 = [0.784240,0.809651,0.793390] #bert_scores2['F1 Score'].tolist()
f1_scores3 = [0.786876,0.813213,0.795723]
categories = ['BERT-large\ntrain=50K steps\ndiffusion=2000 steps', 'BERT-base\ntrain=200K steps\ndiffusion=2000 steps', 'BERT-base\ntrain=200K steps\ndiffusion=500 steps']
values = [np.average(f1_scores), np.average(f1_scores2), np.average(f1_scores3)]

# Create a bar plot
fig, ax = plt.subplots()
bars = ax.bar(categories, values, color='skyblue',width=0.8)

# Add data labels
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), va='bottom')

# Set plot title and labels
ax.set_title('DiffusionLM output')
ax.set_xlabel('Categories')
ax.set_ylabel('BERT Score')
ax.set_ylim([0.6,0.9])

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('super_pretty_bar_chart.png', dpi=300)