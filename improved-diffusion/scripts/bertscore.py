from bert_score import score
import pandas as pd
import numpy as np
import json
import csv
import random
import matplotlib.pyplot as plt
import os
import datetime
import time
from evaluate import load
bertscore = load("bertscore")


curtime = '6' #datetime.datetime.now().strftime('%Y%m%d-%H%M')


def make_dirs():
    for i in DIFFUSION_STEPS:
        outdir=f"/home/selinali/diffusion_lm/improved-diffusion/out_gen/diff_{i}"
        if not os.path.exists(outdir):
            os.make_dirs(outdir)

def sample_references_from_dataset():
    l=4693
    print('sampling')
    fp = '/home/selinali/diffusion_lm/datasets/e2e_data/src1_test.txt'
    row_ids = np.random.choice(list(range(l)),100) # test set length
    with open(fp, 'r') as f:
        lines = f.readlines()
    with open(ground_truth_file,'w+') as f:
        for i in row_ids:
            f.write(lines[i].split("||")[1])


def pad():
    print('padding')
    with open(ground_truth_file,'r') as f:
        sentences = f.readlines()
    seq={}
    N=len(sentences)
    percentages=[]

    with open(partial_seq_file+'.txt','w+') as f:
        for i in range(N):
            sen=sentences[i]
            word_lst = sen.split()
            l = len(word_lst)
            nheads = np.random.randint(3,6)
            heads=np.random.choice(list(range(l)),nheads)
            gaps=np.random.choice([1,1,2,2,3,3,4,4],nheads)
            count = 0
            # each sentence is padded at 3 locations for a length of 1-5, the locations can overlap.
            for j in range(nheads):
                for k in range(heads[j],min(heads[j]+gaps[j],l)):
                    if word_lst[k]!='PAD' and word_lst[k]!='.' and word_lst[k]!=',':
                        count+=1
                        word_lst[k]='PAD'
            new_sen=' '.join(word_lst)
            percentage=count/l
            seq[str(i)] = {'text':new_sen,'percentage':percentage}
            f.write(new_sen+'\n')
            percentages.append(percentage)
    with open(partial_seq_file+'.json','w+') as f:
        json.dump(seq, f, indent=4)
    plt.hist(percentages)
    plt.savefig('/home/selinali/diffusion_lm/improved-diffusion/out_gen/pad_percentages.png')

    
def run_diffusion(diffusion_step):
    command=f"python3 {script_path} \
    --model_path {model_path} --eval_task_ 'infill' \
    --use_ddim True --eta 1. --verbose pipe --diffusion_steps {diffusion_step} \
    --partial_seq_file {partial_seq_file}.json --notes {curtime}"
    print(command)
    os.system(command)



partial_seq_file="/home/selinali/diffusion_lm/improved-diffusion/out_gen/sample"
ground_truth_file='/home/selinali/diffusion_lm/improved-diffusion/out_gen/infill_testdata.txt'
script_path="/home/selinali/diffusion_lm/improved-diffusion/scripts/infill_parrot.py"


categories = []
values = []
values_max = []
runtimes = []
DIFFUSION_STEPS = [1000,400,200,100,80,40,10,5,2] #
# DIFFUSION_STEPS = [40]



# sample_references_from_dataset()
# pad()



time.sleep(5)
with open(ground_truth_file,'r') as f:
    gt_sentences = f.readlines()
expanded = []
for sen in gt_sentences:
    for i in range(8):
        expanded.append(sen)

model_names=['base']
for I in range(1):
    MODEL_NAME=model_names[I]

    # generate infilling test set
    model_path=f"/home/selinali/diffusion_lm/diffusion-models/{MODEL_NAME}/{MODEL_NAME}.pt"
    categories=[]
    for i in DIFFUSION_STEPS:  
        print(f"loop running diffusion step = {i}")
        categories.append(f'{i}')
        # run_diffusion(diffusion_step=i,)
        output_file = f"/home/selinali/diffusion_lm/improved-diffusion/out_gen/diff_{i}/{MODEL_NAME}.{MODEL_NAME}.pt.infill_infill_{curtime}.txt"
        time_file = f"/home/selinali/diffusion_lm/improved-diffusion/out_gen/diff_{i}/time"
        with open(time_file, 'r') as f:
            lines=f.readlines()
            runtime=float(lines[1])
        print(output_file)
        while True:
            if os.path.exists(output_file):
                with open(output_file,'r') as f:
                    gen_sentences=f.readlines()
                break
            time.sleep(30)
        # Compute BERT scores
        print(len(gen_sentences))
        print(len(expanded))
        bert_scores = results = bertscore.compute(predictions=gen_sentences, references=expanded, model_type="distilbert-base-uncased")
        f1_scores = bert_scores['f1'] 
        values.append(np.average(f1_scores))
        values_max.append(np.std(f1_scores))
        runtimes.append(runtime)
        print(f'{i}:{np.min(f1_scores)}')
        # except:
        #     continue

    # Create a bar plot
    fig, ax = plt.subplots()
    bars1 = ax.bar(categories, values, color='skyblue',width=0.8)
    bars2 = ax.bar(categories, values_max, bottom=values,color='lightcyan',width=0.8)
    x = np.arange(len(categories))
    ax.plot(x, runtimes, label='total runtime')
    # Add data labels
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), va='bottom')

    # Set plot title and labels
    ax.set_title('DiffusionLM output BERT-base')
    ax.set_xlabel('Number of diffusion steps')
    ax.set_ylabel('BERT Score')
    ax.set_ylim([0.6,1])

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig('super_pretty_bar_chart.png', dpi=300)