import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import click

from matplotlib import rcParams

# Set the font family globally to Inter
rcParams['font.family'] = 'Inter'

LARGE = 10000000000

def check_tasks(row, tasks):
    res = {}
    for task in tasks:
        if not pd.isna(row[task]):
            res[task] = row[task]
    return res

def check_architecture(row, architectures):
    return row['Accelerator Model Name'] in architectures

@click.command()
@click.option("--title",required=True,help="Title for the plot")
@click.option("--num-gpu",required=True,help="How many GPUs were used")
@click.option("--architectures",required=True,help="Generate plots for these architectures. Comma-separated list. E.g. H100,A100,L40S")
@click.option("--tasks",required=True,help="The tasks which we want to compare")
@click.option("--table-filename",required=True, help="The JSON table with runtimes")
@click.option("--ylabel", required=True, help="Description of the y-axis")
@click.option("--output-file", required=True, help="File where output graph is stored")
@click.option("--normalize_max",required=False,help="Whether to normalize numbers by maximal value")
@click.option("--normalize_min",required=False,help="Whether to normalize numbers by manimal value")
@click.option("--invert",required=False,help="Whether to invert numbers v -> 1/v")
@click.option("--xlabel",required=False, help="Description of the x-axis")
def main(title,num_gpu,architectures,tasks,table_filename,ylabel,output_file,normalize_max=False,normalize_min=False,invert=False,xlabel="Model"):
    num_gpu = int(num_gpu)
    runtimes=json.load(open(table_filename))#json.load(open("table_training.json"))
    architectures = architectures.split(',')
    tasks = tasks.split(",")

    # Filter out irrelevant information
    runtime_df = pd.DataFrame.from_dict(runtimes)
    data = {task:{arch:LARGE for arch in architectures} for task in tasks}
    for i, row in runtime_df.iterrows():
        if (int(row["Total Accelerators"]) == num_gpu and
            check_tasks(row, tasks) and
            check_architecture(row, architectures)):
            times = check_tasks(row, tasks)
            for task in times:
                data[task][row["Accelerator Model Name"]] = min(times[task],
                                                                data[task][row["Accelerator Model Name"]])

    data = {task:data[task] for task in data
               if not LARGE in data[task].values()}
    
    if invert:
        data = {task:{k:1/v for k, v in data[task].items()} for task in data}        
    if normalize_max:
        data = {task:{k:v/max(data[task].values()) for k, v in data[task].items()} for task in data}
    elif normalize_min:
        data = {task:{k:v/min(data[task].values()) for k, v in data[task].items()} for task in data}

    # The font should be inter...
    
    # Transforming the data into a suitable format for seaborn
    df = pd.DataFrame(data).reset_index().melt(id_vars='index', var_name='Task', value_name='Performance')
    df.rename(columns={'index': 'GPU'}, inplace=True)
    
    # Rename columns to match the latest plot requirements
    #ylabel = "Latency in minutes (lower is better)"
    #xlabel = "Model"
    df_renamed = df.rename(columns={'Task': xlabel, 'Performance': ylabel})
    
    # Darker color palette for bars
    darker_palette = ['#7fb1c3', '#4b728a', '#245866']#, '#405966', '#334c59']
    
    # Plotting with updated labels and style enhancements
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df_renamed, x=xlabel, y=ylabel, hue='GPU', palette=darker_palette)

    # Title and axis labels
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.ylabel(ylabel, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=14,fontweight="bold")
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Legend adjustments
    plt.legend(title='GPU', fontsize=12, title_fontsize=13, loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    
    # Adding annotations to bars with a condition to skip near-zero heights
    for bar in plt.gca().patches:
        height = bar.get_height()
        if height > 0.1:  # Avoid annotating bars with heights near zero
            plt.text(
                bar.get_x() + bar.get_width() / 2, 
                height, 
                f'{height:.1f}x', 
                ha='center', 
                va='bottom', 
                fontsize=12
            )

    # Removing spines and adding light grid lines
    sns.despine(left=True, bottom=True)
    plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_file)

if __name__=="__main__":
    main()
