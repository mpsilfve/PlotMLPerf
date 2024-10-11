import pandas as pd
import json
import re
import click

MAX_TOTAL_ACCELERATORS = 64

def tr_count(x):
    try:
        return int(x.replace(",","")) if type(x) == type("") else x
    except:
        return -1

def tr_latency(x):
    return x
    
def get_column_name_count(row):
    for i in range(100):
        if not f"Unnamed: {i}" in row:
            return i

def get_description_column(row):
    for key in row.keys():
        if type(key) == type("") and "Benchmark  /  Model MLC" in key:
            return key
    return "ERROR"

@click.command()
@click.option("--input-filename",required=True, help="MLPerf Excel filename")
@click.option("--output-filename", required=True, help="Output json filename")
@click.option("--total-accelerators-key", required=True, help="Key used to refer to the total accelerator count in the Excel spreadsheet (probably 'Total Accelerators' or '# of Accelerators').")
@click.option("--accelerator-name-key",required=True,help="Key used to refer to the name of the GPU model. Probably 'Accelerator Model Name' or 'Accelerator'")
@click.option("--models", required=True, help="Comma-separated list of model names")
def main(input_filename, output_filename,total_accelerators_key,accelerator_name_key,models):
    mlperf_table = pd.read_excel(input_filename)
    
    # First step is to recover system info column names which were
    # lost when converting to Excel from Tableau. These can be found
    # as values for columns Unnamed: 0 - Unnamed: X on row 3. X is
    # returned by function get_column_name_count().
    name_row = mlperf_table.iloc[3]
    column_name_count = get_column_name_count(name_row)
    total_columns = len(name_row)
    name_index = {f"Unnamed: {i}":name_row[f"Unnamed: {i}"] for i in range(column_name_count)}
    mlperf_table.rename( columns=name_index, inplace=True )
    models = models.split(",")
    # Second step is to recover model names which were lost when
    # converting to Excel from Tableau. These can be found as vales for
    # columns "Benchmark / Model MLC / Units (copy) / Units" and Unnamed:
    # 17 - Unnamed: 24
    name_row = mlperf_table.iloc[1]
    name_index = {get_description_column(name_row):name_row[get_description_column(name_row)]}#"Benchmark  /  Model MLC  /  Units (copy)  /  Units"]}
    name_index.update({f"Unnamed: {i}":name_row[f"Unnamed: {i}"] for i in range(column_name_count+1,total_columns)})
    mlperf_table.rename( columns=name_index, inplace=True )
    
    mlperf_table[total_accelerators_key] = mlperf_table[total_accelerators_key].transform(tr_count)
    #models = "bert,dlrm_dcnv2,gnn,gpt3,llama2_70b_lora,resnet,ssd,stable_diffusion,unet3d".split(",")
    for model in models:
        mlperf_table[model] = mlperf_table[model].transform(tr_latency)
        
    # We'll then filter out experiments which are not relevant to us:
    
    skip_header = True
    experiments = []
    for i, row in mlperf_table.iterrows():
        if type(row["Public ID"]) == type("") and "-" in row["Public ID"]:
            skip_header = False
        if not skip_header:
            if type(row["Public ID"]) == type("") and row["Public ID"] != "NaN":
                experiments.append({"Public ID":str(row["Public ID"]),
                                    "Accelerator Model Name":re.sub("\\n.*","",str(row[accelerator_name_key]).replace("Accelerator Model Name    ","").replace("Accelerator    ","")),
                                    # "Organization":str(row["Organization"]),
                                    # "Accelerators Per Node":str(row["Accelerators Per Node"]),
                                    "Total Accelerators":-1})                
                experiments[-1].update({m:-1 for m in models})
                skip_results = False
                total_accelerators = 0
                if type(row[total_accelerators_key]) == type(""):
                    total_accelerators = tr_count(row[total_accelerators_key])
                elif type(row[total_accelerators_key]) == type(1.0):
                    total_accelerators = int(row[total_accelerators_key])
                else:
                    total_accelerators = MAX_TOTAL_ACCELERATORS + 1
                experiments[-1]["Total Accelerators"] = total_accelerators
            else:
                for model in models:
                    if "Avg. Result at System Name" in list(row):
                        experiments[-1][model] = row[model]
    experiments = [d for d in experiments if d["Total Accelerators"] <= MAX_TOTAL_ACCELERATORS]
    json.dump(experiments,open(output_filename,"w"))

if __name__=="__main__":
    main()
