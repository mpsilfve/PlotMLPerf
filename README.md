## Extracting data from MLPerf tables

Script to extract the info we need for generating plots from the Excel files MLPerf makes available:

```
$ python3 scripts/extract_benchmark_data.py --help
Usage: extract_benchmark_data.py [OPTIONS]

Options:
  --input-filename TEXT          MLPerf Excel filename  [required]
  --output-filename TEXT         Output json filename  [required]
  --total-accelerators-key TEXT  Key used to refer to the total accelerator
                                 count in the Excel spreadsheet (probably
                                 'Total accelerators' or '# of Accelerators').
                                 [required]
  --accelerator-name-key TEXT    Key used to refer to the name of the GPU
                                 model. Probably 'Accelerator Model Name' or
                                 'Accelerator'  [required]
  --models TEXT                  Comma-separated list of model names
                                 [required]
  --help                         Show this message and exit.
```

Example run:

```
python3 scripts/extract_benchmark_data.py --input-filename assets/mlperf_tables/mlperf_training_40.xlsx \
                                          --output-filename assets/mlperf_json/mlperf_inference_40.json \
					  --total-accelerators-key 'Total Accelerators' \
					  --accelerator-name-key 'Accelerator Model Name' \
					  --models "bert,dlrm_dcnv2,gnn,gpt3,llama2_70b_lora,resnet,ssd,stable_diffusion,unet3d"
```

## Generating plots

Script to generate a visualization from MLPerf data in JSON form:

```
$ python3 scripts/generate_plot.py --help
Usage: generate_plot.py [OPTIONS]

Options:
  --title TEXT           Title for the plot  [required]
  --num-gpu TEXT         How many GPUs were used  [required]
  --architectures TEXT   Generate plots for these architectures. Comma-
                         separated list. E.g. H100,A100,L40S  [required]
  --tasks TEXT           The tasks which we want to compare  [required]
  --table-filename TEXT  The JSON table with runtimes  [required]
  --ylabel TEXT          Description of the y-axis  [required]
  --output-file TEXT     File where output graph is stored  [required]
  --normalize_max TEXT   Whether to normalize numbers by maximal value
  --normalize_min TEXT   Whether to normalize numbers by manimal value
  --invert TEXT          Whether to invert numbers v -> 1/v
  --xlabel TEXT          Description of the x-axis
  --help                 Show this message and exit.

```

Example run:


