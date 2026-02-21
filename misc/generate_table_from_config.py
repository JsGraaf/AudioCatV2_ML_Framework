import yaml
import argparse
import pandas as pd
from tabulate import tabulate
import os

def extract_relevant_info(config):
    preprocessing = config.get("preprocessing", {})
    feature_extraction = config.get("feature_extraction", {})
    training = config.get("training", {})

    model = training.get("model", {})
    optimizer_dict = training.get("optimizer", {})
    optimizer_name = list(optimizer_dict.keys())[0] if optimizer_dict else "Unknown"
    learning_rate = optimizer_dict.get(optimizer_name, {}).get("learning_rate", "N/A")

    return {
        "Target Sample Rate": preprocessing.get("target_rate", "N/A"),
        "Minimum Length": f"{preprocessing.get('min_length', 'N/A')} s",
        "Maximum Length": f"{preprocessing.get('max_length', 'N/A')} s",
        "Frame Length": preprocessing.get("frame_length", "N/A"),
        "Hop Length": preprocessing.get("hop_length", "N/A"),
        "Number of Mel Bands": feature_extraction.get("n_mels", "N/A"),
        "Window Function": feature_extraction.get("window", "N/A"),
        "Overlap": feature_extraction.get("overlap", "N/A"),
        "FFT Size (n_fft)": feature_extraction.get("n_fft", "N/A"),
        "Patch Length": f"{feature_extraction.get('patch_length', 'N/A')} frames",
        "Normalization": feature_extraction.get("norm", "N/A"),
        "Input Shape": str(model.get("input_shape", "N/A")),
        "Batch Size": training.get("batch_size", "N/A"),
        "Epochs": training.get("epochs", "N/A"),
        "Dropout": training.get("dropout", "N/A"),
        "Optimizer": f"{optimizer_name} (lr={learning_rate})",
        "Epochs": training.get("epochs", "N/A"),
    }

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX table from user_config.yaml")
    parser.add_argument("config_path", type=str, help="Path to the user_config.yaml file")
    parser.add_argument("output_name", type=str, help="Filename (without .tex) for the LaTeX table")
    parser.add_argument("--output_path", type=str, default=None, help="Optional output path for the LaTeX table file")
    args = parser.parse_args()

    # Load YAML
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract info and format table
    table_data = extract_relevant_info(config)
    df = pd.DataFrame(list(table_data.items()), columns=["Parameter", "Value"])
    latex_rows = tabulate(df.values, headers=df.columns, tablefmt="latex").splitlines()

    # Insert \hline after every row (skip the first two lines: header + \hline)
    latex_with_lines = []
    for i, line in enumerate(latex_rows):
        latex_with_lines.append(line)
        if (i == 0):
            latex_with_lines[i] = '\\begin{tabular}{l|l}'
        if (line == "\\hline"):
            continue
        
        if i >= 2 and line != "\end{tabular}":  # only add \hline after header and data rows
            latex_with_lines[i] += "\\hline"

    latex_table = "\n".join(latex_with_lines)

    latex_output = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Configuration of the audio preprocessing and model training parameters used in this experiment.}\n"
        "\\label{tab:experiment_" + f"{args.output_name}" + "_config}\n"
        f"{latex_table}\n"
        "\\end{table}"
    )

    # Ensure output directory exists
    os.makedirs("tables", exist_ok=True)
    if (args.output_path != None): 
        output_path = os.path.join(args.output_path, f"{args.output_name}.tex")
    else:
        output_path = os.path.join("tables", f"{args.output_name}.tex") 



    # Save LaTeX table
    with open(output_path, "w") as f:
        f.write(latex_output)

    print(f"LaTeX table written to: {output_path}")

if __name__ == "__main__":
    main()
