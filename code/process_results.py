import argparse
from collections import defaultdict
from functools import reduce
import pandas as pd
from pathlib import Path

"""
TODO:
  * Merging of multiple results files
  * Add Golden col (need to identify golden files)
  * Rename cols: [DONE]
    - veri_aggregated -> {model}-s1
    - p2_aggregated -> {model}-s2
    - CATEGORY1 -> {model}-category
    - ob_aggregated -> {model}-s3-1
    - sub_aggregated -> {model}-s3-2
   => may be best to do this renaming in the annotation script instead?
   => counterpoint, best avoid making big changes to the pipeline
  * Something for working on dir instead of individual files?
    - can work by filenames?
  * Figure out a way to extract human evaluation results from existing files

Program Flow:
  1. Parse args:
      - results files
      - gold labels
  2. Process results
      - identify models and 'speeches' (subj to change)
      - create dict of structure model -> speech -> filepath
      - programmatically load each file
      - add speech identifier col
      - concatenate each speech for each model
  3. Rename model cols
  4. Join results tables resulting in one big table
  vvvvv TODO FROM HERE vvvvv
  5. Add gold column
  6. DONE - ready to compute scores
"""


def rename_and_filter_model_cols(df: pd.DataFrame, model_name: str
                                 ) -> pd.DataFrame:
    """
    Rename cols in results dataframe to contain model name and drop unused
    columns.
    """
    rename_dict = {
        'veri_aggregated': f'{model_name}-s1',
        'p2_aggregated': f'{model_name}-s2',
        'CATEGORY': f'{model_name}-category',
        'ob_aggregated': f'{model_name}-s3-1',
        'sub_aggregated': f'{model_name}-s3-2'
    }
    to_drop = ['ANALYSIS1', 'FACT_PART1', 'VERIFIABLE_REASON1',
               'VERIFIABILITY1', 'CATEGORY1', 'subjectivity', 'objectivity']
    df = df.rename(columns=rename_dict).drop(to_drop, axis=1)
    return df


def bool_s2_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Convert s2 cols to boolean datatype."""
    def val_to_bool(val):
        if not isinstance(val, str):
            return False
        if 'rue' in val.lower():
            return True
        return False
    for col_name in [col for col in df.columns if col.endswith('-s2')]:
        df[col_name] = df[col_name].apply(val_to_bool)
    return df


def merge_result_dfs(dfs: [pd.DataFrame]) -> pd.DataFrame:
    """Merge model dataframes to create one big df"""
    merge_keys = ['SENTENCES', 'SPEECH']
    final_df = reduce(lambda left, right: pd.merge(
        left, right, on=merge_keys, how='outer'), dfs)
    return final_df


def load_file(filename: str) -> pd.DataFrame:
    """Load file as df independent of extension."""
    if filename.endswith('xlsx'):
        df = pd.read_excel(filename)
    else:
        df = pd.read_csv(filename, encoding='utf-8')
    return df


def write_file(output_name: str, df: pd.DataFrame) -> None:
    """Write df to file independent of extension."""
    if output_name.endswith('xlsx'):
        df = df.to_excel(output_name)
    else:
        df = df.to_csv(output_name)
    return df


def combine_gold_labels(gold_label_file: str, df: pd.DataFrame
                        ) -> pd.DataFrame:
    """Combine combined df with gold label file."""
    gold_df = load_file(gold_label_file)
    return merge_result_dfs([gold_df, df])


def main(args):
    # Collect results files by model & speech
    results_files = defaultdict(dict)
    for file in args.results_files:
        speech, model, _ = Path(file).name.rstrip('.csv').split('_')
        results_files[model][speech] = file

    # Concatenate results by model
    model_dfs = []
    for model, speech_dict in results_files.items():
        dfs = []
        # Add speech col
        for speech, filepath in speech_dict.items():
            df = load_file(filepath)
            df.insert(1, 'SPEECH', speech)
            dfs.append(df)
        # Concatenate dfs for the same model
        df_concat = pd.concat(dfs, ignore_index=True)
        df_concat = rename_and_filter_model_cols(df_concat, model_name=model)
        model_dfs.append(df_concat)

    # Join resultant dataframes
    final_df = merge_result_dfs(model_dfs)
    if args.gold_file:
        df_gold = combine_gold_labels(args.gold_file, final_df)
        write_file(args.output, df_gold)
    else:
        write_file(args.output, final_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_files', nargs='+',
                        help='Results CSV or Excel files')
    parser.add_argument('--gold-file', '-g',
                        help='File containing gold labels')
    parser.add_argument('--output', '-o', required=True,
                        help='Output file')
    args = parser.parse_args()
    main(args)
