import argparse
import pandas as pd
import re

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
"""


def rename_model_cols(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    # expects filename to start with model name and contain underscore
    model_name = filename.split('_')[0]
    rename_dict = {
        'veri_aggregated': f'{model_name}-s1',
        'p2_aggregated': f'{model_name}-s2',
        'CATEGORY': f'{model_name}-category',
        'ob_aggregated': f'{model_name}-s3-1',
        'sub_aggregated': f'{model_name}-s3-2'
    }
    df.rename(mapper=rename_dict)
    return df


# TODO can't do this one until I've got more data?
def merge_new_data(self) -> None:
    pass


def merge_new_models(self) -> None:
    pass


def extract_gold_labels(self) -> None:
    pass


def load_file(filename: str) -> pd.DataFrame:
    if filename.endswith('xslx'):
        df = pd.read_excel(filename)
    else:
        df = pd.read_csv(filename, encoding='utf-8')
    return df


def main(args):
    if len(args.filename) > 1:
        pass
    # model_names = [x for x in df.columns if re.match(r'^[\w.]+[-:][\w.]+$', x)]
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='+', help='Path to results file(s)')
    args = parser.parse_args()
    main(args)
