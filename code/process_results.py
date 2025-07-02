import argparse
import pandas as pd

"""
TODO:
  * Merging of multiple results files
  * Add Golden col (need to identify golden files)
  * Rename cols:
    - veri_aggregated -> {model}-s1
    - p2_aggregated -> {model}-s2
    - CATEGORY1 -> {model}-category
    - ob_aggregated -> {model}-s3-1
    - sub_aggregated -> {model}-s3-2
   => may be best to do this renaming in the annotation script instead?
  * Something for working on dir instead of individual files?
"""


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='Path to results file')
    args = parser.parse_args()
    main(args)
