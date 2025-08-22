import argparse
import numpy as np
import pandas as pd
import random
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score)
from process_results import load_file

random.seed(42)


def compute_likelihood(df_to_eval, model_names):
    for model in model_names:
        # Skip incorrectly formatted cols (original llama and zephyr)
        if f'{model}-s1' not in df_to_eval.columns:
            continue
        df_to_eval[f'{model}-s1'].fillna('No')
        df_to_eval[f'{model}-s2'].fillna(False)
        df_to_eval[f'{model}-s3-1'].fillna('Subjective')
        df_to_eval[f'{model}-s3-2'].fillna('Subjective')
        p1 = df_to_eval[f'{model}-s1'].apply(
            lambda x: 1 if pd.notna(x) and "yes" in x.lower() else 0).values
        p2 = []
        for i, c in zip(df_to_eval[f'{model}-s2'],
                        df_to_eval[f'{model}-category']):
            if i or 'C0' not in c:
                p2.append(1)
            else:
                p2.append(0)
        p2 = np.array(p2)
        p3_1 = df_to_eval[f'{model}-s3-1'].apply(
            lambda x: 1 if pd.notna(x) and "objective" in x.lower() else 0
        ).values
        p3_2 = df_to_eval[f'{model}-s3-2'].apply(
            lambda x: 1 if pd.notna(x) and "objective" in x.lower() else 0
        ).values

        df_to_eval[model] = p1 + p2 + 0.5 * p3_1 + 0.5 * p3_2
    return df_to_eval


def mapping(col, neg):
    return [
        0 if i == neg or str(neg).lower() in str(i).lower() else 1
        for i in col]


def mapping_two(l1, l2, neg):
    ret = []
    for i1, i2 in zip(l1, l2):
        if i1 != i2:
            ret.append(random.randint(0, 1))
        elif i1 == neg:
            ret.append(0)
        else:
            ret.append(1)
    return ret


def metrics_for_label(labels, gold, scores_dict, prefix=''):
    # Calculate average metric (deals with S3 labels)
    for label in labels:
        scores_dict[f'{prefix}acc'] = (accuracy_score(
            gold, label) / len(labels)) + scores_dict.get(f'{prefix}acc', 0)
        scores_dict[f'{prefix}recall_pos'] = (
            (recall_score(gold, label) / len(labels)
             ) + scores_dict.get(f'{prefix}recall_pos', 0))
        scores_dict[f'{prefix}precision_pos'] = (
            (precision_score(gold, label) / len(labels)
             ) + scores_dict.get(f'{prefix}precision_pos', 0))
        scores_dict[f'{prefix}recall_neg'] = (
            (recall_score(gold, label, pos_label=0) / len(labels)
             ) + scores_dict.get(f'{prefix}recall_neg', 0))
        scores_dict[f'{prefix}precision_neg'] = (
            (precision_score(gold, label, pos_label=0) / len(labels)
             ) + scores_dict.get(f'{prefix}precision_neg', 0))
        scores_dict[f'{prefix}f1_claim'] = (
            (f1_score(gold, label) / len(labels)
             ) + scores_dict.get(f'{prefix}f1_claim', 0))
        scores_dict[f'{prefix}f1_nonclaim'] = (
            (f1_score(gold, label, pos_label=0) / len(labels)
             ) + scores_dict.get(f'{prefix}f1_nonclaim', 0))
        scores_dict[f'{prefix}macro_f1'] = (
            (f1_score(gold, label, average='macro') / len(labels)
             ) + scores_dict.get(f'{prefix}macro_f1', 0))
        scores_dict[f'{prefix}macro_recall'] = (
            (recall_score(gold, label, average='macro') / len(labels)
             ) + scores_dict.get(f'{prefix}recall_pos', 0))
        scores_dict[f'{prefix}macro_precision'] = (
            (precision_score(gold, label, average='macro') / len(labels)
             ) + scores_dict.get(f'{prefix}precision_pos', 0))

    print(f'{prefix.upper().lstrip('-')} Accuracy score:',
          scores_dict[f'{prefix}acc'])
    print(f'{prefix.upper().lstrip('-')} Recall Score (negative): ',
          scores_dict[f'{prefix}recall_neg'])
    print(f'{prefix.upper().lstrip('-')} Precision Score (negative): ',
          scores_dict[f'{prefix}precision_neg'])
    print(f'{prefix.upper().lstrip('-')} Recall Score (positive): ',
          scores_dict[f'{prefix}recall_pos'])
    print(f'{prefix.upper().lstrip('-')} Precision Score (positive): ',
          scores_dict[f'{prefix}precision_pos'])
    print(f'{prefix.upper().lstrip('-')} Claim F1 Score: ',
          scores_dict[f'{prefix}f1_claim'])
    print(f'{prefix.upper().lstrip('-')} Non-Claim F1 Score: ',
          scores_dict[f'{prefix}f1_nonclaim'])
    print(f'{prefix.upper().lstrip('-')} Macro F1 Score: ',
          scores_dict[f'{prefix}macro_f1'])
    print(f'{prefix.upper().lstrip('-')} Macro Recall Score: ',
          scores_dict[f'{prefix}macro_recall'])
    print(f'{prefix.upper().lstrip('-')} Macro Precision Score: ',
          scores_dict[f'{prefix}macro_precision'])


def main(df, save_to, verbose):
    model_names = [x[:-3] for x in df.columns if x.endswith('-s1')]
    if 'zephyr' in df and 'llama' in df:
        model_names.extend(['zephyr', 'llama'])
    df = compute_likelihood(df, model_names)
    scores_list = []

    print(f'Mean Gold label : {np.mean(df['Golden'])}')

    # Kappa scores can only be calculated if multiple Golden labels are present
    if 'label_1' in df.columns and 'label_2' in df.columns:
        compute_kappa = True
    else:
        compute_kappa = False

    for model_name in model_names:
        scores_dict = {'model': model_name}

        df[f'{model_name}_label'] = df[model_name].apply(
            lambda x: 0 if x <= 1.5 else 1)
        print(f'\n==={model_name}===')

        # support original zephyr and llama result
        if f'{model_name}-s1' not in df:
            print('Accuracy', accuracy_score(
                df['Golden'], df[f'{model_name}_label']))
            if compute_kappa:
                print('Kappa', (
                    cohen_kappa_score(
                        df['label_1'], df[f'{model_name}_label']
                    ) + cohen_kappa_score(
                        df['label_2'], df[f'{model_name}_label']
                    ) / 2))
            continue

        sub_df = df
        # S1 Results ----------------------------------------------------------
        print("S1 label")
        if compute_kappa:
            scores_dict['s1-kappa'] = (
                cohen_kappa_score(
                    sub_df['label_1'],
                    mapping(sub_df[f'{model_name}-s1'], neg='No')
                ) + cohen_kappa_score(
                    sub_df['label_2'],
                    mapping(sub_df[f'{model_name}-s1'], neg='No'))
            ) / 2
            print('S1- Kappa score', scores_dict['s1-kappa'])

        metrics_for_label([mapping(sub_df[f'{model_name}-s1'], neg='No')],
                          sub_df['Golden'], scores_dict, 's1-')

        # S2 Results ----------------------------------------------------------
        print("S2 label")
        if compute_kappa:
            scores_dict['s2-kappa'] = (
                cohen_kappa_score(
                    sub_df['label_1'],
                    mapping(sub_df[f'{model_name}-s2'], neg=False)
                ) + cohen_kappa_score(
                    sub_df['label_2'],
                    mapping(sub_df[f'{model_name}-s2'], neg=False))
            ) / 2
            print('S2- Kappa score', scores_dict['s2-kappa'])

        metrics_for_label([mapping(sub_df[f'{model_name}-s2'], neg=False)],
                          sub_df['Golden'], scores_dict, 's2-')

        # S3 Results ----------------------------------------------------------
        print("S3 label")
        scores_dict['s3-kappa'] = (
            (cohen_kappa_score(
                sub_df['label_1'],
                mapping(sub_df[f'{model_name}-s3-1'], neg='Subjective')
            ) + cohen_kappa_score(
                sub_df['label_1'],
                mapping(sub_df[f'{model_name}-s3-2'], neg='Subjective'))
            ) / 2
        ) + (
            (cohen_kappa_score(
                sub_df['label_2'],
                mapping(sub_df[f'{model_name}-s3-1'], neg='Subjective')
            ) + cohen_kappa_score(
                sub_df['label_2'],
                mapping(sub_df[f'{model_name}-s3-2'], neg='Subjective'))
            ) / 2
        ) / 2
        print('S3- Kappa score', scores_dict['s3-kappa'])

        s3_labels = [
            mapping(sub_df[f'{model_name}-s3-1'], neg='Subjective'),
            mapping(sub_df[f'{model_name}-s3-2'], neg='Subjective')]
        metrics_for_label(s3_labels, sub_df['Golden'], scores_dict, 's3-')

        # Aggregated Results --------------------------------------------------
        print("Aggregated label")
        if compute_kappa:
            scores_dict['agg-human-kappa'] = (
                cohen_kappa_score(
                    sub_df['label_1'], sub_df[f'{model_name}_label']
                ) + cohen_kappa_score(
                    sub_df['label_2'], sub_df[f'{model_name}_label'])) / 2
            print(f'{model_name}-human kappa score',
                  scores_dict['agg-human-kappa'])

        metrics_for_label([sub_df[f'{model_name}_label']],
                          sub_df['Golden'], scores_dict)

        scores_list.append(scores_dict)

    if compute_kappa:
        print('Human kappa', cohen_kappa_score(
            sub_df['label_1'], sub_df['label_2']))
        print('Human accuracy: ', (
            accuracy_score(
                sub_df['Golden'], sub_df['label_1']
            ) + accuracy_score(
                sub_df['Golden'], sub_df['label_2'])) / 2)

    if save_to:
        out_df = pd.DataFrame(scores_list)
        if not verbose:
            out_df = out_df.get(
                ['model', 'macro_f1', 'macro_precision', 'macro_recall',
                 'f1_claim'])
        out_df.to_csv(save_to, index=False)
        print(f'Saved to {save_to}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='Results file to compute scores for')
    parser.add_argument('--save-to', '-s',
                        help='Location to save results to (optional)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Save verbose results (versus just agg macro'
                        'scores)')
    args = parser.parse_args()
    df = load_file(args.filename)
    verbose = args.verbose
    main(df, args.save_to, verbose)
