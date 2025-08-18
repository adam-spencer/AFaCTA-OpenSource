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


def main(df, save_to):
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

        # TODO can this be condensed?
        sub_df = df
        # S1 Results ----------------------------------------------------------
        print("S1 label")
        if compute_kappa:
            print('Kappa score', (
                cohen_kappa_score(
                    sub_df['label_1'],
                    mapping(sub_df[f'{model_name}-s1'], neg='No')
                ) + cohen_kappa_score(
                    sub_df['label_2'],
                    mapping(sub_df[f'{model_name}-s1'], neg='No'))
            ) / 2)
        # scores_dict['s1_acc'] = accuracy_score(
        #     sub_df['Golden'], mapping(sub_df[f'{model_name}-s1'].fillna('No'), neg='No'))
        # print('Accuracy score: ', scores_dict['s1_acc'])

        metrics_for_label([mapping(sub_df[f'{model_name}-s1'], neg='No')],
                          sub_df['Golden'], scores_dict, 's1-')

        # S2 Results ----------------------------------------------------------
        print("S2 label")
        if compute_kappa:
            print('Kappa score', (
                cohen_kappa_score(
                    sub_df['label_1'],
                    mapping(sub_df[f'{model_name}-s2'], neg=False)
                ) + cohen_kappa_score(
                    sub_df['label_2'],
                    mapping(sub_df[f'{model_name}-s2'], neg=False))
            ) / 2)
        # scores_dict['s2_acc'] = accuracy_score(
        #     sub_df['Golden'], mapping(sub_df[f'{model_name}-s2'].fillna(False), neg=False))
        # print('Accuracy score: ', scores_dict['s2_acc'])

        metrics_for_label([mapping(sub_df[f'{model_name}-s2'], neg=False)],
                          sub_df['Golden'], scores_dict, 's2-')

        # S3 Results ----------------------------------------------------------
        print("S3 label")
        if compute_kappa:
            print('kappa score', (
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
            ) / 2)

        # scores_dict['s3_acc'] = (accuracy_score(
        #     sub_df['Golden'],
        #     mapping(
        #         sub_df[f'{model_name}-s3-1'].fillna('Subjective'), neg='Subjective')
        # ) + accuracy_score(
        #     sub_df['Golden'],
        #     mapping(sub_df[f'{model_name}-s3-2'].fillna('Subjective'), neg='Subjective'))) / 2

        # print('acc score', scores_dict['s3_acc'])
        s3_labels = [
            mapping(sub_df[f'{model_name}-s3-1'], neg='Subjective'),
            mapping(sub_df[f'{model_name}-s3-2'], neg='Subjective')]
        metrics_for_label(s3_labels, sub_df['Golden'], scores_dict, 's3-')

        # Aggregated Results --------------------------------------------------
        print("Aggregated label")
        if compute_kappa:
            print(f'{model_name}-human kappa score', (
                cohen_kappa_score(
                    sub_df['label_1'], sub_df[f'{model_name}_label']
                ) + cohen_kappa_score(
                    sub_df['label_2'], sub_df[f'{model_name}_label'])) / 2)

        # scores_dict['acc'] = accuracy_score(
        #     sub_df['Golden'], sub_df[f'{model_name}_label'])
        # print('Accuracy score: ', scores_dict['acc'])
        # scores_dict['recall_neg'] = recall_score(
        #     sub_df['Golden'], sub_df[f'{model_name}_label'], pos_label=0)
        # print('Recall Score (negative): ', scores_dict['recall_neg'])
        # scores_dict['precision_neg'] = precision_score(
        #     sub_df['Golden'], sub_df[f'{model_name}_label'], pos_label=0)
        # print('Precision Score (negative): ', scores_dict['precision_neg'])
        # scores_dict['recall_pos'] = recall_score(
        #     sub_df['Golden'], sub_df[f'{model_name}_label'])
        # print('Recall Score (positive): ', scores_dict['recall_pos'])
        # scores_dict['precision_pos'] = precision_score(
        #     sub_df['Golden'], sub_df[f'{model_name}_label'])
        # print('Precision Score (positive): ', scores_dict['precision_pos'])
        # scores_dict['f1_pos'] = f1_score(
        #     sub_df['Golden'], sub_df[f'{model_name}_label'])
        # print('F1 Score (positive): ', scores_dict['f1_pos'])
        # scores_dict['f1_neg'] = f1_score(
        #     sub_df['Golden'], sub_df[f'{model_name}_label'], pos_label=0)
        # print('F1 Score (negative): ', scores_dict['f1_neg'])
        # scores_dict['macro_f1'] = f1_score(
        #     sub_df['Golden'], sub_df[f'{model_name}_label'], average='macro')
        # print('Macro F1 Score: ', scores_dict['macro_f1'])

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
        out_df.to_csv(save_to, index=False)
        print(f'Saved to {save_to}')


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
        scores_dict[f'{prefix}f1_pos'] = (
            (f1_score(gold, label) / len(labels)
             ) + scores_dict.get(f'{prefix}f1_pos', 0))
        scores_dict[f'{prefix}f1_neg'] = (
            (f1_score(gold, label, pos_label=0) / len(labels)
             ) + scores_dict.get(f'{prefix}f1_neg', 0))
        scores_dict[f'{prefix}macro_f1'] = (
            (f1_score(gold, label, average='macro') / len(labels)
             ) + scores_dict.get(f'{prefix}macro_f1', 0))

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
    print(f'{prefix.upper().lstrip('-')} F1 Score (positive): ',
          scores_dict[f'{prefix}f1_pos'])
    print(f'{prefix.upper().lstrip('-')} F1 Score (negative): ',
          scores_dict[f'{prefix}f1_neg'])
    print(f'{prefix.upper().lstrip('-')} Macro F1 Score: ',
          scores_dict[f'{prefix}macro_f1'])

    # for model_name in model_names:
    #     golden = df.loc[(df[model_name] == 0) | (
    #         df[model_name] == 3), 'Golden'].to_list()
    #     label_1 = df.loc[(df[model_name] == 0) | (
    #         df[model_name] == 3), 'label_1'].to_list()
    #     label_2 = df.loc[(df[model_name] == 0) | (
    #         df[model_name] == 3), 'label_2'].to_list()
    #     label = df.loc[(df[model_name] == 0) | (df[model_name] == 3),
    #                    model_name].apply(lambda x: 0 if x == 0 else 1)
    #     human_kappa = cohen_kappa_score(label_1, label_2)
    #     ai_acc = accuracy_score(golden, label)
    #     ai_kappa = (cohen_kappa_score(label, label_2) + (
    #                 cohen_kappa_score(label, label_1))) / 2
    #     human_acc = (accuracy_score(golden, label_2) + (
    #                  accuracy_score(golden, label_1))) / 2
    #     print('kappa of inconsistent samples', ai_kappa, human_kappa)
    #     print('acc of inconsistent samples', ai_acc, human_acc)
    #     golden = df.loc[(df[model_name] > 0) & (
    #         df[model_name] < 3), 'Golden'].to_list()
    #     label_1 = df.loc[(df[model_name] > 0) & (
    #         df[model_name] < 3), 'label_1'].to_list()
    #     label_2 = df.loc[(df[model_name] > 0) & (
    #         df[model_name] < 3), 'label_2'].to_list()
    #     label = df.loc[(df[model_name] > 0) & (df[model_name] < 3),
    #                    model_name].apply(lambda x: 0 if x <= 1.5 else 1)
    #     human_kappa = cohen_kappa_score(label_1, label_2)
    #     ai_acc = accuracy_score(golden, label)
    #     ai_kappa = (cohen_kappa_score(label, label_2) + (
    #                 cohen_kappa_score(label, label_1))) / 2
    #     human_acc = (accuracy_score(golden, label_2) + (
    #                  accuracy_score(golden, label_1))) / 2
    #     print('kappa of perfect consistent samples', ai_kappa, human_kappa)
    #     print('acc of perfect consistent samples', ai_acc, human_acc)
    #     print('\n')


# def confusion(num_answer, model, data):
    # column_names = [
    #         'veri_Answer_' + str(i) for i in range(1, num_answer + 1)]
#     if data == 0:
#         df = pd.read_excel(
#             f'data/CoT_self-consistency/policlaim_test_{model}_CoT.xlsx'
#         )[column_names + ['Golden']]
#     elif data == 1:
#         df = pd.read_excel(
#             f'data/CoT_self-consistency/clef2021_test_{model}_CoT.xlsx'
#         )[column_names + ['Golden']]
#     np.random.seed(42)
#     rand_labels = np.random.randint(2, size=len(df))
#     df['random'] = rand_labels
#
#     def aggregate_func(row):
#         answer_list = []
#         num = len(column_names)
#         for name in column_names:
#             if row[name].startswith('Yes'):
#                 answer_list.append(1)
#             else:
#                 answer_list.append(0)
#         answer_sum = sum(answer_list)
#         if answer_sum <= num // 2:
#             aggregated_answer = 0
#             confusion_level = answer_sum
#         else:
#             aggregated_answer = 1
#             confusion_level = num - answer_sum
#         return aggregated_answer, confusion_level
#
#     df[['aggregated', 'confusion']] = df.apply(
#         aggregate_func, axis=1, result_type='expand')
#
#     confusion_scores = []
#     random_scores = []
#     percentage = []
#     for i in range(num_answer // 2 + 1):
#         agg = df.loc[df['confusion'] == i, 'aggregated']
#         golden = df.loc[df['confusion'] == i, 'Golden']
#         rand = df.loc[df['confusion'] == i, 'random']
#         print('confusion level', i, accuracy_score(golden, agg),
#               'Random', accuracy_score(rand, agg),
#               'Percentage {:.2f}'.format(100 * len(agg) / len(df)))
#         confusion_scores.append(accuracy_score(golden, agg))
#         random_scores.append(accuracy_score(rand, agg))
#         percentage.append(round(100 * len(agg) / len(df), 2))
#     # print('majority-voted accuracy: ', confusion_scores)
#     # print('random scores: ', random_scores)
#     # print('percentage of each consistency level: ', percentage)
#

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='Results file to compute scores for')
    parser.add_argument('--save-to', '-s',
                        help='Location to save results to (optional)')
    # parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--num_answer", type=int, default=0)
    # parser.add_argument("--model", type=str, default='G3')
    args = parser.parse_args()
    df = load_file(args.filename)
    # if args.num_answer == 0:
    # main(df)
    # else:
    #     # TODO remove or change confusion function
    #     confusion(args.num_answer, args.model, args.data)
    main(df, args.save_to)
