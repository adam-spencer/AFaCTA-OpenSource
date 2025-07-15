import argparse
from collections import Counter
import json
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface.chat_models import ChatHuggingFace
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
import numpy as np
import pandas as pd
from pathlib import Path
import random
import re
import time
import torch

SYSTEM_PROMPT = """You are an AI assistant who helps fact-checkers to identify fact-like information in statements.
"""

PROMPT_PART_2_0905 = """Statements in political speech are usually based on facts to draw reasonable conclusions.

Categories of fact:
C1. Mentioning somebody (including the speaker) did or is doing something specific and objective.
C2. Quoting quantities, statistics, and data.
C3. Claiming a correlation or causation.
C4. Assertion of existing laws or rules of operation.
C5. Pledging a specific future plan or making specific predictions about future.

Please first analyze the objective and subjective information that the following <statement> (from a political speech) covers.
Then extract the fact that the <statement> is based on.
Then carefully reason about if the extracted fact is objectively verifiable.
Finally answer if the fact falls into the above categories (C1 to C5) or not (C0).

Context for <statement> to help you understand it better: "...{context}..."
<statement>: "{sentence}"

Format your answer in JSON with the following keys in order:
{{
    "ANALYSIS": "What are the objective and subjective information that <statement> covers?",
    "FACT_PART": "The extracted fact.",
    "VERIFIABLE_REASON": "Detailed reason about the extracted fact's verifiability. Note that a fact lacks important details or can be interpreted differently is not objectively verifiable. Future plans/pledge (C5) that are specific and clear can be verifiable. Citing others' words is verifiable and falls into C1. ",
    "VERIFIABILITY": "A boolean value indicates the verifiability.",
    "CATEGORY": "C1 to C5, or C0."
}}
"""

PROMPT_PART_1_VERIFIABILITY = """Given the <context> of the following <sentence> from a political speech, does it contain any objective information?

<context>: "...{context}..."
<sentence>: "{sentence}"

Answer with Yes or No only.
"""

PROMPT_OBJECTIVE = """Concisely argue that the following <sentence> from a political speech does contain some objective information.

Context of <sentence> in the speech: "...{context}..."
<sentence>: "{sentence}"
"""


PROMPT_SUBJECTIVE = """Concisely argue that the following <sentence> from a political speech does not contain any objective information.

Context of <sentence> in the speech: "...{context}..."
<sentence>: "{sentence}"
"""

JUDGE_PROMPT = """Two AI assistants are debating about whether the following <sentence> (from a political speech) contains any objectively verifiable information.

Context of <sentence> in the speech: "...{context}..."
<sentence>: "{sentence}"

Assistant A's View: "{assistant_a}"

Assistant B's View: "{assistant_b}"

Based on the above, does <sentence> contain any objectively verifiable information? Which perspective do you align with more closely?
Please reply with "Lean towards A", or "Lean towards B" only."""


def judge_vote(answer_lists):
    final_answer = []
    confusion = []
    for answer_list in answer_lists:
        lean2a_count = 0
        lean2b_count = 0
        for a in answer_list:
            if "lean towards a" in a.lower():
                lean2a_count += 1
            else:
                lean2b_count += 1
        if lean2a_count > lean2b_count:
            final_answer.append("Lean towards A")
            confusion.append(lean2b_count)
        else:
            final_answer.append("Lean towards B")
            confusion.append(lean2a_count)
    return final_answer, confusion


def majority_vote_p1_opinion(answer_lists):
    final_answer = []
    confusion = []
    total_ans_num = len(answer_lists[0])
    for answer_list in answer_lists:
        opinion_count = 0
        fact_count = 0
        mix_count = 0
        for a in answer_list:
            if "Opinion with fact" in a:
                mix_count += 1
            elif "Fact" in a:
                fact_count += 1
            else:
                opinion_count += 1
        candidates = ['Opinion with fact', 'Fact', 'Opinion']
        max_num = np.max([mix_count, fact_count, opinion_count])
        if max_num > total_ans_num // 3 + 1:
            final_answer.append(candidates[np.argmax(
                [mix_count, fact_count, opinion_count])])
        else:
            final_answer.append('Opinion with fact')
        confusion.append(total_ans_num - max_num)
    return final_answer, confusion


def majority_vote_p1_verifiability(answer_lists):
    final_answer = []
    confusion = []
    for answer_list in answer_lists:
        yes_count = 0
        no_count = 0
        for a in answer_list:
            if "yes" in a.lower():
                yes_count += 1
            else:
                no_count += 1
        if yes_count > no_count:
            final_answer.append("Yes")
            confusion.append(no_count)
        else:
            final_answer.append("No")
            confusion.append(yes_count)
    return final_answer, confusion


def majority_vote_p2(answer_lists):
    final_answer = []
    confusion = []
    for answer_list in answer_lists:
        true_count = 0
        false_count = 0
        for a in answer_list:
            if "true" in str(a).lower():
                true_count += 1
            else:
                false_count += 1
        if true_count > false_count:
            final_answer.append("TRUE")
            confusion.append(false_count)
        else:
            final_answer.append("FALSE")
            confusion.append(true_count)
    return final_answer, confusion


def _find_answer(string, name="FACT_PART"):
    for l in string.split('\n'):
        if name in l:
            start = l.find(":") + 3
            end = len(l) - 1
            return l[start:end]
    return string


def parse_part2(all_answers, keys):
    # TODO consider rewriting this using `json` module
    return_lists = {k: [] for k in keys}
    for answers in all_answers:
        lists = {k: [] for k in keys}
        for a in answers:
            try:
                # Attempt to find the start of the JSON object
                json_start_index = a.find('{')
                json_end_index = a.rfind('}') + 1
                if json_start_index != -1 and json_end_index != -1:
                    json_str = a[json_start_index:json_end_index]
                    result_dict = json.loads(json_str)
                else:
                    raise ValueError("No JSON object found")
            except Exception as e:
                result_dict = {
                    k: _find_answer(a, name=k) for k in keys
                }
            for k in keys:
                lists[k].append(result_dict.get(k, ""))  # Use .get for safety
        for k in keys:
            return_lists[k].append(lists[k])
    return return_lists


def batchify_list(input_list, batch_size):
    batches = []
    for i in range(0, len(input_list), batch_size):
        batches.append(input_list[i:i + batch_size])
    return batches


def contextualize_sentences(sentences, window_size=1):
    contexts = []
    # If window_size is 0, just return the sentences themselves as context
    if window_size == 0:
        return sentences

    for i in range(len(sentences)):
        start_index = max(0, i - window_size)
        end_index = min(len(sentences), i + window_size + 1)
        context_sentences = sentences[start_index:end_index]
        contexts.append(" ".join(context_sentences))
    return contexts


def sync_api_call(llm, messages, gen_num, batch_size=10):
    batches = batchify_list(messages, batch_size)
    all_outputs = []
    for b in batches:
        time.sleep(0.1)
        # outputs = llm.generate(b)
        outputs = llm.batch(b)
        output_texts = [[g.text for g in outputs.generations[i]]
                        for i in range(len(outputs.generations))]
        all_outputs.extend(output_texts)
    return all_outputs


def lean_to_answer(answer, first):
    if first == 'objective':
        if "lean towards a" in answer.lower():
            return "Objective"
        elif "lean towards b" in answer.lower():
            return "Subjective"
        else:
            return "Not defined: " + answer
    else:
        if "lean towards b" in answer.lower():
            return "Objective"
        elif "lean towards a" in answer.lower():
            return "Subjective"
        else:
            return "Not defined: " + answer


def debate(args, llm, sentences, contexts):
    if args.load_debate == "":
        objective_prompts = [
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=PROMPT_OBJECTIVE.format(
                    sentence=s, context=c))
            ]
            for s, c in zip(sentences, contexts)
        ]
        objective_outputs = sync_api_call(llm, objective_prompts, 1)
        objective_outputs = [o[0].strip() for o in objective_outputs]

        subjective_prompts = [
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=PROMPT_SUBJECTIVE.format(
                    sentence=s, context=c))
            ]
            for s, c in zip(sentences, contexts)
        ]
        subjective_outputs = sync_api_call(llm, subjective_prompts, 1)
        subjective_outputs = [o[0].strip() for o in subjective_outputs]
        df_debate = pd.DataFrame(
            {"SENTENCES": sentences,
             'subjectivity': subjective_outputs,
             'objectivity': objective_outputs})
        df_debate.to_csv(args.output_name + '_debate.csv',
                         index=False, encoding='utf-8')
    else:
        df_debate = pd.read_csv(
            args.load_debate + '_debate.csv', encoding='utf-8')
        subjective_outputs = [l.strip()
                              for l in df_debate['subjectivity'].to_list()]
        objective_outputs = [l.strip()
                             for l in df_debate['objectivity'].to_list()]

    time.sleep(args.sleep)
    df_debate_results = df_debate

    judge_prompt = JUDGE_PROMPT
    objective_first_prompts = [
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=judge_prompt.format(
                sentence=s, context=c, assistant_a=ob,
                assistant_b=sub))
        ]
        for s, c, ob, sub in zip(sentences, contexts, objective_outputs,
                                 subjective_outputs)
    ]

    objective_first_outputs = sync_api_call(
        llm, objective_first_prompts, args.num_gen)

    ob_aggregated_answer = [o[0] for o in objective_first_outputs]
    ob_verifiable_answer = [lean_to_answer(
        a, first='objective') for a in ob_aggregated_answer]
    df_debate_results['ob_aggregated'] = ob_verifiable_answer

    time.sleep(args.sleep)

    subjective_first_prompts = [
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=judge_prompt.format(
                sentence=s, context=c, assistant_a=sub,
                assistant_b=ob))
        ]
        for s, c, ob, sub in zip(sentences, contexts, objective_outputs,
                                 subjective_outputs)
    ]

    subjective_first_outputs = sync_api_call(
        llm, subjective_first_prompts, args.num_gen)

    sub_aggregated_answer = [o[0] for o in subjective_first_outputs]
    sub_verifiable_answer = [lean_to_answer(a, first='subjective') for a in
                             sub_aggregated_answer]
    df_debate_results['sub_aggregated'] = sub_verifiable_answer

    df_debate_results.to_csv(
        args.output_name + '_p3_' + str(args.num_gen) + '.csv', index=False,
        encoding='utf-8')
    return df_debate_results


def opinion(args, llm, prompt, sentences):
    fact_opinion_prompts = [
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt.format(sentence=s))
        ]
        for s in sentences
    ]

    opinion_outputs = sync_api_call(
        llm, fact_opinion_prompts, args.num_gen)

    opinion_answers = [o[0] for o in opinion_outputs]
    df_p1_opinion = pd.DataFrame(
        {'SENTENCES': sentences, 'op_aggregated': opinion_answers})

    df_p1_opinion.to_csv(
        args.output_name + '_opinion_p1_' + str(args.num_gen) + '.csv',
        encoding='utf-8', index=False)
    return df_p1_opinion


def verifiability(args, llm, prompt, sentences, contexts):
    verifiability_prompts = [
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt.format(
                sentence=s, context=c))
        ]
        for s, c in zip(sentences, contexts)
    ]

    verifiability_outputs = sync_api_call(
        llm, verifiability_prompts, args.num_gen)

    verifiability_answers = [o[0] for o in verifiability_outputs]
    df_p1_verifiability = pd.DataFrame(
        {'SENTENCES': sentences, 'veri_aggregated': verifiability_answers})

    df_p1_verifiability.to_csv(
        args.output_name + '_ver_p1_' + str(args.num_gen) + '.csv',
        encoding='utf-8', index=False)
    return df_p1_verifiability


def part_2(args, llm, prompt, p2_keys, verifiable_key, sentences,
           contexts):
    part2_prompts = [
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt.format(
                sentence=s, context=c))
        ]
        for s, c in zip(sentences, contexts)
    ]

    part2_outputs = sync_api_call(llm, part2_prompts, args.num_gen)
    answer_lists = parse_part2(part2_outputs, keys=p2_keys)
    if args.num_gen > 1:
        aggregated_answer, confusion = majority_vote_p2(
            answer_lists[verifiable_key])
    else:
        aggregated_answer = [
            l[0] if l else None for l in answer_lists[verifiable_key]]
        confusion = None
    df_p2 = pd.DataFrame({'SENTENCES': sentences})
    for i in range(args.num_gen):
        for k in p2_keys:
            df_p2[k + str(i + 1)] = [
                l[i] if len(l) > i else None for l in answer_lists[k]]
    df_p2['p2_aggregated'] = aggregated_answer
    df_p2['CATEGORY'] = Counter(dict(filter(
        lambda x: x[0].startswith('CATEGORY'), df_p2.items()))).most_common(1)
    if confusion is not None:
        df_p2['p2_confusion'] = confusion

    df_p2.to_csv(args.output_name + '_p2_' + str(args.num_gen) +
                 '.csv', encoding='utf-8', index=False)
    return df_p2


def main(args):
    P1_VERIFIABILITY = PROMPT_PART_1_VERIFIABILITY
    PART_2_PROMPT = PROMPT_PART_2_0905

    PART2_KEYS = ["ANALYSIS", "FACT_PART",
                  "VERIFIABLE_REASON", "VERIFIABILITY", "CATEGORY"]
    verifiable_key = "VERIFIABILITY"

    if args.seed > 0:
        random.seed(args.seed)

    if args.file_name.endswith('xlsx'):
        df = pd.read_excel(args.file_name)
    else:
        df = pd.read_csv(args.file_name, encoding='utf-8')

    sentences = df['SENTENCES'].to_list()
    sentences = [s.strip() for s in sentences]

    if args.sample > 0:
        sentences = random.sample(sentences, args.sample)

    print("INFO: Generating text contexts with a window size of "
          f"{args.context}...")
    contexts = contextualize_sentences(sentences, window_size=args.context)
    print("INFO: Context generation complete.")

    if args.num_gen > 1:
        temperature = 0.7
        do_sample = True
    else:
        temperature = 0.1  # ignored
        do_sample = False

    model = AutoModelForCausalLM.from_pretrained(
        args.llm_name, device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        temperature=temperature,
        max_new_tokens=512,
        do_sample=do_sample
    )
    hf_pipe = HuggingFacePipeline(pipeline=pipe)
    llm = ChatHuggingFace(llm=hf_pipe)

    if not args.skip_p1:
        # Part 1 verifiability
        df_p1_verifiability = verifiability(
            args, llm, P1_VERIFIABILITY, sentences, contexts)
        time.sleep(args.sleep)
    else:
        df_p1_verifiability = pd.read_csv(
            args.load_p1 + '_ver_p1_' + str(args.num_gen) + '.csv',
            encoding='utf-8')

    # Part 2 annotation
    if not args.skip_p2:
        df_p2 = part_2(args, llm, PART_2_PROMPT, PART2_KEYS,
                       verifiable_key, sentences, contexts)
    else:
        df_p2 = pd.read_csv(
            args.load_p2 + '_p2_' + str(args.num_gen) + '.csv',
            encoding='utf-8')

    # Part 3 debate annotation
    if not args.skip_p3:
        df_p3 = debate(args, llm, sentences, contexts)
    else:
        df_p3 = pd.read_csv(
            args.load_p3 + '_p3_' + str(args.num_gen) + '.csv',
            encoding='utf-8')

    df_merged = pd.merge(df_p1_verifiability, df_p2,
                         how='left', on='SENTENCES')
    df_merged = pd.merge(df_merged, df_p3, how='left', on='SENTENCES')
    df_merged.to_csv(
        args.output_name + '_' + str(args.num_gen) + '.csv',
        encoding='utf-8', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default="")
    parser.add_argument("--output_name", type=str, default="")
    parser.add_argument("--load_debate", type=str, default="")
    parser.add_argument("--load_p1", type=str, default="")
    parser.add_argument("--load_p2", type=str, default="")
    parser.add_argument("--load_p3", type=str, default="")
    parser.add_argument(
        "--llm_name", type=str, default="google/gemma-3-12b-it",
        help='Use HF Hub name or path to local installation')
    parser.add_argument("--context", type=int, default=1)
    parser.add_argument("--skip_p1", action="store_true", default=False)
    parser.add_argument("--skip_p2", action="store_true", default=False)
    parser.add_argument("--skip_p3", action="store_true", default=False)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_gen", type=int, default=1)
    parser.add_argument("--sleep", type=int, default=5)
    args = parser.parse_args()

    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        print("✅ Compatible GPU detected, setting matmul precision to 'high'.")
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("ℹ️ No compatible GPU found, using default matmul precision.")
    torch._dynamo.config.cache_size_limit = 64

    if args.output_name == '':
        if (p := Path(args.llm_name)).exists():
            llm_name_fmt = p.name
        else:
            llm_name_fmt = args.llm_name.split('/')[1]
        args.output_name = (f'{re.split(r'[_.]', args.file_name)[0]}'
                            f'_{llm_name_fmt}')

    main(args)
