import os
# import statistics

import numpy as np
import pandas as pd


def read_datum(args, DATUM_DIR):
    df = pd.read_csv(os.path.join(DATUM_DIR, args.datum_emb_fn), header=0)

    if args.nonWords:
        df = df[df.is_nonword == 0]
    if args.gpt2:
        df = df[df.in_gpt2 == 1]
    if args.bert:
        df = df[df.in_bert == 1]
    if args.bart:
        df = df[df.in_bart == 1]
    if args.glove:
        df = df[df.in_glove == 1]

    # df = df[df.in_roberta == 1]

    df_cols = df.columns.tolist()
    embedding_columns = df_cols[df_cols.index('0'):]
    df = df[~df['word'].isin(['sp', '{lg}', '{ns}', '{inaudible}'])]
    df = df.dropna()

    df['embeddings'] = df[embedding_columns].values.tolist()
    df = df.drop(columns=embedding_columns)

    if args.word_value == 'bottom':
        df = df.dropna(subset=['gpt2_xl_target_prob', 'human_target_prob'])
        denom = 3
        if args.pilot == 'GPT2':
            pred = df.gpt2_xl_target_prob
        elif args.pilot == 'mturk':
            pred = df.human_target_prob
        m = sorted(pred)
        # med = statistics.median(m)
        datum = df[
            pred <= m[np.ceil(len(m) / denom)],
            ['word', 'onset', 'offset', 'accuracy', 'speaker', 'embeddings']]
    elif args.word_value == 'all':
        datum = df[[
            'word', 'onset', 'offset', 'accuracy', 'speaker', 'embeddings'
        ]]
    else:
        df = df.dropna(subset=['gpt2_xl_target_prob', 'human_target_prob'])
        denom = 3
        if args.pilot == 'GPT2':
            pred = df.gpt2_xl_target_prob
        elif args.pilot == 'mturk':
            pred = df.human_target_prob
        m = sorted(pred)
        # med = statistics.median(m)
        datum = df[
            pred >= m[len(m) - np.ceil(len(m) / denom)],
            ['word', 'onset', 'offset', 'accuracy', 'speaker', 'embeddings']]

    return datum
