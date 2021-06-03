import os
import sys

import numpy as np
import pandas as pd

# import statistics


def read_datum(args):
    """Read and process the datum based on input arguments

    Args:
        args (namespace): commandline arguments

    Raises:
        Exception: args.word_value should be one of ['top', 'bottom', 'all']

    Returns:
        DataFrame: processed datum
    """
    # Load datum file
    df = pd.read_csv(os.path.join(args.DATUM_DIR, args.datum_emb_fn), header=0)

    if args.replication:
        fold_pickle = 'fold' + str(args.fold_idx) + '.pickle'
        df = pd.read_pickle(
            os.path.join(
                os.getcwd(), 'data', 'podcast',
                '6059e10f2219d67149d59979b0bd636523af3e47ce1104d1716766ed1655fa72',
                fold_pickle))
        df = pd.DataFrame(df)
        df = df[df.dataset.isin(['train', 'dev'])]

    # Filter/Align across language models
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

    # Filter words on minimum frequency
    if args.min_word_freq:
        df = df[df.uncased_freq >= args.min_word_freq]

    # Drop stopwords
    df = df[~df['word'].isin(['sp', '{lg}', '{ns}', '{inaudible}'])]

    if args.replication:
        df['embeddings'] = df.embedding
        df = df.dropna(subset=['embeddings'])
    else:
        # Find, combine and drop embedding columns
        df_cols = df.columns.to_list()
        embedding_columns = df_cols[df_cols.index('0'):]

        df = df.dropna(subset=embedding_columns)

        df['embeddings'] = df[embedding_columns].values.tolist()
        df = df.drop(columns=embedding_columns)

    col_names = [
        'word', 'onset', 'offset', 'accuracy', 'speaker', 'embeddings'
    ]

    if args.word_value == 'all':
        return df[col_names]

    df = df.dropna(subset=['gpt2_xl_target_prob', 'human_target_prob'])
    denom = 3
    if args.pilot == 'GPT2':
        pred = df.gpt2_xl_target_prob
    elif args.pilot == 'mturk':
        pred = df.human_target_prob
    m = sorted(pred)
    # med = statistics.median(m)

    if args.word_value == 'bottom':
        datum = df[pred <= m[np.ceil(len(m) / denom)], col_names]
    elif args.word_value == 'top':
        datum = df[pred >= m[len(m) - np.ceil(len(m) / denom)], col_names]

    return datum
