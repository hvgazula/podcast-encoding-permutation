import os
import socket

import numpy as np
import pandas as pd
'''
This script generate random (static) embeddings of the desired dimension
Requires: A sample datum file from which the metadata is extracted.
Note: Embeddings are generated for each unique word (class) and then propagated
      to other occurrences (instances of the same class)
'''

if __name__ == '__main__':
    # Setting the seed
    np.random.seed(0)

    # Desired dimension for embedding space
    embeddings_dim = [50, 500, 1024, 1600]

    if 'tiger' in socket.gethostname():
        PROJ_DIR = '/projects/HASSON/247/'
        DATUMS_FOLDER = os.path.join(
            PROJ_DIR, 'data/podcast/NY777_111_Part1_conversation1/misc')
    elif 'scotty' in socket.gethostname():
        PROJ_DIR = '/mnt/bucket/labs/hasson/ariel/247/'
        DATUMS_FOLDER = os.path.join(PROJ_DIR, 'models/podcast-datums')
    else:
        print('Please Enter the datums folder:\n')
        DATUMS_FOLDER = input()

    input_file_name = 'podcast-aligned-datum-random-50d.csv'
    file_name, file_ext = os.path.splitext(
        os.path.basename(os.path.join(DATUMS_FOLDER, input_file_name)))

    # Read the original embeddings as dataframes
    df_current = pd.read_csv(input_file_name)
    print(df_current.shape)

    # Convert all words to lowercase
    df_current['word'] = df_current['word'].str.lower()

    # Keep aside the first part of the data_frame (metadata)
    df_cols = df_current.columns.tolist(
    )  # metadata columns + embedding columns (starting with '0')
    df_c_orig_meta = df_current.drop(columns=df_cols[df_cols.index('0'):])
    df_subset_meta = df_current.drop_duplicates(subset=['word'])

    for emb_dim in embeddings_dim:
        print(f'Generating static random embeddings of dimension: {emb_dim}')
        rand_df = pd.DataFrame(
            np.random.uniform(0, 1, size=(df_subset_meta.shape[0], emb_dim)))

        rand_df['word'] = df_subset_meta['word'].values

        # Propagate embeddings to other instances
        sklearn_df = df_c_orig_meta.merge(rand_df, on='word')

        # Save output to a datum file
        output_file_name = '-'.join(file_name.split('-')[:-1]) + '-' + str(
            emb_dim) + 'd-unif01-hg' + file_ext
        sklearn_df.to_csv(output_file_name, index=False)
