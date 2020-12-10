#!/usr/bin/env python
# coding: utf-8

import os
import socket

import pandas as pd
from sklearn.decomposition import PCA

if __name__ == '__main__':

    # input_file_name = 'podcast-aligned-datum-bert-large-uncased-whole-word-masking.csv'
    # input_file_name = 'podcast-aligned-datum-gpt2-xl-c_1024-previous.csv'
    input_file_name = 'podcast-aligned-datum-glove-50d.csv'

    pca_flag = 0
    concat_flag = 0
    avg_flag = 1
    shuffle_flag = 1

    host_name = socket.gethostname()

    if 'tiger' in host_name:
        PROJ_DIR = '/projects/HASSON/247'
        DATUMS_FOLDER = os.path.join(
            PROJ_DIR, 'data/podcast/NY777_111_Part1_conversation1/misc')
    elif 'scotty' in host_name:
        PROJ_DIR = '/mnt/bucket/labs/hasson/ariel/247'
        DATUMS_FOLDER = os.path.join(PROJ_DIR, 'models/podcast-datums')
    else:
        DATUMS_FOLDER = os.getcwd()

    if pca_flag:
        pca_dim = 50

    file_name, file_ext = os.path.splitext(
        os.path.basename(os.path.join(DATUMS_FOLDER, input_file_name)))

    pca_output_file_name = file_name + '-pca-50d-hg' + file_ext
    avg_output_file_name = file_name + '-InClassAvg-hg' + file_ext
    shuffle_output_file_name = file_name + '-InClassShuffle-hg' + file_ext

    # Read the original embeddings as dataframes
    df_current = pd.read_csv(input_file_name)
    print(df_current.shape, df_current['word'].nunique())
    print(df_current.shape, df_current['lemmatized_word'].nunique())

    # Convert lemmatized_words to lowercase
    df_current.loc[:,
                   'lemmatized_word'] = df_current.loc[:,
                                                       'lemmatized_word'].str.strip(
                                                       )
    df_current.loc[:,
                   'lemmatized_word'] = df_current.loc[:,
                                                       'lemmatized_word'].str.lower(
                                                       )
    print(df_current.shape, df_current['lemmatized_word'].nunique())

    # specify the embedding columns
    # embedding_columns = ['x' + str(item) for item in raange(500)]  # + ['z' + str(item) for item in range(1, 1551)]
    df_cols = df_current.columns.tolist()
    embedding_columns = df_cols[df_cols.index('0'):]

    # Reading only the embedding columns from the dataframe
    df_c_orig = df_current[embedding_columns]

    # Keep aside the first part of the data_frame for later concatenation
    df_c_orig_meta = df_current.drop(columns=df_c_orig.columns)

    # Uncomment this block of code to do +/- concatenation
    if concat_flag:
        concat_width = 5
        prev_df = []
        for prev_pos in range(5, 0, -1):
            prev_df.append(df_c_orig.shift(prev_pos))

        future_df = []
        for future_pos in range(-1, -6, -1):
            future_df.append(df_c_orig.shift(future_pos))

        prev_df = pd.concat(prev_df, axis=1)
        future_df = pd.concat(future_df, axis=1)

        df_c_orig = pd.concat([prev_df, df_c_orig, future_df], axis=1)

    # Dropping rows/embeddings with all nans
    print("Shape before dropping nans")
    print(df_c_orig.shape)

    df_c_orig = df_c_orig.dropna(how='any')

    print("\nShape after dropping nans")
    print(df_c_orig.shape)

    print("\nShape after dropping duplicates")
    df_c_orig.drop_duplicates()
    print(df_c_orig.shape)

    if 'glove' not in input_file_name:
        if pca_flag:
            # Initialize PCA object
            pca = PCA(n_components=pca_dim, svd_solver='auto')

            # Transform the data
            output1 = pca.fit_transform(df_c_orig)

            # saving the outputs back into a csv files
            out_columns = [str(item) for item in range(50)]
            output1_df = pd.DataFrame(output1,
                                      index=df_c_orig.index,
                                      columns=out_columns)

            sklearn_df = df_c_orig_meta.join(output1_df)
            sklearn_df.to_csv(pca_output_file_name, index=False)
        else:
            # remove all columns except 'lemmatized_word' and 'embeddings'
            sklearn_df = df_current.drop(
                columns=df_c_orig_meta.columns.difference(['lemmatized_word']))

        if avg_flag:
            # Code to take average of words
            # calculate in class average
            df_avg = sklearn_df.groupby(['lemmatized_word'
                                         ]).mean().reset_index()

            # create dataframe to save
            df_avg = df_c_orig_meta.merge(df_avg,
                                          on='lemmatized_word',
                                          how='left')
            df_avg.to_csv(avg_output_file_name, index=False)

        if shuffle_flag:
            # Code to shuffle in class
            # perform in class shuffle
            df_shuffle = sklearn_df
            for word in df_shuffle['lemmatized_word'].unique():
                m = df_shuffle['lemmatized_word'] == word
                word_df = df_shuffle[df_shuffle['lemmatized_word'] == word]
                df_shuffle[m] = word_df.sample(frac=1).to_numpy()

            # create dataframe to save
            df_shuffle = df_c_orig_meta.merge(df_shuffle,
                                              on='lemmatized_word',
                                              left_index=True,
                                              right_index=True)
            df_shuffle.to_csv(shuffle_output_file_name, index=False)
