import argparse

import pandas as pd
from gensim.models import KeyedVectors

from frameAxis import FrameAxis


def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate Moral Foundation Scores Using FrameAxis Approach.')

    parser.add_argument('--input_file',
                        help='Path to the dataset .csv file containing input text documents in a column.')

    parser.add_argument('--dict_type', type=str, default='mfd',
                        help='Dictionary for calculating FrameAxis Scores. Possible values are: emfd, mfd, mfd2')

    parser.add_argument('--word_embedding_model',
                        help='Path to the word embedding model used to map words to a vector space.')

    parser.add_argument('--output_file',
                        help='The path for saving the MF scored output CSV file.')

    parser.add_argument('--docs_colname',
                        help='The column containing the text docs to score with moral foundations.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print("Running FrameAxis Moral Foundations scores")
    args = parse_arguments()

    IN_PATH = args.input_file
    DICT_TYPE = args.dict_type
    if DICT_TYPE not in ["emfd", "mfd", "mfd2", "customized"]:
        raise ValueError(
            f'Invalid dictionary type received: {DICT_TYPE}, dict_type must be one of \"emfd\", \"mfd\", \"mfd2\", \"customized\"')
    OUT_CSV_PATH = args.output_file
    DOCS_COL = args.docs_colname
    if args.word_embedding_model is not None:
        W2V_PATH = args.word_embedding_model
        model = KeyedVectors.load_word2vec_format(W2V_PATH, binary=False)
    else:
        print('Downloading word embedding model: word2vec-google-news-300')
        import gensim.downloader

        model = gensim.downloader.load('word2vec-google-news-300')

    data = pd.read_csv(IN_PATH, lineterminator='\n')

    fa = FrameAxis(mfd=DICT_TYPE, w2v_model=model)
    mf_scores = fa.get_fa_scores(df=data, doc_colname=DOCS_COL, tfidf=False, format="virtue_vice",
                                 save_path=OUT_CSV_PATH)
