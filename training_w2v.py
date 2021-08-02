def make_bigrams(sentences: list):
    '''
    :param sentences: list of list of tokens
    :return: the sentences in which collocoations (bigrams) are replaced with a unified concatenated by underscore e.g. 'united', 'states' -> 'united_states'
    '''
    from gensim.models.phrases import Phrases
    phrases = Phrases(sentences, min_count=3,
                      threshold=100)  # , connector_words=gensim.models.phrases.ENGLISH_CONNECTOR_WORDS) # higher threshld fewer phrases
    results = []
    for doc in sentences:
        results.append(phrases[doc])
    return results


def make_trigrams(bigram_sentences: list):
    from gensim.models.phrases import Phrases
    trigram_model = Phrases(bigram_sentences, threshold=40)
    results = []
    for doc in bigram_sentences:
        results.append(trigram_model[doc])
    return results

def w2v_update(data: list, save_path_new=None, pretrained_w2v=None):
    '''
    :param data: list of strings (documents)
    :param save_path_new: path to save new word2vec model
    :return: new_model, old_model
    '''
    import pandas as pd
    # preprocess
    preprocessed_texts = preprocess_text(pd.Series(data))
    # form bigrams
    tokens = [doc.split(" ") for doc in preprocessed_texts]
    preprocessed_texts = make_bigrams(tokens)
    # form trigrams
    preprocessed_texts = make_trigrams(preprocessed_texts)
    # training word2vec model
    new_model, old_model = w2v_old_gensim(sentences_tokenized=preprocessed_texts, pretrained_path=google_news_w2v,
                                          save_path=save_path_new)
    return new_model, old_model

def w2v_update_gensim(sentences_tokenized, pretrained_path=None, save_path=None):
    from gensim.models import KeyedVectors, Word2Vec
    '''
    Train a w2v model on sentences_tokenized, if pretrained_path (path to a pretrained word2vec model) \\
    is provided, then update the model based on that and the given documents
    :param sentences_tokenized: List of lists of tokens
    :param pretrained_path: Path to pretrained model file
    :param save_path: The path to save the new model 
    :return: the new model (updated model), the old model (baseline)
    '''
    if pretrained_path:
        new_model = Word2Vec(size=300, min_count=1)
        new_model.build_vocab(sentences_tokenized)
        print('count of vocab before update: ', len(new_model.wv.vocab))
        total_examples = new_model.corpus_count
        model = KeyedVectors.load_word2vec_format(pretrained_path, binary=True)
        print('count of vocab for google news: ', len(model.wv.vocab))
        new_model.build_vocab([list(model.vocab.keys())], update=True)
        print('count of vocab after update: ', len(new_model.wv.vocab))
        # todo play with lockf
        print('intersecting')
        new_model.intersect_word2vec_format(pretrained_path, binary=True, lockf=1.0)
        print('training')
        new_model.train(sentences_tokenized, total_examples=total_examples, epochs=new_model.epochs)
        print('count of vocab at the end: ', len(new_model.wv.vocab))
    else:
        new_model = Word2Vec(sentences_tokenized, size=300, min_count=15)
        model = None

    if save_path:
        new_model.wv.save_word2vec_format(save_path, binary=False)
        print('New w2v model saved to {}'.format(save_path))
    return new_model, model


if __name__ == '__main__':
    google_news_w2v = "./GoogleNews-vectors-negative300.bin"
    w2v_update(save_path_new="updated_word_embeddings.txt", pretrained_w2v=google_news_w2v)
