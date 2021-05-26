from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from preprocess.preprocess import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer


class FrameAxis:

    def __init__(self, mfd=None, w2v_model=None):

        self.model = w2v_model
        self.vocab = self.model.vocab
        if mfd == "emfd":
            emfd = pd.read_csv(
                'moral_foundation_dictionaries/eMFD_wordlist.csv')
            self.axes, mfs = self._get_emfd_axes(emfd)
        elif mfd == "mfd":
            mfd = pd.read_csv(
                'moral_foundations_dictionaries/MFD_orignial.csv')
            self.axes, mfs = self._get_axes(mfd)
        # todo add "mfd2"
        self.cos_sim_dict = {'authority': {}, 'fairness': {}, 'general_morality': {}, 'harm': {}, 'ingroup': {},
                             'liberty': {}, 'purity': {}}

    def vocab_sim_axes(self):
        rows = []
        for word in self.vocab:
            row_dict = {'token': word}
            for mf in self.axes.keys():
                row_dict[mf] = self.cos_sim(self.model[word], self.axes[mf])
            rows.append(row_dict)

        df_sim = pd.DataFrame(rows)
        return df_sim

    def cos_sim(self, a, b):
        dot = np.dot(a, b)
        norma = np.linalg.norm(a)
        normb = np.linalg.norm(b)
        cos = dot / (norma * normb)
        return cos

    def _get_axes(self, all_mft_expand):
        # todo make it right for centroids
        axes = {}
        mfs = []
        all_vecs = []
        grp = all_mft_expand.groupby('category')
        for mf, mf_group in grp:
            print(mf)
            mfs.append(mf)
            diffs = []
            for index1, row1 in mf_group.iterrows():
                for index2, row2 in mf_group.iterrows():
                    if row1['polarization'] == 'virtues' and row2['polarization'] == 'vices':
                        try:
                            diffs.append(self.model[row1['word']] - self.model[row2['word']])
                        except KeyError:
                            pass

            diffs = np.array(diffs)
            # TODO scale the vectors
            mean = np.mean(diffs, axis=0)
            all_vecs.append(mean)
            axes[mf] = mean

        return axes, mfs

    def _get_emfd_axes(self, eMFD):
        print('Building Moral Foundation Axes from eMFD')
        mfs = set()
        mf_p = []
        for col in eMFD.columns:
            if col.endswith('_p'):
                mfs.add(col.split('_')[0])
                mf_p.append(col)

        axes = {}
        centroids = {}
        for index, row in eMFD.iterrows():
            mf = pd.to_numeric(row[mf_p]).idxmax()
            mf = mf.split('_')[0]
            try:
                vec = self.model[row['word']]
            except:
                continue
            mf_vice = mf + '.vice'
            mf_virtue = mf + '.virtue'
            if row[mf + '_sent'] > 0:
                if mf_virtue not in centroids:
                    centroids[mf_virtue] = [vec]
                else:
                    centroids[mf_virtue].append(vec)
            else:
                if mf_vice not in centroids:
                    centroids[mf_vice] = [vec]
                else:
                    centroids[mf_vice].append(vec)

        for mf in mfs:
            mf_vice = mf + '.vice'
            mf_virtue = mf + '.virtue'
            centroids[mf_virtue] = np.mean(np.array(centroids[mf_virtue]), axis=0)
            centroids[mf_vice] = np.mean(np.array(centroids[mf_vice]), axis=0)
            axes[mf] = centroids[mf_virtue] - centroids[mf_vice]

        return axes, mfs

    def framing_scores(self, doc_tokens, mf, B_T=None):
        bias_score = 0.0
        intensity_score = 0.0
        freq = Counter(doc_tokens)
        doc_tokens_set = list(set(doc_tokens))
        sum_freq = 0.0
        for token in doc_tokens_set:
            sum_freq += freq[token]
            bias_score += (freq[token] * self.cos_sim(self.model[token], self.axes[mf]))
            if B_T is not None:
                intensity_score += (freq[token] * (self.cos_sim(self.model[token], self.axes[mf]) - B_T) ** 2)

        bias_score /= sum_freq
        intensity_score /= sum_freq
        return bias_score, intensity_score

    def get_tfidf(self, doc_idx, token):
        if token in self.tfidf:
            return self.tfidf[token].iloc[doc_idx]
        else:
            return 0.0

    def get_avg_tfidf(self, token):
        if token in self.tfidf:
            return self.avg_tfidf[token]
        else:
            return 0.0

    def framing_scores_tfidf(self, doc_tokens, mf, B_T=None, doc_idx=None):
        bias_score = 0.0
        intensity_score = 0.0
        doc_tokens_set = list(set(doc_tokens))
        sum_tfidf = 0.0
        for token in doc_tokens_set:
            if B_T:
                tfidf_doc_token = self.get_tfidf(doc_idx, token)
            else:
                tfidf_doc_token = self.get_avg_tfidf(token)
            sum_tfidf += tfidf_doc_token
            if token not in self.cos_sim_dict[mf]:
                self.cos_sim_dict[mf][token] = self.cos_sim(self.model[token], self.axes[mf])
            bias_score += tfidf_doc_token * self.cos_sim_dict[mf][token]
            if B_T is not None:
                intensity_score += tfidf_doc_token * (self.cos_sim_dict[mf][token] - B_T) ** 2

        bias_score /= sum_tfidf
        intensity_score /= sum_tfidf
        return bias_score, intensity_score

    def framing_scores_set(self, doc_tokens, mf, B_T=None):
        # todo what was this for?
        doc_tokens = list(set(doc_tokens))
        bias_score = 0.0
        intensity_score = 0.0
        freq = Counter(doc_tokens)
        doc_tokens_set = list(set(doc_tokens))
        sum_freq = 0.0
        for token in doc_tokens_set:
            sum_freq += freq[token]
            bias_score += (freq[token] * self.cos_sim(self.model[token], self.axes[mf]))
            if B_T is not None:
                intensity_score += (freq[token] * (self.cos_sim(self.model[token], self.axes[mf]) - B_T) ** 2)

        bias_score /= sum_freq
        intensity_score /= sum_freq
        return bias_score, intensity_score

    def calc_tfidf(self, docs):
        vectorizer = TfidfVectorizer(max_df=.8, min_df=20, sublinear_tf=True)
        vectors = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names()
        # dense = vectors.todense()
        # denselist = dense.tolist()
        denselist = vectors.toarray()
        tfidf = pd.DataFrame(denselist, columns=feature_names)
        return tfidf

    def doc_scores(self, docs, baseline_docs, tfidf=False):
        if tfidf:
            self.tfidf = self.calc_tfidf(docs)
            print('tfidf', self.tfidf.shape)
            self.avg_tfidf = self.tfidf.mean(axis=0)
        biases = pd.DataFrame()
        intensities = pd.DataFrame()
        docs = docs.str.lower()

        if baseline_docs:
            all_baseline_docs = ' '.join(baseline_docs)
            all_docs_tokens = [x for x in all_baseline_docs.split() if x in self.vocab]

        for mf in self.axes.keys():
            print(mf)
            mf_scores_bias = []
            mf_scores_intensity = []
            if tfidf and baseline_docs:
                B_T, _ = self.framing_scores_tfidf(doc_tokens=all_docs_tokens, mf=mf)
            elif baseline_docs:
                B_T, _ = self.framing_scores(doc_tokens=all_docs_tokens, mf=mf)
            else:
                B_T = 0.0
            print('B_T = {}'.format(B_T))
            for idx in range(len(docs)):
                if idx % 100000 == 0:
                    print(f'Current doc_idx: {idx}/ Total: {len(docs)}')
                doc = docs[idx]
                doc_tokens = [x for x in doc.split() if x in self.vocab]
                if len(doc_tokens) == 0:
                    score_bias, score_intensity = (np.nan, np.nan)
                    # print('nan doc:', doc)
                else:
                    if tfidf:
                        score_bias, score_intensity = self.framing_scores_tfidf(doc_tokens, mf, B_T, idx)
                    else:
                        score_bias, score_intensity = self.framing_scores(doc_tokens, mf, B_T)
                mf_scores_bias.append(score_bias)
                mf_scores_intensity.append(score_intensity)
            biases['bias_{}'.format(mf)] = mf_scores_bias
            intensities['intensity_{}'.format(mf)] = mf_scores_intensity
        return biases, intensities

    def get_fa_scores(self, df, doc_colname, save_path=None, tfidf=False,
                      format="virtue_vice"):
        df = df.reset_index(drop=True)
        docs = df[doc_colname]
        print(f'Preprocessing column {doc_colname}')
        docs = preprocess(docs).reset_index(drop=True)
        baseline_docs = []  # todo docs.sample(frac=0.3, random_state=157).reset_index(drop=True)
        # todo build the w2v model
        print('Let\'s calculate bias and intensity')
        bias, intensity = self.doc_scores(docs=docs, baseline_docs=baseline_docs, tfidf=tfidf)
        print('total size: ', df.shape[0])
        print('any NaN in bias?', np.isnan(bias.values).sum())  # Nan means empty docs, we should remove them
        print('any NaN in intensity?', np.isnan(intensity.values).sum())

        fa_scores = pd.concat([df, bias, intensity], axis=1)

        fa_scores = fa_scores.dropna(subset=bias.columns.tolist() + intensity.columns.tolist()).reset_index(drop=True)
        print('NAN scores dropped, new size:', fa_scores.shape[0])

        if format == "virtue_vice":  # todo replace this for with conditional new column dataframe acts
            df_virtue_vice = []
            for index, row in fa_scores.iterrows():
                row_virtue_vice = {}
                for mf in self.axes.keys():
                    if row[f'bias_{mf}'] < 0:
                        row_virtue_vice[f'{mf}.vice'] = row[
                            f'intensity_{mf}']
                        row_virtue_vice[f'{mf}.virtue'] = 0
                    else:
                        row_virtue_vice[f'{mf}.virtue'] = row[f'intensity_{mf}']
                        row_virtue_vice[f'{mf}.vice'] = 0
                df_virtue_vice.append(row_virtue_vice)

            df_virtue_vice = pd.DataFrame(df_virtue_vice)
            fa_scores = pd.concat([fa_scores, df_virtue_vice], axis=1)
            print('After addding vice-virtue scores, the shape:', fa_scores.shape)

        if save_path:
            if len(save_path.split('/')) > 1:
                output_dir = '/'.join(save_path.split('/')[:-1])
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            fa_scores.to_csv(save_path, index=None, header=True)
            print('Moral Foundations FrameAxis scores saved to {}'.format(save_path))
        else:
            print('not saving the fa scores.')
        return fa_scores
