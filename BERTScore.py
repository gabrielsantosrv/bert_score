from bert_score.bert_score import score
import numpy as np

class BERTScore:
    def __init__(self, lang='pt', metric='recall'):
        self.lang = lang
        self.metric = metric

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  res (list) : list of dictionaries with image ic and tokenized hypothesis / candidate sentence
                gts (dict)  : dictionary with key <image id> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """


        cands = []
        refs = []
        for res_id in res:
            # tokenized hypothesis / candidate sentence
            hypo = res_id['caption']

            # tokenized reference sentence
            ref = gts[res_id['image_id']]

            # Sanity check.
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) > 0)

            cands.append((' '.join(hypo)).strip())
            refs.append((' '.join(ref)).strip())

        output = score(cands, refs, lang=self.lang, verbose=True, rescale_with_baseline=True, idf=True)
        precision, recall, f1_scores = output

        if self.metric == 'recall':
            scores = recall
        elif self.metric == 'precision':
            scores = precision
        else:
            scores = f1_scores

        scores = np.array(scores)
        return scores.mean(), scores
