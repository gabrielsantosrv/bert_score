from bert_score import score
import numpy as np

class BERTScore:
    def __init__(self, lang='pt', metric='recall'):
        self.lang = lang
        self.metric = metric

    def compute_score(self, reference_sents, generated_sents):
        """
        Main function to compute CIDEr score
        :param  res (list) : list of dictionaries with image ic and tokenized hypothesis / candidate sentence
                gts (dict)  : dictionary with key <image id> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """
        print("======== generated sentences ========\n", generated_sents)
        print("======== reference sentences ========\n", reference_sents)
        output = score(generated_sents, reference_sents, lang=self.lang, verbose=True,
                       rescale_with_baseline=True, idf=True)
        precision, recall, f1_scores = output

        if self.metric == 'recall':
            scores = recall
        elif self.metric == 'precision':
            scores = precision
        else:
            scores = f1_scores

        scores = np.array(scores)
        return scores.mean(), scores
