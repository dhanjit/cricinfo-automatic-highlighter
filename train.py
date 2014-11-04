import string
import json
import argparse
import re
import nltk
import pickle
from commentary import process_commentary, FeatureExtractor, Classifier

def main(args):
    learn_trainingset(args.trainingfile, args.custominfofile, args.topN_words, args.classifier_file)

def learn_trainingset(trainingfile, custominfofile, topN_words, classifier_file):
    with open(trainingfile,'r') as f:
        training_data = json.load(f)
    tagged_commentaries = []

    with open(custominfofile, 'r') as f:
        custom_info = json.load(f)

    for trainingset in training_data:
        for commentary in trainingset['commentary']:
            processed_commentary = process_commentary(commentary, custom_info, 'train')
            if not processed_commentary:
                continue
            tagged_commentaries.append(processed_commentary)

    wordset = getwordset( tagged_commentaries, topN_words )

    feature_extractor = FeatureExtractor(custom_info, wordset)

    training_set = nltk.classify.apply_features(feature_extractor.featureset, tagged_commentaries)

    classifier = nltk.NaiveBayesClassifier.train(training_set)

    # Dark magic: change classifier's __class__ to our own subclass
    classifier.__class__ = Classifier

    # Add our own feature extractor into the classifier,
    # to be retrieved when classifying test data.
    classifier.set_feature_extractor(feature_extractor)

    with open(classifier_file, 'wb') as f:
        pickle.dump(classifier, f)

def getwordset(tagged_commentaries, topN_words):
    allwords = set()
    allwords_fd = nltk.probability.FreqDist()
    allwords_tagged_fd = nltk.probability.ConditionalFreqDist()

    for tagged_commmentary in tagged_commentaries:
        for word in tagged_commmentary[0]:
            allwords.add(word)
            allwords_fd[word] += 1
            allwords_tagged_fd[tagged_commmentary[1]][word] += 1

    pos_word_count = allwords_tagged_fd[True].N()
    neg_word_count = allwords_tagged_fd[False].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}

    for word, freq in allwords_fd.items():
        pos_score = nltk.metrics.BigramAssocMeasures.chi_sq(allwords_tagged_fd[True][word], (freq, pos_word_count), total_word_count)
        neg_score = nltk.metrics.BigramAssocMeasures.chi_sq(allwords_tagged_fd[False][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    best_words = set( word for word, score in sorted(word_scores.items(), key=lambda item: item[1], reverse=True)[:topN_words] )

    return { 'allwords': allwords, 'best_words': best_words }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--train', help="Specify training data set file, format: JSON", action='store', dest='trainingfile', default='data/trainingset.json')
    parser.add_argument('-c', '--custom', help="Specify custom information file", action='store', dest='custominfofile', default='data/custom_information.json')
    parser.add_argument('-l', '--classifier', help="Specify classifier output file", action='store', dest='classifier_file', default='classifier.pickle')
    parser.add_argument('-b', '--topN_words', help="Set a limit to take the most N number of informative words", action='store', dest='topN_words', default=50,type=int)
    args = parser.parse_args()

    main(args)
