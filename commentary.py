import re
import string
import nltk

def process_commentary(commentary, custom_info, method):
    if 'text' not in commentary or 'ball' not in commentary or 'isHighlight' not in commentary:
        return False

    if any( word in commentary['text'] for word in custom_info['invalid commentary'] ) or commentary['ball'] == 0:
        return False

    text = commentary['text']
    for phrase in custom_info['combinable phrases']:
        pattern = re.compile(phrase, re.IGNORECASE)
        combinedphrase = phrase.replace(' ','')
        text = pattern.sub(combinedphrase,text)

    # print(text)
    punctuations = string.punctuation
    valid_punctuations = '$!'
    token_punctuations = '!'
    for punctuation in valid_punctuations:
        punctuations = punctuations.replace( punctuation,'')

    for punctuation in punctuations:
        text = text.replace(punctuation,' ')

    for punctuation in token_punctuations:
        text = text.replace(punctuation, ' ' + punctuation + ' ' )

    words = text.split()
#    print(words)
#    print( [ word for word in words if word not in string.punctuation ] )

    def strip_markers(word):
        if word[:2] == word[-2:] == '$$':
            return word[2:-2]
        else:
            return word
    # print(words)
    filtered_words = [ word.lower() for word in words if strip_markers(word).lower() not in nltk.corpus.stopwords.words('english') ]
    filtered_words = [ word for word in filtered_words if word not in custom_info['players'] ]
    #print('\t'+str(filtered_words))
    if all( word[:2] == '$$' or word[-2:] == '$$' for word in filtered_words ):
        return False
    #print(filtered_words)

    commentary[ 'processed_text' ] = filtered_words
    if method == 'train':
        return ( commentary , commentary['isHighlight'])
    else:
        return commentary

class FeatureExtractor:

    # Create a FeatureExtractor with some custom info
    def __init__(self, custom_info, wordset, feature_config ):
        self.custom_info = custom_info
        self.wordset = wordset

        self.length_factor = feature_config.get( 'length_factor', 10 )
        self.length_norm = feature_config.get( 'length_norm', 2 )
        self.exciting_count_factor = feature_config.get( 'exciting_count_factor', 2 )

    # Return the feature set of the commentary
    def featureset(self, commentary):
        words = commentary['processed_text']
        features = {}

        features[ '$exciting$'] = False

        features[ '$exciting-count$' ] = 0

        features[ '$important-event$' ] = False

        features[ '$longlength$' ] = ( len(commentary['processed_text']) / self.length_factor > self.length_norm )

        features[ '$syntactically-important$' ] = False

        for word in words:

            if word in self.custom_info['exciting words']:
                features[ '$exciting$' ] = True
                features[ '$exciting-count$' ] += 1

            if word in self.custom_info['important events']:
                features[ '$important-event$' ] = True

            if word in self.custom_info['syntactically important tokens']:
                features[ '$syntactically-important$' ] = True

        features[ '$exciting-count$' ] = round( features[ '$exciting-count$' ] / self.exciting_count_factor )

        def bigram_exists( bigram ):
            bigram = bigram.split()
            return

        for word in self.wordset['best_words']:
            if ' ' in word:
                bigram = word.split()
                features['contains(%s)' % word ] = any( bigram[0] is words[i] and bigram[1] is words[i+1] for i in range(0,len(words)-1) )
            else:
                features['contains(%s)' % word ] = ( word in words )

        return features

# subclass nltk's Naive Bayes Classifier, so that we can
# pickle our FeatureExtractor in the same object.
class Classifier(nltk.NaiveBayesClassifier):

    def set_feature_extractor(self, feature_extractor):
        self.feature_extractor = feature_extractor
