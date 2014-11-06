import re
import string
import nltk

def process_commentary(commentary, custom_info, method):
    if ": \\\"" in commentary['text'] or commentary['ball'] == 0:
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
    #print('\t'+str(filtered_words))
    if all( word[:2] == '$$' or word[-2:] == '$$' for word in filtered_words ):
        return False
    #print(filtered_words)

    if method == 'train':
        return (filtered_words, commentary['isHighlight'])
    else:
        return filtered_words

    # commentary['ball'] = BeautifulSoup(commentary['formattedball']).p.text.replace('\n',' ').strip()
    # soup = BeautifulSoup(commentary['formattedtext'])
    # commentary['rawtext'] = soup.p.text.replace('\n',' ').strip()
    # commentary['wordlist'] = [ word.lower() for word in commentary['rawtext'].split() if word not in stopwords('english') ]
    # if method == 'train':
    #     taggedset.append( (commentary['wordlist'],commentary['isHighlight']) )
    # elif method == 'test':
    #     taggedset.append( (commentary['wordlist'],commentary['isHighlight']) )
    # commentary['boldphrases'] = soup.p.find_all(True,recursive=False) # bold font/commsImportant. do something later
    # return commentary

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
        words = set(commentary)
        features = {}
        custom_event_words = { 'six': [ '$$six$$' ], 'four': [ '$$four$$' ], 'out': [ '$$out$$' ], 'drop':[ '$$drop$$', '$$dropped$$' ] }
        features[ '$exciting$'] = False
        features[ '$exciting-count$' ] = 0
        features[ '$important$' ] = False
        features[ '$longlength$' ] = ( len(commentary) / self.length_factor > self.length_norm )

        # print( commentary )
        for exciting_word in self.custom_info['exciting words']:
            if exciting_word in words:
                features[ '$exciting$' ] = True
                # print( 'found ' + exciting_word )
                features[ '$exciting-count$' ] += 1
                words.remove(exciting_word)

        features[ '$exciting-count$' ] = int( features[ '$exciting-count$' ] / self.exciting_count_factor )

        if '!' in words:
            features[ '$exclamation$' ] = True
            words.remove('!')
        else:
            features[ '$exclamation$' ] = False

        # print( features  )
        # print( '.................................' )
        features[ '$highlight-event$' ] = any( word[:2] == word[-2:] == '$$' for word in words )
        words = set( word for word in words if not (word[:2] == word[-2:] =='$$') )

        for word in self.wordset['best_words']:
            features['contains(%s)' % word ] = ( word in words )
        return features

# subclass nltk's Naive Bayes Classifier, so that we can
# pickle our FeatureExtractor in the same object.
class Classifier(nltk.NaiveBayesClassifier):

    def set_feature_extractor(self, feature_extractor):
        self.feature_extractor = feature_extractor
