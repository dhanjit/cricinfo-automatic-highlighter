import string
import json
import argparse
import re
import nltk

def main(args):
    learn_trainingset(args.trainingfile, args.testfile, args.outputfile, args.custominfofile)

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
    for punctuation in valid_punctuations:
        punctuations = punctuations.replace( punctuation,'')

    for punctuation in punctuations:
        text = text.replace(punctuation,' ')

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
    if all( [ word[:2] == word[-2:] == '$$' for word in filtered_words ] ):
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

def learn_trainingset(trainingfile, testfile, outputfile, custominfofile):
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

    allwords = set()
    for tagged_commmentary in tagged_commentaries:
        [ allwords.add(word) for word in tagged_commmentary[0] ]

    #print(allwords)
    def feature_extractor( commentary ):
        words = set(commentary)
        features = {}
        custom_event_words = { 'six': [ '$$six$$' ], 'four': [ '$$four$$' ], 'out': [ '$$out$$' ], 'drop':[ '$$drop$$', '$$dropped$$' ] }
        features[ '$exciting$'] = False
        features[ '$important$' ] = False
        features[ '$longlength$' ] = ( len(commentary) / 10 > 2 )

        for exciting_word in custom_info['exciting words']:
            features['$exciting$'] = ( exciting_word in words )
            if exciting_word in words:
                words.remove(exciting_word)

        features[ '$highlight-event$' ] = any( word[:2] == word[-2:] == '$$' for word in words )
        words = set( word for word in words if not (word[:2] == word[-2:] =='$$') )

        for word in allwords:
            features['contains(%s)' % word ] = ( word in words )
        return features

    training_set = nltk.classify.apply_features(feature_extractor, tagged_commentaries)
    classifier = nltk.NaiveBayesClassifier.train(training_set)

    with open(testfile,'r') as f:
        test_data = json.load(f)

    result = []
    for testset in test_data:
        highlight_result = []
        #print(testset)
        informative_features = classifier.show_most_informative_features(n=1000)
        print( informative_features )

        for commentary in testset['commentary']:
            print(commentary)
            preprocessed_test = process_commentary(commentary, custom_info,'test')
            if not preprocessed_test:
                continue

            features = feature_extractor(preprocessed_test)
            classifier_result = classifier.classify(features)
            print( classifier_result )
            print( '.....................................' )
            commentary['isHighlight'] = classifier_result
            highlight_result.append( commentary )

        testset['commentary'] = highlight_result
        result.append(testset)

    with open( outputfile, 'w') as outfile:
        json.dump(testset,outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--train', help="Specify training data set file, format: JSON", action='store', dest='trainingfile',required=True)
    parser.add_argument('-o', '--output', help="Specify output file", action='store',default='output.json',dest='outputfile')
    parser.add_argument('-t', '--test', help="Specify test file", action='store',dest='testfile',required=True)
    parser.add_argument('-c', '--custom', help="Specify custom information file", action='store', dest='custominfofile', required=True)
    args = parser.parse_args()

    main(args)
