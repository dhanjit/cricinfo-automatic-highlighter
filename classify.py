import pickle
import argparse
import json
from commentary import process_commentary
import re
import sys

def main(args):

    if (args.testfile == '-'):
        f = sys.stdin
    else:
        f = open(args.testfile,'r')
    test_data = json.load(f)
    f.close()

    with open(args.classifier) as f:
        classifier = pickle.load(f)

    with open(args.custominfofile) as f:
        custom_info = json.load(f)

    result = []
    for testset in test_data:
        highlight_result = []
        #print(testset)
        if (args.outputfile != '-'):
            informative_features = classifier.show_most_informative_features(n=1000)
            print( informative_features )

        for commentary in testset['commentary']:
            preprocessed_test = process_commentary(commentary, custom_info, 'test')
            if not preprocessed_test:
                #print(commentary)
                #print(preprocessed_test)
                continue

            features = classifier.feature_extractor.featureset(preprocessed_test)
            classifier_result = classifier.classify(features)
            # print( commentary )
            # print( classifier_result )
            # print( '.....................................' )
            commentary['isHighlight'] = classifier_result
            highlight_result.append( commentary )

        testset['commentary'] = highlight_result
        result.append(testset)

    if (args.outputfile == '-'):
        outfile = sys.stdout
    else:
        outfile = open(args.outputfile, 'w')
    json.dump(testset,outfile)
    outfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--classifier', help="Specify trained classifier", action='store',dest='classifier', default='classifier.pickle')
    parser.add_argument('-t', '--test', help="Specify test file. Use '-' to read from stdin", action='store',dest='testfile', default='data/testset.json')
    parser.add_argument('-o', '--output', help="Specify output file. Use '-' to print to stdout", action='store',default='output.json',dest='outputfile')
    parser.add_argument('-c', '--custom', help="Specify custom information file", action='store', dest='custominfofile', default='data/custom_information.json')

    args = parser.parse_args()
    main(args)
