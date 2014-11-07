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

    with open(args.classifier, 'rb') as f:
        classifier = pickle.load(f)

    with open(args.custominfofile) as f:
        custom_info = json.load(f)

    result = []
    for testset in test_data:
        testset['commmentary'] = classify_list( classifier, testset['commentary'], custom_info, args.verbose )
        result.append(testset)

    if (args.outputfile == '-'):
        outfile = sys.stdout
    else:
        outfile = open(args.outputfile, 'w')
    json.dump(testset,outfile)
    outfile.close()

def classify_list(classifier, commentary_list, custom_info, verbose_mode):
    highlight_result = []
    firstball_found = False
    lastball_found = True
    lastball = commentary_list[-1]['ball']

    for commentary in commentary_list:
        processed_result = process_commentary(commentary, custom_info, 'test')

        if not processed_result:
            continue

        features = classifier.feature_extractor.featureset(commentary)
        prob_dist = classifier.prob_classify(features)

        commentary['isHighlight'] = prob_dist.max()
        commentary['score' ] = prob_dist.prob( True )

        if not firstball_found and commentary['ball'] == 0.1 :
            firstball_found = True
            commentary['isHighlight'] = True

        if not lastball_found and commentary['ball'] == lastball :
            lastball_found = True
            commentary['isHighlight'] = True

        if verbose_mode:
            # if features['$highlight-event$'] and not commentary['isHighlight']:
            #     commentary['false result'] = True
            commentary['features'] = features

        highlight_result.append( commentary )

    return highlight_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--classifier', help="Specify trained classifier", action='store',dest='classifier', default='classifier.pickle')
    parser.add_argument('-t', '--test', help="Specify test file. Use '-' to read from stdin", action='store',dest='testfile', default='data/testset.json')
    parser.add_argument('-o', '--output', help="Specify output file. Use '-' to print to stdout", action='store',default='output.json',dest='outputfile')
    parser.add_argument('-c', '--custom', help="Specify custom information file", action='store', dest='custominfofile', default='data/custom_information.json')
    parser.add_argument('-v', '--verbose', help="Specify if verbose output required, will add feature info to each commentary in json output", action='store_true', dest='verbose', default=False )

    args = parser.parse_args()
    main(args)
