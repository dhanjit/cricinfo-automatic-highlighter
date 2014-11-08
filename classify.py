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

    test_data['commentary'] = classify_list( classifier, test_data['commentary'], custom_info, args.min_score, args.verbose )

    if args.sort_and_limit:
        total_commentary = len(test_data['commentary'])
        sorted_commentary = sorted( test_data['commentary'], key=lambda commentary: commentary['score'], reverse=True)

        test_data['commentary'] = []

        for highlight in sorted_commentary[:int(total_commentary/5)]:
            highlight['isHighlight'] = True
            test_data['commentary'].append( highlight )

        for highlight in sorted_commentary[int(total_commentary/5):]:
            highlight['isHighlight'] = False
            test_data['commentary'].append( highlight )

        test_data['commentary'] = sorted( test_data['commentary'], key=lambda commentary: commentary['index'] )

        firstball_found = lastball_found = False
        for i in range(0,len(test_data['commentary'])):
            if test_data['commentary'][i]['ball'] == 0.1 and not firstball_found:
                test_data['commentary'][i]['isHighlight'] == True
                firstball_found = True
            if test_data['commentary'][i]['ball'] == test_data['commentary'][-1]['ball'] and not lastball_found:
                test_data['commentary'][i]['isHighlight'] = True
                lastball_found = True

    test_data['average_score'] = 0
    for commentary in test_data['commentary']:
        test_data['average_score'] += commentary['score']
    test_data['average_score'] /= len(test_data['commentary'])

    if (args.outputfile == '-'):
        outfile = sys.stdout
    else:
        outfile = open(args.outputfile, 'w')

    json.dump(test_data,outfile)
    outfile.close()

def classify_list(classifier, commentary_list, custom_info, min_score, verbose_mode):
    highlight_result = []
    firstball_found = False
    lastball_found = True
    lastball = commentary_list[-1]['ball']

    index = 0
    for commentary in commentary_list:
        processed_result = process_commentary(commentary, custom_info, 'test')

        if not processed_result:
            continue

        features = classifier.feature_extractor.featureset(commentary)
        prob_dist = classifier.prob_classify(features)

        commentary['score'] = prob_dist.prob( True )
        commentary['isHighlight'] = commentary['score'] > min_score

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

        commentary['index'] = index
        index += 1
        highlight_result.append( commentary )

    return highlight_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--classifier', help="Specify trained classifier", action='store',dest='classifier', default='classifier.pickle')
    parser.add_argument('-t', '--test', help="Specify test file. Use '-' to read from stdin", action='store',dest='testfile', default='data/testset.json')
    parser.add_argument('-o', '--output', help="Specify output file. Use '-' to print to stdout", action='store',default='output.json',dest='outputfile')
    parser.add_argument('-c', '--custom', help="Specify custom information file", action='store', dest='custominfofile', default='data/custom_information.json')
    parser.add_argument('-v', '--verbose', help="Specify if verbose output required, will add feature info to each commentary in json output", action='store_true', dest='verbose', default=False )
    parser.add_argument('-s', '--sort-and-limit', help="Specify whether to sort and limit the classified highlights", action='store_true', dest='sort_and_limit', default=False)
    parser.add_argument('-m', '--min-score', help="Specify min score for setting a commentary as a highlight", action = 'store', dest='min_score',default=0.5, type=float)

    args = parser.parse_args()
    main(args)
