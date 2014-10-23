import json
import argparse
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

def main(args):
    raw_trainingset = get_rawtrainingset(args.trainingfile)
    ( commentaries, tag ) = get_taggedset(rawtrainingset)


def get_rawtrainingset(trainingfile):
    with open(args.trainingfile,'r') as f:
        rawtrainingset = json.load(f)

    taggedset = []

    def process_commentary(commentary):
        commentary['ball'] = BeautifulSoup(commentary['formattedball']).p.text.replace('\n',' ').strip()
        soup = BeautifulSoup(commentary['formattedtext'])
        commentary['rawtext'] = soup.p.text.replace('\n',' ').strip()
        commentary['wordlist'] = [ for word in commentary['rawtext'].split() if word not in stopwords('english') ]
        taggedset.append(commentary['wordlist'],commentary['isHighlight'])
        commentary['boldphrases'] = soup.p.find_all(True,recursive=False) # bold font/commsImportant. do something later
        return commentary

    rawtrainingset = [ [ process_commentary(commentary) for commentary in match ] for match in rawtrainingset ]


    return rawtrainingset


def get_taggedset(rawtrainingset):
     = [ [ (commentary['text'],commentary['isHighlight']) for commentary in match ] for match in rawtrainingset ]


def learn_trainingset():


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--train', help="Specify training data set file, format: JSON", action='store', destination='trainingfile',required=True)
    parser.add_argument('-o', '--output', help="Specify output file", action='store',default='output',destination='outputfile')
    parser.add_argument('-t', '--test', help="Specify test file", action='store',destination='testfile',required=True)
    args = parse.parse_args()

    main(args)
