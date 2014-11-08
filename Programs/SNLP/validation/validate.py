import json
import argparse

def get_highlight_balls_from_json(filename):
	highlights = []
	with open(filename) as data_file:
		data = json.load(data_file)
	for item in data['commentary']:
		if(item['isHighlight'] == True):
			highlights.append(str(item['ball']))
	return highlights

def compute_metrics(l1, l2):
	s1 = set(l1)
	s2 = set(l2)
	a = len( s1.intersection(s2) )
	missed_highlights = s1 - s1.intersection(s2) 
	comp = float(a)/len( s2 )
	acc = float(a)/len( s1 )
	return comp,acc,list(missed_highlights)	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--classifier', help="Specify classifier output, format: JSON", action='store', dest='classifier_filename', default='output58-1.json')
	parser.add_argument('-v', '--video', help="Specify manual output, format: JSON", action='store', dest='manual_filename', default='58-1.json')
	parser.add_argument('-o', '--output', help="Specify validator output file", action='store', dest='output_file', default='validation.json')
	args = parser.parse_args()

	classifier_filename = args.classifier_filename
	manual_filename = args.manual_filename
	output_file = args.output_file
	metrics = {}

	classifier_highlights = get_highlight_balls_from_json(classifier_filename)
	manual_highlights = get_highlight_balls_from_json(manual_filename)
	completeness,accuracy,missed_highlights = compute_metrics(classifier_highlights, manual_highlights)
	metrics['completeness'] = completeness
	metrics['accuracy'] = accuracy
	metrics['missed_highlights'] = missed_highlights
	print metrics
	with open(output_file, 'wb') as fo:
		json.dump(metrics, fo)	
