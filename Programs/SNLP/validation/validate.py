import json

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
	classifier_filenames = ['output58-1.json', 'output58-2.json', 'output59-1.json', 'output59-2.json']
	manual_filenames = ['58-1.json', '58-2.json', '59-1.json', '59-2.json']
	keys = ['58-1','58-2','59-1','59-2']
	metrics = {}
	for x in range(len(classifier_filenames)):
		classifier_highlights = get_highlight_balls_from_json(classifier_filenames[x])
		manual_highlights = get_highlight_balls_from_json(manual_filenames[x])
		completeness,accuracy,missed_highlights = compute_metrics(classifier_highlights, manual_highlights)
		# print completeness, accuracy
		metrics[keys[x]] = [completeness, accuracy, missed_highlights]
	print metrics
	with open('validation.json', 'wb') as fo:
		json.dump(metrics, fo)	
