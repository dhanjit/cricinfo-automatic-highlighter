from bs4 import BeautifulSoup
import requests
import json

def isFloat(x):
	try:
		a = float(x)
	except ValueError:
		return False
	else:
		return True

def process(coms):
	keys = []
	for com in coms:
		s = ""
		for t in com.contents:
			if(isinstance (t, str)):
				s += ' ' + t
			else:
				foo = t.text.split()
				for f in foo:
					s += ' $$' + f + '$$'

		keys.append(s.replace('\n', '').strip());

	ball = 0
	commentary = []
	for key in keys:
		if(isFloat(key)):
			ball = key
		else:
			o = {}
			o['ball'] = ball
			o['text'] = key.replace('\n', '').strip()
			commentary.append(o)

	return commentary

def crawlUrl(url, file):
	d = {}
	response = requests.get(url)
	soup = BeautifulSoup(response.content)

	d['url'] = url
	d['name'] = soup.find('p', {'class' : 'teamText'}).text.replace('\n', '').strip()
	d['innings'] = soup.find('a', {'class' : 'activeTab'}).text.replace('\n', '').strip()
	d['commentary'] = process(soup.findAll('p', {'class' : 'commsText'}))

	with open(file, 'w', encoding='utf-8') as f:
		json.dump(d, f)

# crawlUrl("http://www.espncricinfo.com/indian-premier-league-2014/engine/match/729281.html?innings=1;view=commentary", "data1.json")
# crawlUrl("http://www.espncricinfo.com/indian-premier-league-2014/engine/match/729281.html?innings=2;view=commentary", "data2.json")
crawlUrl("http://www.espncricinfo.com/indian-premier-league-2014/engine/match/729295.html?innings=1;view=commentary", "test1.json")
