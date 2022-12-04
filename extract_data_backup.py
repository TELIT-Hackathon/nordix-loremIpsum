import json
import pprint

with open("data.json", 'r', encoding='utf-8') as f:
	data = json.load(f)

extracted = dict()
for elem in data:
	main_info = dict()
	name = elem['summary']['name'].lower()
	main_info[name] = dict()
	main_info[name]['title'] = elem['summary']['title'].lower()
	main_info[name]['rating'] = elem['summary']['rating']
	main_info[name]['noOfReviews'] = elem['summary']['noOfReviews']
	main_info[name]['description'] = elem['summary']['description'].lower()
	main_info[name]['focus'] = dict()
	for item in elem['focus']:
		main_info[name]['focus'][item['title'].lower()] = [tag['name'].lower() for tag in item['values']]

	main_info[name]['portfolio'] = dict()
	for item in elem['portfolio']:
		main_info[name]['portfolio'][item['title'].lower()] = item['description'].lower()
	# main_info['portfolio']
	extracted.update(main_info)
	pprint.pprint(extracted)
	exit()


# dict_keys(['url', 'summary', 'focus', 'portfolio', 'verification', 'reviews', 'websiteUrl'])

for comp in extracted:
	print(comp)
