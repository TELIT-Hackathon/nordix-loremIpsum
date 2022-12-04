import json
import pprint
from gensim.parsing.preprocessing import remove_stopwords

with open("data.json", 'r', encoding='utf-8') as f:
	data = json.load(f)

extracted = dict()
lst_focus = list()
for elem in data:
	main_info = dict()
	name = elem['summary']['name'].lower()
	main_info[name] = dict()
	main_info[name]['text_info'] = list()
	# main_info[name]['text_info'].append(elem['summary']['title'].lower())
	main_info[name]['rating'] = elem['summary']['rating']
	main_info[name]['n_of_reviews'] = elem['summary']['noOfReviews']
	main_info[name]['text_info'].append(elem['summary']['description'].lower())

	for item in elem['focus']:
	# 	main_info[name]['text_info'].append(item['title'].lower())
		main_info[name]['text_info'] += [tag['name'].lower() for tag in item['values']]
		lst_focus += [tag['name'].lower() for tag in item['values']]

	for item in elem['portfolio']:
		# main_info[name]['text_info'].append(item['title'].lower())
		main_info[name]['text_info'].append(item['description'].lower())

	for review in elem['reviews']:
		main_info[name]['text_info'].append(review['name'].lower())
		# main_info[name]['text_info'].append(review['project']['name'].lower())
		# main_info[name]['text_info'].append(review['project']['category'].lower())
		main_info[name]['text_info'].append(review['project']['description'].lower())
		# main_info[name]['text_info'].append(review['review']['comments'].lower())
		# for cont in review['content']:
		# 	main_info[name]['text_info'].append(cont['text'].lower())
		# print(review['content'])
		# main_info[name]['text_info'].append(review['content']['comments'].lower())

	extracted.update(main_info)
	# pprint.pprint(extracted)

print(len(extracted.keys()))

for i in extracted:
	new_list = list()
	for string in extracted[i]['text_info']:
		new_list.append(remove_stopwords(string))
	extracted[i]['text_info'] = new_list

with open("extracted_cut_focus.json", 'w', encoding='utf-8') as f:
	json.dump(extracted, f, indent=4)


with open("focus.txt", 'w', encoding='utf-8') as f:
	for name in lst_focus:
		f.write(name + '\n')
