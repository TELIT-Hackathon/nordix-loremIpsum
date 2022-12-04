import json

f = open('data.json', encoding="utf8")
d = json.load(f)
for i in d[0].keys():
    print(i)