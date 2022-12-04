import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from pathlib import Path
from test_bert import *

app = Flask(__name__)

# hi = {'hello': 'world!'}


@app.route('/')
def upload_form():
    return render_template('template.html')


@app.route('/stats', methods=['GET', 'POST'])
def get_stats():
    args = request.args
    description = args.get("name")
    # data = read_clean_vectorize_dataset('../extracted_cut.json')
    with open("../extracted_v.json", "r",encoding="utf8") as outfile:
        data = json.load(outfile)
    with open("../data.json", "r",encoding="utf8") as outfile2:
        full_data = json.load(outfile2)
    full_data_img = {}
    for item in full_data:
        full_data_img[item["summary"]["name"]] = item["summary"]["logo"]
    # print("loaded")
    x = find_min_distance_company_top(make_torch_vector(description).cpu().detach().numpy(), data)
    # for key, value in x:
    #     print(key, value )
    d = {}
    for k,v in x:
       d.setdefault(k, []).append(v)
    # print('\n\n\n')
    res_sort, res_score = sort_companies_by_key_words(description, d)
    # for key, value in res_sort.items():
    #     print(key, value)
    # print('\n\n\n')
    # for key, value in res_score.items():
    #     print(key, value)
    # print('\n\n\n')
    r_dict, _ = sort_by_rating(res_sort, data)
    return render_template('template_res.html', r_dict=r_dict, orig_data=data, images=full_data_img)
    # for key, value in r.items():
    #     print(key, value)
    # print('\n\n\n')
    # for key, value in b.items():
    #     print(key, value)
    # print('\n\n\n')
    # print(find_min_distance_company(make_torch_vector(description).cpu().detach().numpy(), data))
    # print(cosine_between_texts(description, description_to_comp))



app.run()