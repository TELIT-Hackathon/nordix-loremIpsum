import torch
from transformers import BertModel, BertTokenizer
from scipy.spatial import distance
import json
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
porter = PorterStemmer()

description = "We designed & developed the mobile application for iZZi Guide that helps tour companies manage and allocate their guides in a more efficient way. The tour guide talent pool offers access to thousands of registered and licensed tour guides.\n \nAbout iZZi Guide\niZZi guide is an easy-to-use booking tool that connects hotels with licensed tour guides through offering a personalized service by matching the right guides with the right guests.\n \nIndustry: Travel\nProject: Mobile application\nClient: iZZi Guide\n \nWorking model: Discovery project\n \n \niZZi Guide helps tour companies manage and allocate their guides in a more efficient way.\niZZi Guide helps tour companies manage and allocate their guides in a more efficient way. The tour guide talent pool offers access to thousands of registered and licensed tour guides.\n \nThe challenge\nTravel companies struggle with having reliable sources of local tour guides and maintaining their talent pools. Effectively matching guides to correct tours, coordinating details and taking change requests is an everyday paint point for organizers.\nThe challenge was finding a way to get licensed guide profiles saved into system, building a system that could match tour jobs with multiple guides and help the guides reply to jobs in an efficient way.\n \nThe solution\nWe ran a discovery project starting with task analysis workshops to identify all the possible users of the platform, their current tasks and our matching solutions.\nWe progressed quickly into low fidelity wire-framing stages to define screens needed based on mapped user flows.\n \nWe tested and looped back for iterations of the information architecture and updated the wire-frames and prototypes accordingly.\n \nRead the full case study here: https://sevenpeakssoftware.com/ux-design-izzi-guide-bangkok/"
# description = "Project Requirements: 1. Design an MVP healthcare application using Hyperledger, GO, and ERC-20 token technologies. 2. Develop a secure and user-friendly interface to collect and store personal information for deep data analysis. 3. Ensure the application is GDPR and HIPPA compliant for data privacy and security. 4. Implement blockchain technology for data storage and transactions. 5. Develop a system to reward users with crypto for sharing their information. 6. Ensure scalability and performance of the application. 7. Develop appropriate testing and quality assurance processes. 8. Develop detailed documentation for the application. 9. Develop and execute a plan for deployment of the application."
# description_to_comp = "Seven Peaks Software developed the native Android & iOS apps along with a supporting application layer allowing sales agents to create accurate car insurance quotations, manage their existing customers, and finalize policy sales – right from their mobile phones.\n \nAbout Viriyah Insurance\nViriyah Insurance has effectively nurtured the knowledge, expertise, and experiences of its employees and organizations, having been a market leader in the insurance industry for 20 consecutive years.\n \nIndustry: Insurance\nProject: Mobile application\nClient: Viriyah Insurance\n \nWorking model: Discovery project\n \nOpening up direct sales channels for car insurance products on mobile\nWe created native Android and iOS apps along with a supporting application layer to allow sales agents to create and deliver accurate and up to date car insurance quotations, manage their existing customers, as well as finalize policy sales for immediate coverage. Everything right from their mobile phones.\n \nThe challenge\nThe client has eagerly embraced mobile applications for many of their business processes from accident reporting and claims filing, to parts procurement, distribution and delivery.\nHowever after all this time their sales channel remained firmly fixed to desktop-formatted websites, which has limited sales operations of their staff and indirect agents to when they are close to a bulky device capable of handling full screen web applications.\n \nThe solution\nBy developing native Android and iOS apps we were able to allow the client to leverage the benefit of a connected world where estimations, sales, quotations and renewals can all be handled from mobile devices.\n \nTechnologies such as GPS, high-resolution cameras on mobile devices.....\n \nRead the full case study here: https://sevenpeakssoftware.com/ux-design-viriyah-insurance-bangkok/"
# description_to_comp = "Grabbd is a fun way to remember and save places you love and want to try.The App allows you to create your own list of go to places, easily tags the places you grab let it be a new one or one of your all time favourites. The App allows you to create your own list of go to places, easily tags the places you grab let it be a new one or one of your all time favourites. It also let's you follow your friends and family and other foodies on grabbd you can see what they have to say about various place and what’s more you can access this information from anywhere on the earth!\nGrabbd is availabe on iOS for download!"
# description_to_comp = "We designed & developed the mobile application for iZZi Guide that helps tour companies manage and allocate their guides in a more efficient way. The tour guide talent pool offers access to thousands of registered and licensed tour guides.\n \nAbout iZZi Guide\niZZi guide is an easy-to-use booking tool that connects hotels with licensed tour guides through offering a personalized service by matching the right guides with the right guests.\n \nIndustry: Travel\nProject: Mobile application\nClient: iZZi Guide\n \nWorking model: Discovery project\n \n \niZZi Guide helps tour companies manage and allocate their guides in a more efficient way.\niZZi Guide helps tour companies manage and allocate their guides in a more efficient way. The tour guide talent pool offers access to thousands of registered and licensed tour guides.\n \nThe challenge\nTravel companies struggle with having reliable sources of local tour guides and maintaining their talent pools. Effectively matching guides to correct tours, coordinating details and taking change requests is an everyday paint point for organizers.\nThe challenge was finding a way to get licensed guide profiles saved into system, building a system that could match tour jobs with multiple guides and help the guides reply to jobs in an efficient way.\n \nThe solution\nWe ran a discovery project starting with task analysis workshops to identify all the possible users of the platform, their current tasks and our matching solutions.\nWe progressed quickly into low fidelity wire-framing stages to define screens needed based on mapped user flows.\n \nWe tested and looped back for iterations of the information architecture and updated the wire-frames and prototypes accordingly.\n \nRead the full case study here: https://sevenpeakssoftware.com/ux-design-izzi-guide-bangkok/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
model = model.to(device)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def make_torch_vector(text):

    description_encoded = tokenizer.encode(text)
    description_encoded = description_encoded[:512]
    description_encoded_tensor = torch.LongTensor(description_encoded)
    # Set the device to GPU (cuda) if available, otherwise stick with CPU
    description_encoded_tensor = description_encoded_tensor.to(device)
    model.eval()

    description_encoded_tensor_ids = description_encoded_tensor.unsqueeze(0)
    try:
        out = model(input_ids=description_encoded_tensor_ids)
    except Exception as e:
        print(e)
        print(len(description_encoded))
        raise RuntimeError()
    hidden_states = out[2]
    cat_sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()

    return cat_sentence_embedding


def cosine_between_texts(original_text, compare_to):
    vector1 = make_torch_vector(original_text)
    vector2 = make_torch_vector(compare_to)

    return distance.cosine(vector1.cpu().detach().numpy(), vector2.cpu().detach().numpy())

def read_clean_vectorize_dataset(json_path):
    f = open(json_path, encoding="utf8")
    data_dict = json.load(f)
    index = 0
    l = len(data_dict.keys())
    for key, value in data_dict.items():
        res_dict = {}
        # print(data_dict)
        for text in value["text_info"]:
            line = text.split(' ')
            if len(line) > 500:
                print('qwerty', len(line))
                l1 = ' '.join(line[0:len(line)//2])
                l2 = ' '.join(line[len(line) // 2:])
                res_dict[l1] = make_torch_vector(l1).cpu().detach().numpy()
                res_dict[l2] = make_torch_vector(l2).cpu().detach().numpy()
            else:
                res_dict[text] = make_torch_vector(text).cpu().detach().numpy()
        data_dict[key]["text_info"] = res_dict
        index+=1
        print("{} {}/{}".format(key, index, l))
    return data_dict

def find_min_distance_company(vector_to_comp, data_dict):
    min_dict = {}
    indexes = {}
    for key, value in data_dict.items():
        dist_list = []
        for _, vector in value["text_info"].items():
            dist_list.append(distance.cosine(vector_to_comp, vector))
        min_dict[key] = min(dist_list)
        index = dist_list.index(min_dict[key])
        indexes[key] = index
        # list(data_dict[key]["text_info"].keys())[indexes[min(min_dict, key=min_dict.get)]]
    return min(min_dict, key=min_dict.get), list(data_dict[key]["text_info"].keys())[indexes[min(min_dict, key=min_dict.get)]], min_dict[min(min_dict, key=min_dict.get)]


def find_min_distance_company_top(vector_to_comp, data_dict):
    min_dict = {}
    indexes = {}
    for key, value in data_dict.items():
        dist_list = []
        for _, vector in value["text_info"].items():
            dist_list.append(distance.cosine(vector_to_comp, vector))
        index = dist_list.index(min(dist_list))
        # indexes[key] = index
        min_dict[key] = (min(dist_list), list(data_dict[key]["text_info"].keys())[index])
        # list(data_dict[key]["text_info"].keys())[indexes[min(min_dict, key=min_dict.get)]]
    min_dict = {k: v for k, v in sorted(min_dict.items(), key=lambda item: item[1][0])}

    return list(min_dict.items())[0:30]


def sort_companies_by_key_words(text, top_dict):
    lines = []
    list_of_tech = []
    with open("focus.txt", 'r') as fp:
        lines = fp.readlines()
        token_words_text = word_tokenize(text)
        stem_text = []
        for word in token_words_text:
            stem_text.append(porter.stem(word))
            # stem_text.append(" ")
        # stem_text = "".join(stem_text)
        for i in range(len(lines)):
            stem_sentence = []
            lines[i] = lines[i].replace('\n','')
            for k in range(50):
                lines[i] = lines[i].replace('<i>{}%</i>'.format(k), '')
            lines[i] = lines[i].lower()
            token_words_top = word_tokenize(lines[i])
            for word in token_words_top:
                # stem_sentence.append(porter.stem(word))
                # stem_sentence.append(" ")
                list_of_tech.append(porter.stem(word))
            # lines[i] = "".join(stem_sentence)

    list_of_tech = list(set(list_of_tech))
    # print(list_of_tech, stem_text)
    result_score = {}
    for key, value in top_dict.items():
        # print(value)
        descr_tokens = value[0][1].split(' ')
        for token in descr_tokens:
            if token in list_of_tech and token in stem_text:
                if key not in result_score:
                    result_score[key] = 0
                else:
                    result_score[key] += 1
    # print(result_score)
    result_score = {k: v for k, v in sorted(result_score.items(), key=lambda item: item[1], reverse=True)}
    result_dict = {}
    for key in result_score.keys():
        result_dict[key] = top_dict[key]

    return result_dict, result_score

def sort_by_rating(top_dict, main_dict):
    score = {}
    Q = 5
    for key, item in top_dict.items():
        p = main_dict[key]["rating"]
        q = main_dict[key]["n_of_reviews"]
        score[key] = (5*p/10 + 5*(1 - np.exp(-q/Q)))
    score = {k: v for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)}

    res = {}
    for key in score.keys():
        res[key] = top_dict[key]

    return res, score


# data = read_clean_vectorize_dataset('extracted_cut.json')


with open("extracted_v.json", "r") as outfile:
    data = json.load(outfile)
print("loaded")

x = find_min_distance_company_top(make_torch_vector(description).cpu().detach().numpy(), data)
for key, value in x:
    print(key, value)
d = {}
for k, v in x:
   d.setdefault(k, []).append(v)
print('\n\n\n')
res_sort, res_score = sort_companies_by_key_words(description, d)
for key, value in res_sort.items():
    print(key, value)
print('\n\n\n')
for key, value in res_score.items():
    print(key, value)
print('\n\n\n')
r, b = sort_by_rating(res_sort, data)
for key, value in r.items():
    print(key, value)
print('\n\n\n')
for key, value in b.items():
    print(key, value)
print('\n\n\n')


