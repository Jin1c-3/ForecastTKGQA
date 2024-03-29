# Python已经内嵌pickle
# import pickle5 as pickle
import pickle as pickle

# ff = pickle.load(open('test.pickle', 'rb'))
# 项目的相对路径不正确
ff = pickle.load(open("Data/ForecastTKGQuestions/entity_prediction/test.pickle", "rb"))

# for item in ff[0]:
#     print(item)

# # Check whichever question you like
# a = ff[3]
# print(a)
# print(a["paraphrases"])  # Question text
# # print(a["choices"][0]["paraphrases"])
# # print(a["choices"][1]["paraphrases"])
# # print(a["choices"][2]["paraphrases"])
# # print(a["choices"][3]["paraphrases"])

for item in ff[0]:
    print(item, end="\t")

print()

for i in range(5):
    for j in range(len(ff[0])):
        print(ff[i][list(ff[0])[j]],end="\t")
    print()    
