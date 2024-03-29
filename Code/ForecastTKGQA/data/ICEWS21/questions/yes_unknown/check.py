import pickle as pickle

ff = pickle.load(open("Data/ForecastTKGQuestions/yes_unknown/test.pickle", "rb"))
# for item in ff[0]:
#     print(item)

# # Check whichever question you like
# a = ff[3]
# print(a)
# print(a["paraphrases"]) # Question text
# # print(a["choices"][0]["paraphrases"])
# # print(a["choices"][1]["paraphrases"])
# # print(a["choices"][2]["paraphrases"])
# # print(a["choices"][3]["paraphrases"])

for item in ff[0]:
    print(item, end="\t")

print()

for i in range(5):
    for j in range(len(ff[0])):
        print(ff[i][list(ff[0])[j]], end="\t")
    print()
