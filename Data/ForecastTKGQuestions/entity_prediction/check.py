import pickle5 as pickle

ff = pickle.load(open('test.pickle', 'rb'))
for item in ff[0]:
    print(item)

# Check whichever question you like
a = ff[3]
print(a)
print(a["paraphrases"]) # Question text
# print(a["choices"][0]["paraphrases"])
# print(a["choices"][1]["paraphrases"])
# print(a["choices"][2]["paraphrases"])
# print(a["choices"][3]["paraphrases"])