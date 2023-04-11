import pickle
with open("/Users/macm2/PycharmProjects/FLLM_black_box/result/score_dictionaries/clients_1_3_PPL-XL_Zlib.txt", 'rb') as fp:
    top_scores=pickle.load(fp)
    print(top_scores)
file = open("Fruits.obj",'rb')
object_file = pickle.load(file)
file.close()