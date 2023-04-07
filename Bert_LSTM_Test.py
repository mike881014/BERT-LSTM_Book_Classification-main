import torch
import os
from transformers import BertModel, BertTokenizer
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import warnings
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

class Book_to_sent(nn.Module):
  def __init__(self):
    super(Book_to_sent, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask,
    )
    return pooled_output

class book_LSTM(nn.Module):
    def __init__(self, input_dim, n_hidden, seq_len, n_layers=2):
        super(book_LSTM, self).__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first= False,
            bidirectional=False,
            dropout=0.3
        )
        self.linear1 = nn.Linear(in_features=n_hidden, out_features=int(n_hidden/2))
        self.linear2 = nn.Linear(in_features=int(n_hidden/2), out_features=len(CLASS_NAMES.keys()))

    def reset_hidden_state(self,batch_size):            #API沒寫錯，只是batch_size的實際意義不明
        self.batch_size = batch_size
        self.h = torch.zeros(self.n_layers, batch_size, self.n_hidden).to(device)
        self.c = torch.zeros(self.n_layers, batch_size, self.n_hidden).to(device)

    def forward(self, sequences):
        lstm_out, _ = self.lstm(sequences.reshape(self.seq_len,self.batch_size,-1), (self.h, self.c))
        linear1 = self.linear1(lstm_out)
        #y_pred = self.linear2(lstm_out[-1])  #用[seq_len,batch,dim] 來輸入，要拿出last來比較輕鬆。(都在最後一維(垂直看，一個col就是一筆資料從頭到尾))
        y_pred = self.linear2(linear1[-1])

        return y_pred

class BookDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sent = str(self.data[item])
        encoding = self.tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,  # 2句話所以這邊是True
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True,  # 有給max_length沒給這個他一直叫，所以給一下
            is_pretokenized=False,
            return_tensors='pt',
        )
        return {
            'sent': sent,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

def bert_data_loader(data,tokenizer, max_len, batch_size): #多一個BatchSize
  data = BookDataset(
    data= data,
    tokenizer= tokenizer,
    max_len= max_len
  )
  return DataLoader(
    data,
    batch_size= batch_size,
    num_workers= 0      #這裡我給他多幾個就出問題
  )

def find_books():
    books = []
    for path,_,filenames in os.walk(os.getcwd()):
        category = path.split("\\")[-1]
        if category in CLASS_NAMES.keys() or category == "書": #"書"放的是傳過來的書
            for name in filenames:
                books.append([category,name,category+"/"+name])#[category,name,dir]

    return books

def bert_comp(dir,model,tokenizer):
    with open(dir,"r",encoding="utf-8") as b:
        data = b.read()
    data = data.replace("\n","").split("。")                 #用"。"分句

    max_len = len(max(data,key=len))
    if max_len > 510:
        max_len = 510
    batch_size = 16
    data = bert_data_loader(data,tokenizer,max_len,batch_size)#[sent,input_ids,attention_mask]
    model.eval()
    vector = []
    with torch.no_grad():
        for d in data:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            outputs = model(                                #outputs = [batch_size,768]
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            for i in outputs.cpu().numpy():
                vector.append(i)
    return vector

def bert(name):
    original_dir = os.getcwd()
    os.chdir("data")
    with open(name+".txt","w+",encoding="utf-8") as f:   #先建好黨
        os.chdir(original_dir)
        os.chdir(name)
        books = find_books()                                #books = [category,name,dir]
        model = Book_to_sent()
        model = model.to(device)
        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        for i in range(0,len(books)):

            try:
                vector = bert_comp(books[i][2],model,tokenizer)#vector = [[768],[768],[768]...[768]]
                f.write("{}\t".format(books[i][0]))
                f.write("{}\t".format(books[i][1]))
                for v in range(0,len(vector)-1):
                    for s in range(0,len(vector[v])-1):        #vector[v] = [768]
                        f.write("{},".format(vector[v][s]))
                    f.write("{}".format(vector[v][-1]))
                    f.write("|")

                for s in range(0, len(vector[-1]) - 1):         # vector[v] = [768]
                    f.write("{},".format(vector[-1][s]))
                f.write("{}".format(vector[-1][-1]))
                f.write("\n")                                   # category+"\t"+name+"\t"+vector(每句用"|"隔開)
                print("寫入 --{}".format(books[i][1]))
            except:
                continue

    os.chdir(original_dir)

def LSTM_data_loader(category, name, vector, batch_size):#list 中append {}
    data = []               # [{b*c,b*n,b*v},{},{}...]
    count = 0
    while count + batch_size < len(category):
        temp = {}
        temp["target"]= category[count:count+batch_size]
        temp["name"] = name[count:count + batch_size]
        temp["vector"] = vector[count:count + batch_size]
        count += batch_size
        data.append(temp)

    if len(category) - count <= batch_size :
        temp = {}
        temp["target"] = category[count:]
        temp["name"] = name[count:]
        temp["vector"] = vector[count:]
        data.append(temp)

    return data

def LSTM_data_pre(file,seq_len,batch_size):
    og_path = os.getcwd()
    os.chdir("data")

    with open(file + ".txt","r",encoding="utf-8") as f:
        temp = f.readlines()

    n_examples = len(temp)

    category = [i.split("\t")[0] for i in temp]
    name = [i.split("\t")[1] for i in temp]
    vector = [i.split("\t")[2] for i in temp]
    del temp
    vector = [i.split("|") for i in vector] # vector[book][sent]
    padd = [float(0) for i in range(768)]

    for book in range(0,len(vector)):             # book
        for sent in range(0,len(vector[book])):   # to torch.tensor
            vector[book][sent] = list(map(float, vector[book][sent].split(",")))
        while len(vector[book]) < seq_len:
            vector[book].append(padd)
        while len(vector[book]) > seq_len:
            del vector[book][-1]
    # print(np.array(vector).shape)

    # dataloader
    data = LSTM_data_loader(category, name, vector, batch_size)
    # test
    t = data[0]
    print(len(t["target"]))                                             # torch.tensor([CLASS_NAMES[i] for i in t["target"]], dtype=torch.long)
    print(len(t["name"]))                                               # ,t["name"])
    print(len(t["vector"]),len(t["vector"][0]),len(t["vector"][0][0]))
    # print(torch.from_numpy(np.array(t["vector"][0][0])).float()) #這樣才不會都歸0

    os.chdir(og_path)
    return data,n_examples

def v_lstm(data,model,n_examples):
    model = model.eval()
    history = {"name":[],"ans":[],"pre":[],
               "人文社會":[0], "心理成長":[0], "史地傳記":[0], "冒險推理":[0], "科幻靈異":[0], "科普百科":[0], "旅遊紀實":[0], "商業理財":[0], "感性抒情":[0],
                "詩詞國學":[0], "運動休閒":[0], "醫療健康":[0], "藝術設計":[0],
               "top3":[]} #top3 = bool
    correct_predictions = 0
    book_count = 0
    with torch.no_grad():
        for d in data:#{target,name,vector}
            #print("name",d["name"])
            if d["name"] == ['']:#next book 該統計一下這次的結果了 ，換下一本才會extend
                n_examples -= 1

                history["name"].extend(i.replace("[SubtitleTools.com]", "") for i in temp["name"])
                history["ans"].extend(i for i in temp["target"])
                l = [history[i][book_count] for i in list(history.keys())[3:-1]]
                history["pre"].append(list(CLASS_NAMES.keys())[np.argmax(l)])
                # top3###
                if CLASS_NAMES[history["ans"][book_count]] in np.argpartition(l, -3)[-3:]:
                    history["top3"].append(1)
                else:
                    history["top3"].append(0)
                '''print("--", book_count + 1)
                for i in list(history.keys()):
                    print(i, len(history[i]))'''
                for i in list(history.keys())[3:-1]:
                    history[i].append(0)

                book_count += 1
                continue

            temp = d        #要記錄的時候用
            vector = torch.from_numpy(np.array(d["vector"])).float().to(device)
            targets = torch.tensor([CLASS_NAMES[i] for i in d["target"]], dtype=torch.long).to(device)  #Ans

            model.reset_hidden_state(len(targets))
            outputs = model(vector)              #All
            _, preds = torch.max(outputs, dim=1) #predict

            correct_predictions += torch.sum(preds == targets)
            history[list(history.keys())[3 + preds]][book_count] += 1 #紀錄投票
    for i in list(history.keys())[3:-1]:
        history[i].pop(-1)

    return  correct_predictions.double() / n_examples,history # 回傳統計數據

def LSTM(num,model_path,layer):
    seq_len = num  # 暫定
    batch_size = 1  #設1就會學了...好像這東西不是真正的Batch size。他會背輸入的格式

    og_dir = os.getcwd()

    print("-" * 10, " Data preprocess ", "-" * 10, "\n")
    t_data, t_n_examples = LSTM_data_pre("test5_"+str(num), seq_len, batch_size)#output : dataloader{target(String),name,vector},n_examples
    print(len(t_data), t_n_examples)

    print("-" * 10, " Build LSTM model ", "-" * 10,"\n")
    model = book_LSTM(
        input_dim=768,
        n_hidden= 512,
        seq_len=seq_len,
        n_layers=layer
    )
    model.load_state_dict(torch.load(model_path))#load model
    model = model.to(device)


    print("-" * 10, " Start Testing ", "-" * 10, "\n")
    os.chdir("output")

    acc,history = v_lstm(t_data, model,t_n_examples)

    df = pd.DataFrame.from_dict(history)
    df.to_csv(r'Test_Output_'+str(num)+'.txt')
    with open("Test_Output_acc_confusion_matrix_"+str(num)+".txt","w+") as f:
        f.write("--book")
        f.write("real :"+ str([CLASS_NAMES[i] for i in history["ans"]]) +"\n")  #整本的
        f.write("pre :" + str([CLASS_NAMES[i] for i in history["pre"]]) + "\n")
        f.write("per"+str(num)+"_Acc :" + str(acc.cpu().numpy()) + "\n")                        #每n句的

    os.chdir(og_dir)

def LSTM_find_best(num,model_path,layer,epochs):
    seq_len = num  # 暫定
    batch_size = 1  #設1就會學了...好像這東西不是真正的Batch size。他會背輸入的格式

    og_dir = os.getcwd()
    top1 = []

    print("-" * 10, " Data preprocess ", "-" * 10, "\n")
    t_data, t_n_examples = LSTM_data_pre("test5_"+str(num), seq_len, batch_size)#output : dataloader{target(String),name,vector},n_examples
    print(len(t_data), t_n_examples)

    print("-" * 10, " Build LSTM model ", "-" * 10,"\n")
    model = book_LSTM(
        input_dim=768,
        n_hidden= 512,
        seq_len=seq_len,
        n_layers=layer
    )
    model = model.to(device)

    os.chdir("output")

    for i in range(1,epochs):
        model.load_state_dict(torch.load(model_path+r"\BertLSTM_for_LSTM_epoch"+str(i)+".bin"))#load model

        acc,history = v_lstm(t_data, model,t_n_examples)
        temp = accuracy_score([CLASS_NAMES[i] for i in history["ans"]],[CLASS_NAMES[i] for i in history["pre"]])
        print("epoch",i,"top1 : ",temp)
        top1.append(temp)

    with open("Layer2 "+str(num)+" top1.txt","w+") as f:
        f.write("top1 acc: " + str(top1))

    with open("Layer2 "+str(num)+" top1_ver2.txt","w+") as f:
        for i in range(len(top1)):
            f.write("epoch "+str(i+1)+" : "+str(top1[i])+"\n")

    os.chdir(og_dir)

def divide(mode , num):
    og = os.getcwd()
    os.chdir("data/")
    with open(mode+"_" + str(num) + ".txt", "w+", encoding="utf-8") as new:
        with open(mode+".txt", "r", encoding="utf-8") as old:
            while True:
                data = old.readline()
                if not data:
                    break

                category, name, vector = data.split("\t")
                print("處理 --", name)
                vector = vector.replace("\n", "\n\t\t0.2")         #每本書中間隔一個 (\t\t) => category:"",name:"",vector:"0.2"
                vector = vector.split("|")  # 切成一句一句

                over = len(vector) % num  # 最後一剩多少句
                steps = len(vector) // num  # 完整分成多少
                if steps != 0:
                    for s in range(steps):
                        new.write(category + "\t" + name + "\t")
                        for sent in range(s * num, (s + 1) * num - 1):
                            new.write(vector[sent] + "|")
                        new.write(vector[(s + 1) * num - 1] + "\n")

                if over != 0:
                    new.write(category + "\t" + name + "\t")
                    for sent in range(len(vector) - over, len(vector) - 1):
                        new.write(vector[sent] + "|")
                    new.write(vector[len(vector) - 1] + "\n")
    os.chdir(og)

if __name__ == '__main__':
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #torch.backends.cudnn.enabled = False
    print(torch.cuda.get_device_name(0))
    print('\n')
    #----------------------------------------------------------------------------------
    model_path_ver2 = r"C:\Users\user\PycharmProjects\class\BERT_LSTM\output" # 需更改
    #----------------------------------------------------------------------------------
    PRE_TRAINED_MODEL_NAME = 'bert-base-chinese'
    CLASS_NAMES = {"人文社會":0, "心理成長":1, "史地傳記":2, "冒險推理":3, "科幻靈異":4, "科普百科":5, "旅遊紀實":6, "商業理財":7, "感性抒情":8,
                   "詩詞國學":9, "運動休閒":10, "醫療健康":11, "藝術設計":12}

    num = [10]                          #每多少分一段(判斷一次)
    layer = 2
    #epochs = 10
    best_epoch = [10]###########

    start = datetime.now()
    #--------------Part 1---------------用來看哪個epoch最好(人工看XD)
    #########################################################################################################
    #Create test data
    temp = ["test5"]
    for i in temp:
        bert(i)                                 # 有做過就可以不用跑。

    #Create different versions of test5
    for i in num :
        print("--","test5","--",i)              # 每本書中間隔一個 (\t\t) => category:"",name:"",vector:"0.2"
        divide("test5",i)                       #(mode,num)有做過就可以不用跑
    #########################################################################################################
    #Find best
    #for i in num:
        #LSTM_find_best(i, model_path_ver2+" "+str(i), layer,epochs+1)
        #LSTM_find_best(i, model_path_ver2, layer, epochs + 1)

    # --------------Part 2---------------展生數據
    # Test LSTM
    #start = datetime.now()
    for i in range(len(num)):
        model_path = model_path_ver2+r"\BertLSTM_for_LSTM_epoch" + str(best_epoch[i]) + ".bin"
        print(num[i])
        print(model_path)
        LSTM(num[i],model_path,layer)

    print("執行時間 : ",datetime.now() - start)