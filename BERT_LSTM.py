import torch
import os
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import random
import warnings
from datetime import datetime
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
'''
class LSTMDataset(Dataset):
    def __init__(self, category, name, vector):
        self.category = category
        self.name = name
        self.vector = vector
    def __len__(self):
        return len(self.vector)
    def __getitem__(self, item):
        category = torch.tensor(CLASS_NAMES[self.category[item]],dtype=torch.long)
        name = self.name[item]
        vector = self.vector[item]
        return {
            'target': category,
            'name': name,
            'vector': vector,
        }

def LSTM_data_loader(category, name, vector, batch_size):
  data = LSTMDataset(
      category=category,
      name=name,
      vector=vector
  )
  return DataLoader(
    data,
    batch_size= batch_size,
    num_workers= 0
  )'''

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
        if category in CLASS_NAMES.keys():
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
                f.write("\n")                                   #category+"\t"+name+"\t"+vector(每句用"|"隔開)
                print("寫入 --{}".format(books[i][1]))
            except:
                continue

    os.chdir(original_dir)

def LSTM_data_loader(category, name, vector, batch_size):#list 中append {}
    data = []               #[{b*c,b*n,b*v},{},{}...]
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

def average(data,manul=None):#平均分配
    dis = {i:0 for i in list(CLASS_NAMES.keys())}#紀錄
    data = [i.split("\t") for i in data]
    temp = []

    for i in data:                  #count
        dis[i[0]] += 1
    for k,v in dis.items():         #print()
        print(k," : ",v)
    if manul==None:
        min_class = min(list(dis.values()))   #find min
    else:
        min_class = manul
    print("min : ",min_class)

    random.shuffle(data)
    dis = {i: 0 for i in list(CLASS_NAMES.keys())}

    for i in data:
        if dis[i[0]] <= min_class:
            temp.append(i)
            dis[i[0]] += 1

    print("--平均整理後")
    for k, v in dis.items():  # print()
        print(k, " : ", v)

    return temp

def LSTM_data_pre(file,seq_len,batch_size,train = False,manul=None):
    og_path = os.getcwd()
    os.chdir("data")

    with open(file + ".txt","r",encoding="utf-8") as f:
        temp = f.readlines()
    #print(temp[-1])

    if train:
        temp = average(temp,manul)

    random.shuffle(temp)
    n_examples = len(temp)

    if not train:
        category = [i.split("\t")[0] for i in temp]
        name = [i.split("\t")[1] for i in temp]
        vector = [i.split("\t")[2] for i in temp]
    else:
        category = [i[0] for i in temp]
        name = [i[1] for i in temp]
        vector = [i[2] for i in temp]

    del temp
    vector = [i.split("|") for i in vector] #vector[book][sent]
    padd = [float(0) for i in range(768)]

    for book in range(0,len(vector)):             #book
        for sent in range(0,len(vector[book])):   #to torch.tensor
            vector[book][sent] = list(map(float, vector[book][sent].split(",")))
        while len(vector[book]) < seq_len:
            vector[book].append(padd)
        while len(vector[book]) > seq_len:
            del vector[book][-1]
    #print(np.array(vector).shape)

    #dataloader
    data = LSTM_data_loader(category, name, vector, batch_size)
    #test
    t = data[0]
    print(len(t["target"]))                                             #torch.tensor([CLASS_NAMES[i] for i in t["target"]], dtype=torch.long)
    print(len(t["name"]))                                               #,t["name"])
    print(len(t["vector"]),len(t["vector"][0]),len(t["vector"][0][0]))
    #print(torch.from_numpy(np.array(t["vector"][0][0])).float())

    os.chdir(og_path)
    return data,n_examples

def t_lstm(data,model,loss_fn,optimizer,scheduler,n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data:
        vector = torch.from_numpy(np.array(d["vector"])).float().to(device)
        targets = torch.tensor([CLASS_NAMES[i] for i in d["target"]], dtype=torch.long).to(device)
        model.reset_hidden_state(len(targets))
        #print(len(d["vector"]), len(d["vector"][0]), len(d["vector"][0][0]))

        outputs = model(vector)

        _, preds = torch.max(outputs, dim=1)  # 看看出來的答案對不對，計算loss
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)  # 記錄正確數
        losses.append(loss.item())  # 這邊也只是在記錄

        optimizer.zero_grad()  # 這個我目前不懂
        loss.backward()  # 更新
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 防止梯度爆炸(我不確定需不需要)
        optimizer.step()  # 跟他們說要到下一步了
        scheduler.step()

    return correct_predictions.double() / n_examples, np.mean(losses)  # 回傳統計數據

def v_lstm(data,model,loss_fn,n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data:
            vector = torch.from_numpy(np.array(d["vector"])).float().to(device)
            targets = torch.tensor([CLASS_NAMES[i] for i in d["target"]], dtype=torch.long).to(device)
            model.reset_hidden_state(len(targets))

            outputs = model(vector)

            _, preds = torch.max(outputs, dim=1)  # 看看出來的答案對不對，計算loss
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)  # 記錄正確數
            losses.append(loss.item())  # 這邊也只是在記錄

    return correct_predictions.double() / n_examples, np.mean(losses)  # 回傳統計數據

def LSTM(num,layer,epoch,model_path=None,manul=None):
    #seq_len = 1000  # 暫定
    #seq_len = 500  # 暫定
    #seq_len = 250  # 暫定
    seq_len = num
    batch_size = 1  #設1就會學了...好像這東西不是真正的Batch size。他會背輸入的格式
    epochs = epoch
    overfit = 0.1

    history = {"t_acc":[],"t_loss":[],"v_acc":[],"v_loss":[],"best_epoch":[-1,-1,10000]}#best_epoch = [epoch,val_acc,acc_diff]
    og_dir = os.getcwd()

    print("-" * 10, " Data preprocess ", "-" * 10, "\n")
    t_data, t_n_examples = LSTM_data_pre("train_"+str(num),seq_len,batch_size,train=True,manul=manul) #output : dataloader{target(String),name,vector},n_examples
    #t_data, t_n_examples = LSTM_data_pre("val_"+str(num),seq_len,batch_size,train=True,manul=manul) #純粹測試
    v_data,v_n_examples = LSTM_data_pre("val_"+str(num), seq_len, batch_size)

    print(len(t_data),t_n_examples)

    print("-" * 10, " Build LSTM model ", "-" * 10,"\n")
    model = book_LSTM(
        input_dim=768,
        n_hidden= 512,
        seq_len=seq_len,
        n_layers=layer
    )
    if model_path :
        print("-"*10,"Continue Training","-"*10,"\n")
        model.load_state_dict(torch.load(model_path))  # load model
    model = model.to(device)

    print("-" * 10, " Set optimizer,scheduler,loss_fn ", "-" * 10,"\n")
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(t_data) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(device)

    print("-" * 10, " Start training ", "-" * 10, "\n")
    os.chdir("output")
    for i in range(epochs):
        print("epoch : {} / {}".format(i + 1, epochs))
        t_acc, t_loss = t_lstm(t_data, model, loss_fn, optimizer, scheduler,t_n_examples)
        print("--train acc: {} loss: {}".format(t_acc, t_loss))

        v_acc, v_loss = v_lstm(v_data, model, loss_fn, v_n_examples)
        print("--val acc: {} loss: {}".format(v_acc, v_loss))

        history["t_acc"].append(t_acc)
        history["t_loss"].append(t_loss)
        history["v_acc"].append(v_acc)
        history["v_loss"].append(v_loss)

        #if abs(t_acc-v_acc) <= overfit and v_acc >= history["best_epoch"][1]: #best_epoch = [epoch,val_acc,acc_diff]
        if v_acc >= history["best_epoch"][1]:  # best_epoch = [epoch,val_acc,acc_diff]
            history["best_epoch"][0] = i+1
            history["best_epoch"][1] = v_acc
            history["best_epoch"][2] = abs(t_acc-v_acc)

        #Save model
        torch.save(model.state_dict(), 'BertLSTM_for_LSTM_epoch'+str(i+1)+'.bin')

    #Save Acc,loss,best_epoch
    with open("History.txt","w+") as f:
        for key in history.keys():
            f.write(key + " : ")
            for i in range(len(history[key])):
                f.write("{},".format(history[key][i]))
            f.write("\n\n")
        f.write("#best_epoch = [epoch,acc_differance]")

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
                vector = vector.replace("\n", "")
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
    torch.backends.cudnn.enabled = False
    print(torch.cuda.get_device_name(0))
    print(device)
    print('\n')

    PRE_TRAINED_MODEL_NAME = 'bert-base-chinese'
    CLASS_NAMES = {"人文社會":0, "心理成長":1, "史地傳記":2, "冒險推理":3, "科幻靈異":4, "科普百科":5, "旅遊紀實":6, "商業理財":7, "感性抒情":8,
                   "詩詞國學":9, "運動休閒":10, "醫療健康":11, "藝術設計":12}



    num = 10 #每n句分割
    layer = 2 # LSTM層數
    epoch = 10 # 訓練的次數，例:10會產生出10個model
    model_path = None
    manul = None        ########沒有要用 舊社 None

    ######################################################################
    # Create train,val,test data
    temp = ["train", "val"]
    for i in temp:
        bert(i)                    #有做過就可以不用跑

    print("--num :",num)
    print("--epoch :",epoch)
    print("--manul :", manul)
    for i in ["train","val"]:
        print("--",i)
        divide(i,num)           #(mode,num)有做過就可以不用跑
    ######################################################################

    #Train LSTM
    start = datetime.now()
    LSTM(num,layer,epoch,model_path,manul=manul)
    print("執行時間 : ",datetime.now() - start)