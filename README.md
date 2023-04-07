# BERT-LSTM_Book_Classification

## Setup

Categories = ["人文社會", "心理成長", "史地傳記", "冒險推理", "科幻靈異", "科普百科", "旅遊紀實", "商業理財", "感性抒情",
                   "詩詞國學", "運動休閒", "醫療健康", "藝術設計"]
1. **Create folders at //BERT-LSTM_Book_Classification/**

    - val
        - folders * number of Categories. **named accordingly**.
    - train
        - folders * number of Categories. **named accordingly**.
    - test5
        - folders * number of Categories. **named accordingly**.
    - data
        - Null . Used for placing **vectors created by BERT**, and than **used by LSTM**. 
        
2.  __Place *.txt of books in their belonged Category.__

    - val : Books for validation
    - train : Books for training
    - test5 : Books for testing

Or you can just get the data here [Google Drive](https://drive.google.com/drive/folders/1FK3ticrrJrRbRTkR6eI24ZxU0y8Y1sSq?usp=sharing).

## Train

Run **BERT_LSTM.py**

Output : 

  1. **model checkpoint** for each epoch. //Output/BertLSTM_for_LSTM_epoch*.bin
  2. History.txt **Contans Loss and Accuracy for each epoch**

## Test
Run **BERT_LSTM_Test.py**

Output :

  1. Details on prediction (CSV structure). ex : **//Output/Test_Output_10_epoch9.txt**
  
      - name : Book name
      - ans  : Real Category
      - pre  : Predicted Category
      - All Category names : How many times the model think it was a certain category
      - top3 : Whether the real answer is in the top 3 by vote count or not (true : 1,false : 0)
      
  2. List of real & predicted answer. Plus accuracy of every **N** sentences. **NOT THE ACCURACY OF THE MODEL**

## If only used for prediction

**Setup**<p>
      1. Delete all folders but 1,in the test5 folder. **Folder name needs to be one of the categories**<p>
      2. Put books you want to predict in the only folder test5 has.<p>

Run **BERT_LSTM_Test.py**

Output : Same as in **Test**

## Strongly recomend using GPU
