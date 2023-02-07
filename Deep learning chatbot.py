#!/usr/bin/env python
# coding: utf-8

# # EDA

# In[1]:


f=open('dialogs.txt','r')
file = f.read()


# In[2]:


qna_list = [f.split('\t') for f in file.split('\n')]
print(qna_list[:5])


# In[3]:


que = [x[0] for x in qna_list]
ans = [x[1] for x in qna_list]

print(que[:5])
print(ans[:5])
print(len(que))
print(len(ans))


# In[4]:


from tensorflow.keras.preprocessing.text import Tokenizer


# In[5]:


tk = Tokenizer(oov_token="<OOV>")
tk.fit_on_texts(que+ans)


# In[6]:


tk.index_word


# In[7]:


tk.word_index


# In[8]:


tk.texts_to_sequences(['zaf'])


# In[9]:


que_seq = tk.texts_to_sequences(que)
ans_seq = tk.texts_to_sequences(ans)


# In[10]:


import matplotlib.pyplot as plt
plt.hist([len(s) for s in que_seq])


# In[11]:


plt.hist([len(s) for s in ans_seq])


# In[12]:


MAX_SEQ_LEN = 12
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[13]:


pad_que_seq = pad_sequences(que_seq,maxlen=MAX_SEQ_LEN,padding='post')
pad_ans_seq = pad_sequences(ans_seq,maxlen=MAX_SEQ_LEN,padding='post')


# In[14]:


print(pad_que_seq[:5])
print(type(pad_que_seq))
print(pad_ans_seq[:5])
print(type(pad_ans_seq))


# In[15]:


# Use tk.word_index (if tk.word_index changes texts_to_sequences uses changed dict)
tk.texts_to_sequences(['i have'])


# In[16]:


for key in tk.word_index:
    tk.word_index[key] += 2


# In[17]:


tk.word_index['<PAD>']=0
tk.word_index['<S>']=1
tk.word_index['<E>']=2


# In[18]:


a={1:2,3:4}
b = {a[k]:k for k in a}


# In[19]:


tk.index_word = {tk.word_index[key]:key for key in tk.word_index}


# In[20]:


que_seq = tk.texts_to_sequences(que)
ans_seq = tk.texts_to_sequences(ans)
pad_que_seq = pad_sequences(que_seq,maxlen=MAX_SEQ_LEN,padding='post')
pad_ans_seq = pad_sequences(ans_seq,maxlen=MAX_SEQ_LEN,padding='post')


# In[21]:


print(pad_que_seq.shape)
print(pad_ans_seq.shape)


# In[22]:


import numpy as np
dec_input_seq = np.array([[1]+list(s[:-1]) for s in pad_ans_seq.copy()])
dec_target_seq = [list(s) for s in pad_ans_seq.copy()]
# print(dec_target_seq)
for i,l in enumerate(dec_target_seq):
    try:
        dec_target_seq[i][l.index(0)]=2
    except:
        dec_target_seq[i][-1]=2

dec_target_seq = np.array(dec_target_seq)
print(dec_input_seq.shape)
print(dec_target_seq.shape)


# In[23]:


enc_input_seq = pad_que_seq.copy()
print(enc_input_seq[:5])
print(dec_input_seq[:5])
print(dec_target_seq[:5])
assert type(enc_input_seq) == type(dec_input_seq) and type(dec_input_seq) == type(dec_target_seq)
assert enc_input_seq.shape == dec_input_seq.shape and dec_input_seq.shape == dec_target_seq.shape
print(enc_input_seq.shape)


# In[24]:


import tensorflow as tf
vocab_size = len(tk.word_index)
print(vocab_size)


# # MODELING

# In[25]:


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.enc_units = enc_units
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.pre_bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.enc_units,return_sequences=True, return_state=True))
        
    def call(self, x):
        # x shape == (batch_size, seq len)
        # h,c = hidden
        # h c shape == (batch_size, enc_units)
        x = self.embedding(x)
        # embedded x shape == (batch_size, seq_len, emb_dim)
        x,fh,fs,bh,bs = self.pre_bi_lstm(x)
        # x shape == (batch_size, seq_len, enc_units*2)
        # fh,fs,bh,bs shape == (batch_size, enc_units)
        return x



class Attention(tf.keras.layers.Layer):
    def __init__(self,seq_len):
        super(Attention, self).__init__()
#         self.units = units
        self.seq_len = seq_len
        self.repeat = tf.keras.layers.RepeatVector(self.seq_len)
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.d1 = tf.keras.layers.Dense(10,activation='tanh')
        self.d2 = tf.keras.layers.Dense(1,activation='relu')
        self.softmax = tf.keras.layers.Softmax(axis=1)
        
    def call(self, a, s_prev):
        # a.shape == (batch_size, seq_len, enc_units*2)
        # s_prev.shape == (batch_size, dec_units)
        s_prev = self.repeat(s_prev)
        # s_prev.shape == (batch_size, seq_len, dec_unit)
        a_s = self.concat([a,s_prev])
        # a_s.shape == (batch_size, seq_len, dec_units*2)
        out = self.d1(a_s)
        # out.shape == (batch_size, seq_len, dense units)
        out = self.d2(out)
        # out.shape == (batch_size, seq_len, dense units==1)
        attention_score = self.softmax(out)
        # attention_score.shape == out.shape
        context_vector = attention_score * a
        # context_vector.shape == (batch_size, seq_len, dec_units)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        # context_vector.shape == (batch_size, dec_units) == decoder input
        context_vector=tf.expand_dims(context_vector,1)
        return context_vector, attention_score


class Decoder(tf.keras.layers.Layer):
    def __init__(self,vocab_size, dec_units, batch_size):
        super(Decoder, self).__init__()
#         self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.dec_units = dec_units
        self.lstm = tf.keras.layers.LSTM(self.dec_units,return_state=True)
        self.d1 = tf.keras.layers.Dense(self.vocab_size,activation='softmax')

    def call(self,x,hidden):
        h,c = hidden
        # h,c shape == (batch_size, dec_units)
        # x.shape == (batch_size, 1, dec_units)
        output, new_h, new_c = self.lstm(x,initial_state=[h,c])
        # 3 output shape == (batch_size, dec_units)
        output = self.d1(output)
        # output shape == (batch_size, vocab_size)
        return output, new_h, new_c
    
class AttentionMachineTranslationModel(tf.keras.Model):
    def __init__(self,vocab_size, embedding_dim, enc_units, dec_units, batch_size, seq_len,end_token=2):
        super(AttentionMachineTranslationModel, self).__init__()
        self.end_token=end_token
        self.encoder = Encoder(vocab_size,embedding_dim,enc_units,batch_size)
        self.attention = Attention(seq_len)
        self.decoder = Decoder(vocab_size,dec_units,batch_size)
        self.enc_units = enc_units
        self.dec_units = dec_units
        
    def call(self, x):
        a = self.encoder(x)
        predict = []
        h = tf.zeros((x.shape[0],self.dec_units))
        c = tf.zeros((x.shape[0],self.dec_units))
        for t in range(x.shape[1]):
            context, attention = self.attention(a,h)
            output, h, c = self.decoder(context, [h,c])
            predict.append(output)
        return tf.stack(predict,axis=1)
    
    def predict(self, x):
        a = self.encoder(x)
        predict = []
        attentions = []
        h = tf.zeros((x.shape[0],self.dec_units))
        c = tf.zeros((x.shape[0],self.dec_units))
        for t in range(x.shape[1]):
            context, attention = self.attention(a,h)
            output, h, c = self.decoder(context, [h,c])
            predict.append(output)
            attentions.append(attention)
        return [tf.stack(predict,axis=1),attentions]
        
    


# In[26]:


model = AttentionMachineTranslationModel(2523,512,128,256,128,12,2)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='acc')])
history=model.fit(enc_input_seq[:-13],dec_target_seq[:-13],batch_size=128,epochs=300)


# In[27]:


def pred():
    query = input()
    query_list = [query for _ in range(128)]
    # print(query_list)
    query_seq = tk.texts_to_sequences(query_list)
#     print(query_seq)
    query_seq = pad_sequences(query_seq,maxlen=MAX_SEQ_LEN,padding='post')
    pad_index = query_seq[0].tolist().index(0)
    pred,attentions = model.predict(np.array(query_seq))
    one_pred = pred.numpy()[0]
    pred_index = one_pred.argmax(axis=-1)
    for end_in,i in enumerate(pred_index):
        if tk.index_word[i]=='<E>':
            end_index = end_in
            break
        print(tk.index_word[i],end=' ')


# In[29]:


print("hello")


# In[ ]:


while True:
    pred()


# In[ ]:




