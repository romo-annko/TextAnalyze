# -*- coding:utf-8 -*-

# Japanese corpus is downloaded from the url below
# https://github.com/Hironsan/ja.text8

import io
import os
import MeCab
import numpy as np
from keras.layers import Activation, Dense, Dropout, Embedding, LSTM
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.utils import plot_model
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from gensim.models import word2vec

c_path = './data/ja.text8'
path = './data/teacher.txt'
sentences = []
batch_size = 40
batch_for_fit = 32
epochs = 50
word_length = 100
hidden_layer_size = 128
predict_size = 100
model_name = 'lstm_text_model.h5'

# テキストファイル内の文章を形態素解析し、単語レベルで配列化
print('テキストファイルの読み込みを開始します...')
with io.open(path, encoding='utf-8') as f:
    text = f.read()
m = MeCab.Tagger('-Ochasen')
m.parse('')
node = m.parseToNode(text)
while node:
    print(node.surface)
    sentences.append(node.surface)
    node = node.next
num_iterations = len(sentences) - batch_size

# コーパスから辞書作成
print('コーパスの読み込みとWord2Vecによる辞書作成を開始します...')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
corpus = word2vec.Text8Corpus(os.path.join(DATA_DIR, 'ja.text8'), 50)
dictionary = word2vec.Word2Vec(corpus, size=word_length, min_count=30)
if not os.path.exists(model_name):
    # 教師データの作成
    print('教師データの作成を開始します...')
    X = np.empty((num_iterations,batch_size,word_length))
    Y = np.empty((num_iterations,word_length))
    for i in range(num_iterations):
        for j in range(batch_size):
            try:
                X[i,j,:] = dictionary.wv[sentences[i+j]]
            except KeyError:
                X[i,j,:] = dictionary.wv['*']
    print('input_matrix --> dim:{}, shape:{}, size:{}'.format(X.ndim,X.shape,X.size))
    for i,w in enumerate(sentences[batch_size:]):
        try:
            Y[i,:] = dictionary.wv[w]
        except:
            Y[i,:] = dictionary.wv['*']
    print('input_matrix --> dim:{}, shape:{}, size:{}'.format(X.ndim,X.shape,X.size))
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=38)
    # Sequentialモデルの作成
    model = Sequential()
    model.add(LSTM(hidden_layer_size,input_shape=(batch_size, word_length)))
    model.add(Dense(word_length,activation='softmax'))
    model.summary()
    # 作成したモデルを.pngファイルで保存
    #plot_model(model,'lstm_model.png')
    # 学習を開始
    print('モデルの学習を開始します...')
    optimizer = RMSprop(lr=0.02)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    model.fit(Xtrain, Ytrain, batch_size=batch_for_fit, epochs=epochs, verbose=1, validation_data=(Xtest, Ytest))
    model.save(model_name)
else:
    model = load_model(model_name)
    seed_text = input('自動生成のため、元となる文章を入力してください：')
    sentences = []
    node = m.parseToNode(seed_text)
    while node:
        print(node.surface)
        sentences.append(node.surface)
        node = node.next
    # 入力した文章をbatch_sizeと同じ配列長に調整
    if len(sentences) > batch_size:
        for i in range(len(sentences)-batch_size):
            sentences.pop(0)
    else:
        if len(sentences) < batch_size:
            for i in range(batch_size-len(sentences)):
                sentences.insert(0,'*')
    vec_result = np.empty((predict_size+batch_size,word_length))
    vec_input = np.empty((1,batch_size,word_length))
    for i, w in enumerate(sentences):
        try:
            vec_result[i,:] = dictionary.wv[w]
            vec_input[0,i,:] = dictionary.wv[w]
        except KeyError:
            vec_result[i,:] = dictionary.wv['*']
            vec_input[0,i,:] = dictionary.wv['*']

    for i in range(predict_size):
        y_pred = model.predict(vec_input,verbose=0)
        word_pred = str(dictionary.similar_by_vector(np.reshape(y_pred,(word_length,)),topn=1)[0][0])
        vec_input = np.concatenate((vec_input[:,1:,:],np.reshape(dictionary.wv[word_pred],(1,1,word_length))),axis=1)
        vec_result[batch_size+i,:] = np.reshape(dictionary.wv[word_pred],(1,word_length))

    for i in range(batch_size,predict_size):
        seed_text = seed_text + str(dictionary.similar_by_vector(vec_result[i,:],topn=1)[0][0])

    print(seed_text)
