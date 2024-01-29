import pandas as pd
import re


import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense,SimpleRNN
from tensorflow.keras.optimizers import Adam
# 读取CSV文件
dataset_fake = pd.read_csv('E://2//archive//Fake.csv')
dataset_true = pd.read_csv('E://2//archive//True.csv')

data = []

# 处理假新闻数据
for _, row in dataset_fake.iterrows():
    text = re.sub(r'[^\w\s]', '', row['text'])
    text = text.lower()
    data.append([text, 0])

# 处理真新闻数据
for _, row in dataset_true.iterrows():
    text = re.sub(r'[^\w\s]', '', row['text'])
    text = text.lower()
    data.append([text, 1])

# 将数据转换为DataFrame
df = pd.DataFrame(data, columns=['text', 'label'])

# 文本和标签
texts = df['text'].values
labels = df['label'].values

# 划分训练集和测试集
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 构建词汇表
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts_train)

# 将文本转换为序列
sequences_train = tokenizer.texts_to_sequences(texts_train)

sequences_test = tokenizer.texts_to_sequences(texts_test)

# 填充序列，使长度相同
max_len = max(len(seq) for seq in sequences_train)
sequences_train = pad_sequences(sequences_train, maxlen=max_len)
sequences_test = pad_sequences(sequences_test, maxlen=max_len)
print(len(tokenizer.word_index) + 1)
# 创建模型
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 128, input_length=max_len))
model.add(LSTM(32))
model.add(Dense(1, activation='hard-sigmoid'))

# 编译模型
optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])



# 训练模型
model.fit(sequences_train, labels_train, batch_size=32, epochs=10, validation_data=(sequences_test, labels_test))

# 在测试集上评估模型
loss, accuracy = model.evaluate(sequences_test, labels_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)
model.save('model.h5')

