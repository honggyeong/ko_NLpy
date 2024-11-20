import numpy as np
import pandas as pd
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

okt = Okt()


data = [
    ("배달앱으로 주문하기", "성인"),
    ("키오스크 이용법 배우기", "성인"),
    ("어린이 그림 그리기", "어린이"),
    ("청소년 프로그래밍 교육", "청소년"),
    ("디지털 기기 활용법", "성인"),
    ("스마트폰 사용법", "성인"),
    ("어르신 스마트폰 교육", "노인"),
    ("청소년 영어 교육", "청소년"),
]

df = pd.DataFrame(data, columns=["education_name", "age_group"])






def preprocess_text(text):
    return " ".join(okt.nouns(text))

df["processed_text"] = df["education_name"].apply(preprocess_text)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["processed_text"])
X = tokenizer.texts_to_sequences(df["processed_text"])

X = pad_sequences(X, padding='post')


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["age_group"])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64, input_length=X.shape[1]))
model.add(LSTM(32, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])




history = model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))






def predict_age_group(education_name):
    processed_text = preprocess_text(education_name)
    seq = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(seq, maxlen=X.shape[1], padding='post')
    prediction = model.predict(padded)
    predicted_label = np.argmax(prediction, axis=1)
    return label_encoder.inverse_transform(predicted_label)








"""test"""
new_education_name = "배달앱 주문하기"
predicted_age_group = predict_age_group(new_education_name)
print(f"'{new_education_name}' 교육에 해당하는 연령대는: {predicted_age_group[0]}")
