#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd

# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# Check the column names to ensure they are as expected
print(data.columns)

# Drop unnecessary columns (Unnamed columns)
data = data[['class', 'message']]

# Rename the columns for clarity
data.columns = ['label', 'text']

# Map labels to binary values (ham = 0, spam = 1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Inspect the cleaned dataset
print(data.head())


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Initialize the Tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

# Convert text to sequences (tokens)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding sequences to ensure equal length
max_len = 100  # You can adjust this depending on your dataset
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

# Check the shape of the processed data
print("Training data shape:", X_train_pad.shape)
print("Testing data shape:", X_test_pad.shape)


# In[10]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Define the LSTM model
model = Sequential()

# Embedding layer: Maps words to vectors of fixed size
model.add(Embedding(input_dim=5000, output_dim=64, input_length=max_len))

# LSTM layer: Processes the sequential data
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))

# Fully connected layer: For binary classification (ham vs spam)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Summary of the model
model.summary()


# In[11]:


# Train the model
history = model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_pad, y_test)
print("Test Accuracy:", accuracy)


# In[12]:


from sklearn.metrics import classification_report, confusion_matrix

# Predictions
y_pred = model.predict(X_test_pad)
y_pred = (y_pred > 0.5).astype(int)  # Thresholding at 0.5 for binary classification

# Classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[ ]:




