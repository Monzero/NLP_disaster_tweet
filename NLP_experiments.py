import pandas as pd
import numpy as np
import os
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional, SimpleRNN, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score
import tensorflow as tf
import matplotlib.pyplot as plt



###############################################################################
#############             Drafting user functions             #################
###############################################################################

def clean_text(text):

    text = text.lower()     # Convert text to lowercase
    text = text.replace('\n', ' ') # Remove newline characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#','', text) # Remove user @ references and '#' from text   
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove punctuations and numbers
    text = re.sub(r'\s+', ' ', text).strip() # Remove multiple spaces
    return text



###############################################################################
#############             Reading Data                        #################
###############################################################################

train     = pd.read_csv('C:/Users/Monil/OneDrive/Desktop/MSDS/07_PML_MSDS_422/Assignment_9/data/train.csv')
test     = pd.read_csv('C:/Users/Monil/OneDrive/Desktop/MSDS/07_PML_MSDS_422/Assignment_9/data/test.csv')

###############################################################################
#########              Data preprocessing                         #############
###############################################################################
#train.head(50) # We observe some NaN in the keyword and the location columns
#train.describe()
#train.isna().sum()

train['keyword'].fillna('NA',inplace=True)
train['location'].fillna('NA',inplace=True)

####    Check sample feeds on both sides of target   ###
#train[train['target']==1]['text'].values[0] #'Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all'
#train[train['target']==0]['text'].values[0] #"What's up man?"


# Apply the cleaning function to the text data
train['text'] = train['text'].apply(clean_text)
test['text'] = test['text'].apply(clean_text)

# Tokenization and Padding
vocab_size        = 10000
embedding_dim     = 128
max_length        = 100
trunc_type        = 'post'
padding_type      = 'post'
oov_tok           = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train['text'])

train_sequences = tokenizer.texts_to_sequences(train['text'])
test_sequences = tokenizer.texts_to_sequences(test['text'])

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Preparing labels
train_labels = train['target'].values

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(train_padded, train_labels, test_size=0.2, random_state=42)




#############################################################################################################
###############                   LSTM Model 3                                            ###################
#############################################################################################################


# Build Model 3
model3 = Sequential([
    Embedding(vocab_size, 128, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dense(64, activation='tanh'),
    #Dense(64),
    #LeakyReLU(alpha=0.01),
    Dense(1, activation='sigmoid')
])

# Compile Model 3
model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train Model 3
history3 = model3.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stop])

# Evaluate Model 3
val_loss3, val_accuracy3 = model3.evaluate(X_val, y_val)
print(f'Model 3 Validation Accuracy: {val_accuracy3:.4f}')

set_threshold = 0.5

# Evaluate Model 3
val_preds3 = (model3.predict(X_val) > set_threshold).astype(int)
val_f1_3 = f1_score(y_val, val_preds3)
print(f'Model 3 Validation F1 Score: {val_f1_3:.4f}')

proba3 = model3.predict(test_padded).flatten()


###############################################################################
########       Plot the probability                        ####################
##############################################################################

plt.figure(figsize=(10, 6))
plt.plot(proba3, label='Predicted Values')
plt.title('Predicted Values')
plt.xlabel('Index')
plt.ylabel('Prediction')
plt.legend()
plt.show()

###############################################################################
########       Make threshold and label                    ####################
##############################################################################


# Predict on test data
predictions3 = (model3.predict(test_padded) > set_threshold).astype(int).flatten()

# Count the number of 0s
count_zeros = np.sum(predictions3 == 0)
# Count the number of 1s
count_ones = np.sum(predictions3 == 1)

print(f"Number of 0s: {count_zeros}")
print(f"Number of 1s: {count_ones}")


# Prepare submission
submission3 = pd.DataFrame({'id': test['id'], 'target': predictions3})
submission3.to_csv('submission_9.csv', index=False)


