
#!/usr/bin/env python
# edgar_allen_poe_text_generator.py

# Jeremy Chow
# 7/19/2019


# Commandline arg parsing
import sys
# Standard manipulations
import numpy as np
from numpy import array
# Neural Net Preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Neural Net Layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding

from pickle import load

# Define model
def load_text_gen_model():
	vocab_size = 15713
	train_len = 19
	model_3 = Sequential([
	    Embedding(vocab_size+1, 50, input_length=train_len),
	    LSTM(150, return_sequences=True),
	    LSTM(150),
	    Dense(150, activation='relu'),
	    Dense(vocab_size, activation='softmax')
	])

	# Load weights
	return model_3

def gen(seq,max_len = 20):
    ''' Generates a sequence given a string seq using specified model until the total sequence length
    reaches max_len'''
    model = load_text_gen_model()
    model.load_weights('models/model_3_weights_colab.hdf5')
    print("Output: ")
    # Tokenize the input string
    with open("models/tokenizer.pkl", "rb") as f:
    	tokenizer = load(f)
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    tokenized_sent = tokenizer.texts_to_sequences([seq])
    max_len = max_len+len(tokenized_sent[0])
    # If sentence is not as long as the desired sentence length, we need to 'pad sequence' so that
    # the array input shape is correct going into our LSTM. the `pad_sequences` function adds 
    # zeroes to the left side of our sequence until it becomes 19 long, the number of input features.
    while len(tokenized_sent[0]) < max_len:
        padded_sentence = pad_sequences(tokenized_sent[-19:],maxlen=19)
        op = model.predict(np.asarray(padded_sentence).reshape(1,-1))
        tokenized_sent[0].append(op.argmax()+1)
        
    return " ".join(map(lambda x : reverse_word_map[x],tokenized_sent[0]))
def process_input(user_input):
	if len(user_input) == 1:
		print('Script takes form: edgar_allen_poe_text_generator.py [string] [number_of_words_to_predict]')
		return
	if not isinstance(user_input[1],str):
		print("Not a string!")
		return
	if len(user_input)>2:
		print(gen(user_input,int(user_input[2])))
	else:
		print(gen(user_input))


if __name__ == "__main__":
	
	process_input(sys.argv)
