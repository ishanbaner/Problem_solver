import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Attention
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
df = pd.read_csv("/home/ishan/Downloads/mawps_train.csv")
total_problem_phrases = list(df['Question'])
total_formula_phrases = list(df['Equation'])

# Add start and end tokens to the formula phrases
formula_phrases = [f'<start> {phrase} <end>' for phrase in total_formula_phrases]

# Tokenization and padding function
def tokenize_and_pad(sequences, max_len):
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(sequences)
    sequences = tokenizer.texts_to_sequences(sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences, tokenizer

# Get maximum sequence lengths
max_len_problem = max(len(seq.split()) for seq in total_problem_phrases)
max_len_formula = max(len(seq.split()) for seq in formula_phrases)

# Tokenize and pad the input and output sequences
input_sequences, input_tokenizer = tokenize_and_pad(total_problem_phrases, max_len_problem)
output_sequences, output_tokenizer = tokenize_and_pad(formula_phrases, max_len_formula)

# Model parameters
embedding_dim = 256
lstm_units = 256
num_encoder_tokens = len(input_tokenizer.word_index) + 1
num_decoder_tokens = len(output_tokenizer.word_index) + 1

# Encoder model
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(num_encoder_tokens, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(lstm_units, return_state=True, return_sequences=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder model
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(num_decoder_tokens, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

# Attention mechanism
attention = Attention()
context_vector = attention([decoder_outputs, encoder_outputs])

# Concatenate context vector and decoder output
decoder_combined_context = Concatenate(axis=-1)([context_vector, decoder_outputs])

# Final Dense layer for prediction
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_combined_context)

# Define the Seq2Seq model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# One-hot encode the output sequences for training
def one_hot_encode(sequences, num_classes):
    return np.array([tf.keras.utils.to_categorical(seq, num_classes=num_classes) for seq in sequences])

decoder_target_data = one_hot_encode(output_sequences[:, 1:], num_decoder_tokens)

# Train the model
model.fit(
    [input_sequences, output_sequences[:, :-1]],
    decoder_target_data,
    batch_size=64,
    epochs=50,  # Increase the epochs for better training
    validation_split=0.2
)

# Inference models for translation
# Encoder model
encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

# Decoder model for inference
decoder_state_input_h = Input(shape=(lstm_units,))
decoder_state_input_c = Input(shape=(lstm_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Reuse the encoder outputs
decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs)
decoder_states_inf = [state_h_inf, state_c_inf]

# Attention mechanism for inference
context_vector_inf = attention([decoder_outputs_inf, encoder_outputs])
decoder_combined_context_inf = Concatenate(axis=-1)([context_vector_inf, decoder_outputs_inf])

# Final Dense layer for inference
decoder_outputs_inf = decoder_dense(decoder_combined_context_inf)

decoder_model = Model(
    [decoder_inputs, encoder_outputs] + decoder_states_inputs,
    [decoder_outputs_inf] + decoder_states_inf)

# Improved decode_sequence function
def decode_sequence(input_seq, max_len=100, skip_unk=True, verbose=False):  
    """
    Decode the input sequence using the trained seq2seq model.

    Args:
    - input_seq: The input sequence to decode.
    - max_len: Maximum length for decoding to prevent infinite loops.
    - skip_unk: Whether to skip <unk> tokens during decoding.
    - verbose: Whether to print out probabilities at each step (for debugging purposes).

    Returns:
    - decoded_sentence: The final predicted sentence without <start> and <end> tokens.
    """
    # Run the encoder model to get the initial hidden state
    encoder_outputs_inf, h, c = encoder_model.predict(input_seq)

    # Initialize the target sequence with the start token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = output_tokenizer.word_index['<start>']

    stop_condition = False
    decoded_sentence = ''
    iteration_count = 0

    while not stop_condition:
        # Predict the next token and update states
        output_tokens, h, c = decoder_model.predict([target_seq, encoder_outputs_inf, h, c])

        # Get the token with the highest probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = output_tokenizer.index_word.get(sampled_token_index, '<unk>')
        prob = output_tokens[0, -1, sampled_token_index]  # Get the probability of the selected token

        if verbose:
            print(f"Sampled word: {sampled_word}, Probability: {prob:.4f}")

        # Add the sampled word to the decoded sentence, skip <unk> if skip_unk is enabled
        if sampled_word not in ['<start>', '<unk>'] or not skip_unk:
            decoded_sentence += ' ' + sampled_word

        # End the loop if <end> token or max length is reached
        if sampled_word == '<end>' or iteration_count >= max_len:
            stop_condition = True

        # Update the target sequence (of length 1)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        iteration_count += 1

    # Clean up <start> and <end> tokens
    return decoded_sentence.replace('<start>', '').replace('<end>', '').strip()


'''
# Function to decode sequences
# Function to decode sequences with an added iteration limit
def decode_sequence(input_seq, max_len=100):  # Adding max_len to prevent infinite loop
    encoder_outputs_inf, h, c = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = output_tokenizer.word_index['<start>']

    stop_condition = False
    decoded_sentence = ''
    max_iterations = max_len  # Set a reasonable limit to the number of iterations

    iteration_count = 0  # Keep track of iterations

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq, encoder_outputs_inf, h, c])

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = output_tokenizer.index_word.get(sampled_token_index, '')

        if sampled_word:
            decoded_sentence += ' ' + sampled_word

        # Check if stop condition is met: either <end> token or max iterations
        if sampled_word == '<end>' or iteration_count >= max_iterations:
            stop_condition = True

        # Update the target sequence (of length 1)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        iteration_count += 1  # Increment iteration count

    return decoded_sentence.replace('<start>', '').replace('<end>', '').strip()
'''
# Function to translate input sentences
def translate_input(input_sentence):
    input_seq = pad_sequences(input_tokenizer.texts_to_sequences([input_sentence]), maxlen=max_len_problem)
    decoded_sentence = decode_sequence(input_seq)
    return decoded_sentence

# Test the translation function
r=random.randint(0,1700)
test_sentence = total_problem_phrases[r]  # Replace with your test input
translated_sentence = translate_input(test_sentence)

print(f'Input: {test_sentence}')
print(f'Translation: {translated_sentence}')
print("Correct:",total_formula_phrases[r])
