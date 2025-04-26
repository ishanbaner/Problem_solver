A simple Seq2seq model which uses Encoder-Decoder LSTM models with Attention.
It takes math word problems as input, replaces the numbers with variables while storing their values, and returns a sequence in the form of a formula.
The solver method further solves the formula using the values of the variables.
To train, I used 1772 MAWPS and 1664 ASDiv word problems. I randomly sampled 3336 problems from the total lot and used the remaining 100 for testing. 
The future plan is to use a larger dataset and switch to transformers.
