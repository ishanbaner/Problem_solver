A simple Seq2seq model which uses Encoder Decoder LSTM models along with Attention.
It takes math word problems as input, replaces the numbers with variables while storing their values as well and returns a sequence in the form of a formula.
The solver method further solves the formula using the values of the variables.
To train, I have used 1772 MAWPS word problems and for testing I have used 250 SigmaDolphin questions.
Future plan is to use a larger dataset and maybe switch to transformers.
