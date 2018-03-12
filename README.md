# pytorch-rnet
An RNET implementation for question answering in PyTorch . https://www.microsoft.com/en-us/research/publication/mrc/


# Model Dimensions

R_NET


HIDDEN STATES -> hidden state of question and passage will be zero initially, also zero for gated and self matching 


1st step is BIRNN for both passage and question
So dimensions are 
Embeddings for questions and passage is 300 
So question is => question_len x 300
So passage is => passage_len x 300

After BI RNN

u_question => question_len x 150
u_passage => passage_len x 150 

(150 as its a bi directional rnn)

After gated attention based recurrent network 

So hidden state of rnn is 2 , 1 , 75

For tanh all of them are of size 1 , 75
c_t is 1, 75

final_input size is 1, 225 ( as u_question + c_t )

So input for rnn is of size 1, 225 and hidden state is of size 2, 1 ,75

And output to the next network will be hidden state of rnn ,
But as shape is 2,1,75

We reshape to get it as 1, 150

HENCE

v => passage_len x 150


After self matching network



So hidden state of rnn is 2 , 1 , 75
Reshape that to 1, 150

For tanh all of them are of size 1 , 75
c_t is 1, 75

final_input size is 1, 225 ( as u_question + c_t )

So input for rnn is of size 1, 225 and hidden state is of size 2, 1 ,75

And output to the next network will be hidden state of rnn ,
But as shape is 2,1,75

We reshape to get it as 1, 150

HENCE

h => passage_len x 75

POINTER NETWORK
Hidden layer is pooling of question vector and is of size
1,1,75

a_t => 1 x passage_len c_t => 1,1,75
