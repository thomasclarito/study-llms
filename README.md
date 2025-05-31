# Learning LLMS

This repository holds my code and resources I'm using to learn about large language models.

## Transformer

This section largely follows Andrej Karpathy's lecture: ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY) 

### Notes
We break up text/data into tokens (tokenization). In this example we use individual characters. However there are richer ways to tokenize data: SentencePiece, tiktoken, etc.

When training a transformer we use random chunks, up to a maximum length, of the training set (block size). A transformer will never receive more than block size inputs when predicting the next character. There are multiple examples packed into a chunk. We train it to make predictions at every position of the chunk. At position 1 predict the char at position 2. At position 2, using position 1 and 2 for context, predict character at position 3. And so on, with context as large as the block size. This teaches the tranformer to predict characters with only one character of context to block size characters of context. It is also more efficient.
As we sample chunks of texts, we create many batches of chunks of data for efficiency. The chunks are still processed independent of each other.

#### Self Attention
We want to tokens to "communicate" with each other. We want to couple the tokens such that tokens only communicate with previous tokens. The easiest way to do this is to average the channels of the current tokens and previous tokens. This would result in a feature vector of the token in the context of its history. This way is very lossy, we lose a lot of information. 
To make this implementation we use matrix multiplication. We use a lower triangular matrix of ones, then divide by the sum across each row, we are able to compute the average very quickly. We can use softmax to implemenent this:
    - Start with a matrix of zeros and a lower triangular matrix
    - Use masked fil on the zero matrix with the lower triangular matrix to fill the upper triangular part with -inf
    - Apply softmax on the rows of the matrix (softmax(v) = e^(v(i)) / (sum_{j=1,K} e^(v(i))))
This is a very interesting implementation because we can think of the initial zeros matrix as an affinity matrix between tokens. This tells us how much from the past token do we want to aggregate. In addition by setting the upper triangular part to -inf, we are saying that tokens we will not aggregate tokens these tokens. In other words, the we are implementing that the past cannot communicate with tokens from the future.