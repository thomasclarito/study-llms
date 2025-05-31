"""
This script teaches us simple tokenization and encoding of text data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)  # For reproducibility


# Read input data
with open("./data/input.txt", "r") as f:
    text = f.read()

# The set of unique characters in the text
chars = sorted(list(set(text)))
# Number of possible characters
vocab_size = len(chars)

# Mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}  # char to index
itos = {i: ch for i, ch in enumerate(chars)}  # index to char
# Encode the text into integers
encode = lambda s: [stoi[c] for c in s]
# Decode integers back to text
decode = lambda l: "".join([itos[i] for i in l])

# Encode the entire text (using pytorch) into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Split the data into train and test sets
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

block_size = 8  # The maximum context length for the model
train_data[: block_size + 1]

# An example of block size, context, and target of the transformer
x = train_data[:block_size]  # Inputs to the transformer
y = train_data[1 : block_size + 1]  # Targets for the transformer
for t in range(block_size):
    context = x[: t + 1]
    target = y[t]
    print(f"Context: {context}, Target: {target}")


# The following code is an example of how to create batches of data of length block size for training
batch_size = 4
block_size = 8


def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else test_data
    # Generate a batch size numbere of random starting block indices
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Create a tensor of shape (batch_size, block_size) for inputs x
    x = torch.stack([data[i : i + block_size] for i in ix])
    # Create a tensor of shape (batch_size, block_size) for targets y
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


xb, yb = get_batch("train")
print("inputs:", xb.shape)
print(xb)
print("targets:", yb.shape)
print(yb)
print("-----")

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, : t + 1]
        target = yb[b, t]
        print(f"Context: {context}, Target: {target}")


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Create a linear layer that maps from vocab_size to vocab_size
        # Input is a token index, output is logits for the next token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # Get the embeddings for the input indices
        # Logits = scores for the next character in the sequence
        # In this case, we predict the next character based on the current character
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # Measures the quality of the logits against the targets
            loss = F.cross_entropy(
                logits, targets
            )  # cross entropy function expects (Batch, Channel, Time) format

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx: (B, T) array of indices in the current context

        # Generate new tokens based on the input indices
        for _ in range(max_new_tokens):
            # Get the predited logits for the current indices
            logits, loss = self(idx)
            # Focus on the last time step
            logits = logits[:, -1, :]  # Logits shape: (B, C)
            # Softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution to get the next token
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append the next token to the input indices
            idx = torch.cat((idx, idx_next), dim=1)  # idx shape: (B, T + 1)
        return idx


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
# Should be (batch_size, block_size, vocab_size)
#           (Batch, Time, Channel)
print("Logits shape:", logits.shape)
print("Loss:", loss)  # Print the loss value

init_idx = torch.zeros((1, 1), dtype=torch.long)  # Start with a single token (0)
print(
    decode(m.generate(init_idx, max_new_tokens=100)[0].tolist())
)  # Generate 100 new tokens from the model

# Training the model
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for step in range(10000):
    # Get a batch of data
    xb, yb = get_batch("train")

    # Evaluate loss
    logits, loss = m(xb, yb)
    # Zero the gradients
    optimizer.zero_grad(set_to_none=True)  # Set to None for better performance
    # Backward pass
    loss.backward()
    # Update the model parameters
    optimizer.step()

    if step % 1000 == 0:
        print(f"Step {step}, Loss: {loss.item()}")  # Print the loss every 1000 steps

print(
    decode(m.generate(init_idx, max_new_tokens=300)[0].tolist())
)  # Generate 100 new tokens from the model

# Self attention
B, T, C = 4, 8, 2  # Batch size, Time steps, Channels
x = torch.randn(B, T, C)  # Random input tensor
x.shape

# Want x[b, t] = mean_{i<=t} x[b, i]
xbow = torch.zeros(B, T, C)  # Initialize the output tensor
# This is an inefficient implementation. We can use matrix multiplication to make it more efficient.
for b in range(B):
    for t in range(T):
        context = x[b, : t + 1]  # Get the context up to time t
        xbow[b, t] = torch.mean(
            context, 0
        )  # Compute the mean across the time dimension

# Efficient implementation using matrix multiplication
wei = torch.tril(torch.ones(T, T))  # Lower triangular matrix for attention weights
wei = wei / wei.sum(dim=1, keepdim=True)  # Normalize the weights
# Matrix multiplication to get the same result:
# PyTorch does a batch matrix multiply
# (B, T, T) @ (B, T, C) -> (B, T, C)
xbow2 = wei @ x

# xbow should be equal to xbow2
torch.allclose(xbow, xbow2)  # Should be True

# Implementation using softmax
tril = torch.tril(torch.ones(T, T))  # Lower triangular matrix
# Begins as all zeros
wei = torch.zeros(T,T)
# Mask the upper triangular part to -inf
wei = wei.masked_fill(tril == 0, float("-inf"))
# Apply softmax to get attention weights
wei = F.softmax(wei, dim=-1)  # Normalization operation
