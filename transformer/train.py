
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