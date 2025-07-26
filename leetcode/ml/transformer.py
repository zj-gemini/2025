sentence = "Life is short, eat dessert first"

tokens = sentence.replace(",", "").split()
print("tokens:", tokens)

vocab = sorted(set(tokens))
print("vocab:", vocab)

dc = {s: i for i, s in enumerate(vocab)}
print(dc)

# Next, we use this dictionary to assign an integer index to each word:
import torch

sentence_int = torch.tensor([dc[token] for token in tokens])
print(sentence_int)


# Now, using the integer-vector representation of the input sentence, we can use an embedding layer to encode the inputs into a real-vector embedding.
torch.manual_seed(123)
embed = torch.nn.Embedding(6, 2)
print("embed:", embed.weight)
embedded_sentence = embed(sentence_int).detach()
print("embedded_sentence:", embedded_sentence)
print("embedded_sentence shape:", embedded_sentence.shape)


# Defining the Weight Matrices
# d represents the size of each word vector, x
d = embedded_sentence.shape[1]
print("d:", d)
# Since we are computing the dot-product between the query and key vectors, these two vectors have to contain the same number of elements (dq=dk)
d_q, d_k, d_v = 4, 4, 8
W_query = torch.nn.Parameter(torch.rand(d_q, d))
W_key = torch.nn.Parameter(torch.rand(d_k, d))
W_value = torch.nn.Parameter(torch.rand(d_v, d))

queries = W_query.matmul(embedded_sentence.T).T
keys = W_key.matmul(embedded_sentence.T).T
values = W_value.matmul(embedded_sentence.T).T

print("queries.shape:", queries.shape)
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

#  we compute ω(i,j) as the dot product between the query and key sequences, ω(i,j)=q(i) dot k(j).
omega_2 = queries[1].matmul(keys.T)
print("omega_2:", omega_2.shape)
print(omega_2)

import torch.nn.functional as F

attention_weights_2 = F.softmax(omega_2 / d_k**0.5, dim=0)
print("attention_weights_2:", attention_weights_2.shape)
print(attention_weights_2)

# Finally, we compute the context vector as the weighted sum of the value vectors, where the weights are given by the attention weights.
context_vector_2 = attention_weights_2.matmul(values)

print("context_vector_2:", context_vector_2.shape)
print(context_vector_2)
