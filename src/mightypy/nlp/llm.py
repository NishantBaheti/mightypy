"""
LLM
----
"""

import torch
from torch import nn
from tokkit import PyBytePairTokenizer
from mightypy.nlp.dataset import CustomDatasetLoader



class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dims, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, embedding_dims)  # V x E
        self.linear = nn.Linear(embedding_dims, vocab_size)  # E x V

    def forward(self, X: torch.Tensor):
        embeds = self.embedding(X)  # (B, T, M)
        logits = self.linear(embeds)
        return logits


class Attention(nn.Module):
    def __init__(self, d_model, d_query, d_key):
        super().__init__()
        self._d_model = d_model
        self._d_query = d_query
        self._d_key = d_key

    def forward(self, X):

        w_q = torch.rand(self._d_query, self._d_model, requires_grad=True) * 1e-1
        # alternatively
        # w_q = torch.nn.Linear(d_model, bias=False)

        w_k = torch.rand(self._d_key, self._d_model, requires_grad=True) * 1e-1
        w_v = torch.rand(self._d_model, self._d_model, requires_grad=True) * 1e-1

        q = torch.matmul(X, w_q.T)
        # alternatively
        # q = w_q(X)
        k = torch.matmul(X, w_k.T)
        v = torch.matmul(X, w_v.T)

        scaled_dot_product = (q @ k.T) / torch.sqrt(
            torch.tensor(self._d_key, dtype=torch.int32)
        )
        # print(scaled_dot_product.shape)
        scaled_dot_product_probs = torch.softmax(scaled_dot_product, dim=0)
        attn_out = scaled_dot_product_probs @ v
        return attn_out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_query, d_key, masked=False):
        super().__init__()
        self._n_heads = n_heads
        self._d_model = d_model
        self._d_query = d_query
        self._d_key = d_key
        self._masked = masked

    def forward(self, X: torch.Tensor):

        w_q = (
            torch.rand(self._n_heads, self._d_query, self._d_model, requires_grad=True)
            * 1e-1
        )  # (H, Q, M)
        w_k = (
            torch.rand(self._n_heads, self._d_key, self._d_model, requires_grad=True)
            * 1e-1
        )  # (H, K, M)
        w_v = (
            torch.rand(self._n_heads, self._d_model, self._d_model, requires_grad=True)
            * 1e-1
        )  # (H, V, M)

        # transpose last two dims (H, Q, M) -> (H, M, Q) := (H, T, M) x (H, M, Q) = (H, T, Q)
        q = torch.bmm(X, w_q.transpose(-2, -1))
        # print(X.shape, w_q.shape, q.shape)

        k = torch.bmm(X, w_k.transpose(-2, -1))  # (H, T, K)
        v = torch.bmm(X, w_v.transpose(-2, -1))  # (H, T, V)

        scaled_dot_product = (q @ k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self._d_key, dtype=torch.int32)
        )  # (H, T, Q) x (H, K, T) := (H, T, T) i.e. Q = K

        if self._masked:
            scaled_dot_product = scaled_dot_product.masked_fill(
                torch.tril(scaled_dot_product) == 0, float("-inf")
            )
        scaled_dot_product_probs = torch.softmax(
            scaled_dot_product, dim=-1
        )  # (H, Q, K)

        # print("A ", scaled_dot_product_probs.shape, scaled_dot_product_probs.sum(dim=-1))
        attn_out = scaled_dot_product_probs @ v
        return attn_out


class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def _positional_encoding(self, n_tokens, d_model, scale=10_000):
        """
        * Rows - Positions (sentence length, number of tokens in input sentence)
        * Columns - Dimensions (Dimensions of embedding or models)
        """
        p = torch.zeros((n_tokens, d_model))
        positions = torch.arange(n_tokens).unsqueeze(1)
        denominator = 1 / torch.pow(
            scale, torch.arange(0, d_model, 2).unsqueeze(0) / d_model
        )

        if d_model % 2 == 0:
            end_idx = denominator.shape[1]
        else:
            end_idx = denominator.shape[1] - 1

        # for even indexes
        p[:, 0::2] = torch.sin(positions * denominator)

        # for odd indexes
        p[:, 1::2] = torch.cos(positions * denominator[:, :end_idx])
        return p

    def forward(self, X):
        n_tokens, d_model = X.shape
        pe = self._positional_encoding(n_tokens, d_model)
        return X + pe


class FFN(nn.Module):
    def __init__(self, in_units, out_units):
        super().__init__()
        self.linear = torch.nn.Linear(in_units, out_units, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        # X = (H, T, M)
        X = self.linear(X)
        X = self.relu(X)
        return X


class RepeatBlock(nn.Module):
    def __init__(self, n_heads, d_model, d_key, d_query):
        super().__init__()
        self._n_heads = n_heads
        self._d_model = d_model
        self._d_key = d_key
        self._d_query = d_query
        self._masked_multi_head_attn = MultiHeadAttention(
            self._n_heads, self._d_model, self._d_query, self._d_key, masked=True
        )
        self._multi_head_attn = MultiHeadAttention(
            self._n_heads, self._d_model, self._d_query, self._d_key, masked=False
        )
        self._layer_norm = torch.nn.LayerNorm(self._d_model)
        self._feed_forward = FFN(in_units=self._d_model, out_units=self._d_model)

    def forward(self, X):

        # X = (H, T, M)

        X_masked_attn_out = self._masked_multi_head_attn(X)
        X = X + X_masked_attn_out
        X = self._layer_norm(X)

        X_attn_out = self._multi_head_attn(X)
        X = X + X_attn_out
        X = self._layer_norm(X)

        X_ffn_out = self._feed_forward(X)
        X = X + X_ffn_out
        X = self._layer_norm(X)
        return X


class LLM(nn.Module):
    def __init__(self, n_heads, d_model, d_key, d_query, n_x, vocab_size):
        super().__init__()
        self._n_heads = n_heads
        self._d_model = d_model
        self._d_key = d_key
        self._d_query = d_query
        self._n_x = n_x
        self._vocab_size = vocab_size
        self._pe = PositionalEncoding()
        self._linear = torch.nn.Linear(self._d_model, self._vocab_size, bias=False)
        self._repeat_block = RepeatBlock(
            self._n_heads, self._d_model, self._d_key, self._d_query
        )

    def forward(self, X: torch.Tensor):
        X = self._pe.forward(X)  # (T, M)
        # print(X.shape)
        X = X.repeat(self._n_heads, 1, 1)  # (H, T, M)
        # print(X.shape)
        for _ in range(self._n_x):
            X = self._repeat_block(X)

        # flatten head and tokens # (H, T, M) -> (H*T, M)
        X = self._linear(X.view(-1, self._d_model))  

        # we need logits so removing softmax
        # X = torch.softmax(X, dim=0)
        return X

    def generate(self, context, max_tokens):
        pass


def train(data_loader, llm_model, emb_model, loss_fn, optimizer, epochs, device):
    for _ in range(epochs):
        running_loss = 0.0
        for X_idx, y_idx in data_loader:
            input_embeddings = emb_model.embedding(X_idx)
            pred_logits = llm_model.forward(input_embeddings)
            # print(pred_logits.shape, y_idx.shape)
            loss = loss_fn(pred_logits.mean(axis=0), y_idx.to(device).view(-1)[0])  # Compute loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(loss.item())
        print(f"Epoch Loss: {running_loss:.6f}")



# @torch.no_grad()
# def test():
#     pass


if __name__ == "__main__":
    
    N_HEADS = 6
    D_MODEL = 100
    D_KEY = 7
    D_QUERY = 7
    N_X = 5
    CONTEXT_LENGTH = 100
    
    EPOCHS = 10
    device = "cpu"

    # input_sentence = "Hello, how are you?"

    tokenizer = PyBytePairTokenizer()
    VOCAB_SIZE = tokenizer.size
    url = "https://raw.githubusercontent.com/NishantBaheti/tokkit/refs/heads/main/datasets/raw/combined.txt"
    dataloader = CustomDatasetLoader(url, tokenizer, context_length=CONTEXT_LENGTH)
    embedding_model = Word2Vec(VOCAB_SIZE, D_MODEL)
    # for X, y in dataloader:
    #     input_tokens = X
    #     break
    
    llm_model = LLM(
        n_heads = N_HEADS,
        d_model = D_MODEL,
        d_key = D_KEY,
        d_query = D_QUERY,
        n_x = N_X,
        vocab_size = VOCAB_SIZE
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(llm_model.parameters(), lr=0.001)

    # out = model.forward(input_embeddings)
    # print(out.shape)
    # print(out.sum(dim=0))

    train(dataloader, llm_model, embedding_model, loss_fn, optimizer, EPOCHS, device)

