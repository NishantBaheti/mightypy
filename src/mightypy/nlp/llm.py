"""
LLM
----
"""

import torch
from torch import nn
from tokkit import PyBytePairTokenizer
from mightypy.nlp.dataset import CustomDataset


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dims, device="cpu"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dims, device=device)  # V x E
        self.linear = nn.Linear(embedding_dims, vocab_size, device=device)  # E x V

    def forward(self, X: torch.Tensor):
        embeds = self.embedding(X)  # (B, T, M)
        logits = self.linear(embeds)
        return logits


class Attention(nn.Module):
    def __init__(self, d_model, d_query, d_key, device="cpu"):
        super().__init__()
        self._d_model = d_model
        self._d_query = d_query
        self._d_key = d_key
        self._device = device
        self.w_q = torch.rand(self._d_query, self._d_model, requires_grad=True, device=self._device) * 1e-1
        # alternatively
        # w_q = torch.nn.Linear(d_model, bias=False)
        self.w_k = torch.rand(self._d_key, self._d_model, requires_grad=True, device=self._device) * 1e-1
        self.w_v = torch.rand(self._d_model, self._d_model, requires_grad=True, device=self._device) * 1e-1


    def forward(self, X):
        q = torch.matmul(X, self.w_q.T)
        # alternatively
        # q = w_q(X)
        k = torch.matmul(X, self.w_k.T)
        v = torch.matmul(X, self.w_v.T)

        scaled_dot_product = (q @ k.T) / torch.sqrt(
            torch.tensor(self._d_key, dtype=torch.int32)
        )
        # print(scaled_dot_product.shape)
        scaled_dot_product_probs = torch.softmax(scaled_dot_product, dim=0)
        attn_out = scaled_dot_product_probs @ v
        return attn_out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_query, d_key, masked=False, device="cpu"):
        super().__init__()
        self._n_heads = n_heads
        self._d_model = d_model
        self._d_query = d_query
        self._d_key = d_key
        self._masked = masked
        self._device = device
        self.w_q = nn.Parameter(
            torch.rand(self._n_heads, self._d_query, self._d_model, requires_grad=True, device=self._device)
            * 1e-1
        )  # (H, Q, M)
        self.w_k = nn.Parameter(
            torch.rand(self._n_heads, self._d_key, self._d_model, requires_grad=True, device=self._device)
            * 1e-1
        )  # (H, K, M)
        self.w_v = nn.Parameter(
            torch.rand(self._n_heads, self._d_model, self._d_model, requires_grad=True, device=self._device)
            * 1e-1
        )  # (H, V, M)

    def forward(self, X: torch.Tensor):

        # transpose last two dims (H, Q, M) -> (H, M, Q) := (H, T, M) x (H, M, Q) = (H, T, Q)
        q = torch.bmm(X, self.w_q.transpose(-2, -1))
        # print(X.shape, w_q.shape, q.shape)

        k = torch.bmm(X, self.w_k.transpose(-2, -1))  # (H, T, K)
        v = torch.bmm(X, self.w_v.transpose(-2, -1))  # (H, T, V)

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

class PositionalEmbedding(nn.Module):
    pass

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, context_len= 10_000, scale=10_000, device="cpu"):
        super().__init__()
        self.scale = scale
        self._device = device
        self._context_len = context_len
        self._d_model = d_model
        self.pe = self._positional_encoding()

    def _positional_encoding(self,):
        """
        * Rows - Positions (sentence length, number of tokens in input sentence)
        * Columns - Dimensions (Dimensions of embedding or models)
        """
        p = torch.zeros((self._context_len, self._d_model), device=self._device)
        positions = torch.arange(self._context_len).unsqueeze(1)
        denominator = 1 / torch.pow(
            self.scale, torch.arange(0, self._d_model, 2).unsqueeze(0) / self._d_model
        )

        if self._d_model % 2 == 0:
            end_idx = denominator.shape[1]
        else:
            end_idx = denominator.shape[1] - 1

        # for even indexes
        p[:, 0::2] = torch.sin(positions * denominator)

        # for odd indexes
        p[:, 1::2] = torch.cos(positions * denominator[:, :end_idx])
        return p

    def forward(self, X):
        shape = X.shape
        return X + self.pe[:shape[0],:shape[1]]


class FFN(nn.Module):
    def __init__(self, in_units, out_units, device="cpu"):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_units, in_units * 4, bias=True, device=device)
        self.relu = torch.nn.ReLU().to(device)
        self.linear2 = torch.nn.Linear(in_units * 4, out_units, bias=True, device=device)

    def forward(self, X):
        # X = (H, T, M)
        X = self.linear1(X)
        X = self.relu(X)
        X = self.linear2(X)
        return X


class RepeatBlock(nn.Module):
    def __init__(self, n_heads, d_model, d_key, d_query, device):
        super().__init__()
        self._n_heads = n_heads
        self._d_model = d_model
        self._d_key = d_key
        self._d_query = d_query
        self._device = device
        self._masked_multi_head_attn = MultiHeadAttention(
            self._n_heads, self._d_model, self._d_query, self._d_key, masked=True, device=self._device
        )
        self._multi_head_attn = MultiHeadAttention(
            self._n_heads, self._d_model, self._d_query, self._d_key, masked=False, device=self._device
        )
        self._layer_norm = torch.nn.LayerNorm(self._d_model, device=self._device)
        self._feed_forward = FFN(in_units=self._d_model, out_units=self._d_model, device=self._device)

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
    def __init__(self, n_heads, d_model, d_key, d_query, n_x, vocab_size, device="cpu"):
        super().__init__()
        self._device = device
        self._n_heads = n_heads
        self._d_model = d_model
        self._d_key = d_key
        self._d_query = d_query
        self._n_x = n_x
        self._vocab_size = vocab_size
        self._pe = PositionalEncoding(d_model=self._d_model, device=self._device)
        self._linear = torch.nn.Linear(self._d_model, self._vocab_size, bias=False, device=self._device)
        self._repeat_block = RepeatBlock(
            self._n_heads, self._d_model, self._d_key, self._d_query, device=self._device
        )

    def forward(self, X: torch.Tensor):
        X = self._pe.forward(X)  # (T, M)
        # print(X.shape)

        X = X.repeat(self._n_heads, 1, 1)  # (H, T, M)
        # print(X.shape)
        
        for _ in range(self._n_x):
            X = self._repeat_block(X)
        
        X = self._linear(X) 
        # flatten head and tokens # (H, T, M) -> (H*T, M)
        # X = X.view(-1, self._d_model)

        # we need logits so removing softmax
        # X = torch.softmax(X, dim=0)
        return X


def train(data_loader, llm_model, emb_model, loss_fn, optimizer, epochs, device):
    for _ in range(epochs):
        running_loss = 0.0
        for X_idx, y_idx in data_loader:
            # print(X_idx, y_idx)

            input_embeddings = emb_model.embedding(X_idx).to(device)
            # print(input_embeddings.shape)

            pred_logits = llm_model.forward(input_embeddings)

            pred_logits = pred_logits.permute(0, 2, 1)
            H, C, T = pred_logits.shape

            y_idx = y_idx.repeat(H, T)
            # print(pred_logits.shape, y_idx.shape)
            
            loss = loss_fn(pred_logits, y_idx)  # Compute loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(loss.item())
            
        print(f"Epoch Loss: {running_loss:.6f}")

@torch.no_grad()
def generate(llm_model: LLM, emb_model: Word2Vec, tokenizer: PyBytePairTokenizer, context, max_tokens, top_k, temperature, device="cpu"):
    idxs = torch.tensor(tokenizer.encode(context), dtype=torch.int64).to(device)
    for _ in range(max_tokens):
        embeddings = emb_model.embedding(idxs).to(device)
        logits = llm_model.forward(embeddings)  # (H, T, M)

        # as it is an autoregressive model, we need to get the last token's logits
        final_tokens_logits = logits[:, -1, :]  # Last token's logits from the last layer (H, M)
        
        # # top k sampling
        # # torch.topk returns top k values sorted and their indices for each head
        top_values, _ = torch.topk(final_tokens_logits, top_k, dim=-1)

        # print(final_tokens_logits.shape, top_values.shape, top_indices.shape)
        least_values = top_values[:, [-1]]
        print(top_values)
        final_tokens_logits[final_tokens_logits < least_values] = float("-inf")
        
        print(final_tokens_logits)

        # Apply temperature scaling
        # higher the temperature, more scaled down the logits and more random the output
        final_probs = torch.softmax(final_tokens_logits / temperature, dim=-1)

        next_token = torch.multinomial(final_probs, num_samples=1)

        # print(idxs.shape, next_token.shape)
        
        # Append next token to sequence
        idxs = torch.cat([idxs, next_token.view(-1)], dim=0)
    return tokenizer.decode(idxs.cpu().tolist())  # tokenizer.decode(idxs.cpu().numpy())




# @torch.no_grad()
# def test():
#     pass
