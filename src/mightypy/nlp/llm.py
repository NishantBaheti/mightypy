"""
LLM
----
"""

import torch
from torch import nn
from tokkit import PyBytePairTokenizer

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dims, device="cpu"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dims, device=device)  # V x E
        self.linear = nn.Linear(embedding_dims, vocab_size, device=device)  # E x V

    def forward(self, X: torch.Tensor):
        embeds = self.embedding(X)  # (B, T, M)
        logits = self.linear(embeds)
        return logits


class BatchMultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_value, d_key, masked=False, device="cpu"):
        # From attention is all you need paper secton 3.2.2 Multi-Head Attention
        super().__init__()
        self._n_heads = n_heads
        self._d_model = d_model
        self._d_v = d_value
        self._d_k = d_key
        self._masked = masked
        self._device = device 

        # TODO : this can be optimized by nn.Linear layer
        self.w_q = nn.Parameter(
            torch.rand(self._n_heads, self._d_k, self._d_model, requires_grad=True, device=self._device)
            * 1e-1
        )  # (H, K, M)
        self.w_k = nn.Parameter(
            torch.rand(self._n_heads, self._d_k, self._d_model, requires_grad=True, device=self._device)
            * 1e-1
        )  # (H, K, M)
        self.w_v = nn.Parameter(
            torch.rand(self._n_heads, self._d_v, self._d_model, requires_grad=True, device=self._device)
            * 1e-1
        )  # (H, V, M)
        self.w_o = nn.Parameter(
            torch.rand(self._n_heads * self._d_v, self._d_model, requires_grad=True, device=self._device)
            * 1e-1
        ) # (H*V, M)

    def forward(self, X: torch.Tensor):

        X = X.unsqueeze(1) # (B, T, M) -> (B, 1, T, M)

        # Q transpose last two dims (H, K, M) -> (H, M, K) 
        # (B, 1, T, M) x (H, M, K) = (B, H, T, K)
        q = X @ self.w_q.transpose(-2, -1)
        # print(X.shape, w_q.shape, q.shape)

        k = X @ self.w_k.transpose(-2, -1) # (B, H, T, K)
        v = X @ self.w_v.transpose(-2, -1)  # (B, H, T, V)

        # scaled dot product
        # (B, H, T, K) x (B, H, K, T) := (B, H, T, T) 
        scaled_dot_product = (q @ k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self._d_k, dtype=torch.float32, device=self._device)
        )

        if self._masked:
            scaled_dot_product = scaled_dot_product.masked_fill(
                torch.tril(scaled_dot_product) == 0, float("-inf")
            )

        scaled_dot_product_probs = torch.softmax(
            scaled_dot_product, dim=-1
        )  # (B, H, T, T)

        # print("A ", scaled_dot_product_probs.shape, scaled_dot_product_probs.sum(dim=-1))

        attn_out = scaled_dot_product_probs @ v # (B, H, T, T) x (B, H, T, V) = (B, H, T, V)
        attn_out = attn_out.permute(0, 2, 1, 3) # (B, H, T, V) -> (B, T, H, V)
        B, T, H, V = attn_out.shape
        out_projection = attn_out.reshape(B, T, H*V) @ self.w_o # (B, T, H * V) x (H * V, M)= (B, T, M)
        return out_projection

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

    def _positional_encoding(self):
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
        _, row, col = X.shape # (B, T, M)
        return X + self.pe[:row,:col]


class FFN(nn.Module):
    def __init__(self, in_units, out_units, device="cpu"):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_units, in_units * 4, bias=True, device=device)
        self.relu = torch.nn.ReLU().to(device)
        self.linear2 = torch.nn.Linear(in_units * 4, out_units, bias=True, device=device)

    def forward(self, X):
        # X = (B, T, M)
        X = self.linear1(X)
        X = self.relu(X)
        X = self.linear2(X)
        return X


class RepeatBlock(nn.Module):
    def __init__(self, n_heads, d_model, d_key, d_value, device):
        super().__init__()
        self._n_heads = n_heads
        self._d_model = d_model
        self._d_key = d_key
        self._d_value = d_value
        self._device = device
        self._masked_multi_head_attn = BatchMultiHeadAttention(
            self._n_heads, self._d_model, self._d_value, self._d_key, masked=True, device=self._device
        )
        self._multi_head_attn = BatchMultiHeadAttention(
            self._n_heads, self._d_model, self._d_value, self._d_key, masked=False, device=self._device
        )
        self._layer_norm = torch.nn.LayerNorm(self._d_model, device=self._device)
        self._feed_forward = FFN(in_units=self._d_model, out_units=self._d_model, device=self._device)

    def forward(self, X):

        # X: (B, T, M)

        X_masked_attn_out = self._masked_multi_head_attn(X) # (B, T, M)
        X = X + X_masked_attn_out # (B, T, M)
        X = self._layer_norm(X) # (B, T, M)

        X_attn_out = self._multi_head_attn(X) # (B, T, M)
        X = X + X_attn_out # (B, T, M)
        X = self._layer_norm(X) # (B, T, M)

        X_ffn_out = self._feed_forward(X) # (B, T, M)
        X = X + X_ffn_out # (B, T, M)
        X = self._layer_norm(X) # (B, T, M)
        return X


class LLM(nn.Module):
    def __init__(self, n_heads, d_model, d_key, d_value, n_x, vocab_size, device="cpu"):
        super().__init__()
        self._device = device
        self._n_heads = n_heads
        self._d_model = d_model
        self._d_key = d_key
        self._d_value = d_value
        self._n_x = n_x
        self._vocab_size = vocab_size
        self._pe = PositionalEncoding(d_model=self._d_model, device=self._device)
        self._linear = torch.nn.Linear(self._d_model, self._vocab_size, bias=False, device=self._device)
        self._repeat_block = RepeatBlock(
            self._n_heads, self._d_model, self._d_key, self._d_value, device=self._device
        )

    def forward(self, X: torch.Tensor):
        X = self._pe.forward(X)  # (B, T, M)
        # print(X.shape)

        # Repeat is an unncessary step as Multihead Attention Layer must handle this inherently
        # IT WILL CAUSE A BIG PROBLEM IN LATER STAGES
        # Where we want last layer's output to be a single token, (this will give token proabability from each head :-( )
        # X = X.repeat(self._n_heads, 1, 1)  # (H, T, M)
        # # print(X.shape)
        
        for _ in range(self._n_x):
            X = self._repeat_block(X) # (B, T, M)
        
        X = self._linear(X) # (B, T, V)

        # we need logits so removing softmax
        # X = torch.softmax(X, dim=0)
        return X[:, [-1], :]


def train(data_loader, llm_model: LLM, emb_model, loss_fn, optimizer, epochs, device):
    for _ in range(epochs):
        running_loss = 0.0
        for X_idx, y_idx in data_loader:
            # print(X_idx.shape, y_idx.shape) (B, T), (B, 1)

            input_embeddings = emb_model.embedding(X_idx).to(device) # (B, T, M)
            # print(input_embeddings.shape)

            pred_logits = llm_model.forward(input_embeddings) # (B, T, C) -> (B, 1, C)

            # (B, T, C) -> (B, C, T)
            pred_logits = pred_logits.permute(0, 2, 1) # (B, C, 1)

            # B, C, T = pred_logits.shape
            # y_idx = y_idx.repeat(B, T)

            # print(pred_logits.shape, y_idx.shape)
            
            loss = loss_fn(pred_logits, y_idx)  # Compute loss (B, C, 1) , (B, 1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # print(loss.item())
            
        print(f"Epoch Loss: {running_loss:.6f}")

@torch.no_grad()
def generate(llm_model: LLM, emb_model: Word2Vec, tokenizer: PyBytePairTokenizer, context, max_tokens, top_k, temperature, device="cpu"):
    idxs = torch.tensor(tokenizer.encode(context), dtype=torch.int64).to(device)
    for _ in range(max_tokens):
        embeddings = emb_model.embedding(idxs).to(device) # (T, M)
        # print(embeddings.shape)

        logits = llm_model.forward(embeddings.unsqueeze(0))  # (B, T, M) := (1, T, M) 

        # as it is an autoregressive model, we need to get the last token's logits
        final_tokens_logits = logits[:, -1, :]  # Last token's logits from the last layer = (1, M)
        # print(final_tokens_logits.shape)
        
        # # top k sampling
        # # torch.topk returns top k values sorted and their indices for each head
        top_values, _ = torch.topk(final_tokens_logits, top_k, dim=-1)

        # print(final_tokens_logits.shape, top_values.shape, top_indices.shape)
        least_values = top_values[:, [-1]]
        # print(top_values)

        final_tokens_logits[final_tokens_logits < least_values] = float("-inf") # (1, M)
        
        # print(final_tokens_logits)

        # Apply temperature scaling
        # higher the temperature, more scaled down the logits and more random the output
        final_probs = torch.softmax(final_tokens_logits / temperature, dim=-1)  # (1, M)

        next_token = torch.multinomial(final_probs, num_samples=1)  # (1, 1)

        # print(idxs.shape, next_token.shape)
        # Append next token to sequence
        idxs = torch.cat([idxs, next_token.view(-1)], dim=0)
    return tokenizer.decode(idxs.cpu().tolist())  # tokenizer.decode(idxs.cpu().numpy())
