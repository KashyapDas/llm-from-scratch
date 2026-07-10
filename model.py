import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from rotary_embedding_torch import RotaryEmbedding

# ==========================================
# 1. Modern Transformer Components
# ==========================================

class SwiGLU(nn.Module):
    """Modern FeedForward replacing standard ReLU/GELU MLPs."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.act = nn.SiLU() # PyTorch native Swish

    def forward(self, x):
        # SwiGLU formula: Down( Act(Gate(x)) * Up(x) )
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class ModernAttention(nn.Module):
    """Grouped Query Attention with RoPE and FlashAttention."""
    def __init__(self, dim, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        
        # Projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        # Third-party RoPE module
        self.rope = RotaryEmbedding(dim=self.head_dim)

    def forward(self, x):
        B, L, D = x.shape
        
        # Project and reshape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE to Queries and Keys
        q = self.rope.rotate_queries_or_keys(q)
        k = self.rope.rotate_queries_or_keys(k)
        
        # Transpose for Attention: [Batch, Heads, SeqLen, HeadDim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # GQA: Repeat K and V to match Q's head count
        num_groups = self.num_heads // self.num_kv_heads
        if num_groups > 1:
            k = k.repeat_interleave(num_groups, dim=1)
            v = v.repeat_interleave(num_groups, dim=1)
        
        # PyTorch Native SDPA (FlashAttention enabled by default if available)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Reshape back and project out
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(attn_out)


class ModernBlock(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, hidden_dim):
        super().__init__()
        # Native RMSNorm (Requires PyTorch >= 2.4)
        self.norm1 = nn.RMSNorm(dim) 
        self.attn = ModernAttention(dim, num_heads, num_kv_heads)
        self.norm2 = nn.RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ModernLanguageModel(nn.Module):
    def __init__(self, vocab_size, dim, depth, num_heads, num_kv_heads, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            ModernBlock(dim, num_heads, num_kv_heads, hidden_dim) for _ in range(depth)
        ])
        self.norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Weight tying (common optimization)
        self.embed.weight = self.lm_head.weight

    def forward(self, x, targets=None):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

# ==========================================
# 2. Data Preparation (Shakespeare)
# ==========================================

def get_shakespeare_data():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Simple character-level tokenizer
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, vocab_size

# ==========================================
# 3. Training Loop
# ==========================================

if __name__ == "__main__":
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # Load data
    data, vocab_size = get_shakespeare_data()
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    # Hyperparameters
    batch_size = 16
    block_size = 128
    dim = 256
    depth = 4
    num_heads = 8
    num_kv_heads = 2 # GQA: 8 query heads, 2 KV heads
    hidden_dim = int(8 * dim / 3) # Standard Llama SwiGLU expansion
    learning_rate = 3e-4
    epochs = 1000

    # Initialize model
    model = ModernLanguageModel(
        vocab_size=vocab_size, dim=dim, depth=depth, 
        num_heads=num_heads, num_kv_heads=num_kv_heads, hidden_dim=hidden_dim
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i : i + block_size] for i in ix])
        y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    # Training
    model.train()
    for iter in range(epochs):
        xb, yb = get_batch("train")
        
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter % 100 == 0:
            print(f"Step {iter} | Loss: {loss.item():.4f}")

    print("Training complete!")

