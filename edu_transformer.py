#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║           EDU-TRANSFORMER — FULL COURSE                      ║
║                                                              ║
║  One script = complete GPT architecture training             ║
║  Trains in seconds. Learn by experimenting.                  ║
║                                                              ║
║  MODES:                                                      ║
║  1. Training + Generation (basic)                            ║
║  2. Debug Mode (tensor inspection step by step)              ║
║  3. Config Comparison (3 models side by side)                ║
║  4. Interactive Playground (generate anything)               ║
║  5. Exercises (quizzes + tasks)                              ║
║  6. Attention Visualization                                  ║
║  7. Embedding Analysis                                       ║
║  8. Ablation Study (disable components, see what breaks)     ║
║  9. Save/Load Model                                          ║
║  0. Full Course (everything in order)                        ║
║                                                              ║
║  Requirements: pip install torch numpy                       ║
╚══════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import json
import sys

# ============================================================
# CONFIG - EXPERIMENT HERE!
# ============================================================
CONFIG = {
    # ── Architecture ──────────────────────────────────────
    "d_model": 32,          # embedding dimension
                             #   4-8:   too small, model can't learn
                             #   16-32: sweet spot for micro-model
                             #   64-128: more capacity, slower
                             #   768:   GPT-2 Small
                             #   12288: GPT-4 (estimated)

    "n_heads": 2,           # number of attention heads
                             #   MUST divide d_model!
                             #   1: single perspective
                             #   2: two perspectives (e.g. grammar + semantics)
                             #   4+: more relationship aspects
                             #   GPT-2: 12 heads, GPT-3: 96 heads

    "n_layers": 2,          # number of transformer layers (blocks)
                             #   1: simple patterns (bigrams)
                             #   2: complex dependencies
                             #   4+: abstract concepts
                             #   GPT-2: 12, GPT-3: 96

    "d_ff": 64,             # feed-forward inner dimension
                             #   Usually 4× d_model
                             #   This is "compute power" per token
                             #   Too small = bottleneck
                             #   Too large = overfitting on small data

    "dropout": 0.1,         # dropout (regularization)
                             #   0.0: none, risk of overfitting
                             #   0.1: standard
                             #   0.3-0.5: strong regularization
                             #   Disables random neurons during training

    "max_seq_len": 32,      # max sequence length
                             #   GPT-2: 1024, GPT-4: 128000

    # ── Training ──────────────────────────────────────────
    "epochs": 300,          # how many times model sees the data
                             #   50:  underfitting (too few)
                             #   200-300: sweet spot
                             #   1000+: overfitting (memorizes, doesn't learn)

    "lr": 0.001,            # learning rate (step size)
                             #   0.01:   large steps, unstable
                             #   0.001:  standard for Adam
                             #   0.0001: small steps, stable but slow
                             #   Too large = "overshoots" the minimum
                             #   Too small = gets stuck

    "batch_size": 4,        # sentences per batch
                             #   1: SGD, noisy gradient
                             #   4-8: mini-batch, good compromise
                             #   all: full batch, stable but slow

    # ── Generation ────────────────────────────────────────
    "temperature": 0.8,     # controls generation randomness
                             #   0.1: deterministic (always the same)
                             #   0.5: fairly confident
                             #   0.8: creativity balance
                             #   1.0: standard sampling
                             #   1.5+: "creative chaos"
                             #   Technically: divides logits before softmax

    "top_k": 5,             # how many top tokens to consider
                             #   1:   greedy (always the best)
                             #   3-5: limited choice
                             #   0:   no limit (full sampling)

    "max_gen_len": 20,      # max tokens to generate
}

# ============================================================
# TRAINING DATA - ADD YOUR OWN!
# ============================================================
CORPUS = """
the cat sits on the mat
the dog sits on the carpet
the cat likes milk
the dog likes bones
the cat and the dog sit on the couch
the small cat sits on the big mat
the big dog sits on the small carpet
the cat drinks milk from the bowl
the dog eats bones in the garden
the small cat likes warm milk
the big dog likes big bones
the cat sits on the couch and drinks milk
the dog sits on the carpet and eats bones
the cat and the dog like to sleep on the couch
the small cat and the big dog sit on the mat
the cat likes to sleep on the couch
the dog likes to sleep in the garden
the small cat drinks warm milk from the bowl
the big dog eats big bones in the garden
the cat sits on the mat and likes milk
"""


# ============================================================
# TOKENIZER
# ============================================================
class SimpleTokenizer:
    """
    TOKENIZER — converts text to numbers and back.

    ╔══════════════════════════════════════════════════════╗
    ║  "the cat sits on the mat"                           ║
    ║       ↓ encode()                                     ║
    ║  [3, 4, 5, 6, 3, 7]                                 ║
    ║       ↓ model processes numbers                      ║
    ║  [3, 4, 5, 6, 3, 7, 8]                               ║
    ║       ↓ decode()                                     ║
    ║  "the cat sits on the mat carpet"                    ║
    ╚══════════════════════════════════════════════════════╝

    Types of tokenizers:
    - THIS ONE: word-level (1 word = 1 token) — simplest
    - BPE: subword (GPT-2, GPT-3) — "playing" → "play" + "ing"
    - SentencePiece: (LLaMA, T5) — statistical splitting

    Special tokens:
    - <PAD> (0): padding (equalizes sequence lengths)
    - <BOS> (1): Begin Of Sequence (sentence start)
    - <EOS> (2): End Of Sequence (sentence end)
    """
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2}
        self.idx2word = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>"}
        self.vocab_size = 3

    def build_vocab(self, text):
        words = text.lower().split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

    def encode(self, text):
        words = text.lower().split()
        return [self.word2idx.get(w, 0) for w in words]

    def decode(self, ids):
        words = []
        for idx in ids:
            word = self.idx2word.get(idx, "<UNK>")
            if word == "<EOS>":
                break
            if word not in ("<PAD>", "<BOS>"):
                words.append(word)
        return " ".join(words)

    def print_vocab(self):
        print(f"\n📚 Vocabulary ({self.vocab_size} tokens):")
        print("   ID → Word")
        print("   " + "─" * 20)
        for idx in sorted(self.idx2word.keys()):
            print(f"   {idx:3d} → '{self.idx2word[idx]}'")


# ============================================================
# POSITIONAL ENCODING
# ============================================================
class PositionalEncoding(nn.Module):
    """
    POSITIONAL ENCODING — adds position information.

    ╔══════════════════════════════════════════════════════╗
    ║  PROBLEM: Attention treats tokens as a SET,          ║
    ║  not a SEQUENCE. It doesn't know "cat" comes         ║
    ║  before "sits".                                      ║
    ║                                                      ║
    ║  SOLUTION: Add unique "positional DNA"               ║
    ║  to each token.                                      ║
    ║                                                      ║
    ║  Position 0: [sin(0), cos(0), sin(0), cos(0), ...]  ║
    ║  Position 1: [sin(1), cos(1), sin(0.01), cos(0.01)] ║
    ║  Position 2: [sin(2), cos(2), sin(0.02), cos(0.02)] ║
    ║                                                      ║
    ║  Each position has a UNIQUE sinusoidal pattern.      ║
    ║  The model LEARNS to interpret it.                   ║
    ╚══════════════════════════════════════════════════════╝

    Formulas:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Why sinusoids?
    - Unique for each position
    - Model can easily learn "position X is 3 tokens after Y"
      (because sin(a+b) is a linear combination of sin(a) and cos(b))
    - Works for sequences longer than seen in training

    ALTERNATIVE: Learned positional embeddings (GPT-2)
    - Separate nn.Embedding for positions
    - Simpler, but doesn't generalize beyond seen lengths
    """
    def __init__(self, d_model, max_len=128, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.enabled = True  # for ablation study

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if self.enabled:
            x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================
# MULTI-HEAD SELF-ATTENTION
# ============================================================
class MultiHeadAttention(nn.Module):
    """
    MULTI-HEAD SELF-ATTENTION — the heart of the transformer!

    ╔══════════════════════════════════════════════════════╗
    ║  Question: "For token X, which other tokens          ║
    ║            should I pay attention to?"                ║
    ║                                                      ║
    ║  Example: "the cat sits on the big mat"              ║
    ║                                                      ║
    ║  Token "sits" looks at:                              ║
    ║    "cat"  → 0.6 (WHO sits? → subject!)               ║
    ║    "sits" → 0.2 (itself)                             ║
    ║    "on"   → 0.1                                      ║
    ║    "mat"  → 0.1 (WHERE it sits?)                     ║
    ╚══════════════════════════════════════════════════════╝

    Q-K-V mechanism (database analogy):
    ┌─────────────────────────────────────────────────────┐
    │ Query (Q) = "What am I looking for?"                │
    │   → Like a search engine query                      │
    │                                                     │
    │ Key (K) = "What do I offer?"                        │
    │   → Like a page title                               │
    │                                                     │
    │ Value (V) = "What content do I carry?"              │
    │   → Like page content                               │
    │                                                     │
    │ score = Q·K (how well query matches the offer)      │
    │ output = score × V (weighted content)               │
    └─────────────────────────────────────────────────────┘

    MULTI-HEAD = we do this N times in parallel:
    ┌──────────────────────────────────────────────────────┐
    │ Head 1: may look at GRAMMATICAL relationships       │
    │         (subject-verb)                               │
    │                                                     │
    │ Head 2: may look at SEMANTIC relationships          │
    │         (cat-milk, dog-bones)                        │
    │                                                     │
    │ Results are combined → fuller picture                │
    └──────────────────────────────────────────────────────┘

    Formula: Attention(Q,K,V) = softmax(Q @ K^T / √d_k) @ V

    Why / √d_k?
    - Without scaling, dot-product grows with dimension
    - Large values → softmax gives ~[0, 0, 1, 0, 0]
    - Too "sharp" — gradient vanishes
    - √d_k normalizes this
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})!"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.enabled = True  # for ablation study

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None

    def forward(self, x, mask=None):
        if not self.enabled:
            return x

        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        self.attn_weights = attn.detach()
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        return self.W_o(context)


# ============================================================
# FEED-FORWARD NETWORK
# ============================================================
class FeedForward(nn.Module):
    """
    FEED-FORWARD NETWORK — each token's "brain".

    ╔══════════════════════════════════════════════════════╗
    ║  Attention says: "look at these tokens"              ║
    ║  FFN says: "here's what that means"                  ║
    ║                                                      ║
    ║  d_model → d_ff → d_model                           ║
    ║  32      → 64   → 32                                ║
    ║  (info)  → (expand + ReLU) → (compress)             ║
    ║                                                      ║
    ║  Each token processed INDEPENDENTLY.                 ║
    ║  (unlike Attention, which connects tokens)           ║
    ╚══════════════════════════════════════════════════════╝

    Why 4× expansion?
    - Gives "workspace" for computation
    - Empirically: 4× works well
    - GPT-2: d_model=768, d_ff=3072 (4×)

    ReLU vs GELU:
    - ReLU: max(0, x) — simpler, faster
    - GELU: x·Φ(x) — smoother, used in GPT-2+
    - We use ReLU here for clarity
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.enabled = True
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if not self.enabled:
            return x
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ============================================================
# TRANSFORMER BLOCK
# ============================================================
class TransformerBlock(nn.Module):
    """
    TRANSFORMER BLOCK — one "floor" of the building.

    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║  Input ──┐                                           ║
    ║          ├──→ LayerNorm → Attention → Dropout ──┐    ║
    ║          └──────────────────────────────────────+──→  ║
    ║                                                │     ║
    ║  ──────────┐                                   │     ║
    ║            ├──→ LayerNorm → FFN → Dropout ─────┐    ║
    ║            └───────────────────────────────────+──→  ║
    ║                                               Output ║
    ╚══════════════════════════════════════════════════════╝

    RESIDUAL CONNECTION (x + sublayer(x)):
    - "Highway" for gradients
    - Model can decide "this layer adds nothing" → passes x through
    - Without this, deep networks don't train
    - Invented in ResNet (2015)

    LAYER NORM vs BATCH NORM:
    - BatchNorm: normalizes across batch (popular in CV)
    - LayerNorm: normalizes across dimension (standard in NLP)
    - Why LayerNorm? Because sequences have different lengths

    PRE-NORM vs POST-NORM:
    - Post-Norm: norm(x + sublayer(x)) — original paper
    - Pre-Norm: x + sublayer(norm(x)) — more stable, used in GPT-2+
    - We use Pre-Norm here
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_out)

        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_out)

        return x


# ============================================================
# MICRO-TRANSFORMER (full model)
# ============================================================
class MicroTransformer(nn.Module):
    """
    COMPLETE AUTOREGRESSIVE LANGUAGE MODEL.

    ╔══════════════════════════════════════════════════════╗
    ║  "the cat sits on" → ???                             ║
    ║                                                      ║
    ║  [3, 4, 5, 6] ← Token IDs                           ║
    ║      ↓                                               ║
    ║  Embedding: ID → 32-dimensional vector               ║
    ║      ↓                                               ║
    ║  + Positional Encoding (add position info)           ║
    ║      ↓                                               ║
    ║  TransformerBlock 1 (attention + FFN)                 ║
    ║      ↓                                               ║
    ║  TransformerBlock 2 (attention + FFN)                 ║
    ║      ↓                                               ║
    ║  LayerNorm                                           ║
    ║      ↓                                               ║
    ║  Linear → [0.01, 0.02, 0.7, 0.05, ...] ← logits    ║
    ║                          ↑                           ║
    ║                        "mat" ← highest logit         ║
    ╚══════════════════════════════════════════════════════╝

    AUTOREGRESSIVE = generates token by token:
    1. "the cat"           → predicts "sits"
    2. "the cat sits"      → predicts "on"
    3. "the cat sits on"   → predicts "the"
    4. "the cat sits on the" → predicts "mat"
    """
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 max_seq_len, dropout, debug=False):
        super().__init__()

        self.d_model = d_model
        self.debug = debug

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

        self.total_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_causal_mask(self, seq_len):
        """
        CAUSAL MASK — prevents "looking into the future".

        When generating "the cat sits on ___":
        - "the"  CAN see: [the]
        - "cat"  CAN see: [the, cat]
        - "sits" CAN see: [the, cat, sits]
        - "on"   CAN see: [the, cat, sits, on]
        - "___"  CAN see: [the, cat, sits, on, ___]

        Matrix (1=visible, 0=blocked):
        ┌──────────────────────────┐
        │  1  0  0  0  0   ← the  │
        │  1  1  0  0  0   ← cat  │
        │  1  1  1  0  0   ← sits │
        │  1  1  1  1  0   ← on   │
        │  1  1  1  1  1   ← ___  │
        └──────────────────────────┘

        WITHOUT MASK: model "cheats" — sees the answer!
        (Try disabling it in the ablation study)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        return mask

    def forward(self, x, use_causal_mask=True):
        seq_len = x.size(1)

        if use_causal_mask:
            mask = self.make_causal_mask(seq_len).to(x.device)
        else:
            mask = None

        # ── Debug: show each step ──
        if self.debug:
            print(f"\n{'═'*60}")
            print(f"🔍 DEBUG: Forward Pass")
            print(f"{'═'*60}")
            print(f"  Input token IDs: {x[0].tolist()}")
            print(f"  Shape: {x.shape}  (batch_size, seq_len)")

        # Embedding
        h = self.embedding(x) * math.sqrt(self.d_model)
        if self.debug:
            print(f"\n  📊 After Embedding:")
            print(f"     Shape: {h.shape}  (batch, seq_len, d_model)")
            print(f"     Values[0][0][:8]: {[round(v, 3) for v in h[0][0][:8].tolist()]}")
            print(f"     × √d_model = × {math.sqrt(self.d_model):.2f} (scaling)")

        # Positional encoding
        h = self.pos_encoding(h)
        if self.debug:
            print(f"\n  📊 After Positional Encoding:")
            print(f"     Shape: {h.shape}")
            print(f"     Values[0][0][:8]: {[round(v, 3) for v in h[0][0][:8].tolist()]}")

        # Transformer blocks
        for i, layer in enumerate(self.layers):
            h = layer(h, mask)
            if self.debug:
                print(f"\n  📊 After TransformerBlock {i+1}:")
                print(f"     Shape: {h.shape}")
                print(f"     Mean: {h.mean().item():.4f}, Std: {h.std().item():.4f}")
                if layer.attention.attn_weights is not None:
                    aw = layer.attention.attn_weights[0]
                    print(f"     Attention weights shape: {aw.shape} "
                          f"(n_heads, seq_len, seq_len)")

        # Final norm + projection
        h = self.final_norm(h)
        logits = self.output_proj(h)

        if self.debug:
            print(f"\n  📊 Logits (output):")
            print(f"     Shape: {logits.shape}  (batch, seq_len, vocab_size)")
            probs = F.softmax(logits[0, -1, :], dim=-1)
            top5 = torch.topk(probs, 5)
            print(f"     Top-5 predictions for last position:")
            for prob, idx in zip(top5.values, top5.indices):
                print(f"       {prob.item():.3f} → token {idx.item()}")
            print(f"{'═'*60}")

        return logits

    @torch.no_grad()
    def generate(self, tokenizer, prompt, max_len=20,
                 temperature=0.8, top_k=0, verbose=False):
        """
        TEXT GENERATION — autoregressive.

        ╔══════════════════════════════════════════════════╗
        ║  Step 1: "the cat"                               ║
        ║    → model says: P("sits")=0.4, P("likes")=0.3  ║
        ║    → we sample: "sits"                           ║
        ║                                                  ║
        ║  Step 2: "the cat sits"                          ║
        ║    → model: P("on")=0.7, P("and")=0.1           ║
        ║    → we sample: "on"                             ║
        ║                                                  ║
        ║  Step 3: "the cat sits on"                       ║
        ║    → model: P("the")=0.8, P("a")=0.1            ║
        ║    → we sample: "the"                            ║
        ║                                                  ║
        ║  Result: "the cat sits on the mat"               ║
        ╚══════════════════════════════════════════════════╝

        TEMPERATURE:
        logits = [2.0, 1.0, 0.5] → softmax → [0.57, 0.28, 0.15]

        temp=0.5: logits/0.5 = [4.0, 2.0, 1.0] → [0.84, 0.11, 0.04]  ← confident
        temp=1.0: logits/1.0 = [2.0, 1.0, 0.5] → [0.57, 0.28, 0.15]  ← normal
        temp=2.0: logits/2.0 = [1.0, 0.5, 0.25] → [0.42, 0.27, 0.20] ← random

        TOP-K:
        Only considers the K best tokens.
        Prevents generating total garbage.
        """
        self.eval()
        tokens = [tokenizer.word2idx["<BOS>"]] + tokenizer.encode(prompt)
        tokens = torch.tensor([tokens], dtype=torch.long)

        if verbose:
            print(f"\n  🎲 Generating: '{prompt}'")

        for step in range(max_len):
            input_tokens = tokens[:, -CONFIG["max_seq_len"]:]
            logits = self(input_tokens)
            next_logits = logits[0, -1, :] / max(temperature, 0.01)

            # Top-K filtering
            if top_k > 0:
                topk_vals, topk_idx = torch.topk(next_logits, min(top_k, next_logits.size(0)))
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(0, topk_idx, topk_vals)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            if verbose:
                word = tokenizer.idx2word.get(next_token.item(), "?")
                top3_probs, top3_idx = torch.topk(probs, min(3, probs.size(0)))
                top3_words = [tokenizer.idx2word.get(i.item(), "?") for i in top3_idx]
                top3_str = ", ".join(
                    f"'{w}':{p:.2f}" for w, p in zip(top3_words, top3_probs)
                )
                print(f"     Step {step+1}: chosen='{word}' "
                      f"(top-3: {top3_str})")

            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == tokenizer.word2idx["<EOS>"]:
                break

        return tokenizer.decode(tokens[0].tolist())


# ============================================================
# DATA PREPARATION
# ============================================================
def prepare_data(corpus, tokenizer):
    """
    NEXT-TOKEN PREDICTION — the training objective.

    ╔══════════════════════════════════════════════════════╗
    ║  Sentence: "the cat sits on the mat"                 ║
    ║                                                      ║
    ║  Sequence: [BOS, the, cat, sits, on, the, mat, EOS]  ║
    ║                                                      ║
    ║  Input:  [BOS, the, cat,  sits, on,  the, mat ]      ║
    ║  Target: [the, cat, sits, on,   the, mat, EOS ]      ║
    ║           ↑    ↑    ↑     ↑     ↑    ↑    ↑          ║
    ║  Model predicts the NEXT token!                      ║
    ║                                                      ║
    ║  This is the ONLY objective of GPT. All "intelligence"║
    ║  emerges from this simple task at enormous scale.    ║
    ╚══════════════════════════════════════════════════════╝
    """
    sentences = [s.strip() for s in corpus.strip().split("\n") if s.strip()]

    sequences = []
    for sent in sentences:
        tokens = ([tokenizer.word2idx["<BOS>"]]
                  + tokenizer.encode(sent)
                  + [tokenizer.word2idx["<EOS>"]])
        sequences.append(tokens)

    max_len = min(max(len(s) for s in sequences), CONFIG["max_seq_len"])
    padded = []
    for seq in sequences:
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq = seq + [0] * (max_len - len(seq))
        padded.append(seq)

    data = torch.tensor(padded, dtype=torch.long)
    inputs = data[:, :-1]
    targets = data[:, 1:]

    return inputs, targets, sentences


# ============================================================
# TRAINING
# ============================================================
def train(model, inputs, targets, config, verbose=True):
    """
    TRAINING LOOP

    ╔══════════════════════════════════════════════════════╗
    ║  1. Forward:  model(input) → predictions             ║
    ║  2. Loss:     CrossEntropy(predictions, target)      ║
    ║  3. Backward: compute gradients (∂loss/∂weights)     ║
    ║  4. Update:   weights -= lr × gradients              ║
    ║  5. Repeat                                           ║
    ╚══════════════════════════════════════════════════════╝

    CROSS-ENTROPY LOSS:
    - Measures how far prediction is from truth
    - If model says P("mat")=0.9 and truth is "mat" → low loss
    - If model says P("mat")=0.01 and truth is "mat" → high loss
    - loss = -log(P(correct_token))

    PERPLEXITY = e^loss
    - "How many tokens the model hesitates between"
    - Perplexity 1 = perfect (always knows)
    - Perplexity 19 = random guessing (vocab_size=19)
    - GPT-3 on benchmarks: ~20-30 (but vocab_size=50257!)

    ADAM OPTIMIZER:
    - SGD + momentum + adaptive learning rate
    - Each weight gets its OWN learning rate
    - Standard in deep learning since 2015

    GRADIENT CLIPPING:
    - Clips gradients if too large
    - Prevents "gradient explosion"
    - max_norm=1.0 is standard
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    start_time = time.time()
    history = {"loss": [], "perplexity": []}

    if verbose:
        print(f"\n🏋️ Training: {config['epochs']} epochs, lr={config['lr']}")
        print("─" * 60)

    for epoch in range(config["epochs"]):
        total_loss = 0
        n_batches = 0

        indices = torch.randperm(inputs.size(0))
        for i in range(0, len(indices), config["batch_size"]):
            batch_idx = indices[i:i + config["batch_size"]]
            batch_in = inputs[batch_idx]
            batch_tgt = targets[batch_idx]

            logits = model(batch_in)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                batch_tgt.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        history["loss"].append(avg_loss)
        history["perplexity"].append(perplexity)

        if verbose and ((epoch + 1) % max(config["epochs"] // 10, 1) == 0
                        or epoch == 0):
            elapsed = time.time() - start_time
            bar_len = int(30 * (1 - min(avg_loss / 3.0, 1.0)))
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"  Epoch {epoch+1:4d}/{config['epochs']} │ "
                  f"Loss: {avg_loss:.4f} │ "
                  f"PPL: {perplexity:7.2f} │ "
                  f"[{bar}] │ {elapsed:.1f}s")

    total_time = time.time() - start_time
    if verbose:
        print(f"  {'─'*56}")
        print(f"  ✅ Done in {total_time:.1f}s │ "
              f"Final loss: {avg_loss:.4f} │ PPL: {perplexity:.2f}")

    return history


# ============================================================
# MODE 1: BASIC TRAINING + GENERATION
# ============================================================
def mode_basic():
    """Basic mode: build, train, generate."""
    print("\n" + "═" * 60)
    print("  📚 MODE 1: Basic Training + Generation")
    print("═" * 60)

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(CORPUS)
    tokenizer.print_vocab()

    print(f"\n🔨 Creating model...")
    model = MicroTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        n_layers=CONFIG["n_layers"],
        d_ff=CONFIG["d_ff"],
        max_seq_len=CONFIG["max_seq_len"],
        dropout=CONFIG["dropout"],
    )
    print(f"   {model.total_params:,} parameters")
    print(f"   {CONFIG['n_layers']} layers × {CONFIG['n_heads']} heads")
    print(f"   d_model={CONFIG['d_model']}, d_ff={CONFIG['d_ff']}")

    inputs, targets, sentences = prepare_data(CORPUS, tokenizer)
    print(f"\n📊 Data: {len(sentences)} sentences")

    history = train(model, inputs, targets, CONFIG)

    print(f"\n🎲 Generation (temp={CONFIG['temperature']}, "
          f"top_k={CONFIG['top_k']}):")
    print("─" * 60)
    prompts = ["the cat", "the dog", "the cat sits", "the dog likes",
               "the small", "the big"]
    model.eval()
    for prompt in prompts:
        results = set()
        for _ in range(3):
            text = model.generate(
                tokenizer, prompt,
                max_len=CONFIG["max_gen_len"],
                temperature=CONFIG["temperature"],
                top_k=CONFIG["top_k"]
            )
            results.add(text)
        print(f"\n  '{prompt}' →")
        for r in results:
            print(f"    • {r}")

    return model, tokenizer, history


# ============================================================
# MODE 2: DEBUG MODE
# ============================================================
def mode_debug():
    """Tensor inspection at every stage of the forward pass."""
    print("\n" + "═" * 60)
    print("  🔍 MODE 2: Debug Mode — Forward Pass Inspection")
    print("═" * 60)
    print("""
    You'll see EXACTLY what happens to the data:
    - Tensor shapes at every stage
    - Numeric values
    - How attention "looks"
    - What the predictions are
    """)

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(CORPUS)

    model = MicroTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        n_layers=CONFIG["n_layers"],
        d_ff=CONFIG["d_ff"],
        max_seq_len=CONFIG["max_seq_len"],
        dropout=CONFIG["dropout"],
        debug=True
    )

    # Train briefly
    inputs, targets, _ = prepare_data(CORPUS, tokenizer)
    saved_debug = model.debug
    model.debug = False
    print("\n⏳ Quick training (50 epochs)...")
    train(model, inputs, targets, {**CONFIG, "epochs": 50}, verbose=False)
    model.debug = saved_debug

    # Now debug forward pass
    sentence = "the cat sits on"
    print(f"\n🔍 Debug forward pass for: '{sentence}'")
    tokens = [tokenizer.word2idx["<BOS>"]] + tokenizer.encode(sentence)
    x = torch.tensor([tokens], dtype=torch.long)
    model.eval()

    # Show token mapping
    print(f"\n  Text: '{sentence}'")
    print(f"  Tokens: {[tokenizer.idx2word.get(t, '?') for t in tokens]}")
    print(f"  IDs: {tokens}")

    with torch.no_grad():
        logits = model(x)

    # Show predictions
    probs = F.softmax(logits[0, -1, :], dim=-1)
    top10 = torch.topk(probs, min(10, probs.size(0)))
    print(f"\n  🎯 Next token prediction after '{sentence}':")
    for prob, idx in zip(top10.values, top10.indices):
        word = tokenizer.idx2word.get(idx.item(), "?")
        bar = "█" * int(prob.item() * 40)
        print(f"     {prob.item():.3f} [{bar:40s}] '{word}'")

    # Generate with verbose
    print(f"\n  🎲 Step-by-step generation:")
    model.debug = False
    model.generate(tokenizer, "the cat", max_len=6,
                   temperature=CONFIG["temperature"], verbose=True)


# ============================================================
# MODE 3: CONFIG COMPARISON
# ============================================================
def mode_comparison():
    """Compares 3 configurations side by side."""
    print("\n" + "═" * 60)
    print("  📊 MODE 3: Config Comparison")
    print("═" * 60)
    print("""
    We train 3 models with different parameters
    and compare the results.
    """)

    configs = {
        "TINY (d=8, 1 layer, 1 head)": {
            **CONFIG,
            "d_model": 8, "n_heads": 1, "n_layers": 1,
            "d_ff": 16, "epochs": 200,
        },
        "SMALL (d=32, 2 layers, 2 heads)": {
            **CONFIG,
            "d_model": 32, "n_heads": 2, "n_layers": 2,
            "d_ff": 64, "epochs": 200,
        },
        "MEDIUM (d=64, 3 layers, 4 heads)": {
            **CONFIG,
            "d_model": 64, "n_heads": 4, "n_layers": 3,
            "d_ff": 128, "epochs": 200,
        },
    }

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(CORPUS)
    inputs, targets, _ = prepare_data(CORPUS, tokenizer)

    results = {}
    prompts = ["the cat", "the dog likes", "the small cat sits"]

    for name, cfg in configs.items():
        print(f"\n{'─'*60}")
        print(f"  🔨 {name}")

        model = MicroTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            d_ff=cfg["d_ff"],
            max_seq_len=cfg["max_seq_len"],
            dropout=cfg["dropout"],
        )
        print(f"     Parameters: {model.total_params:,}")

        start = time.time()
        history = train(model, inputs, targets, cfg, verbose=False)
        elapsed = time.time() - start

        final_loss = history["loss"][-1]
        final_ppl = history["perplexity"][-1]

        generations = {}
        model.eval()
        for p in prompts:
            generations[p] = model.generate(
                tokenizer, p, max_len=10,
                temperature=0.5, top_k=3
            )

        results[name] = {
            "params": model.total_params,
            "loss": final_loss,
            "ppl": final_ppl,
            "time": elapsed,
            "generations": generations,
        }

        print(f"     Loss: {final_loss:.4f} | PPL: {final_ppl:.2f} | "
              f"Time: {elapsed:.1f}s")

    # Comparison table
    print(f"\n{'═'*60}")
    print(f"  📊 RESULTS COMPARISON")
    print(f"{'═'*60}")
    print(f"  {'Model':<40} {'Params':>8} {'Loss':>7} {'PPL':>7} {'Time':>6}")
    print(f"  {'─'*40} {'─'*8} {'─'*7} {'─'*7} {'─'*6}")
    for name, r in results.items():
        print(f"  {name:<40} {r['params']:>8,} {r['loss']:>7.3f} "
              f"{r['ppl']:>7.2f} {r['time']:>5.1f}s")

    print(f"\n  Generation:")
    for prompt in prompts:
        print(f"\n    Prompt: '{prompt}'")
        for name, r in results.items():
            short_name = name.split("(")[0].strip()
            print(f"      {short_name:<8} → {r['generations'][prompt]}")

    print(f"""
    📝 TAKEAWAYS:
    - More parameters → lower loss (memorizes better)
    - But: on small data, large model = overfitting
    - Time grows with parameters
    - Sweet spot depends on data size!
    """)


# ============================================================
# MODE 4: INTERACTIVE PLAYGROUND
# ============================================================
def mode_interactive():
    """Interactive mode — generate whatever you want."""
    print("\n" + "═" * 60)
    print("  🎮 MODE 4: Interactive Playground")
    print("═" * 60)

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(CORPUS)
    tokenizer.print_vocab()

    model = MicroTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        n_layers=CONFIG["n_layers"],
        d_ff=CONFIG["d_ff"],
        max_seq_len=CONFIG["max_seq_len"],
        dropout=CONFIG["dropout"],
    )

    inputs, targets, _ = prepare_data(CORPUS, tokenizer)
    print(f"\n⏳ Training...")
    train(model, inputs, targets, CONFIG)

    temp = CONFIG["temperature"]
    top_k = CONFIG["top_k"]

    print(f"""
    ╔══════════════════════════════════════════════════════╗
    ║  Type a prompt and generate text!                    ║
    ║                                                      ║
    ║  Commands:                                           ║
    ║    /temp 0.5    — change temperature                 ║
    ║    /topk 3      — change top-k                       ║
    ║    /verbose     — show step-by-step generation       ║
    ║    /vocab       — show vocabulary                    ║
    ║    /retrain 500 — retrain for 500 epochs             ║
    ║    /quit        — exit                                ║
    ╚══════════════════════════════════════════════════════╝
    """)

    verbose = False
    while True:
        try:
            prompt = input(f"  📝 Prompt (temp={temp}, top_k={top_k}): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  👋 Goodbye!")
            break

        if not prompt:
            continue

        if prompt.startswith("/"):
            parts = prompt.split()
            cmd = parts[0].lower()

            if cmd == "/quit":
                print("  👋 Goodbye!")
                break
            elif cmd == "/temp" and len(parts) > 1:
                try:
                    temp = float(parts[1])
                    print(f"  ✅ Temperature: {temp}")
                except ValueError:
                    print("  ❌ Usage: /temp 0.5")
            elif cmd == "/topk" and len(parts) > 1:
                try:
                    top_k = int(parts[1])
                    print(f"  ✅ Top-K: {top_k}")
                except ValueError:
                    print("  ❌ Usage: /topk 3")
            elif cmd == "/verbose":
                verbose = not verbose
                print(f"  ✅ Verbose: {'ON' if verbose else 'OFF'}")
            elif cmd == "/vocab":
                tokenizer.print_vocab()
            elif cmd == "/retrain" and len(parts) > 1:
                try:
                    epochs = int(parts[1])
                    print(f"  ⏳ Retraining for {epochs} epochs...")
                    train(model, inputs, targets,
                          {**CONFIG, "epochs": epochs})
                except ValueError:
                    print("  ❌ Usage: /retrain 500")
            else:
                print("  ❌ Unknown command. "
                      "Available: /temp /topk /verbose /vocab /retrain /quit")
            continue

        # Check if words are in vocabulary
        unknown = [w for w in prompt.lower().split()
                   if w not in tokenizer.word2idx]
        if unknown:
            print(f"  ⚠️ Unknown words: {unknown}")
            print(f"     Use /vocab to see known words")
            continue

        # Generate 3 variants
        model.eval()
        print(f"  Results:")
        results = set()
        for _ in range(5):
            text = model.generate(
                tokenizer, prompt,
                max_len=CONFIG["max_gen_len"],
                temperature=temp,
                top_k=top_k,
                verbose=verbose and len(results) == 0
            )
            results.add(text)
        for r in results:
            print(f"    → {r}")
        print()


# ============================================================
# MODE 5: EXERCISES AND QUIZZES
# ============================================================
def mode_exercises():
    """Learning exercises — quizzes and experiments."""
    print("\n" + "═" * 60)
    print("  🎓 MODE 5: Exercises & Quizzes")
    print("═" * 60)

    score = 0
    total = 0

    def quiz(question, options, correct, explanation):
        nonlocal score, total
        total += 1
        print(f"\n  ❓ Question {total}: {question}")
        for i, opt in enumerate(options):
            print(f"     {i+1}. {opt}")

        try:
            ans = input("     Your answer (1-4): ").strip()
            if ans == str(correct):
                score += 1
                print(f"     ✅ Correct!")
            else:
                print(f"     ❌ Answer: {correct}. {options[correct-1]}")
            print(f"     💡 {explanation}")
        except (EOFError, KeyboardInterrupt):
            print(f"\n     Answer: {correct}. {options[correct-1]}")
            print(f"     💡 {explanation}")

    # ── QUIZ 1: Architecture ──
    print("\n" + "─" * 40)
    print("  📖 Section 1: ARCHITECTURE")
    print("─" * 40)

    quiz(
        "What does Self-Attention do?",
        [
            "Compresses the sequence into a single vector",
            "Lets each token attend to other tokens",
            "Generates random tokens",
            "Normalizes values"
        ],
        2,
        "Attention computes weighted sums — each token 'pays attention' "
        "to those tokens that are relevant to it."
    )

    quiz(
        "Why do we divide by √d_k in attention?",
        [
            "To speed up computation",
            "To prevent dot-products from getting too large (stabilizes softmax)",
            "To reduce the number of parameters",
            "It's optional and changes nothing"
        ],
        2,
        "Without scaling, dot-product grows with dimension. Large values → "
        "softmax gives near one-hot → gradient vanishes."
    )

    quiz(
        "What is the causal mask for?",
        [
            "Speeds up training",
            "Reduces overfitting",
            "Prevents looking into the future (model can't cheat)",
            "Normalizes attention weights"
        ],
        3,
        "Without the mask, the model at position 3 sees the token at position 5 — "
        "it 'knows the answer'. The mask blocks future positions."
    )

    quiz(
        "What is a residual connection (x + sublayer(x))?",
        [
            "An extra normalization layer",
            "A gradient highway — model can pass info through unchanged",
            "A way to reduce parameters",
            "A data augmentation technique"
        ],
        2,
        "Residual connections let gradients 'skip over' layers. "
        "Without them, deep networks don't train (vanishing gradients)."
    )

    # ── QUIZ 2: Training ──
    print("\n" + "─" * 40)
    print("  📖 Section 2: TRAINING")
    print("─" * 40)

    quiz(
        "What is perplexity?",
        [
            "An optimizer name",
            "e^loss — measures model 'uncertainty' (lower = better)",
            "Number of model parameters",
            "Accuracy on the test set"
        ],
        2,
        "Perplexity = 'how many tokens the model hesitates between'. "
        "PPL=1 → always knows. PPL=vocab_size → random guessing."
    )

    quiz(
        "What happens when the learning rate is TOO HIGH?",
        [
            "Model trains slower",
            "Model overshoots the optimum — loss jumps or increases",
            "Nothing changes",
            "Model will be better"
        ],
        2,
        "Too high lr → optimizer takes too large steps and 'overshoots' "
        "the loss minimum. Symptom: loss doesn't decrease or oscillates."
    )

    quiz(
        "Why do we use gradient clipping?",
        [
            "To make the model smaller",
            "To speed up training",
            "To prevent gradient explosion (excessively large weight updates)",
            "To increase learning rate"
        ],
        3,
        "Sometimes the gradient is huge (e.g. 1000). Without clipping, "
        "weights jump dramatically and the model 'explodes' (loss → NaN)."
    )

    # ── QUIZ 3: Generation ──
    print("\n" + "─" * 40)
    print("  📖 Section 3: GENERATION")
    print("─" * 40)

    quiz(
        "What does temperature do in generation?",
        [
            "Changes the number of layers",
            "Controls randomness: low=confident, high=creative",
            "Changes the learning rate",
            "Decides text length"
        ],
        2,
        "Temperature divides logits before softmax. Low temp → "
        "sharpens the distribution (always top token). High → flattens "
        "(all tokens have a chance)."
    )

    quiz(
        "What is top-k sampling?",
        [
            "We take the k longest sentences",
            "We train k models and average them",
            "We only consider the k best tokens, discard the rest",
            "We generate k times and pick the best"
        ],
        3,
        "Top-k limits sampling to the k best tokens. "
        "Top-k=1 is greedy (always the best). Prevents generating "
        "improbable tokens."
    )

    # ── PRACTICAL EXERCISE ──
    print("\n" + "─" * 40)
    print("  📖 Section 4: PRACTICAL EXERCISE")
    print("─" * 40)

    print("""
    🧪 Exercise: Now you'll test it yourself!

    We'll train a model and compare:
    A) Normal model
    B) Without causal mask (model "cheats")
    """)

    input("     Press Enter to run... ")

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(CORPUS)
    inputs, targets, _ = prepare_data(CORPUS, tokenizer)

    # Model A: normal
    print("\n  🅰️ Normal model (with causal mask):")
    model_a = MicroTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=32, n_heads=2, n_layers=2, d_ff=64,
        max_seq_len=32, dropout=0.1,
    )
    train(model_a, inputs, targets, {**CONFIG, "epochs": 200}, verbose=False)

    model_a.eval()
    gen_a = model_a.generate(tokenizer, "the cat", max_len=8, temperature=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        logits_a = model_a(inputs)
        loss_a = criterion(logits_a.reshape(-1, logits_a.size(-1)),
                          targets.reshape(-1)).item()
    print(f"     Loss: {loss_a:.4f} | Generation: '{gen_a}'")

    # Model B: no mask
    print("\n  🅱️ Model WITHOUT mask (model 'cheats'):")
    model_b = MicroTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=32, n_heads=2, n_layers=2, d_ff=64,
        max_seq_len=32, dropout=0.1,
    )

    # Train without mask
    model_b.train()
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=CONFIG["lr"])
    for epoch in range(200):
        indices = torch.randperm(inputs.size(0))
        for i in range(0, len(indices), CONFIG["batch_size"]):
            batch_idx = indices[i:i + CONFIG["batch_size"]]
            logits = model_b(inputs[batch_idx], use_causal_mask=False)
            loss = criterion(logits.reshape(-1, logits.size(-1)),
                           targets[batch_idx].reshape(-1))
            optimizer_b.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
            optimizer_b.step()

    model_b.eval()
    gen_b = model_b.generate(tokenizer, "the cat", max_len=8, temperature=0.5)
    with torch.no_grad():
        logits_b = model_b(inputs, use_causal_mask=False)
        loss_b = criterion(logits_b.reshape(-1, logits_b.size(-1)),
                          targets.reshape(-1)).item()
    print(f"     Loss: {loss_b:.4f} | Generation: '{gen_b}'")

    print(f"""
    📝 WHAT DO YOU SEE?
    - Model B has LOWER loss (because it cheats — sees the answer!)
    - BUT generates WORSE (because during generation there are no future tokens)
    - It's like a student who cheats on a test:
      gets 100% on the test, but can't do anything on their own

    CONCLUSION: Causal mask is CRUCIAL for generation!
    """)

    # ── SCORE ──
    print(f"\n{'═'*60}")
    print(f"  📊 SCORE: {score}/{total} correct answers")
    if score == total:
        print("  🏆 Perfect! You know the transformer architecture!")
    elif score >= total * 0.7:
        print("  👏 Very good! Just a bit more practice.")
    else:
        print("  📚 Read the code descriptions and try again!")
    print(f"{'═'*60}")


# ============================================================
# MODE 6: ATTENTION VISUALIZATION
# ============================================================
def mode_attention_viz():
    """Attention weights visualization."""
    print("\n" + "═" * 60)
    print("  👁️ MODE 6: Attention Visualization")
    print("═" * 60)
    print("""
    We'll see WHAT each attention head "looks at".

    ╔══════════════════════════════════════════════════════╗
    ║  Legend:                                             ║
    ║  ██████ = strong attention (> 0.5) ← "this matters!"║
    ║  ▓▓▓▓▓▓ = moderate (0.3-0.5)                        ║
    ║  ░░░░░░ = weak (0.1-0.3)                             ║
    ║  ...... = minimal (< 0.1)                            ║
    ╚══════════════════════════════════════════════════════╝
    """)

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(CORPUS)

    model = MicroTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        n_layers=CONFIG["n_layers"],
        d_ff=CONFIG["d_ff"],
        max_seq_len=CONFIG["max_seq_len"],
        dropout=CONFIG["dropout"],
    )

    inputs, targets, _ = prepare_data(CORPUS, tokenizer)
    print("  ⏳ Training...")
    train(model, inputs, targets, CONFIG, verbose=False)
    print("  ✅ Done!\n")

    sentences = [
        "the cat sits on the mat",
        "the dog likes bones",
        "the small cat and the big dog",
    ]

    for sentence in sentences:
        model.eval()
        tokens_list = [tokenizer.word2idx["<BOS>"]] + tokenizer.encode(sentence)
        token_names = ["<BOS>"] + sentence.split()
        x = torch.tensor([tokens_list], dtype=torch.long)

        with torch.no_grad():
            _ = model(x)

        print(f"  📝 Sentence: '{sentence}'")

        for layer_idx, layer in enumerate(model.layers):
            attn = layer.attention.attn_weights[0]

            print(f"\n  Layer {layer_idx + 1}:")
            for head in range(attn.size(0)):
                print(f"    Head {head + 1}:")

                # Header
                header = "            " + "".join(
                    f"{w[:7]:>8}" for w in token_names
                )
                print(header)

                for i, name in enumerate(token_names):
                    row_parts = []
                    for j in range(len(token_names)):
                        w = attn[head][i][j].item()
                        if w > 0.5:
                            cell = f"{'██':>5}{w:.2f}"
                        elif w > 0.3:
                            cell = f"{'▓▓':>5}{w:.2f}"
                        elif w > 0.1:
                            cell = f"{'░░':>5}{w:.2f}"
                        else:
                            cell = f"{'··':>5}{w:.2f}"
                        row_parts.append(f"{cell:>8}")
                    print(f"    {name[:10]:>10} {''.join(row_parts)}")

                # Interpretation
                print(f"    → Head {head+1} focuses on: ", end="")
                last_attn = attn[head][-1]
                top_idx = torch.topk(last_attn, min(2, len(token_names))).indices
                top_words = [token_names[i] for i in top_idx]
                print(f"{' and '.join(top_words)} (from last token's perspective)")

        print(f"\n  {'─'*50}")


# ============================================================
# MODE 7: EMBEDDING ANALYSIS
# ============================================================
def mode_embeddings():
    """Analysis of learned embeddings."""
    print("\n" + "═" * 60)
    print("  📐 MODE 7: Embedding Analysis")
    print("═" * 60)
    print("""
    Embedding turns a token ID into a vector of numbers.
    After training, SIMILAR words have SIMILAR vectors!

    ╔══════════════════════════════════════════════════════╗
    ║  Before training: random vectors                     ║
    ║    "cat"  → [0.23, -0.11, 0.87, ...]  (random)      ║
    ║    "dog"  → [-0.45, 0.33, 0.12, ...]  (random)      ║
    ║                                                      ║
    ║  After training: semantically organized!             ║
    ║    "cat"  → [0.45, 0.82, -0.31, ...]                 ║
    ║    "dog"  → [0.42, 0.79, -0.28, ...]  (similar!)     ║
    ║    "milk" → [-0.67, 0.11, 0.55, ...]  (different)    ║
    ╚══════════════════════════════════════════════════════╝
    """)

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(CORPUS)

    model = MicroTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        n_layers=CONFIG["n_layers"],
        d_ff=CONFIG["d_ff"],
        max_seq_len=CONFIG["max_seq_len"],
        dropout=CONFIG["dropout"],
    )

    inputs, targets, _ = prepare_data(CORPUS, tokenizer)
    print("  ⏳ Training...")
    train(model, inputs, targets, CONFIG, verbose=False)
    print("  ✅ Done!\n")

    # Collect embeddings
    words = [w for w in tokenizer.word2idx if w not in ("<PAD>", "<BOS>", "<EOS>")]
    embeddings = {}
    for word in words:
        idx = tokenizer.word2idx[word]
        emb = model.embedding.weight[idx].detach()
        embeddings[word] = emb

    # Similarity table
    print("  Cosine Similarity (1.0=identical, 0=unrelated, -1=opposite):\n")

    header = f"{'':>12}" + "".join(f"{w[:8]:>9}" for w in words)
    print(f"  {header}")
    print(f"  {'─' * (12 + 9 * len(words))}")

    for w1 in words:
        row = f"  {w1[:10]:>10}  "
        for w2 in words:
            sim = F.cosine_similarity(
                embeddings[w1].unsqueeze(0),
                embeddings[w2].unsqueeze(0)
            ).item()
            if w1 == w2:
                row += f"{'1.00':>8} "
            elif sim > 0.7:
                row += f"\033[92m{sim:>8.3f}\033[0m "
            elif sim < -0.3:
                row += f"\033[91m{sim:>8.3f}\033[0m "
            else:
                row += f"{sim:>8.3f} "
        print(row)

    # Top pairs
    print(f"\n  🏆 Most similar word pairs:")
    pairs = []
    for i, w1 in enumerate(words):
        for w2 in words[i+1:]:
            sim = F.cosine_similarity(
                embeddings[w1].unsqueeze(0),
                embeddings[w2].unsqueeze(0)
            ).item()
            pairs.append((sim, w1, w2))
    pairs.sort(reverse=True)

    for sim, w1, w2 in pairs[:5]:
        bar = "█" * int(max(0, sim) * 20)
        print(f"     {w1:>8} ↔ {w2:<8}  {sim:+.3f}  [{bar}]")

    print(f"\n  🔻 Least similar pairs:")
    for sim, w1, w2 in pairs[-5:]:
        print(f"     {w1:>8} ↔ {w2:<8}  {sim:+.3f}")

    print("""
    📝 INTERPRETATION:
    - "cat" and "dog" should be SIMILAR (both animals, both "sit")
    - "sits" and "likes" may be similar (both verbs after subject)
    - "milk" and "bones" — semantically different, but similar role
    - Embeddings learn word ROLES, not just meanings!
    """)


# ============================================================
# MODE 8: ABLATION STUDY
# ============================================================
def mode_ablation():
    """Disable components and see what breaks."""
    print("\n" + "═" * 60)
    print("  🔬 MODE 8: Ablation Study")
    print("═" * 60)
    print("""
    ABLATION STUDY = we disable a component and see
    how much the model degrades.

    If disabling X makes things worse → X is important!

    ╔══════════════════════════════════════════════════════╗
    ║  Testing:                                            ║
    ║  A. Full model (baseline)                            ║
    ║  B. Without positional encoding                      ║
    ║  C. 1 head instead of 2                              ║
    ║  D. 1 layer instead of 2                             ║
    ║  E. Without feed-forward                             ║
    ║  F. Small d_model=8                                  ║
    ╚══════════════════════════════════════════════════════╝
    """)

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(CORPUS)
    inputs, targets, _ = prepare_data(CORPUS, tokenizer)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    results = {}
    test_prompt = "the cat sits"

    def run_ablation(name, **kwargs):
        d_model = kwargs.get("d_model", CONFIG["d_model"])
        n_heads = kwargs.get("n_heads", CONFIG["n_heads"])
        n_layers = kwargs.get("n_layers", CONFIG["n_layers"])
        d_ff = kwargs.get("d_ff", CONFIG["d_ff"])

        model = MicroTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, max_seq_len=CONFIG["max_seq_len"],
            dropout=CONFIG["dropout"],
        )

        if kwargs.get("no_pos_enc"):
            model.pos_encoding.enabled = False
        if kwargs.get("no_ffn"):
            for layer in model.layers:
                layer.feed_forward.enabled = False

        history = train(model, inputs, targets,
                       {**CONFIG, "epochs": 200}, verbose=False)

        model.eval()
        gen = model.generate(tokenizer, test_prompt, max_len=8, temperature=0.5)

        with torch.no_grad():
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)),
                           targets.reshape(-1)).item()
        ppl = math.exp(loss) if loss < 20 else float("inf")

        results[name] = {"loss": loss, "ppl": ppl, "gen": gen,
                        "params": model.total_params}
        print(f"  ✅ {name:<35} Loss: {loss:.4f}  PPL: {ppl:7.2f}  "
              f"→ '{gen}'")

    print("  ⏳ Running tests...\n")

    run_ablation("A. Full model (baseline)")
    run_ablation("B. No positional encoding", no_pos_enc=True)
    run_ablation("C. 1 head (instead of 2)", n_heads=1)
    run_ablation("D. 1 layer (instead of 2)", n_layers=1)
    run_ablation("E. No feed-forward", no_ffn=True)
    run_ablation("F. Small d_model=8", d_model=8, n_heads=1, d_ff=16)

    # Ranking
    baseline_loss = results["A. Full model (baseline)"]["loss"]
    print(f"\n  📊 COMPONENT IMPACT (vs baseline loss {baseline_loss:.4f}):")
    print(f"  {'─'*55}")

    sorted_results = sorted(results.items(), key=lambda x: x[1]["loss"])
    for name, r in sorted_results:
        diff = r["loss"] - baseline_loss
        direction = "📈" if diff > 0.01 else "📉" if diff < -0.01 else "➡️"
        bar_len = int(min(abs(diff) * 20, 20))
        bar = ("🔴" * bar_len if diff > 0 else "🟢" * bar_len) or "⚪"
        print(f"  {direction} {name:<35} Δloss: {diff:+.4f}  {bar}")

    print("""
    📝 TAKEAWAYS:
    - Positional encoding: CRITICAL for word order
      Without it "the cat sits on the mat" = "mat the on sits cat the"

    - Feed-forward: important for information processing
      Attention connects tokens, FFN processes each one individually

    - Number of heads: more heads = more perspectives
      But on small data, 1 head may suffice

    - Depth (layers): more layers = deeper understanding
      But also more parameters and risk of overfitting

    - d_model: too small = bottleneck (too little "memory" per token)
    """)


# ============================================================
# MODE 9: SAVE / LOAD MODEL
# ============================================================
def mode_save_load():
    """Save and load a model."""
    print("\n" + "═" * 60)
    print("  💾 MODE 9: Save / Load Model")
    print("═" * 60)

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(CORPUS)
    inputs, targets, _ = prepare_data(CORPUS, tokenizer)

    # Train
    print("\n  ⏳ Training model...")
    model = MicroTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        n_layers=CONFIG["n_layers"],
        d_ff=CONFIG["d_ff"],
        max_seq_len=CONFIG["max_seq_len"],
        dropout=CONFIG["dropout"],
    )
    train(model, inputs, targets, CONFIG)

    # Generate before saving
    model.eval()
    gen_before = model.generate(tokenizer, "the cat sits", max_len=8, temperature=0.3)
    print(f"\n  Before save: 'the cat sits' → '{gen_before}'")

    # Save
    save_path = "edu_transformer_checkpoint.pt"
    checkpoint = {
        "model_state": model.state_dict(),
        "config": CONFIG,
        "vocab": tokenizer.word2idx,
    }
    torch.save(checkpoint, save_path)
    file_size = os.path.getsize(save_path) / 1024
    print(f"\n  💾 Saved: {save_path} ({file_size:.1f} KB)")
    print(f"     Contains: model weights + config + vocabulary")

    # Load into a NEW model
    print(f"\n  📂 Loading...")
    checkpoint = torch.load(save_path, weights_only=False)

    # Rebuild tokenizer
    new_tokenizer = SimpleTokenizer()
    new_tokenizer.word2idx = checkpoint["vocab"]
    new_tokenizer.idx2word = {v: k for k, v in checkpoint["vocab"].items()}
    new_tokenizer.vocab_size = len(checkpoint["vocab"])

    # Rebuild model
    cfg = checkpoint["config"]
    new_model = MicroTransformer(
        vocab_size=new_tokenizer.vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
    )
    new_model.load_state_dict(checkpoint["model_state"])

    # Generate after loading
    new_model.eval()
    gen_after = new_model.generate(new_tokenizer, "the cat sits",
                                   max_len=8, temperature=0.3)
    print(f"  After load:  'the cat sits' → '{gen_after}'")

    if gen_before == gen_after:
        print(f"\n  ✅ Identical! Model correctly saved and loaded.")
    else:
        print(f"\n  ℹ️ Slight difference (sampling is random when temp>0)")
        print(f"     At temperature=0.0001 they'd be identical.")

    print(f"""
    📝 HOW IT WORKS:
    - torch.save() serializes weights (tensors) to file
    - We also save CONFIG and vocabulary (needed for reconstruction)
    - model.state_dict() = dictionary {{layer_name: tensor}}
    - model.load_state_dict() = load weights into model

    IN REAL PROJECTS you also save:
    - Optimizer state (to resume training)
    - Epoch number
    - Loss history
    - Random seed (for reproducibility)
    """)

    # Cleanup
    try:
        os.remove(save_path)
        print(f"  🗑️ Cleaned up: {save_path}")
    except OSError:
        pass


# ============================================================
# MODE 0: FULL COURSE
# ============================================================
def mode_full_course():
    """All modes in sequence."""
    print("\n" + "═" * 60)
    print("  🎓 MODE 0: FULL COURSE")
    print("═" * 60)
    print("""
    You'll go through EVERYTHING in order:

    1. Basics (building, training, generation)
    2. Debug (what happens inside)
    3. Config comparison
    4. Attention visualization
    5. Embedding analysis
    6. Ablation study
    7. Exercises & quizzes
    8. Save/load

    Press Enter after each section.
    """)

    sections = [
        ("1/8 — Basics", mode_basic),
        ("2/8 — Debug Mode", mode_debug),
        ("3/8 — Config Comparison", mode_comparison),
        ("4/8 — Attention Visualization", mode_attention_viz),
        ("5/8 — Embedding Analysis", mode_embeddings),
        ("6/8 — Ablation Study", mode_ablation),
        ("7/8 — Exercises", mode_exercises),
        ("8/8 — Save/Load", mode_save_load),
    ]

    for i, (name, func) in enumerate(sections):
        print(f"\n{'═'*60}")
        print(f"  Section {name}")
        print(f"{'═'*60}")

        try:
            input("  Press Enter to continue... ")
        except (EOFError, KeyboardInterrupt):
            print("\n  Course interrupted.")
            return

        try:
            func()
        except (EOFError, KeyboardInterrupt):
            pass

    print(f"""

    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║             🏆 COURSE COMPLETE! 🏆                   ║
    ║                                                      ║
    ║  Now you know:                                       ║
    ║  ✅ How the Transformer architecture works            ║
    ║  ✅ What Self-Attention does (Q, K, V)                ║
    ║  ✅ Why causal masking matters                        ║
    ║  ✅ How positional encoding works                     ║
    ║  ✅ What residual connections and LayerNorm do        ║
    ║  ✅ How to train (loss, optimizer, gradient clipping) ║
    ║  ✅ How to generate text (temperature, top-k)        ║
    ║  ✅ How to analyze embeddings                        ║
    ║  ✅ Which components matter most                      ║
    ║                                                      ║
    ║  This is EXACTLY the same architecture as GPT-2/3/4  ║
    ║  Difference: scale (parameters, data, compute)       ║
    ║                                                      ║
    ║  NEXT STEPS:                                         ║
    ║  → nanoGPT (Andrej Karpathy) — full GPT-2            ║
    ║  → "Attention Is All You Need" — original paper      ║
    ║  → Hugging Face Transformers — production models     ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
    """)


# ============================================================
# MAIN MENU
# ============================================================
def main():
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║          🧠 EDU-TRANSFORMER — FULL COURSE            ║
    ║                                                      ║
    ║  One script = complete GPT architecture training     ║
    ║  Trains in seconds. Learn by experimenting.          ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
    """)

    while True:
        print(f"""
    ╔════════════════════════════════════╗
    ║  MENU:                             ║
    ║  0. 🎓 Full course (everything)   ║
    ║  1. 📚 Basic training              ║
    ║  2. 🔍 Debug mode                  ║
    ║  3. 📊 Config comparison           ║
    ║  4. 🎮 Interactive playground      ║
    ║  5. 🎓 Exercises & quizzes         ║
    ║  6. 👁️  Attention visualization     ║
    ║  7. 📐 Embedding analysis          ║
    ║  8. 🔬 Ablation study              ║
    ║  9. 💾 Save/load model             ║
    ║  q. 🚪 Exit                        ║
    ╚════════════════════════════════════╝""")

        try:
            choice = input("\n    Choose mode (0-9, q): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  👋 Goodbye!")
            break

        modes = {
            "0": mode_full_course,
            "1": mode_basic,
            "2": mode_debug,
            "3": mode_comparison,
            "4": mode_interactive,
            "5": mode_exercises,
            "6": mode_attention_viz,
            "7": mode_embeddings,
            "8": mode_ablation,
            "9": mode_save_load,
        }

        if choice == "q":
            print("  👋 Goodbye!")
            break
        elif choice in modes:
            try:
                result = modes[choice]()
            except (EOFError, KeyboardInterrupt):
                print("\n  ↩️ Back to menu...")
            except Exception as e:
                print(f"\n  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("  ❌ Unknown option. Type 0-9 or q.")


if __name__ == "__main__":
    main()