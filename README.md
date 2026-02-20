<div align="center">

# 🧠 edu-transformer

**Learn GPT architecture through experiments, not lectures.**

Trains in 3 seconds. Single Python file. Zero theory without practice.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/tomaszwi66/edu-transformer/pulls)

[Quick Start](#-quick-start) •
[Learning Modes](#-9-learning-modes) •
[Architecture](#-architecture) •
[Learning Path](#-learning-path) •
[FAQ](#-faq)

</div>

---

## 🎯 Who is this for?

- **Beginners** — you want to understand how GPT/ChatGPT works
- **Intermediate** — you know ML basics, want to understand Transformers
- **Practitioners** — you want to quickly prototype and experiment

No prerequisites. No linear algebra required. Everything explained in code comments.

---

## ⚡ Quick Start

```bash
pip install torch numpy
git clone https://github.com/tomaszwi66/edu-transformer.git
cd edu-transformer
python edu_transformer.py
```

Pick a mode from the menu. First time? Type `0` — the full course will guide you through everything.

<pre>
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
║  9. 💾 Save / load model           ║
╚════════════════════════════════════╝
</pre>

---

## 📚 9 Learning Modes

### 1. Basic Training

Build a model, train on simple sentences, generate text. See the entire process from zero to a working language model.

<pre>
🏋️ Training: 300 epochs, lr=0.001
  Epoch    1/300 │ Loss: 2.8912 │ PPL:  18.10
  Epoch  300/300 │ Loss: 0.0523 │ PPL:   1.05
✅ Done in 3.2s

  'the cat sits' →
    • the cat sits on the mat
    • the cat sits on the couch and drinks milk
</pre>

### 2. Debug Mode

Inspect every tensor at every stage. See shapes, values, data flow through the entire forward pass.

<pre>
🔍 DEBUG: Forward Pass
  Input token IDs: [1, 3, 5, 7]
  Shape: torch.Size([1, 4])

  📊 After Embedding:
     Shape: torch.Size([1, 4, 32])  (batch, seq_len, d_model)

  🎯 Prediction after 'the cat sits on':
     0.487 [████████████████████] 'mat'
     0.231 [█████████]            'couch'
     0.098 [████]                 'carpet'
</pre>

### 3. Config Comparison

Automatically trains 3 models (tiny/small/medium) and compares results side by side.

<pre>
  Model                                   Params    Loss     PPL   Time
  TINY (d=8, 1 layer, 1 head)              331   1.203    3.33   0.8s
  SMALL (d=32, 2 layers, 2 heads)         7,891   0.052    1.05   3.1s
  MEDIUM (d=64, 3 layers, 4 heads)       44,627   0.031    1.03   8.4s
</pre>

### 4. Interactive Playground

Type prompts, change parameters on the fly with `/temp`, `/topk`, `/verbose` commands.

<pre>
📝 Prompt (temp=0.8, top_k=5): the cat sits
    → the cat sits on the mat
    → the cat sits on the couch and drinks milk

📝 Prompt: /temp 0.1
✅ Temperature: 0.1
</pre>

### 5. Exercises & Quizzes

9 questions testing your understanding + a practical exercise comparing models with and without causal masking.

<pre>
❓ What does Self-Attention do?
   1. Compresses the sequence into a single vector
   2. Lets each token attend to other tokens  ← ✅
   3. Generates random tokens
   4. Normalizes values

📊 SCORE: 8/9 correct answers
</pre>

### 6. Attention Visualization

Heatmap showing what each attention head "looks at" for a given sentence.

<pre>
  'the cat sits on the mat'     Head 1:
                &lt;BOS&gt;     the     cat   sits     on    mat
       &lt;BOS&gt;  ██0.98  ··0.01  ··0.01  ··0.00  ··0.00  ··0.00
         the  ░░0.12  ██0.85  ··0.02  ··0.01  ··0.00  ··0.00
         cat  ··0.03  ▓▓0.47  ▓▓0.41  ··0.06  ··0.02  ··0.01
        sits  ··0.02  ░░0.15  ▓▓0.38  ▓▓0.35  ··0.07  ··0.03
</pre>

### 7. Embedding Analysis

Cosine similarity table between words. Shows which words the model considers similar after training.

<pre>
🏆 Most similar pairs:
       cat ↔ dog       +0.847  [████████████████]
      sits ↔ likes     +0.723  [██████████████]
</pre>

### 8. Ablation Study

Disables components one by one and measures how much the model degrades.

<pre>
📊 COMPONENT IMPACT:
➡️ A. Full model (baseline)              Δloss: +0.000
📈 B. No positional encoding             Δloss: +0.834  🔴🔴🔴🔴🔴
📈 E. No feed-forward                    Δloss: +0.567  🔴🔴🔴
📈 F. Small d_model=8                    Δloss: +1.456  🔴🔴🔴🔴🔴🔴🔴
</pre>

### 9. Save / Load Model

Save a trained model to file and load it back without retraining.

---

## 🏗️ Architecture

This script implements **exactly the same architecture** as GPT-2/3/4. The only difference is scale.

<pre>
edu-transformer              GPT-4
──────────────────           ──────────────────
19 tokens                    100K tokens
32 dimensions                ~12,288 dimensions
2 heads                      ~96 heads
2 layers                     ~120 layers
~8K parameters               ~1.8T parameters
20 sentences                 the internet
3 seconds                    3 months + $100M
</pre>

<pre>
Input: "the cat sits on"
       ↓
┌─────────────────────────┐
│  Embedding + Position   │  ID → vector + position info
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│  Transformer Block ×N   │
│  ┌───────────────────┐  │
│  │ Multi-Head Attn   │  │  "What to look at?"
│  │ (2 heads)         │  │
│  └────────┬──────────┘  │
│  ┌────────┴──────────┐  │
│  │ Feed-Forward      │  │  "What to do with it?"
│  └───────────────────┘  │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│  Linear → Softmax       │  → "mat" (next token)
└─────────────────────────┘
</pre>

---

## ⚙️ Parameters to Experiment With

Change values in the `CONFIG` section at the top of the script.

### Architecture

| Parameter | Default | What it does | Try |
|-----------|---------|-------------|-----|
| `d_model` | 32 | Embedding dimension — token "memory" | 8, 16, 64, 128 |
| `n_heads` | 2 | Attention heads — number of perspectives | 1, 4 (must divide d_model) |
| `n_layers` | 2 | Model depth — pattern complexity | 1, 3, 4 |
| `d_ff` | 64 | Feed-forward dimension — compute power | 16, 32, 128 |
| `dropout` | 0.1 | Regularization — prevents overfitting | 0.0, 0.3, 0.5 |

### Training

| Parameter | Default | What it does | Try |
|-----------|---------|-------------|-----|
| `epochs` | 300 | How many times the model sees the data | 50, 500, 1000 |
| `lr` | 0.001 | Learning rate — how fast it learns | 0.01, 0.0001 |
| `batch_size` | 4 | Sentences per batch | 1, 2, 8 |

### Generation

| Parameter | Default | What it does | Try |
|-----------|---------|-------------|-----|
| `temperature` | 0.8 | Randomness — 0.1=confident, 2.0=chaos | 0.1, 0.5, 1.5 |
| `top_k` | 5 | Tokens considered — 1=greedy | 1, 3, 0 (no limit) |

---

## 🧪 Experiments to Try

**Easy:**
- Change temperature to 0.1 and 2.0 — compare outputs
- Add your own sentences to CORPUS
- Set `n_heads` to 1 — what changes?

**Medium:**
- Set `d_model=8` — can the model learn anything?
- Set `epochs=1000` — does it memorize too much (overfitting)?
- Add sentences in another language — does it cope?

**Advanced:**
- Comment out positional encoding — what happens to word order?
- Replace ReLU with GELU in FeedForward
- Add 100 sentences — how does it affect quality?

---

## 🗺️ Learning Path

### Week 1: Fundamentals

| Day | What to do | Goal |
|-----|-----------|------|
| 1 | Mode 1 (basic) | Run it, see it work |
| 2 | Mode 2 (debug) | Understand data flow |
| 3 | Mode 5 (quizzes) | Test your understanding |
| 4 | Mode 4 (playground) | Play with temperature, top-k |
| 5 | Mode 8 (ablation) | Build intuition about components |

### Week 2: Deeper Understanding

| Day | What to do | Goal |
|-----|-----------|------|
| 1 | Mode 6 (attention) | Understand "what the model looks at" |
| 2 | Mode 7 (embeddings) | Understand "how the model sees words" |
| 3 | Mode 3 (comparison) | Intuition about scaling |
| 4 | Change CORPUS | Your own data, observe differences |
| 5 | Read the code | Class by class, experiment |

### Week 3: Bridge to Real GPT

| Resource | Description |
|----------|-------------|
| [nanoGPT](https://github.com/karpathy/nanoGPT) | Same concept, larger scale |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | The original paper |
| [Hugging Face](https://huggingface.co) | Production-ready models |
| [Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Best GPT video |

---

## 📁 Project Structure

<pre>
edu-transformer/
├── edu_transformer.py    ← all code (single file, by design)
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore
</pre>

One file. On purpose. So you can open it in any editor and see everything at a glance. No jumping between modules.

---

## ❓ FAQ

**Is this a real Transformer?**
Yes. Same architecture as GPT-2/3/4. The only difference is scale (parameters, data, compute).

**Do I need a GPU?**
No. Trains on CPU in 3 seconds.

**What's next after this script?**
nanoGPT → the original paper → Hugging Face. Learning path described above.

**The model generates nonsense!**
Lower the temperature (`/temp 0.3`) or increase epochs. The model has 19 words in its vocabulary — don't expect ChatGPT.

**Can I use this for teaching?**
Yes. MIT license — do whatever you want. Credit appreciated but not required.

---

## 🤝 Contributing

1. Fork the repository
2. Create a branch (`git checkout -b my-feature`)
3. Make your changes
4. Verify `python edu_transformer.py` works
5. Create a Pull Request

### Ideas for Development

- [ ] Matplotlib visualizations
- [ ] BPE tokenizer (subword)
- [ ] More exercises and quizzes
- [ ] Jupyter notebook with interactive widgets
- [ ] Comparison with RNN/LSTM
- [ ] Multi-language corpus support

---

## 📜 License

MIT — do whatever you want. Details in [LICENSE](LICENSE).

---

<div align="center">

If this helped — leave a ⭐

*You don't need to understand 1.8 trillion parameters. Just understand 8 thousand — the rest is the same architecture.*

</div>
