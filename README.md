# 📢 Transformer-based Spam Classifier (PyTorch)

This project implements a Transformer-based model for binary text classification (spam detection) using **PyTorch**. It walks through the full NLP pipeline, from preprocessing to evaluation, following the core ideas from the Transformer architecture.

---

## 🚀 Objectives

By completing this lab, you will:

- Implement a text preprocessing pipeline (tokenization, vocab building, padding)
- Learn how positional encoding provides sequence order information
- Implement multi-head self-attention and feedforward layers manually
- Build a Transformer block with residual connections and normalization
- Train a spam classifier on tokenized text data
- Evaluate accuracy and performance using test data

---

## 📁 Project Structure

### Key Components:

- **TextPreprocessor**: Tokenizes text, builds vocab, encodes sequences to token IDs
- **PositionalEncoding**: Adds sinusoidal positional info to embeddings
- **MultiHeadAttention**: Implements scaled dot-product self-attention across multiple heads
- **FeedForward**: Position-wise fully connected layer
- **TransformerBlock**: Combines attention, feedforward, layer norm, and dropout
- **SpamTransformer**: Full Transformer model with stacked blocks and classification head
- **Training and Evaluation**: Functions to run training epochs and evaluate on validation/test sets

---

## 🔧 How It Works

1. **Preprocess the Text**  
   Tokenize raw sentences, build a vocabulary, convert to padded sequences.

2. **Build the Model**  
   Embed input, add positional encoding, pass through transformer layers, apply pooling, and classify.

3. **Train and Evaluate**  
   Use standard supervised training with cross-entropy loss and accuracy metrics.

---

## 🔎 Code Overview

- `TextPreprocessor`:
  - Tokenizes and lowercases input
  - Removes punctuation
  - Maps words to indices, handles padding/unknowns

- `PositionalEncoding`:
  - Uses sine/cosine encoding to represent positions

- `MultiHeadAttention`:
  - Projects Q, K, V
  - Applies scaled dot-product attention
  - Concatenates and linearly transforms

- `TransformerBlock`:
  - Attention → dropout → residual → norm
  - Feedforward → dropout → residual → norm

- `SpamTransformer`:
  - Stacks multiple blocks
  - Averages over sequence
  - Final `Linear` layer for binary classification

---

## 🎓 Training Loop

- Standard training with:
  - `optimizer.zero_grad()`
  - Forward pass with attention mask
  - Loss computation (`nn.CrossEntropyLoss()`)
  - Backpropagation and parameter update

---

## 🔬 Evaluation

- Inference using `torch.no_grad()`
- Masking applied during evaluation as well
- Accuracy and average loss returned

---

## 📊 Future Work

- Add padding-aware masking
- Use real-world spam datasets (SMS, email, etc.)
- Add early stopping, LR scheduler, and visualization tools
- Explore multi-class classification or sequence labeling tasks

---

## 📚 References

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- PyTorch documentation: https://pytorch.org
- Coursera NLP: Deep Learning Specialization

