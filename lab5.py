import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from datasets import load_dataset
from transformers import AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"dispositivo: {DEVICE}")


tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

PAD_ID  = tokenizer.pad_token_id
CLS_ID  = tokenizer.cls_token_id
SEP_ID  = tokenizer.sep_token_id
VOCAB_SIZE = tokenizer.vocab_size

MAX_LEN    = 40
SUBSET     = 1000
D_MODEL    = 128
D_FF       = 256
N_CAMADAS  = 2
N_HEADS    = 4
DROPOUT    = 0.1
EPOCHS     = 15
LR         = 1e-3
BATCH_SIZE = 32


print("\ncarregando dataset...")
dataset = load_dataset("Helsinki-NLP/opus_books", "en-pt", split="train")
subset  = dataset.select(range(SUBSET))
print(f"frases carregadas: {len(subset)}")


def tokenizar_par(exemplo):
    src = tokenizer(
        exemplo["translation"]["en"],
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    tgt = tokenizer(
        exemplo["translation"]["pt"],
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "src_ids": src["input_ids"].squeeze(0),
        "tgt_ids": tgt["input_ids"].squeeze(0),
    }

print("tokenizando...")
pares = [tokenizar_par(ex) for ex in subset]

src_tensor = torch.stack([p["src_ids"] for p in pares])
tgt_tensor = torch.stack([p["tgt_ids"] for p in pares])

print(f"shape src_tensor: {src_tensor.shape}")
print(f"shape tgt_tensor: {tgt_tensor.shape}")


def make_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k     = d_model // n_heads
        self.n_heads = n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def _split(self, x):
        b, s, _ = x.size()
        return x.view(b, s, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        Q = self._split(self.Wq(Q))
        K = self._split(self.Wk(K))
        V = self._split(self.Wv(V))
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        b, h, s, d = out.size()
        out = out.transpose(1, 2).contiguous().view(b, s, h * d)
        return self.Wo(out)


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))


class BlocoEncoder(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout):
        super().__init__()
        self.atencao   = MultiHeadAttention(d_model, n_heads)
        self.ffn       = FFN(d_model, d_ff)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.atencao(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class BlocoDecoder(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout):
        super().__init__()
        self.masked_atencao = MultiHeadAttention(d_model, n_heads)
        self.cross_atencao  = MultiHeadAttention(d_model, n_heads)
        self.ffn            = FFN(d_model, d_ff)
        self.norm1          = nn.LayerNorm(d_model)
        self.norm2          = nn.LayerNorm(d_model)
        self.norm3          = nn.LayerNorm(d_model)
        self.dropout        = nn.Dropout(dropout)

    def forward(self, y, Z, tgt_mask=None, src_mask=None):
        y = self.norm1(y + self.dropout(self.masked_atencao(y, y, y, tgt_mask)))
        y = self.norm2(y + self.dropout(self.cross_atencao(y, Z, Z, src_mask)))
        y = self.norm3(y + self.dropout(self.ffn(y)))
        return y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe       = torch.zeros(max_len, d_model)
        posicoes = torch.arange(0, max_len).unsqueeze(1).float()
        divisores = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(posicoes * divisores)
        pe[:, 1::2] = torch.cos(posicoes * divisores)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, n_heads, n_camadas, dropout):
        super().__init__()
        self.d_model   = d_model
        self.emb_src   = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.emb_tgt   = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc   = PositionalEncoding(d_model, dropout=dropout)
        self.encoder   = nn.ModuleList([BlocoEncoder(d_model, d_ff, n_heads, dropout) for _ in range(n_camadas)])
        self.decoder   = nn.ModuleList([BlocoDecoder(d_model, d_ff, n_heads, dropout) for _ in range(n_camadas)])
        self.projecao  = nn.Linear(d_model, vocab_size)

    def encode(self, src, src_mask=None):
        x = self.pos_enc(self.emb_src(src) * math.sqrt(self.d_model))
        for camada in self.encoder:
            x = camada(x, src_mask)
        return x

    def decode(self, tgt, Z, tgt_mask=None, src_mask=None):
        y = self.pos_enc(self.emb_tgt(tgt) * math.sqrt(self.d_model))
        for camada in self.decoder:
            y = camada(y, Z, tgt_mask, src_mask)
        return y

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        Z      = self.encode(src, src_mask)
        saida  = self.decode(tgt, Z, tgt_mask, src_mask)
        return self.projecao(saida)


def run_inference(modelo, src_ids, max_steps=40):
    modelo.eval()
    with torch.no_grad():
        src   = src_ids.unsqueeze(0).to(DEVICE)
        Z     = modelo.encode(src)
        gerado = [CLS_ID]
        for _ in range(max_steps):
            tgt      = torch.tensor([gerado], device=DEVICE)
            tgt_mask = make_causal_mask(tgt.size(1), DEVICE)
            saida    = modelo.decode(tgt, Z, tgt_mask)
            logits   = modelo.projecao(saida[:, -1, :])
            prox     = torch.argmax(logits, dim=-1).item()
            gerado.append(prox)
            if prox == SEP_ID:
                break
        return tokenizer.decode(gerado, skip_special_tokens=True)


modelo    = Transformer(VOCAB_SIZE, D_MODEL, D_FF, N_HEADS, N_CAMADAS, DROPOUT).to(DEVICE)
criterio  = nn.CrossEntropyLoss(ignore_index=PAD_ID)
otimizador = torch.optim.Adam(modelo.parameters(), lr=LR)

n_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
print(f"\nparametros treinaveis: {n_params:,}")

print("\n" + "=" * 60)
print("TAREFA 3 — Training Loop")
print("=" * 60)

historico_loss = []

for epoch in range(1, EPOCHS + 1):
    modelo.train()
    loss_acumulado = 0.0
    n_batches      = 0

    for i in range(0, len(src_tensor), BATCH_SIZE):
        src_batch = src_tensor[i : i + BATCH_SIZE].to(DEVICE)
        tgt_batch = tgt_tensor[i : i + BATCH_SIZE].to(DEVICE)

        tgt_entrada = tgt_batch[:, :-1]
        tgt_alvo    = tgt_batch[:, 1:]

        seq_len  = tgt_entrada.size(1)
        tgt_mask = make_causal_mask(seq_len, DEVICE)

        logits = modelo(src_batch, tgt_entrada, tgt_mask=tgt_mask)

        logits_flat = logits.reshape(-1, VOCAB_SIZE)
        alvo_flat   = tgt_alvo.reshape(-1)

        loss = criterio(logits_flat, alvo_flat)

        otimizador.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        otimizador.step()

        loss_acumulado += loss.item()
        n_batches      += 1

    loss_medio = loss_acumulado / n_batches
    historico_loss.append(loss_medio)
    print(f"  epoca {epoch:02d}/{EPOCHS} — loss: {loss_medio:.4f}")

reducao = ((historico_loss[0] - historico_loss[-1]) / historico_loss[0]) * 100
print(f"\nloss inicial : {historico_loss[0]:.4f}")
print(f"loss final   : {historico_loss[-1]:.4f}")
print(f"reducao      : {reducao:.1f}%")

print("\n" + "=" * 60)
print("TAREFA 4 — Overfitting Test")
print("=" * 60)

frase_teste_en = subset[0]["translation"]["en"]
frase_teste_pt = subset[0]["translation"]["pt"]

src_ids_teste = tokenizer(
    frase_teste_en,
    max_length=MAX_LEN,
    truncation=True,
    padding="max_length",
    return_tensors="pt",
)["input_ids"].squeeze(0)

traducao_gerada = run_inference(modelo, src_ids_teste)

print(f"\nfrase original (en) : {frase_teste_en}")
print(f"traducao esperada   : {frase_teste_pt}")
print(f"traducao gerada     : {traducao_gerada}")