# Lab 5 - Treinamento Fim-a-Fim do Transformer

Este projeto encerra a Unidade I. Ele instancia a arquitetura Transformer construída no Lab 04, conecta-a a um dataset real do Hugging Face e executa o loop completo de treinamento com backpropagation e otimizador Adam, provando que a arquitetura consegue aprender através da queda da função de perda (Cross-Entropy Loss).

## Pré-requisitos

```
pip install torch datasets transformers
```

## Estrutura do Projeto

- `lab5.py`: Pipeline completo — dataset, tokenização, Transformer, training loop e overfitting test.

## Como Executar

```
python3 lab5.py
```

Recomendado rodar no **Google Colab** (gratuito) para maior velocidade. O script detecta automaticamente se há GPU disponível.

O script irá:

1. Carregar as 1.000 primeiras frases do dataset `Helsinki-NLP/opus_books` (en → pt).
2. Tokenizar os pares usando `bert-base-multilingual-cased`.
3. Treinar o Transformer por 15 épocas imprimindo o loss a cada época.
4. Executar o overfitting test com uma frase do conjunto de treino.

## Arquitetura

O Transformer foi reescrito em PyTorch para suportar backpropagation, mantendo exatamente a mesma lógica e nomes das classes dos laboratórios anteriores.

| Componente         | Origem  |
|--------------------|---------|
| SelfAttention      | Lab 1   |
| Causal Mask        | Lab 3   |
| CrossAttention     | Lab 3   |
| FFN + Add & Norm   | Lab 2   |
| BlocoEncoder       | Lab 2   |
| BlocoDecoder       | Lab 4   |
| run_inference      | Lab 4   |

## Hiperparâmetros

| Parâmetro   | Valor |
|-------------|-------|
| d_model     | 128   |
| d_ff        | 256   |
| n_camadas   | 2     |
| n_heads     | 4     |
| epochs      | 15    |
| batch_size  | 32    |
| lr          | 1e-3  |
| max_len     | 40    |
| subset      | 1000  |

## Resultado Esperado

A curva de loss deve cair significativamente ao longo das épocas. O overfitting test demonstra que a arquitetura assimila os padrões do conjunto de treino ao "memorizar" a tradução de uma frase específica.

## Ferramentas utilizadas

- `datasets` e `transformers` (Hugging Face): carregamento do dataset e tokenização (Tarefas 1 e 2).
- `torch`: backpropagation, otimizador Adam e CrossEntropyLoss (Tarefa 3 e 4).
- Arquitetura Transformer: construída integralmente sobre os Labs 1–4.
