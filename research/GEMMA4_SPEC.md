# Gemma 4 26B-A4B Implementation Spec (from actual GGUF dump)

## Architecture: gemma4

## Metadata (from BugTraceAI-Apex-G4-26B-Q4.gguf)

```
general.architecture = gemma4
gemma4.block_count = 30
gemma4.context_length = 262144
gemma4.embedding_length = 2816        (hidden)
gemma4.feed_forward_length = 2112     (dense FFN intermediate)
gemma4.attention.head_count = 16      (query heads, all layers)
gemma4.attention.head_count_kv = [8,8,8,8,8,2, 8,8,8,8,8,2, 8,8,8,8,8,2, 8,8,8,8,8,2, 8,8,8,8,8,2]
gemma4.rope.freq_base = 1000000.0     (global layers)
gemma4.rope.freq_base_swa = 10000.0   (sliding layers)
gemma4.attention.layer_norm_rms_epsilon = 1e-6
gemma4.expert_count = 128
gemma4.expert_used_count = 8
gemma4.attention.key_length = 512     (global head_dim)
gemma4.attention.value_length = 512
gemma4.attention.key_length_swa = 256 (sliding head_dim)
gemma4.attention.value_length_swa = 256
gemma4.final_logit_softcapping = 30.0
gemma4.attention.sliding_window = 1024
gemma4.attention.sliding_window_pattern = [1,1,1,1,1,0, 1,1,1,1,1,0, 1,1,1,1,1,0, 1,1,1,1,1,0, 1,1,1,1,1,0]
    (1 = sliding, 0 = global; layers 5,11,17,23,29 are global)
gemma4.expert_feed_forward_length = 704
gemma4.rope.dimension_count = 512     (global)
gemma4.rope.dimension_count_swa = 256 (sliding)
```

## Layer types

30 layers. Pattern: 5 sliding, 1 global, repeating. Global layers: 5, 11, 17, 23, 29.

Sliding layers: head_dim=256, n_kv_heads=8, sliding_window=1024, RoPE base=10000
Global layers:  head_dim=512, n_kv_heads=2, full attention,       RoPE base=1000000

Global layers have NO attn_v tensor (unified K/V: K output reused as V).

## Tensor layout per layer

### Sliding layer (e.g. blk.0)

```
blk.{i}.attn_norm.weight          F32 [2816]           pre-attention RMSNorm
blk.{i}.attn_q.weight             Q4_K [2816, 4096]    Q proj: 16 heads * 256 dim
blk.{i}.attn_k.weight             Q4_K [2816, 2048]    K proj: 8 heads * 256 dim
blk.{i}.attn_v.weight             Q6_K [2816, 2048]    V proj: 8 heads * 256 dim
blk.{i}.attn_q_norm.weight        F32 [256]            Q RMSNorm (per-head)
blk.{i}.attn_k_norm.weight        F32 [256]            K RMSNorm (per-head)
blk.{i}.attn_output.weight        Q4_K [4096, 2816]    O proj
blk.{i}.post_attention_norm.weight F32 [2816]          post-attention norm
blk.{i}.ffn_norm.weight           F32 [2816]           pre-FFN norm (same as post_attention_norm? separate tensor)
blk.{i}.ffn_gate.weight           Q4_K [2816, 2112]    dense FFN gate
blk.{i}.ffn_up.weight             Q4_K [2816, 2112]    dense FFN up
blk.{i}.ffn_down.weight           Q8_0 [2112, 2816]    dense FFN down
blk.{i}.ffn_gate_inp.weight       F32 [2816, 128]      MoE router
blk.{i}.ffn_gate_inp.scale        F32 [2816]           router pre-scale
blk.{i}.ffn_gate_up_exps.weight   Q4_K [2816, 1408, 128]  fused gate+up, 128 experts (1408 = 704*2)
blk.{i}.ffn_down_exps.weight      Q8_0 [704, 2816, 128]   down for 128 experts
blk.{i}.ffn_down_exps.scale       F32 [128]            per-expert down scale
blk.{i}.layer_output_scale.weight F32 [1]              layer output scale
blk.{i}.post_ffw_norm.weight      F32 [2816]
blk.{i}.post_ffw_norm_1.weight    F32 [2816]           post dense FFN norm
blk.{i}.post_ffw_norm_2.weight    F32 [2816]           post MoE norm
blk.{i}.pre_ffw_norm_2.weight     F32 [2816]           pre MoE router norm
```

### Global layer (e.g. blk.5) - differences only

```
blk.5.attn_q.weight             Q4_K [2816, 8192]    Q proj: 16 heads * 512 dim
blk.5.attn_k.weight             Q4_K [2816, 1024]    K proj: 2 heads * 512 dim
(NO attn_v - K reused as V)
blk.5.attn_q_norm.weight        F32 [512]
blk.5.attn_k_norm.weight        F32 [512]
blk.5.attn_output.weight        Q4_K [8192, 2816]    O proj
```
FFN/MoE tensors are identical to sliding layers.

### Global tensors

```
token_embd.weight    Q6_K [2816, 262144]   token embedding (tied to output)
output_norm.weight   F32 [2816]            final RMSNorm
```
No separate output weight - tied to token_embd.

## Forward pass per layer

1. Pre-attention RMSNorm: x_norm = rmsnorm(x, attn_norm, eps=1e-6)
2. Q/K/V projections
   - Sliding: Q=[M,4096], K=[M,2048], V=[M,2048]
   - Global: Q=[M,8192], K=[M,1024], V=K (no V proj)
3. Q/K RMSNorm (per-head, head_dim sized)
4. RoPE: sliding uses base=10000, dim=256; global uses base=1000000, dim=512
5. Attention:
   - Sliding: only attend to tokens within window=1024
   - Global: attend to all tokens
6. Output projection, add to residual
7. Post-attention norm: post_attention_norm
8. FFN block (see below)
9. Add to residual, apply post_ffw_norm

## FFN block (MoE + dense)

The FFN block has two parallel paths summed:

### Dense FFN path (always active)
```
gate = ffn_gate(x)     # [M, 2112]
up   = ffn_up(x)       # [M, 2112]
hidden = geglu(gate, up)  # gelu_tanh(gate) * up
dense_out = ffn_down(hidden)  # [M, 2816]
dense_out = post_ffw_norm_1(dense_out)
```

### MoE path (top-8 of 128 experts)
```
# Router
x_router = rmsnorm(x, pre_ffw_norm_2, eps=1e-6)   # note: separate norm
x_router = x_router * ffn_gate_inp.scale           # [2816]
x_router = x_router * (1.0 / sqrt(2816))           # scale by 1/sqrt(hidden)
logits = ffn_gate_inp @ x_router                   # [128] expert logits
probs = softmax(logits)                             # over all 128
top8_weights, top8_idx = topk(probs, k=8)
top8_weights = top8_weights / sum(top8_weights)     # renormalize

# Expert computation (per selected expert e):
#   ffn_gate_up_exps[:,:,e] is [2816, 1408] = fused gate+up
#   gate_part = first 704 cols, up_part = next 704 cols
#   expert_hidden = geglu(gate_part, up_part)        # [M, 704]
#   expert_out = ffn_down_exps[:,:,e] @ expert_hidden # [M, 2816]
#   expert_out *= ffn_down_exps.scale[e]              # per-expert scale
#   expert_out *= top8_weight[e]

moe_out = sum of all selected expert outputs
moe_out = post_ffw_norm_2(moe_out)
```

### Combined FFN output
```
ffn_out = dense_out + moe_out
ffn_out = post_ffw_norm(ffn_out)
ffn_out *= layer_output_scale  # scalar
```

## Embedding and output

- Token embedding: scale by sqrt(hidden) = sqrt(2816) after lookup
- Final logit softcapping: logits = tanh(logits / 30.0) * 30.0
- Output weight tied to token_embd

## Activation: GeGLU (gelu_tanh)

```
gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
geglu(gate, up) = gelu_tanh(gate) * up
```

## GGUF dims convention

GGUF dims are reversed: dims[0] is innermost (input), dims[1] is outermost (output).
For 3D tensors (experts): dims[0]=input, dims[1]=output, dims[2]=expert count.

## p-RoPE details

Standard RoPE but with different freq_base per layer type.
The freq exponent denominator is the full head_dim (not rope_dim * proportion).
For global layers: only 25% of dims rotated (rope_dim_count=512, but proportion=0.25 means 128 dims rotated? Need to verify from GGUF - the metadata says rope.dimension_count=512 for global and 256 for sliding, which might mean all dims are rotated.)

Actually from the metadata: rope.dimension_count = 512 (global) and rope.dimension_count_swa = 256 (sliding). These match the head_dims. So all head_dim dimensions get RoPE, just with different freq bases.
