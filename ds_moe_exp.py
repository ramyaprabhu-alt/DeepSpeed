import torch
import torch.nn as nn
from torch.nn import functional as F

import torch
from numpy import random
import argparse
import math
import inspect
from dataclasses import dataclass

parser = argparse.ArgumentParser(description='DeepSpeed ')
parser.add_argument('--n_embd', dest='n_embd', default=12288,type=int,help='Hidden Size')
parser.add_argument('--vocab_size', dest='vocab_size', default=51200,type=int,help='Hidden Size')
parser.add_argument('--n_head', dest='n_head', default=96,type=int, help='Hidden Size')
parser.add_argument('--num_experts', dest='num_experts', type=int, default=16,help='Hidden Size')
parser.add_argument('--bias', dest = 'bias', action='store_true')
parser.set_defaults(bias=False)
parser.add_argument('--dropout', dest='dropout', default=0.0, type = float, help='Hidden Size')
parser.add_argument('--ep_size', dest='ep_size', default=1,type=int, help='Hidden Size')
parser.add_argument('--is_moe', dest = 'is_moe', action='store_true')
parser.set_defaults(is_moe=False)
parser.add_argument('--use_cache', dest = 'use_cache', action='store_true')
parser.add_argument('--top_k_val', dest='top_k_val', default=2, type = int,help='Hidden Size')
parser.add_argument('--block_size', dest='block_size', default=1024, type = int, help='Hidden Size')
parser.add_argument('--n_layers', dest='n_layers', default=1, type = int, help='Hidden Size')
parser.add_argument('--batch_size', dest='batch_size', default=1, type=int,help='Hidden Size')
parser.add_argument('--max_gen_tokens', dest='max_gen_tokens', default=5, type=int,help='Hidden Size')
parser.add_argument('--seq_len', dest='seq_len', default=1024, type = int,help='Hidden Size')
parser.add_argument('--cap_fact', dest='capacity_factor', default=1.0, type = float,help='Hidden Size')
parser.add_argument('--file_name', dest='file_name', default=None, type = str,help='Hidden Size')
args = parser.parse_args()

import deepspeed
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, kv_cache, cache):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        if not kv_cache or cache == {}:
            q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            if(kv_cache):
                cache["prev_k"] = k
                cache["prev_v"] = v
        else:
            q, k, v  = self.c_attn(x[:,-1,:].reshape(B,1,C)).split(self.n_embd, dim=2)
            k = k.view(B, 1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, 1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, 1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            T = 1
            k_prev = cache["prev_k"]
            v_prev = cache["prev_v"]
            if(kv_cache):
                assert bool(cache)
                cache["prev_k"] = torch.cat((cache["prev_k"], k), dim = 2)
                cache["prev_v"] = torch.cat((cache["prev_v"], v), dim = 2)
                k = torch.cat((k_prev, k), dim = 2)
                v = torch.cat((v_prev, v), dim = 2)
               
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # print("flash be flashin")
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        if y.shape[1]>args.seq_len:
            y = y[:,:1,:]
        # output projection
        y = self.resid_dropout(y)
  
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        deepspeed.init_distributed()
        # x = torch.Tensor(random.randint(100, size=(config.batch_size,config.block_size+config.max_gen_tokens, config.n_embd))).to("cuda")
        self.kv_cache =  {}
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        if (config.is_moe):
            self.mlp = deepspeed.moe.layer.MoE(hidden_size=config.n_embd, expert=self.mlp, 
                                               num_experts=config.num_experts ,ep_size=config.ep_size, 
                                               k=config.tok_k_val, eval_capacity_factor=config.capacity_factor,
                                               capacity_factor=config.capacity_factor)



    def forward(self, x):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        x= self.attn(self.ln_1(x), kv_cache = args.use_cache, cache = self.kv_cache)
        # x = x+y
        end.record()
        torch.cuda.synchronize()
        t = start.elapsed_time(end)
       
        # print(type(x))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if not config.is_moe: 
            x = self.mlp(self.ln_2(x))
        elif config.is_moe:
            # print("139")
            # print(x.shape)
            y = self.mlp(self.ln_2(x))
            x = y[0]
        else:
            raise NotImplementedError
        end.record()
        torch.cuda.synchronize()
        t2 = start.elapsed_time(end)
        return x, t, t2
    
    def clear_cache(self):
        self.kv_cache = {}

class TFRMRConfig():
    def __init__(self, args):   
        self.n_embd = args.n_embd
        self.vocab_size = args.vocab_size
        self.n_head = args.n_head
        self.num_experts = args.num_experts
        self.bias = args.bias
        self.dropout = args.dropout
        self.ep_size = args.ep_size
        # print(self.ep_size)
        self.is_moe = args.is_moe
        self.tok_k_val = args.top_k_val
        self.block_size = args.block_size
        self.n_layers = args.n_layers
        self.batch_size = args.batch_size
        self.max_gen_tokens = args.max_gen_tokens
        self.seq_len = args.seq_len
        self.capacity_factor = args.capacity_factor
        self.file_name = args.file_name

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config =  config
        self.transformer = nn.ModuleDict(dict(
                # wte = nn.Embedding(config.vocab_size, config.n_embd, dtype=torch.half).to(torch.half),
                # wpe = nn.Embedding(config.block_size, config.n_embd, dtype=torch.half),
                # drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                ln_f = LayerNorm(config.n_embd, bias=config.bias),
            ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    @torch.no_grad()
    def forward(self, idx, targets=None):
        device = idx.device
        b, t, c = idx.size()
        # assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.half, device=device) # shape (t)

        # forward the GPT model itself
        # tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        # pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        # x = self.transformer.drop(tok_emb + pos_emb)
        x = idx
        for block in self.transformer.h:
            x,t, t2 = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss, t, t2
     
    def clear_cache(self):
        for block in self.transformer.h:
            block.clear_cache()
        


print(args)
# print(type(args.seq_len))
config = TFRMRConfig(args)
model = Model(config)
# model.half()
print(model.eval())
model.half()
model.to("cuda")
deepspeed.init_distributed()
temperature = 1.0
max_new_tokens = 5
top_k = 5
TIME = []
TIME_1 = []
TIME_2 = []
for i in range(3):
    #x = random.randn(1, size=(config.batch_size,256, config.n_embd))
    #y = random.randint(100, size=(config.batch_size,1, config.n_embd))
    # idx = torch.randint(high = 51200, size=(config.batch_size, config.seq_len), dtype=torch.half, device = "cuda")
    # y = torch.randint(high = 51200, size=(config.batch_size, 1), dtype=torch.half, device = "cuda")
    idx = torch.randn(config.batch_size, config.seq_len, config.n_embd, dtype=torch.half, device = "cuda")
    y = torch.randn(config.batch_size, 1, config.n_embd, dtype=torch.half, device = "cuda")
  
    time = []
    time_1 = []
    time_2 = []
    model.clear_cache()
    for j in range(config.max_gen_tokens):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                logits , _, t2, t3 = model(idx)
                end.record()
                torch.cuda.synchronize()
                print(idx.shape)
                t=start.elapsed_time(end)
                time.append(t)
                time_1.append(t2)
                time_2.append(t3)
                logits = logits[:, -1, :] / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
                # append sampled index to the running sequence and continue
                # idx_next = torch.Tensor([[idx[0,0]]])
                # print(idx)
                # print(idx.shape)            
                idx = torch.cat((idx, y), dim=1)


    TIME.append(time)
    TIME_1.append(time_1)
    TIME_2.append(time_2)

time = TIME[1:]
time_1 = TIME_1[1:]
time_2 = TIME_2[1:]
# print(time)
import numpy as np
l = [np.matrix(time).mean(0).tolist()[0][0]]
l.append(sum(np.matrix(time).mean(0).tolist()[0][1:])/len(np.matrix(time).mean(0).tolist()[0][1:]))
l.append(np.matrix(time_1).mean(0).tolist()[0][0])
l.append(sum(np.matrix(time_1).mean(0).tolist()[0][1:])/len(np.matrix(time_1).mean(0).tolist()[0][1:]))
l.append(np.matrix(time_2).mean(0).tolist()[0][0])
l.append(sum(np.matrix(time_2).mean(0).tolist()[0][1:])/len(np.matrix(time_2).mean(0).tolist()[0][1:]))
if (args.is_moe):
    l.append(args.num_experts)
else:
    l.append(0)
l.append(args.seq_len)
l.append(args.batch_size)
print(l)
if config.file_name != None:
    file = config.file_name
elif config.is_moe:
    file="moe_experiment_results.csv"
else:
    file = "dense_experiment_result.csv"

# with open(file, 'a+') as f:
import csv

# with open(file, 'a') as csvfile: 
#     csvwriter = csv.writer(csvfile) 
#     print(type(l))
#     csvwriter.writerow(l)
