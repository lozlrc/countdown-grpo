"""Model and tokenizer construction.

model.name == "tiny-random" builds a small randomly initialized Qwen2-style
model plus a character-level tokenizer, entirely offline — used by the debug
config and the tests. Anything else is treated as a HuggingFace model id.
"""

from __future__ import annotations

import string

import torch
from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers
from tokenizers.models import WordLevel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from grpo.config import ModelCfg

PAD, EOS, UNK = "<pad>", "<eos>", "<unk>"


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(pref: str, device: torch.device) -> torch.dtype:
    if pref != "auto":
        return getattr(torch, pref)
    # bf16 training without a proper mixed-precision setup is unstable on
    # MPS; keep full precision off-GPU.
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def build_char_tokenizer() -> PreTrainedTokenizerFast:
    """Character-level tokenizer covering printable ASCII. No downloads."""
    alphabet = sorted(set(string.digits + string.ascii_letters + string.punctuation + " \n"))
    vocab = {PAD: 0, EOS: 1, UNK: 2}
    for ch in alphabet:
        vocab[ch] = len(vocab)
    tok = Tokenizer(WordLevel(vocab, unk_token=UNK))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tok.decoder = decoders.Fuse()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token=PAD, eos_token=EOS, unk_token=UNK
    )
    fast.padding_side = "left"
    return fast


def build_tiny_model(
    vocab_size: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    seed: int | None = None,
) -> Qwen2ForCausalLM:
    if seed is not None:
        torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        pad_token_id=0,
        eos_token_id=1,
        attn_implementation="eager",
    )
    return Qwen2ForCausalLM(cfg)


def load_model_and_tokenizer(cfg: ModelCfg, device: torch.device):
    if cfg.name == "tiny-random":
        tokenizer = build_char_tokenizer()
        model = build_tiny_model(vocab_size=len(tokenizer), seed=cfg.init_seed)
        model.to(device)
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = pick_dtype(cfg.dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.name, dtype=dtype, attn_implementation=cfg.attn_implementation
    )
    model.to(device)
    return model, tokenizer
