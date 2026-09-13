"""Package data: tokenizer (char-level & BPE) dan dataset untuk language modeling."""

from .tokenizer import CharTokenizer
from .bpe_tokenizer import BPETokenizer

_TOKENIZERS = {"char": CharTokenizer, "bpe": BPETokenizer}


def get_tokenizer_class(name: str):
    """Ambil class tokenizer berdasarkan nama di config ("char" atau "bpe")."""
    try:
        return _TOKENIZERS[name]
    except KeyError:
        raise ValueError(
            f"tokenizer tidak dikenal: {name!r} (pilihan: {list(_TOKENIZERS)})"
        ) 