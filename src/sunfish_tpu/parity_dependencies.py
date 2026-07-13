"""Exact direct dependency contract for the Stage-0 PyTorch parity host."""

PARITY_RUNTIME_VERSIONS = {
    "accelerate": "1.14.0",
    "huggingface-hub": "1.23.0",
    "numpy": "2.5.1",
    "safetensors": "0.8.0",
    "sentencepiece": "0.2.2",
    "tokenizers": "0.23.0",
    "torch": "2.13.0",
    "transformers": "5.13.0",
}

MODEL_CLASS = "DiffusionGemmaForBlockDiffusion"
TRANSFORMERS_RELEASE = "5.13.0"
