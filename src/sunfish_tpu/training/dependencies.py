"""Dependency-free source and version contract for the training runtime."""

GEMMA_SOURCE_COMMIT = "09e7b48ae88720f6236b8266c7213eb51bb62b87"

RUNTIME_VERSIONS = {
    "dialog": "1.1.0",
    "etils": "1.14.0",
    "flax": "0.12.7",
    "gemma": "4.1.0",
    "google-cloud-storage": "3.12.1",
    "grain": "0.2.18",
    "hackable-diffusion": "1.0.1",
    "jax": "0.10.2",
    "jaxlib": "0.10.2",
    "kauldron": "1.4.4",
    "numpy": "2.4.6",
    "optax": "0.2.8",
    "orbax-checkpoint": "0.12.1",
    "sentencepiece": "0.2.1",
}

TPU_ONLY_RUNTIME_VERSIONS = {
    "libtpu": "0.0.42.1",
}
