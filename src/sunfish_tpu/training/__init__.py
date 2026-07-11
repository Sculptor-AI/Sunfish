"""Production training harness for Sunfish.

The modules in this package are intentionally separate from ``sunfish`` core:
they import JAX, Flax, Grain, Orbax, Gemma, and Kauldron on TPU workers.  The
lightweight run specification and record envelope remain stdlib-only so they
can be validated before a JAX backend is touched.
"""

from sunfish_tpu.training.spec import HarnessConfig, Phase

__all__ = ["HarnessConfig", "Phase"]
