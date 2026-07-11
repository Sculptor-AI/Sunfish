"""Parameter-budget calculations for DiffusionGemma-style MoE students."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DiffusionMoEBudget:
    """Compute total and active parameters after structured expert pruning."""

    hidden_size: int = 2816
    expert_intermediate_size: int = 704
    num_layers: int = 30
    source_experts: int = 128
    source_top_k: int = 8
    # Audited from the released shard headers on 2026-07-10 (see
    # reference/upstream/): total 25,823,781,228 minus the 572,794,416-param
    # vision tower. Includes the 60 encoder/decoder per-layer scalars the
    # original design estimate missed.
    source_text_parameters: int = 25_250_986_812

    @property
    def parameters_per_expert_per_layer(self) -> int:
        # Gated MLP: gate projection, up projection, and down projection.
        return 3 * self.hidden_size * self.expert_intermediate_size

    @property
    def parameters_per_expert(self) -> int:
        return self.parameters_per_expert_per_layer * self.num_layers

    @property
    def source_sparse_expert_parameters(self) -> int:
        return self.parameters_per_expert * self.source_experts

    @property
    def source_router_parameters(self) -> int:
        return self.router_parameters(self.source_experts)

    @property
    def shared_text_parameters(self) -> int:
        """Parameters invariant to expert count, excluding sparse banks/router."""
        return (
            self.source_text_parameters
            - self.source_sparse_expert_parameters
            - self.source_router_parameters
        )

    def router_parameters(self, experts: int) -> int:
        # Router projection, d-wide router scale, and one scale per expert.
        per_layer = self.hidden_size * experts + self.hidden_size + experts
        return self.num_layers * per_layer

    def estimate(self, *, experts: int, top_k: int) -> dict[str, int | float]:
        if experts <= 0:
            raise ValueError("experts must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if experts > self.source_experts:
            raise ValueError("experts cannot exceed source_experts")
        if top_k > experts:
            raise ValueError("top_k cannot exceed experts")

        router = self.router_parameters(experts)
        total = self.shared_text_parameters + router + experts * self.parameters_per_expert
        active = self.shared_text_parameters + router + top_k * self.parameters_per_expert
        return {
            "experts": experts,
            "top_k": top_k,
            "parameters_per_expert": self.parameters_per_expert,
            "shared_text_parameters": self.shared_text_parameters,
            "router_parameters": router,
            "total_parameters": total,
            "active_parameters": active,
            "total_billions": round(total / 1_000_000_000, 3),
            "active_billions": round(active / 1_000_000_000, 3),
        }

    def describe(self, *, experts: int, top_k: int) -> dict[str, object]:
        return {
            "assumptions": asdict(self),
            "estimate": self.estimate(experts=experts, top_k=top_k),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    budget = DiffusionMoEBudget()
    print(json.dumps(budget.describe(experts=args.experts, top_k=args.top_k), indent=2))


if __name__ == "__main__":
    main()
