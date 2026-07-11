from pathlib import Path
import tomllib
import unittest

from sunfish_tpu.training.dependencies import (
    GEMMA_SOURCE_COMMIT,
    RUNTIME_VERSIONS,
    TPU_ONLY_RUNTIME_VERSIONS,
)


class TpuDependencyLockTests(unittest.TestCase):
    def test_every_review_dependency_is_exactly_pinned(self):
        root = Path(__file__).resolve().parents[1]
        with (root / "pyproject.toml").open("rb") as source:
            project = tomllib.load(source)
        requirements = project["project"]["optional-dependencies"]["tpu"]
        expected_names = {
            "dialog",
            "etils",
            "flax",
            "gemma",
            "google-cloud-storage",
            "grain",
            "hackable-diffusion",
            "jax",
            "jaxlib",
            "kauldron",
            "libtpu",
            "numpy",
            "optax",
            "orbax-checkpoint",
            "sentencepiece",
        }
        names = {
            requirement.split(" @ ", 1)[0].split("==", 1)[0].split("[", 1)[0]
            for requirement in requirements
        }
        self.assertEqual(names, expected_names)
        for requirement in requirements:
            if " @ " in requirement:
                self.assertRegex(requirement, r"@[0-9a-f]{40}$")
            else:
                self.assertIn("==", requirement)

    def test_lock_matches_project_direct_pins(self):
        root = Path(__file__).resolve().parents[1]
        with (root / "pyproject.toml").open("rb") as source:
            project = tomllib.load(source)
        project_pins = set(project["project"]["optional-dependencies"]["tpu"])
        lock_pins = {
            line
            for line in (root / "requirements-tpu.lock").read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        }
        self.assertEqual(lock_pins, project_pins)

    def test_bootstrap_separates_unreleased_gemma_from_resolver(self):
        root = Path(__file__).resolve().parents[1]
        base = (root / "requirements-tpu-base.lock").read_text(encoding="utf-8")
        source = (root / "requirements-gemma-source.lock").read_text(encoding="utf-8")
        bootstrap = (root / "scripts/bootstrap_tpu.sh").read_text(encoding="utf-8")
        self.assertNotIn("gemma @", base)
        self.assertRegex(source, r"gemma @ .*@[0-9a-f]{40}")
        self.assertIn("--no-deps --requirement requirements-gemma-source.lock", bootstrap)

    def test_runtime_contract_matches_every_direct_distribution(self):
        root = Path(__file__).resolve().parents[1]
        pins = {}
        for line in (root / "requirements-tpu.lock").read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            if " @ " in line:
                name, source = line.split(" @ ", 1)
                pins[name] = "4.1.0"
                self.assertTrue(source.endswith(GEMMA_SOURCE_COMMIT))
            else:
                requirement_name, version = line.split("==", 1)
                name = requirement_name.split("[", 1)[0]
                pins[name] = version
        self.assertEqual(pins, {**RUNTIME_VERSIONS, **TPU_ONLY_RUNTIME_VERSIONS})


if __name__ == "__main__":
    unittest.main()
