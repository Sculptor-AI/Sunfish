from pathlib import Path
import tomllib
import unittest


class TpuDependencyLockTests(unittest.TestCase):
    def test_every_review_dependency_is_exactly_pinned(self):
        root = Path(__file__).resolve().parents[1]
        with (root / "pyproject.toml").open("rb") as source:
            project = tomllib.load(source)
        requirements = project["project"]["optional-dependencies"]["tpu"]
        expected_names = {
            "etils",
            "gemma",
            "google-cloud-storage",
            "grain",
            "hackable-diffusion",
            "jax",
            "jaxlib",
            "kauldron",
            "libtpu",
            "orbax-checkpoint",
        }
        names = {requirement.split("==", 1)[0].split("[", 1)[0] for requirement in requirements}
        self.assertEqual(names, expected_names)
        self.assertTrue(all("==" in requirement for requirement in requirements))

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


if __name__ == "__main__":
    unittest.main()
