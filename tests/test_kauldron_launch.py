import contextlib
import io
import unittest
from unittest import mock

from sunfish_tpu import kauldron_launch
from sunfish_tpu.tpu_preflight import Check


class KauldronLaunchTests(unittest.TestCase):
    def test_topology_check_precedes_kauldron_import(self):
        events = []

        def checks(**kwargs):
            del kwargs
            events.append("distributed-initialize-and-topology")
            return [Check("jax-distributed-initialize", "pass", "ok")]

        def run_module(name, **kwargs):
            del kwargs
            events.append(f"import:{name}")

        with (
            mock.patch.object(kauldron_launch, "_jax_checks", side_effect=checks),
            mock.patch.object(kauldron_launch.runpy, "run_module", side_effect=run_module),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            kauldron_launch.main(
                [
                    "--expected-devices",
                    "64",
                    "--expected-processes",
                    "8",
                    "--",
                    "--cfg=config.py",
                ]
            )

        self.assertEqual(
            events,
            ["distributed-initialize-and-topology", "import:kauldron.main"],
        )


if __name__ == "__main__":
    unittest.main()
