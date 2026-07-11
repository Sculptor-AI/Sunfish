import unittest
from unittest import mock

from sunfish_tpu.tpu_preflight import (
    Check,
    _jax_checks,
    _topology_checks,
    report,
    validate_gcs_uri,
)


class _FakeDevice:
    def __init__(self, process_index, platform="tpu"):
        self.process_index = process_index
        self.platform = platform
        self.device_kind = "fake-tpu"


class _FakeArray:
    def __init__(self, values):
        self.values = values

    def block_until_ready(self):
        return self

    def __getitem__(self, index):
        return self.values[index]


class _FakeDistributed:
    def __init__(self, events):
        self.events = events
        self.initialized = False

    def is_initialized(self):
        return self.initialized

    def initialize(self):
        self.events.append("initialize")
        self.initialized = True


class _FakeNumpy:
    int32 = "int32"

    def __init__(self, events, distributed):
        self.events = events
        self.distributed = distributed

    def ones(self, shape, dtype):
        self.events.append("jax.numpy.ones")
        if not self.distributed.initialized:
            raise AssertionError("backend touched before distributed initialize")
        return [1] * shape[0]


class _FakeLax:
    def __init__(self, device_count):
        self.device_count = device_count

    def psum(self, value, axis_name):
        del value, axis_name
        return self.device_count


class _FakeJax:
    def __init__(self, *, processes=2, devices_per_process=2):
        self.events = []
        self.distributed = _FakeDistributed(self.events)
        self._processes = processes
        self._devices_per_process = devices_per_process
        self._devices = [
            _FakeDevice(process)
            for process in range(processes)
            for _ in range(devices_per_process)
        ]
        self.lax = _FakeLax(len(self._devices))

    def devices(self):
        self.events.append("devices")
        if not self.distributed.initialized:
            raise AssertionError("jax.devices called before distributed initialize")
        return self._devices

    def local_devices(self):
        self.events.append("local_devices")
        return self._devices[: self._devices_per_process]

    def process_count(self):
        return self._processes

    def process_index(self):
        return 0

    def pmap(self, function, axis_name):
        self.events.append("pmap")

        def mapped(values):
            return _FakeArray([function(value) for value in values])

        return mapped


class GcsUriTests(unittest.TestCase):
    def test_valid_project_prefix(self):
        self.assertEqual(
            validate_gcs_uri("gs://sunfish-checkpoints/training/stage-0"),
            ("sunfish-checkpoints", "training/stage-0"),
        )

    def test_bucket_root_and_non_gcs_are_rejected(self):
        for value in ("gs://bucket", "https://bucket/path", "gs:///path"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_gcs_uri(value)


class ReportTests(unittest.TestCase):
    def test_failure_controls_readiness(self):
        payload = report(
            [
                Check("one", "pass", "ok"),
                Check("two", "warn", "check this"),
                Check("three", "fail", "broken"),
            ]
        )
        self.assertFalse(payload["ready"])
        self.assertEqual(payload["summary"], {"pass": 1, "warn": 1, "fail": 1})

    def test_warnings_do_not_block_local_inspection(self):
        payload = report([Check("cpu", "warn", "not a TPU")])
        self.assertTrue(payload["ready"])


class DistributedJaxTests(unittest.TestCase):
    def test_initialize_happens_before_backend_access(self):
        fake_jax = _FakeJax()
        fake_jnp = _FakeNumpy(fake_jax.events, fake_jax.distributed)

        def import_module(name):
            if name == "jax":
                return fake_jax
            if name == "jax.numpy":
                return fake_jnp
            raise AssertionError(name)

        with mock.patch(
            "sunfish_tpu.tpu_preflight.importlib.import_module",
            side_effect=import_module,
        ):
            checks = _jax_checks(
                require_tpu=True,
                require_distributed=True,
                expected_devices=4,
                expected_processes=2,
                expected_local_devices=2,
            )

        self.assertLess(fake_jax.events.index("initialize"), fake_jax.events.index("devices"))
        self.assertTrue(all(check.status == "pass" for check in checks), checks)

    def test_topology_rejects_missing_process_index(self):
        fake_jax = _FakeJax()
        fake_jax.distributed.initialized = True
        for device in fake_jax._devices[-2:]:
            device.process_index = 0
        fake_jnp = _FakeNumpy(fake_jax.events, fake_jax.distributed)
        checks = _topology_checks(
            fake_jax,
            fake_jnp,
            require_tpu=True,
            expected_devices=4,
            expected_processes=2,
            expected_local_devices=2,
        )
        by_name = {check.name: check for check in checks}
        self.assertEqual(by_name["jax-unique-process-indices"].status, "fail")

    def test_real_psum_result_is_gated(self):
        fake_jax = _FakeJax()
        fake_jax.distributed.initialized = True
        fake_jax.lax.device_count = 3
        fake_jnp = _FakeNumpy(fake_jax.events, fake_jax.distributed)
        checks = _topology_checks(
            fake_jax,
            fake_jnp,
            require_tpu=True,
            expected_devices=4,
            expected_processes=2,
            expected_local_devices=2,
        )
        by_name = {check.name: check for check in checks}
        self.assertEqual(by_name["jax-global-psum"].status, "fail")


if __name__ == "__main__":
    unittest.main()
