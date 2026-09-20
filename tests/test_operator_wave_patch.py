import math
import struct
import unittest

from operator_training.wave_patch import (
    DEFAULT_EXTENT_M,
    DEFAULT_GRID_SIZE,
    DEFAULT_SPACING_M,
    WAVE_PATCH_VERSION,
    WavePatchError,
    WavePatchFrame,
    WavePatchSpec,
    decode_wave_patch,
    decode_wave_patch_header,
    encode_wave_patch,
    encode_wave_patch_message,
    patch_extent_matches,
)


class WavePatchSpecTests(unittest.TestCase):
    def test_default_spec_is_valid(self):
        spec = WavePatchSpec()
        spec.validate()
        self.assertEqual(spec.size, DEFAULT_GRID_SIZE)
        self.assertAlmostEqual(spec.spacing_m, DEFAULT_SPACING_M)

    def test_spacing_formula(self):
        spec = WavePatchSpec(size=65, extent_m=128.0)
        self.assertAlmostEqual(spec.spacing_m, 2.0)
        spec = WavePatchSpec(size=33, extent_m=64.0)
        self.assertAlmostEqual(spec.spacing_m, 2.0)

    def test_even_size_rejected(self):
        with self.assertRaises(WavePatchError):
            WavePatchSpec(size=64).validate()

    def test_size_below_four_rejected(self):
        with self.assertRaises(WavePatchError):
            WavePatchSpec(size=3).validate()

    def test_extent_too_small_rejected(self):
        with self.assertRaises(WavePatchError):
            WavePatchSpec(size=5, extent_m=4.0).validate()

    def test_extent_too_large_rejected(self):
        with self.assertRaises(WavePatchError):
            WavePatchSpec(size=5, extent_m=2000.0).validate()

    def test_to_dict_round_trip(self):
        spec = WavePatchSpec()
        d = spec.to_dict()
        self.assertEqual(d["size"], spec.size)
        self.assertAlmostEqual(d["spacing_m"], spec.spacing_m)


class WavePatchEncodeTests(unittest.TestCase):
    def _make_frame(self, *, size=5, sim_t=1.5):
        spec = WavePatchSpec(size=size, extent_m=(size - 1) * 2.0)
        heights = tuple(float(i * 0.01) for i in range(size * size))
        return WavePatchFrame(
            sim_time_s=sim_t,
            centre_north_m=12.5,
            centre_east_m=-7.25,
            spec=spec,
            heights=heights,
        )

    def test_round_trip(self):
        frame = self._make_frame()
        blob = encode_wave_patch(frame)
        decoded = decode_wave_patch(blob)
        self.assertEqual(decoded.spec.size, frame.spec.size)
        self.assertAlmostEqual(decoded.sim_time_s, frame.sim_time_s, places=5)
        self.assertEqual(len(decoded.heights), len(frame.heights))
        for a, b in zip(decoded.heights, frame.heights):
            self.assertAlmostEqual(a, b, places=5)

    def test_header_size(self):
        frame = self._make_frame(size=5)
        blob = encode_wave_patch(frame)
        # 19 header bytes + 5*5*4 = 100 data bytes = 119.
        self.assertEqual(len(blob), 19 + 25 * 4)

    def test_default_size_under_20kib(self):
        frame = self._make_frame(size=DEFAULT_GRID_SIZE,
                                 sim_t=DEFAULT_GRID_SIZE)
        blob = encode_wave_patch(frame)
        # Design target: ~17 KiB per frame for the default 65x65 patch.
        self.assertLess(len(blob), 17 * 1024 + 256)
        self.assertGreater(len(blob), 16 * 1024)

    def test_message_format(self):
        frame = self._make_frame(size=5)
        msg = encode_wave_patch_message(frame, session_id="abc", epoch=3)
        head, payload = msg.split(b"\n", 1)
        meta = decode_wave_patch_header(head)
        self.assertEqual(meta["type"], "wave_patch")
        self.assertEqual(meta["session_id"], "abc")
        self.assertEqual(meta["epoch"], 3)
        self.assertEqual(meta["sim_time_s"], 1.5)
        self.assertEqual(meta["spec"]["size"], 5)
        decoded = decode_wave_patch(payload)
        self.assertEqual(len(decoded.heights), len(frame.heights))
        for a, b in zip(decoded.heights, frame.heights):
            self.assertAlmostEqual(a, b, places=5)

    def test_decode_rejects_short_blob(self):
        with self.assertRaises(WavePatchError):
            decode_wave_patch(b"\x01" + b"\x00" * 10)

    def test_decode_rejects_version_mismatch(self):
        frame = self._make_frame(size=5)
        blob = bytearray(encode_wave_patch(frame))
        blob[0] = 99
        with self.assertRaises(WavePatchError):
            decode_wave_patch(bytes(blob))

    def test_decode_rejects_truncated_payload(self):
        frame = self._make_frame(size=5)
        blob = encode_wave_patch(frame)
        truncated = blob[:-4]
        with self.assertRaises(WavePatchError):
            decode_wave_patch(truncated)


class WavePatchContentTests(unittest.TestCase):
    def test_centre_at_origin(self):
        spec = WavePatchSpec(size=5, extent_m=8.0)
        heights = tuple(0.0 for _ in range(25))
        frame = WavePatchFrame(sim_time_s=0.0, centre_north_m=0.0,
                               centre_east_m=0.0, spec=spec, heights=heights)
        self.assertEqual(frame.spec.size, 5)
        self.assertEqual(len(frame.heights), 25)

    def test_extent_matches(self):
        a = WavePatchSpec(size=33, extent_m=64.0)
        b = WavePatchSpec(size=33, extent_m=64.05)
        # Slightly different extent: spacing differs but size matches.
        self.assertFalse(patch_extent_matches(a, b))
        c = WavePatchSpec(size=33, extent_m=64.0)
        self.assertTrue(patch_extent_matches(a, c))


class WavePatchFromOceanTests(unittest.TestCase):
    """Validate that the ocean sampler yields consistent heights."""

    def test_known_ocean_constant_eta(self):
        from ocean import Ocean
        sea = Ocean(Hs=0.0, Tp=6.0, seed=42)
        spec = WavePatchSpec(size=5, extent_m=8.0)
        # With Hs=0, the ocean surface is identically zero everywhere.
        heights = sample_wave_heights(sea, 0.0, 0.0, spec, 0.0)
        self.assertEqual(len(heights), spec.size * spec.size)
        for h in heights:
            self.assertAlmostEqual(float(h), 0.0, places=6)

    def test_vectorized_matches_pointwise_directional(self):
        from ocean_directional import DirectionalOcean
        from operator_training.wave_patch import (
            sample_wave_heights as vectorized,
            _sample_wave_heights_loop as loop,
        )
        sea = DirectionalOcean(Hs=0.8, Tp=6.0, seed=42)
        spec = WavePatchSpec(size=5, extent_m=8.0)
        a = vectorized(sea, 12.0, -3.0, spec, 1.5)
        b = loop(sea, 12.0, -3.0, spec, 1.5)
        self.assertEqual(len(a), 25)
        self.assertTrue(any(abs(h) > 1e-6 for h in a))
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=5)

    def test_1d_ocean_extrudes_along_east(self):
        from ocean import Ocean
        sea = Ocean(Hs=1.5, Tp=6.0, seed=42)
        spec = WavePatchSpec(size=5, extent_m=8.0)
        heights = sample_wave_heights(sea, 0.0, 0.0, spec, 0.0)
        n = spec.size
        for i in range(n):
            row0 = heights[i * n]
            for j in range(n):
                self.assertAlmostEqual(heights[i * n + j], row0, places=6)


def sample_wave_heights(sea, centre_north_m, centre_east_m, spec, sim_t):
    """Import-on-call helper to avoid numpy requirement at module import."""
    from operator_training.wave_patch import sample_wave_heights as _impl
    return _impl(sea, centre_north_m, centre_east_m, spec, sim_t)


if __name__ == "__main__":
    unittest.main()
