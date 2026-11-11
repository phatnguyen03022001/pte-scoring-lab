"""Compatibility regression for the released faster-whisper/PyAV/CT2 stack."""
import io
import unittest
import wave

import ctranslate2
from faster_whisper import decode_audio


class NativeDependencyCompatibilityTests(unittest.TestCase):
    def test_faster_whisper_decodes_wav_with_pyav_metadata_errors(self):
        stream = io.BytesIO()
        with wave.open(stream, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 1600)
        decoded = decode_audio(io.BytesIO(stream.getvalue()), sampling_rate=16000)
        self.assertEqual(len(decoded), 1600)

    def test_ctranslate2_cpu_int8_is_supported(self):
        self.assertIn("int8", ctranslate2.get_supported_compute_types("cpu"))


if __name__ == "__main__":
    unittest.main()
