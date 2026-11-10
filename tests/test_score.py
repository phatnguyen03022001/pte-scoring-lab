"""Pure-function tests: no model download and no network."""

import unittest

from score import compare_words, parse_silence_log


class WordScoringTests(unittest.TestCase):
    def test_equal_ignores_case_and_punctuation(self):
        result = compare_words("Hello, World!", "hello world")
        self.assertEqual(result["word_error_rate"], 0)
        self.assertEqual(result["word_match_percent"], 100)

    def test_newlines_match_spaces(self):
        result = compare_words("Yellow is optimistic.\nYet it is bright.", "Yellow is optimistic. Yet it is bright.")
        self.assertEqual(result["word_error_rate"], 0)

    def test_substitution_and_deletion(self):
        result = compare_words("The team will publish tomorrow", "The team publish today")
        self.assertEqual(result["correct_words"], 3)
        self.assertEqual(result["deletions"], 1)
        self.assertEqual(result["substitutions"], 1)

    def test_insertion_is_not_rewarded(self):
        result = compare_words("hello world", "hello my lovely world")
        self.assertEqual(result["word_error_rate"], 1)
        self.assertEqual(result["word_match_percent"], 0)

    def test_empty_transcript_scores_zero(self):
        self.assertEqual(compare_words("hello", "")["word_match_percent"], 0)

    def test_invalid_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            compare_words("...!!", "hello")


class SilenceTests(unittest.TestCase):
    def test_closed_and_trailing_pauses(self):
        log = """
        [silencedetect] silence_start: 1.0
        [silencedetect] silence_end: 1.5 | silence_duration: 0.5
        [silencedetect] silence_start: 4.0
        """
        self.assertEqual(parse_silence_log(log, 5.0), (2, 1.5))

    def test_no_silence(self):
        self.assertEqual(parse_silence_log("", 3.0), (0, 0.0))


if __name__ == "__main__":
    unittest.main()
