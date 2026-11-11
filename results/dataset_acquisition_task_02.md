# TASK-02 Dataset Acquisition and Integrity

**Assessment date:** 2026-11-11
**Scope:** Authorized dataset acquisition, local integrity verification, provenance, and readiness for TASK-03.
**Pearson agreement:** NOT VERIFIED.

## SpeechOcean762

The official [OpenSLR SLR101 record](https://www.openslr.org/101/) lists SpeechOcean762 under CC BY 4.0, with five independent human raters and 5,000 English utterances. The raw archive was acquired from the official [OpenSLR download](https://www.openslr.org/resources/101/speechocean762.tar.gz) and preserved under the ignored `.local/datasets/speechocean762/` directory. Its advertised and downloaded size was 520,810,923 bytes. The local archive SHA-256 is `876c34b828c2ebcc35ba2ff1ba5947d50e6b0162fc4a4af91db0942422b3bdd5`. OpenSLR does not publish a trusted checksum, so this hash records local file integrity only.

The current author repository source was resolved to commit [`613968e3b0b789fc33936fb5eba1973176ba7d11`](https://github.com/jimbozhang/speechocean762/commit/613968e3b0b789fc33936fb5eba1973176ba7d11). The archive contains 5,245 WAV files; 5,000 are referenced by the official split manifests and have complete labels, transcripts, and speaker IDs. The other 245 files are readable mono 16 kHz PCM WAVs but are not referenced by `wav.scp`; they remain preserved and are not assigned inferred labels.

The existing `dataset_foundation.py` integrity utility passed on the extracted archive: 5,000/5,000 referenced audio files decoded, all 5,000 utterances had labels, transcripts, and speaker IDs, all 250 speakers were identified, and the official train/test partitions each contained 2,500 utterances with no speaker or utterance overlap. No malformed or missing labels, duplicate referenced audio, or integrity issues were reported. The full manifest and integrity report remain local-only.

`scores-detail.json` matches the author repository's committed Git blob and retains the five individual ratings and phone sequences. Source documentation describes sentence accuracy, fluency, prosody, total, and completeness ratings; word accuracy/stress; and phone accuracy encodings. `scores.json` is a processed average/median, not the original expert record. Its OpenSLR archive content differs from the previously recorded QUALITY-05 hash and from the current author repository file. The archive version is preserved, the variant is documented locally, and the processed aggregate is not treated as raw expert ground truth.

SpeechOcean762 is **not an independent final evaluation set** for this project: the released GOPT checkpoint was trained/selected using this corpus, including its test split. Use it for development and compatibility checks only.

## Existing EpaDB samples

The existing QUALITY-08 and QUALITY-09 EpaDB manifests and source files were verified locally against their recorded hashes. Their historical benchmark results were not rerun or overwritten. A speaker-count inconsistency in the QUALITY-09 local manifest wording was found; the actual record set is the authoritative count, and the details remain in the ignored local research area. No EpaDB recordings, labels, manifests, or derived benchmark results are included in this public repository report. EpaDB remains under [CC BY-NC 4.0](https://github.com/donggangj/EpaDB), so its samples and results remain restricted to the permitted non-commercial research use.

## Restricted acquisition decisions

### L2-ARCTIC

The official [PSI Lab access page](https://psi.engr.tamu.edu/l2-arctic-corpus/) documents the CC BY-NC 4.0 license and requires agreement to the terms followed by a form with the requester's name, email, and affiliation; it uses reCAPTCHA and emails a direct download link. The official [documentation](https://psi.engr.tamu.edu/l2-arctic-corpus-docs/) describes version 5.0, 24 L1 English speakers, four Vietnamese-L1 speakers, 3,599 manually examined utterances with phone error annotations, and a small spontaneous suitcase-story subset. The standard read-speech TextGrids are forced alignments; the separate manual annotation TextGrids contain the human corrections and error tags.

**ACCESS BLOCKED pending Owner action.** No registration was submitted and no license was accepted. The dataset is not present locally.

### Cambridge Speak & Improve Corpus 2025

The official [Cambridge research dataset page](https://researchdatasets.cambridge.org/datasets/speak-and-improve-corpus-2025) describes approximately 315 hours and 45,000+ spontaneous learner utterances, holistic CEFR-style speaking marks from non-operational examiners, and a manually transcribed subset. Access requires completing a registration form and agreeing to the license. The license grants non-commercial research and educational use, prohibits transfer or redistribution, and requires prior Cambridge approval before releasing derived artifacts.

**ACCESS BLOCKED pending Owner action and license review.** No form was submitted and no agreement accepted. This corpus is relevant to holistic speaking/fluency work, but its license does not authorize use in a commercial product or release of derived artifacts without approval.

## Bounded additional discovery

The [MRPL2 Non-native Speech Corpus v1.1 record](https://zenodo.org/records/22690685) describes manually annotated phonetic mispronunciations across 4,500 utterances and 23 speakers. Its files are restricted and require a request; no license is shown on the record. It is a potentially useful future candidate, but it was not acquired or used.

## Readiness and limits

- **DATA ACQUISITION:** SpeechOcean762 acquired from OpenSLR; existing EpaDB samples verified locally. L2-ARCTIC and Cambridge remain access-blocked.
- **DATA INTEGRITY:** SpeechOcean762 referenced audio/annotation joins pass. Its 245 unreferenced WAVs and `scores.json` source variant are documented and excluded from raw expert-label claims.
- **EXPERT ANNOTATION QUALITY:** SpeechOcean762 retains the original five-rater `scores-detail.json`. Existing EpaDB source annotations remain unchanged in local-only storage.
- **INDEPENDENT BENCHMARK READINESS:** Conditional. SpeechOcean762 is checkpoint-exposed; EpaDB provides limited speaker-separated non-commercial research evidence. No acquired Vietnamese-L1 corpus is available yet.
- **PEARSON AGREEMENT:** NOT VERIFIED. None of these corpora provides Pearson-scored PTE outcomes.

**TASK-03 readiness: CONDITIONAL GO** for research using the verified, licensed local resources, subject to their license terms and the limits above. Independent generalization and Pearson alignment remain future evidence requirements.
