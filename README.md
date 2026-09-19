<div align="center">

# ArabicDeXlit

### Undo Arabic ASR transliteration — turn `انترن` back into `intern`, `ايه اي` back into `AI`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/Arabic-DeXlit/blob/main/notebooks/ArabicDeXlit_Train_Colab.ipynb)
[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-ArabicDeXlit--Corpus-yellow)](https://huggingface.co/datasets/mohammedaly22/ArabicDeXlit-Corpus)
[![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-coming%20soon-lightgrey)](https://huggingface.co/mohammedaly22)

[![PyPI](https://img.shields.io/badge/pip-coming%20soon-lightgrey?logo=pypi&logoColor=white)](https://github.com/MohammedAly22/Arabic-DeXlit)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Dataset License](https://img.shields.io/badge/data-CC--BY--SA--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

**[Train it on Colab](https://colab.research.google.com/github/MohammedAly22/Arabic-DeXlit/blob/main/notebooks/ArabicDeXlit_Train_Colab.ipynb)** ·
[Dataset](https://huggingface.co/datasets/mohammedaly22/ArabicDeXlit-Corpus) ·
[How it works](#how-it-works) ·
[Quick start](#quick-start)

</div>

---

## The problem

Arabic ASR models handle code-switching badly. When a speaker says an English word, the model
writes it with Arabic letters. The transcript is phonetically reasonable and practically useless —
you cannot search it, index it, or hand it to an English NLP tool.

<table>
<tr><th align="left">What the ASR gives you</th></tr>
<tr><td dir="rtl">

أول شغل ليا بعد التخرج كنت **انترن** في **او اي اي** اللي هي **اورانج انوفيشن ايجيبت** و بعدها بكام شهر جالي **اوفر**

</td></tr>
<tr><th align="left">What ArabicDeXlit gives back</th></tr>
<tr><td dir="rtl">

أول شغل ليا بعد التخرج كنت **`intern`** في **`OIE`** اللي هي **`Orange Innovation Egypt`** و بعدها بكام شهر جالي **`offer`**

</td></tr>
</table>

And the part that makes it safe to deploy:

<table>
<tr><th align="left">Pure Arabic in</th><th align="left">Pure Arabic out</th></tr>
<tr>
<td dir="rtl">أنا رايح البيت دلوقتي عشان تعبان</td>
<td dir="rtl">أنا رايح البيت دلوقتي عشان تعبان</td>
</tr>
<tr><td colspan="2" align="center"><i>byte-for-byte identical — guaranteed by construction, not by training</i></td></tr>
</table>

---

## How it works

Most people would reach for a seq2seq model. That is the wrong tool here, for two reasons: a
decoder is free to rewrite **any** token, so "pure Arabic must pass through unchanged" can only
ever be *approximately* learned; and it decodes every sentence, paying latency whether or not there
is anything to fix.

ArabicDeXlit splits the problem in two.

```
                    ┌────────────────────────────────────────────┐
   ASR sentence ───▶│  STAGE 1 · Detector                        │
                    │  token tagger over an Arabic encoder       │
                    │  every token → O  or  B-/I-<category>      │
                    └────────────────────────────────────────────┘
                                      │
                         all tags O?  ├── yes ──▶ return the input string, untouched
                                      │            (stage 2 never runs)
                                      no
                                      ▼
                    ┌────────────────────────────────────────────┐
                    │  STAGE 2 · Converter                       │
                    │  ~5M-param char transformer                │
                    │  runs ONLY on the flagged spans            │
                    └────────────────────────────────────────────┘
                                      │
                                      ▼
                            rebuilt sentence
```

**Stage 1 — the detector.** A token classifier over [MARBERTv2](https://huggingface.co/UBC-NLP/MARBERTv2),
chosen because it is pretrained on *dialectal* Arabic rather than MSA. Each token gets one tag.
`O` means *copy this verbatim*.

**Stage 2 — the converter.** A tiny character-level transformer mapping Arabic characters to Latin.
Character-level because the mapping is phonetic, not lexical: a word-level model could only emit
English it had already seen, whereas the real long tail here is company names, products and jargon.

### Why this design earns its keep

| Your requirement | How the architecture delivers it |
|---|---|
| **Pure Arabic must be unchanged** | `O` = copy verbatim. An all-`O` sentence short-circuits and returns the original string **before** any conversion runs. Structural, not learned. |
| **Near-zero added latency** | One encoder pass, then decoding over a few *short spans* rather than the whole sentence. Monolingual Arabic skips stage 2 entirely. |
| **Show me what it changes** | The tags **are** the explanation — they name exactly which words will change, and why. Plotted every evaluation. |
| **Never corrupt untouched text** | Non-span tokens never reach the converter. The worst a mis-firing detector can do is convert a span it should have left alone. |

### Two task-specific additions

**Script-feature injection.** Whether a character is Arabic, Latin, a digit or punctuation is
*perfectly known* at inference — it is a property of the string, not something to infer. Feeding it
in as an explicit embedding frees the encoder from rediscovering it, and helps sharply on the
already-Latin and digit cases that must be left alone.

**A copy gate.** A scalar head per token predicting "is this token untouched", trained jointly with
the tag head. It gives a calibrated, directly thresholdable pass-through signal, and at inference it
can veto spurious edits — the conservative direction for a model that must not corrupt text.

### Span categories

The category is passed to the converter, because identical characters convert differently depending
on it (`ايه اي` is `AI` as an acronym, but `eh ay` as an ordinary word).

| Category | Input | Output |
|:--|:--|:--|
| `CS` | `انترن` | `intern` |
| `ACRONYM` | `ايه اي` | `AI` |
| `ENTITY` | `اورانج انوفيشن ايجيبت` | `Orange Innovation Egypt` |
| `EMAIL` | `احمد ات جيميل دوت كوم` | `ahmed@gmail.com` |
| `NUMBER` | `تو زيرو تو فور` | `2024` |

---

## Quick start

### Train it on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/Arabic-DeXlit/blob/main/notebooks/ArabicDeXlit_Train_Colab.ipynb)

Click the badge and run the notebook top to bottom. It pulls the prebuilt corpus from the Hub, runs
a one-minute smoke test, trains both stages, and evaluates.

| GPU | Config (chosen automatically) | Approx. time |
|:--|:--|:--|
| A100 | `configs/detector_base.yaml` | ~1–1.5 h |
| L4 | `configs/detector_base.yaml` | ~2–3 h |
| T4 | `configs/detector_t4.yaml` | ~3–4 h |

### Train locally

```bash
git clone https://github.com/MohammedAly22/Arabic-DeXlit.git
cd Arabic-DeXlit
pip install -r requirements.txt

# 1. Pull the prebuilt corpus (~68 MB) from the Hub
python -c "import sys; sys.path.insert(0,'src'); \
from arabic_dexlit.data.hub import download_corpus; download_corpus()"

# 2. Verify the whole pipeline in ~1 minute before committing a GPU
python scripts/train.py --stage both --smoke --no-wandb

# 3. Train
python scripts/train.py --stage both --config configs/detector_base.yaml

# 4. Evaluate on the held-out test set
python scripts/evaluate.py --model-dir outputs/detector --data data/processed/test.jsonl
```

### Use a trained model

```python
from arabic_dexlit.inference.pipeline import DeXlitPipeline

pipe = DeXlitPipeline.from_pretrained("outputs/detector", device="cuda")

out = pipe.predict("كنت انترن في او اي اي و بعدها جالي اوفر")
print(out.text)      # كنت intern في OIE و بعدها جالي offer
print(out.changed)   # True
print(out.spans)     # [{'category': 'CS', 'original': 'انترن', 'converted': 'intern'}, ...]

# Pure Arabic comes back untouched
out = pipe.predict("أنا رايح البيت دلوقتي")
print(out.changed)   # False
```

---

## The dataset

[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20datasets-ArabicDeXlit--Corpus-yellow)](https://huggingface.co/datasets/mohammedaly22/ArabicDeXlit-Corpus)

```python
from datasets import load_dataset
ds = load_dataset("mohammedaly22/ArabicDeXlit-Corpus")
```

| | |
|:--|:--|
| **Examples** | 475,945 (train 428,376 · validation 23,601 · test 23,968) |
| **Dialects** | 8 — MSA, Egyptian, Gulf, Levantine, Iraqi, Maghrebi, Sudanese, Yemeni |
| **Pass-through rows** | 30% — pure Arabic where the correct answer is *change nothing* |
| **Labelled spans** | 1,383,667 |
| **License** | CC-BY-SA-4.0 |

### The core idea: invert the corpus

Parallel data for this task barely exists. But corpora of the **target** form — Arabic with English
correctly in Latin script — are plentiful. So the pipeline runs the problem backwards:

```
target (we have this)    كنت intern في Orange
        │ transliterate the English into Arabic script
        ▼
source (we make this)    كنت انترن في اورانج
        │ record where each rewrite landed
        ▼
tags                     O     B-CS   O   B-CS
```

This matters for correctness, not just convenience. Because spans are recorded **during**
generation rather than recovered afterwards by alignment, the labels cannot silently drift out of
sync with the text — the failure mode that quietly poisons hand-aligned corpora.

The transliterator is deliberately **stochastic**, because real ASR is inconsistent: `intern` →
`انترن` / `انتيرن` / `انترين`. Training across epochs therefore exposes the model to the variation
it will actually meet, instead of one canonical spelling it could memorise.

### Sources

| Source | Contribution | License |
|:--|:--|:--|
| [SDAIA ArE-CSTD](https://huggingface.co/datasets/SDAIANCAI/Ar-En-Code-Switching-Textual-Dataset) | 310,578 rows — MSA, Saudi, Egyptian | CC-BY-SA-4.0 |
| Gemini synthesis | 22,584 rows — the missing dialects and rare categories | generated here |
| Arabic Wikipedia + dialectal ASR transcripts | 142,783 rows — pass-through examples | CC-BY-SA |

**Why synthesis was needed.** SDAIA covers three dialects and is almost entirely plain word-level
code-switching. Measured over 20K of its sentences: ~60,000 ordinary spans but only **178 acronyms,
9 entities and zero emails or numbers**. Generation was aimed narrowly at those gaps, not at bulk
volume SDAIA already supplies.

**Why external monolingual text was needed.** SDAIA is ~99% code-switched by construction, so it
cannot supply the pass-through half of the task. Those rows are additionally **screened**: a
sentence already containing transliterated English (`الميتينغ`, `الويكند`) is rejected, since using
it as a pass-through example would teach the exact opposite of the task.

### Verified quality

Each checked on the released build, not assumed:

| Check | Result |
|:--|:--|
| Sentence-family overlap, train ∩ test | **0** |
| Sentence-family overlap, train ∩ validation | **0** |
| Rows where `len(tags) != len(src_tokens)` | **0** |
| Tags outside the 11-tag schema | **0** |
| Pass-through rows where `src != tgt` | **0** |

Splits are assigned by hashing a *normalised* sentence key, so families of near-identical sentences
(common in LLM-generated corpora) cannot straddle a split and inflate the score.

---

## Monitoring

Training logs to [Weights & Biases](https://wandb.ai). Beyond loss and accuracy, four diagnostics
are plotted at every evaluation:

| Plot | What it answers |
|:--|:--|
| **Edit decisions** | *Which words does the model change?* The sentence, each token shaded by the edit chosen, disagreements underlined in red. |
| **Attention** | What the encoder actually attends to, last layer, head-averaged. |
| **Tag confusion** | Which tag is being confused with which, row-normalised. |
| **Safety curves** | Pass-through accuracy and false-edit rate across training. |

### The two metrics that matter

| Metric | Meaning | Target |
|:--|:--|:--|
| `passthrough_accuracy` | Of sentences that must be returned untouched, the fraction that were. | **≥ 0.99** |
| `false_edit_rate` | Share of ordinary tokens the model wanted to edit. | **≤ 0.01** |

A model with excellent F1 and poor pass-through accuracy is **unusable** — it corrupts ordinary
Arabic. Checkpoints are therefore selected on a composite score:

```
score = span_f1 − 0.5 × false_edit_rate
```

---

## Repository layout

```
src/arabic_dexlit/
├── schema.py              tag scheme; O = identity, the pass-through guarantee
├── data/
│   ├── translit.py        English → Arabic script (manufactures training inputs)
│   ├── pairing.py         builds aligned (input, tags, target) triples
│   ├── sources.py         corpus acquisition + monolingual screening
│   ├── synth.py           Gemini synthesis for missing dialects/categories
│   ├── build.py           leak-free splitting and dataset assembly
│   ├── hub.py             pull the prebuilt corpus from the Hub
│   └── dialects.py        dialect registry and generation prompts
├── model/
│   ├── detector.py        stage 1: tagger + script features + copy gate
│   └── converter.py       stage 2: tiny char-level transformer
├── training/
│   ├── dataset.py         torch datasets; sub-word label projection
│   ├── metrics.py         span F1 + the safety metrics
│   ├── viz.py             diagnostic plots
│   ├── train_detector.py  stage-1 loop
│   └── train_converter.py stage-2 loop
└── inference/
    └── pipeline.py        end-to-end; enforces pass-through in code

scripts/    build_dataset.py · synthesize_data.py · export_dataset.py · train.py · evaluate.py
configs/    detector_base.yaml · detector_t4.yaml · converter_base.yaml
notebooks/  ArabicDeXlit_Train_Colab.ipynb
tests/      test_pairing.py · test_models.py
```

## Tests

```bash
python tests/test_pairing.py    # data invariants
python tests/test_models.py     # model shapes, gradients, decoding
```

The data tests assert the invariants that would otherwise fail **silently** — tag/token alignment,
span round-tripping, and that pure Arabic is an exact identity.

---

## Licensing

Code is **Apache-2.0**. The corpus is **CC-BY-SA-4.0**, inherited from SDAIA's share-alike term.

One deliberate exclusion: [Casablanca](https://huggingface.co/datasets/UBC-NLP/Casablanca) is an
excellent multi-dialect corpus that annotates code-switched words in *both* scripts, but it is
CC-BY-**NC-ND**. The no-derivatives term forbids republishing transliterated derivatives, so it is
not part of the released dataset. It remains suitable as a held-out evaluation set, cited rather
than redistributed.

## Citation

```bibtex
@misc{arabicdexlit2026,
  title  = {ArabicDeXlit: Undoing Transliteration in Arabic ASR Code-Switching},
  author = {Mohammed Aly},
  year   = {2026},
  url    = {https://github.com/MohammedAly22/Arabic-DeXlit}
}
```

Built on [SDAIA ArE-CSTD](https://huggingface.co/datasets/SDAIANCAI/Ar-En-Code-Switching-Textual-Dataset)
and [MARBERTv2](https://huggingface.co/UBC-NLP/MARBERTv2).
