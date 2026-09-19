"""ArabicDeXlit -- undo Arabic ASR transliteration.

Arabic ASR systems transcribe spoken English words using Arabic letters:
"intern" comes out as انترن, "AI" as ايه اي. ArabicDeXlit is a small drop-in
post-processor that puts them back into English, while leaving genuinely Arabic
text byte-identical.
"""

__version__ = "0.1.0"
