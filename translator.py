import os
import sys
import logging
import warnings
from typing import Union, List

# Suppress HuggingFace's mistaken mistral regex warning for NLLB
warnings.filterwarnings("ignore")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    import io
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Ensure PyTorch CUDA DLLs are found by CTranslate2 on Windows
if sys.platform == "win32":
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib) and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(torch_lib)
    except Exception:
        pass

import ctranslate2
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# Common language alias mapping
LANG_MAP = {
    "en": "eng_Latn", "english": "eng_Latn",
    "ur": "urd_Arab", "urdu": "urd_Arab",
    "fr": "fra_Latn", "french": "fra_Latn",
    "es": "spa_Latn", "spanish": "spa_Latn",
    "de": "deu_Latn", "german": "deu_Latn",
    "ar": "ara_Arab", "arabic": "ara_Arab",
    "hi": "hin_Deva", "hindi": "hin_Deva",
    "zh": "zho_Hans", "chinese": "zho_Hans",
    "ru": "rus_Cyrl", "russian": "rus_Cyrl",
    "ja": "jpn_Jpan", "japanese": "jpn_Jpan",
    "pt": "por_Latn", "portuguese": "por_Latn",
    "it": "ita_Latn", "italian": "ita_Latn",
    "tr": "tur_Latn", "turkish": "tur_Latn",
    "ko": "kor_Hang", "korean": "kor_Hang",
    "fa": "pes_Arab", "persian": "pes_Arab"
}

# Global instances for fast reuse
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "nllb-200-600M-int8")
_TRANSLATOR = None
_TOKENIZER = None

def _get_engine():
    """Initializes and caches the translator and tokenizer, auto-downloading if missing."""
    global _TRANSLATOR, _TOKENIZER
    if _TRANSLATOR is None:
        model_file = os.path.join(_MODEL_DIR, "model.bin")
        if not os.path.exists(_MODEL_DIR) or not os.path.exists(model_file):
            print(f"[Notice] Model not found locally. Auto-downloading NLLB-200 INT8 (~680 MB)...")
            from huggingface_hub import snapshot_download
            os.makedirs(_MODEL_DIR, exist_ok=True)
            snapshot_download(
                repo_id="JustFrederik/nllb-200-distilled-600M-ct2-int8",
                local_dir=_MODEL_DIR
            )
            print("[OK] Model downloaded successfully!\n")

        # Check for CUDA GPU availability
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        compute_type = "auto" if device == "cuda" else "int8"
        threads = min(os.cpu_count() or 4, 8)

        _TRANSLATOR = ctranslate2.Translator(
            _MODEL_DIR,
            device=device,
            compute_type=compute_type,
            intra_threads=threads
        )
        # Note: Do NOT pass fix_mistral_regex=True on NLLB - it corrupts SentencePiece tokens!
        _TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_DIR)
    return _TRANSLATOR, _TOKENIZER

def _normalize_lang(code: str) -> str:
    cleaned = code.strip().lower()
    return LANG_MAP.get(cleaned, code.strip())

def translate(content: Union[str, List[str]], to_lang: str, from_lang: str) -> Union[str, List[str]]:
    """
    Translates text between two specified languages.
    Automatically handles multi-line strings preserving paragraph formatting.

    Parameters:
        content   (str | list[str]): Text or list of texts to translate.
        to_lang   (str): Target language (e.g. 'urdu', 'ur', 'urd_Arab', 'english', 'en').
        from_lang (str): Source language (e.g. 'english', 'en', 'eng_Latn', 'french', 'fr').

    Returns:
        str | list[str]: Translated content.
    """
    translator, tokenizer = _get_engine()

    src_code = _normalize_lang(from_lang)
    tgt_code = _normalize_lang(to_lang)
    tokenizer.src_lang = src_code

    # Helper function to translate a flat list of sentences in batch
    def _translate_batch_internal(items: List[str]) -> List[str]:
        if not items:
            return []
        
        tokenized_batch = []
        indices_to_translate = []
        
        for idx, text in enumerate(items):
            clean = text.strip()
            if clean:
                encoded = tokenizer(clean)
                tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
                tokenized_batch.append(tokens)
                indices_to_translate.append(idx)

        if not tokenized_batch:
            return items

        target_prefix = [[tgt_code] for _ in range(len(tokenized_batch))]
        translations = translator.translate_batch(
            tokenized_batch,
            target_prefix=target_prefix,
            beam_size=4,
            batch_type="tokens",
            max_batch_size=2048
        )

        translated_outputs = []
        for trans in translations:
            out_tokens = trans.hypotheses[0]
            if len(out_tokens) > 0 and out_tokens[0] == tgt_code:
                out_tokens = out_tokens[1:]
            decoded = tokenizer.decode(tokenizer.convert_tokens_to_ids(out_tokens), skip_special_tokens=True)
            translated_outputs.append(decoded)

        # Reconstruct output preserving empty lines
        final_results = list(items)
        for orig_idx, trans_text in zip(indices_to_translate, translated_outputs):
            final_results[orig_idx] = trans_text
        return final_results

    # If input is a single string containing newlines, translate line-by-line in batch
    if isinstance(content, str):
        if "\n" in content:
            lines = content.split("\n")
            translated_lines = _translate_batch_internal(lines)
            return "\n".join(translated_lines)
        else:
            res = _translate_batch_internal([content])
            return res[0] if res else ""

    # If input is a list of strings
    return _translate_batch_internal(content)


if __name__ == "__main__":
    sample_text = "Hello, how are you?\nMy name is Saad Asif.\nWhat is your name?"
    print(f"Original Text:\n{sample_text}\n")

    # English -> Urdu
    urdu_out = translate(sample_text, to_lang="urdu", from_lang="english")
    print(f"[Urdu]:\n{urdu_out}\n")

    # English -> French
    french_out = translate(sample_text, to_lang="french", from_lang="english")
    print(f"[French]:\n{french_out}\n")

    # English -> Spanish
    spanish_out = translate(sample_text, to_lang="spanish", from_lang="english")
    print(f"[Spanish]:\n{spanish_out}\n")
