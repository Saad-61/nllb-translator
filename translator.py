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
    # English & Urdu
    "en": "eng_Latn", "english": "eng_Latn",
    "ur": "urd_Arab", "urdu": "urd_Arab",
    # European Languages
    "fr": "fra_Latn", "french": "fra_Latn",
    "es": "spa_Latn", "spanish": "spa_Latn",
    "de": "deu_Latn", "german": "deu_Latn",
    "it": "ita_Latn", "italian": "ita_Latn",
    "pt": "por_Latn", "portuguese": "por_Latn",
    "nl": "nld_Latn", "dutch": "nld_Latn",
    "ru": "rus_Cyrl", "russian": "rus_Cyrl",
    "uk": "ukr_Cyrl", "ukrainian": "ukr_Cyrl",
    "pl": "pol_Latn", "polish": "pol_Latn",
    "cs": "ces_Latn", "czech": "ces_Latn",
    "sv": "swe_Latn", "swedish": "swe_Latn",
    "da": "dan_Latn", "danish": "dan_Latn",
    "fi": "fin_Latn", "finnish": "fin_Latn",
    "no": "nob_Latn", "norwegian": "nob_Latn",
    "el": "ell_Grek", "greek": "ell_Grek",
    "ro": "ron_Latn", "romanian": "ron_Latn",
    "hu": "hun_Latn", "hungarian": "hun_Latn",
    # Middle Eastern & Central Asian
    "ar": "arb_Arab", "arabic": "arb_Arab",
    "fa": "pes_Arab", "persian": "pes_Arab", "farsi": "pes_Arab",
    "tr": "tur_Latn", "turkish": "tur_Latn",
    "he": "heb_Hebr", "hebrew": "heb_Hebr",
    "az": "azj_Latn", "azerbaijani": "azj_Latn",
    "uz": "uzn_Latn", "uzbek": "uzn_Latn",
    "kk": "kaz_Cyrl", "kazakh": "kaz_Cyrl",
    "ku": "ckb_Arab", "kurdish": "ckb_Arab",
    # South Asian Languages
    "hi": "hin_Deva", "hindi": "hin_Deva",
    "sd": "snd_Arab", "sindhi": "snd_Arab",
    "ps": "pbt_Arab", "pashto": "pbt_Arab",
    "pa": "pan_Guru", "punjabi": "pan_Guru",
    "bn": "ben_Beng", "bengali": "ben_Beng",
    "ks": "kas_Arab", "kashmiri": "kas_Arab",
    "gu": "guj_Gujr", "gujarati": "guj_Gujr",
    "mr": "mar_Deva", "marathi": "mar_Deva",
    "ta": "tam_Taml", "tamil": "tam_Taml",
    "te": "tel_Telu", "telugu": "tel_Telu",
    "ml": "mal_Mlym", "malayalam": "mal_Mlym",
    "kn": "kan_Knda", "kannada": "kan_Knda",
    "ne": "npi_Deva", "nepali": "npi_Deva",
    "si": "sin_Sinh", "sinhala": "sin_Sinh",
    # East & Southeast Asian
    "zh": "zho_Hans", "chinese": "zho_Hans", "chinese_traditional": "zho_Hant",
    "ja": "jpn_Jpan", "japanese": "jpn_Jpan",
    "ko": "kor_Hang", "korean": "kor_Hang",
    "id": "ind_Latn", "indonesian": "ind_Latn",
    "ms": "zsm_Latn", "malay": "zsm_Latn",
    "vi": "vie_Latn", "vietnamese": "vie_Latn",
    "th": "tha_Thai", "thai": "tha_Thai",
    "tl": "tgl_Latn", "tagalog": "tgl_Latn", "filipino": "tgl_Latn",
    "my": "mya_Mymr", "burmese": "mya_Mymr",
    # African Languages
    "sw": "swh_Latn", "swahili": "swh_Latn",
    "so": "som_Latn", "somali": "som_Latn",
    "ha": "hau_Latn", "hausa": "hau_Latn",
    "yo": "yor_Latn", "yoruba": "yor_Latn",
    "ig": "ibo_Latn", "igbo": "ibo_Latn",
    "am": "amh_Ethi", "amharic": "amh_Ethi",
    "zu": "zul_Latn", "zulu": "zul_Latn"
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
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
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
    print(f"Original Text: \"{sample_text}\"\n")

    # English -> Urdu
    urdu_out = translate(sample_text, to_lang="urdu", from_lang="english")
    print(f"[Urdu]    : {urdu_out}")

    # English -> French
    french_out = translate(sample_text, to_lang="french", from_lang="english")
    print(f"[French]  : {french_out}")

    # English -> Spanish
    spanish_out = translate(sample_text, to_lang="spanish", from_lang="english")
    print(f"[Spanish] : {spanish_out}")

    # English -> Arabic
    arabic_out = translate(sample_text, to_lang="arabic", from_lang="english")
    print(f"[Arabic]  : {arabic_out}")
