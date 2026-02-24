from pathlib import Path
from tokenizers import Tokenizer


class TigrinyaTokenizer:
    """
    Tigrinya Tokenizer Library

    Provides:
        - word_tokenize(text)
        - char_tokenize(text)

    """

    def __init__(self):
        # Load pre-trained BPE tokenizer for word-level tokenization
        pkg_dir = Path(__file__).parent
        tokenizer_path = pkg_dir / "tokenizer.json"

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Pre-trained tokenizer not found at {tokenizer_path}. "
                "Ensure tokenizer.json is included in the package."
            )

        self._bpe_tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # ---------------------------------------
    # WORD LEVEL TOKENIZATION (BPE based)
    # ---------------------------------------
    def word_tokenize(self, text: str):
        """
        Tokenizes text using the pre-trained BPE tokenizer.

        Returns:
            List[str] — subword tokens
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string.")

        encoding = self._bpe_tokenizer.encode(text)
        return encoding.tokens

    # ---------------------------------------
    # CHARACTER LEVEL TOKENIZATION
    # ---------------------------------------
    def char_tokenize(self, text: str):
        """
        Tokenizes text at character level.

        Returns:
            List[str] — individual characters
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string.")

        # Preserve spaces as tokens
        return list(text)