import yaml
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders
import importlib.resources as pkg_resources
from . import configs  # the folder inside your package

class TigrinyaTokenizer:
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.tokenizer_path = Path("outputs/tokenizer/tokenizer.json")

        # Load config from package
        self.cfg = self.load_config()

        # Load or train tokenizer
        if self.tokenizer_path.exists():
            self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        else:
            print("[INFO] No existing tokenizer found, will train new one.")
            self.train()

    def load_config(self):
        # Load YAML config from package resources
        with pkg_resources.open_text(configs, "bpe_50k.yaml") as f:
            return yaml.safe_load(f)

    def train(self):
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.normalizers import Sequence, NFC
        from tokenizers.decoders import BPEDecoder
        import os

        corpus_file = "data/processed/normalized.txt"
        if not Path(corpus_file).exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.normalizer = Sequence([NFC()])
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.decoder = BPEDecoder()

        trainer = BpeTrainer(
            vocab_size=self.cfg["tokenizer"]["vocab_size"],
            min_frequency=self.cfg["tokenizer"]["min_frequency"],
            special_tokens=self.cfg["special_tokens"]
        )

        tokenizer.train(files=[corpus_file], trainer=trainer)

        self.tokenizer = tokenizer
        self.tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(self.tokenizer_path))

    def tokenize(self, text: str):
        return self.tokenizer.encode(text).tokens