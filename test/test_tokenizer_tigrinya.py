from tigrinya_tokenizer import TigrinyaTokenizer

# Initialize your library
tokenizer = TigrinyaTokenizer()


# -------------------------------
# 1️⃣ Word-Level Test Words
# -------------------------------

TEST_WORDS = [
    "ሰላም",
    "ትግርኛ",
    "ኣደርካ",
    "ሕብረት",
    "መንግስቲ",
    "ምምሕዳር",
    "ትምህርቲ",
    "ኤርትራ",
    "ሃገር",
    "ፍቕሪ",
    "ጸሓፊ",
    "ቤት",
    "ስራሕ",
    "ኣቦ",
    "ኣይተ"
]


def test_word(word):
    print("=" * 60)
    print(f"Original: {word}")

    tokens = tokenizer.word_tokenize(word)
    char_tokens = tokenizer.char_tokenize(word)

    print(f"Word Tokens : {tokens}")
    print(f"Char Tokens : {char_tokens}")

    # Basic validation checks
    if not tokens:
        print("❌ Word tokenization FAILED (empty output)")
    else:
        print("✅ Word tokenization OK")

    if not char_tokens:
        print("❌ Char tokenization FAILED (empty output)")
    else:
        print("✅ Char tokenization OK")


def run_word_tests():
    print("\nRunning Word-Level Tokenization Tests\n")
    for word in TEST_WORDS:
        test_word(word)


# -------------------------------
# 2️⃣ Sentence-Level Tests
# -------------------------------

SENTENCES = [
    "ሰላም ኩን ኣደርካ?",
    "ኣብ ትግርኛ መምህራን ኣሎዉ።",
    "ትምህርቲ ኣገዳሲ ኢዩ፣ ወላ'ውን ጠቃሚ ኢዩ",
]


def run_sentence_tests():
    print("\nRunning Sentence-Level Tests\n")
    for sentence in SENTENCES:
        test_word(sentence)


if __name__ == "__main__":
    run_word_tests()
    run_sentence_tests()