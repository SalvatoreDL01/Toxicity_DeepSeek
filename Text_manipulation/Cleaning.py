import re
import string


def clean_continuation(text: str) -> str:
    """Ripulisce la continuation per evitare problemi di formattazione."""
    # Rimuove \n
    text = text.replace("\n", " ").strip()
    # Sostituisce doppi apici interni con apostrofi
    text = text.replace('"', "'")
    # Evita spazi multipli
    text = re.sub(r"\s+", " ", text)
    return text


def remove_overlap_from_str2(str1, str2, min_overlap_ratio=0.5):
    if str2.count(str1) > 0:
        str2 = str2.replace(str1, '')

    clean1 = str1.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'").replace("—", "-")
    # Replace ellipsis with a single space
    clean1 = clean1.replace("...", " ")

    clean2 = str2.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'").replace("—", "-")
    # Replace ellipsis with a single space
    clean2 = clean2.replace("...", " ")

    # Rimuove la punteggiatura provvisoriamente
    clean1 = ' '.join(re.sub(rf"[{re.escape(string.punctuation)}]", "", clean1).split())
    clean2 = ' '.join(re.sub(rf"[{re.escape(string.punctuation)}]", "", clean2).split())

    words_str2 = str2.split()

    words_clenaed_2 = len(words_str2)

    if clean2.count(clean1) > 0:
        words_clenaed_2 = clean2.find(clean1) + len(clean1)

    return ' '.join(words_str2[-words_clenaed_2:])

    return str2


def truncate_to_20_tokens(prompt, continuation, tokenizer, max_new_tokens=20):
    """
    Truncate continuation to have exactly max_new_tokens new tokens
    beyond what's already in the prompt.
    """
    # Tokenize both prompt and continuation
    prompt_tokens = tokenizer.encode(prompt)
    full_text = prompt + continuation
    full_tokens = tokenizer.encode(full_text)

    # Calculate how many new tokens we have
    new_tokens_count = len(full_tokens) - len(prompt_tokens)

    if new_tokens_count <= max_new_tokens:
        return continuation  # Already within limit

    # Truncate to exactly max_new_tokens new tokens
    end_index = len(prompt_tokens) + max_new_tokens
    truncated_tokens = full_tokens[len(prompt_tokens):end_index]

    # Decode back to text (only the new tokens part)
    truncated_continuation = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    return truncated_continuation


def remove_boilerplate_continuation(text: str) -> str:
    """
    Removes boilerplate continuations like:
    "Of course, here is the continuation of the text..."
    or variants (with optional punctuation, capitalization, etc.).
    """
    pattern = re.compile(
        r"""
        ^\s*
        Of\ course\b[.,:!]*\s*             # MUST start with "Of course"
        (?:Here(?:\s+is|['’]s))\s+         # "Here is" or "Here's"
        (?:(?:the|a)\s+)?                  # optional article
        continuation\b                     # "continuation"
        (?:\s+of\b(?:\s+(?:that|the)\s+text\b)?)?  # optional "of (that|the) text"
        (?:\s*,?\s*written\s+in\s+the\s+style\s+of\s+[^.:]+)?  # optional style phrase
        \s*[:.]?\s*                        # optional trailing ':' or '.'
        """,
        re.IGNORECASE | re.VERBOSE
    )
    return re.sub(pattern, "", text).strip()
