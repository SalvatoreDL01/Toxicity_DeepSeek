import re

def clean_continuation(text: str) -> str:
    """Ripulisce la continuation per evitare problemi di formattazione."""
    # Rimuove \n
    text = text.replace("\n", " ").strip()
    # Sostituisce doppi apici interni con apostrofi
    text = text.replace('"', "'")
    # Evita spazi multipli
    text = re.sub(r"\s+", " ", text)
    return text
