import re

SYMBOL_REPLACEMENTS: dict[str, str] = {
    "&": " and ",
    "+": " plus ",
    "@": " at ",
}

# Corporate/legal suffixes to strip, ordered longest-first to avoid partial matches.
ENTITY_SUFFIXES: list[str] = [
    "limited liability company",
    "corporation",
    "incorporated",
    "limited",
    "company",
    "llc",
    "inc",
    "ltd",
    "plc",
    "corp",
    "co",
    "lp",
    "llp",
    "gmbh",
    "ag",
    "sa",
    "srl",
    "bv",
    "nv",
    "pty",
]

# Regex: match any suffix at end of string, optionally preceded by comma/period/space.
# Word boundary (\b) ensures we don't match partial words (e.g., "sago" won't match "sa").
_suffix_alternatives = "|".join(re.escape(s) for s in ENTITY_SUFFIXES)
ENTITY_SUFFIX_PATTERN: re.Pattern[str] = re.compile(
    rf"[,.\s]*\b({_suffix_alternatives})\b[.]?\s*$",
    re.IGNORECASE,
)
