from __future__ import annotations

import re
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Pattern, Tuple, Optional

from api.scholarqa.llms import constants

# ---------------------------------------------------------------------------
# ⚙️ Configuration
# ---------------------------------------------------------------------------
DEFAULT_YAML = Path(__file__).with_name("venues.yaml")
# Default LLM model to use via litellm. Adjust to your org/model names.
DEFAULT_MODEL = constants.GPT_41_MINI

# ---------------------------------------------------------------------------
# 🛠  Regex‑based normaliser
# ---------------------------------------------------------------------------
class RegexNormalizer:
    """Compile the YAML mapping once and provide `match(text)`."""

    def __init__(self, yml_path: Path | str | None = None) -> None:
        yml_path = Path(yml_path) if yml_path else DEFAULT_YAML
        if not yml_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {yml_path}")
        with yml_path.open("r", encoding="utf-8") as fh:
            raw: Dict[str, List[str]] = yaml.safe_load(fh)

        # Pre‑compile regex patterns (case‑insensitive)
        compiled: Dict[str, List[Pattern[str]]] = {}
        for canon, patterns in raw.items():
            compiled[canon] = [re.compile(p, flags=re.I) for p in patterns]
        self.patterns = compiled

    # ---------------------------------------------------------------------
    def match(self, venue: str) -> Optional[str]:
        """Return canonical key if *any* pattern matches, else None."""
        for canon, pats in self.patterns.items():
            if any(p.search(venue) for p in pats):
                return canon
        return None

# ---------------------------------------------------------------------------
# 🤖 LLM fallback normaliser
# ---------------------------------------------------------------------------
class LLMBasedVenueNormalizer:
    """Query an LLM via litellm to guess the canonical venue key."""

    _SYSTEM_PROMPT = (
        "You are a BibTeX venue normalizer familar with NLP, ML, AI venues. Return **only** the canonical short "
        "key (e.g., ACL, EMNLP, ICML, OTHER) that best matches the input. "
        "for arxiv return arxiv."
        "Respond with a single word and no extra commentary."
    )

    def __init__(self, valid_keys: List[str], model: str = DEFAULT_MODEL):
        try:
            import litellm  # noqa: F401 – runtime import to avoid hard dep if unused
        except ImportError as err:
            raise ImportError("litellm is required for LLM fallback. Install via 'pip install litellm'" ) from err
        self._litellm = __import__("litellm")
        self.valid_keys = valid_keys
        self.model = model

    # ---------------------------------------------------------------------
    def normalize(self, venue: str, temperature: float = 0.0) -> str | None:
        """Return canonical key or None if model returns unexpected value."""
        response = self._litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": venue},
            ],
            max_tokens=4,
            temperature=temperature,
        )
        key = (
            response["choices"][0]["message"]["content"].strip().upper()
        )
        return key if key in self.valid_keys or key == "OTHER" else None


class VenueNormalizer:
    """Combine regex layer and optional LLM fallback."""

    def __init__(
        self,
        yml_path: Path | str | None = None,
        use_llm_fallback: bool = True,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.regex_norm = RegexNormalizer(yml_path)
        self.use_llm = use_llm_fallback
        if use_llm_fallback:
            self.llm_norm = LLMBasedVenueNormalizer(list(self.regex_norm.patterns.keys()), model=model)

    # ---------------------------------------------------------------------
    def normalize(self, venue: str) -> str:
        """Return canonical key or 'OTHER'."""
        venue = venue.strip()
        match = self.regex_norm.match(venue)
        if match:
            return match
        if self.use_llm:
            guess = self.llm_norm.normalize(venue)
            if guess:
                return guess
        return "OTHER"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python venue_normalizer.py <venue string>")
        sys.exit(1)

    vn_str = " ".join(sys.argv[1:])
    norm = VenueNormalizer()
    canonical = norm.normalize(vn_str)
    print(canonical)
