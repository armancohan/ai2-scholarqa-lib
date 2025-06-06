import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml
from dotenv import load_dotenv

from api.scholarqa.rag.citation_utils import RegexNormalizer, VenueNormalizer

load_dotenv()

# Use the actual venues.yml for regex tests
VENUES_YAML = Path(__file__).parent.parent / "venues.yml"

@pytest.fixture(scope="module")
def regex_normalizer():
    return RegexNormalizer(VENUES_YAML)

@pytest.fixture(scope="module")
def venue_normalizer():
    # Use_llm_fallback False for pure regex tests
    return VenueNormalizer(VENUES_YAML, use_llm_fallback=False)

@pytest.mark.parametrize("input_str,expected", [
    ("Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing", "EMNLP"),
    ("Annual Meeting of the Association for Computational Linguistics", "ACL"),
    ("ICML 2023", "ICML"),
    ("Advances in Neural Information Processing Systems 34", "NeurIPS"),
    ("IEEE International Conference on Data Engineering", "ICDE"),
    ("Transactions of the Association for Computational Linguistics", "TACL"),
])
def test_regex_normalizer_matches(regex_normalizer, input_str, expected):
    assert regex_normalizer.match(input_str) == expected

def test_regex_normalizer_no_match(regex_normalizer):
    assert regex_normalizer.match("Completely Unknown Venue") is None

def test_venue_normalizer_regex_only(venue_normalizer):
    assert venue_normalizer.normalize("Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing") == "EMNLP"
    assert venue_normalizer.normalize("Unknown Conference 2023") == "OTHER"

@patch("api.scholarqa.rag.citation_utils.LLMBasedVenueNormalizer.normalize")
def test_venue_normalizer_llm_fallback(mock_llm):
    # Simulate LLM returning a valid key
    mock_llm.return_value = "ACL"
    vn = VenueNormalizer(VENUES_YAML, use_llm_fallback=True)
    assert vn.normalize("Association for Computational Linguistics Annual Meeting") == "ACL"
    # Simulate LLM returning None (unexpected value)
    mock_llm.return_value = None
    assert vn.normalize("Unknown Conference 2023") == "OTHER"

@patch("api.scholarqa.rag.citation_utils.LLMBasedVenueNormalizer.normalize")
def test_venue_normalizer_llm_other(mock_llm):
    # Simulate LLM returning 'OTHER'
    mock_llm.return_value = "OTHER"
    vn = VenueNormalizer(VENUES_YAML, use_llm_fallback=True)
    assert vn.normalize("Some random venue") == "OTHER" 