"""contains tests for the multilingual.py module"""
import pytest

from stanza.tests import TEST_MODELS_DIR

from stanza import MultilingualPipeline


@pytest.fixture(name="en_de_fr_texts")
def text_list() -> list[str]:
    return ["english text for language detection",
            "deutscher text für spracherkennung",
            "texte en français pour la reconnaissance vocale"]


def test_remove_lang_from_cache(en_de_fr_texts):
    """checks that cached models are freed correctly"""
    nlp = MultilingualPipeline(
        model_dir=TEST_MODELS_DIR, max_cache_size=2
    )
    results = []
    for text in en_de_fr_texts:
        results.append(nlp(text))
    assert len(results) == 3
