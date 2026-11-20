"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest
from app.core import (
    milestone_1,
    tokenize,
    clean_text,
    build_corpus,
    get_vocabulary,
    START_TOKEN,
    END_TOKEN,
)


# Milestone 1: Compile a corpus of cooking recipe instructions.
def test_milestone_1():
    # Test milestone_1 compiles default corpus
    result = milestone_1()
    assert isinstance(result, dict)
    assert "corpus" in result
    assert "vocabulary" in result
    assert result["num_instructions"] > 0
    assert result["vocab_size"] > 0

    corpus = result["corpus"]
    # Check that sentences have boundary tokens
    first_sentence = corpus[0]
    assert first_sentence[0] == START_TOKEN
    assert first_sentence[-1] == END_TOKEN

    # Test custom corpus input
    custom_recipes = [
        "Preheat the oven to 375 degrees.",
        "Whisk the eggs gently."
    ]
    custom_result = milestone_1(custom_recipes, n_gram_order=2)
    assert custom_result["num_instructions"] == 2
    assert "preheat" in custom_result["vocabulary"]
    assert "whisk" in custom_result["vocabulary"]

    # Test tokenization and clean text
    assert clean_text("  Heat   olive   oil.  ") == "Heat olive oil."
    tokens = tokenize("Chop 2 onions, finely!", n_gram_order=3, add_boundaries=False)
    assert tokens == ["chop", "2", "onions", "finely"]


# Milestone 2: Implement n-gram counting and maximum-likelihood probabilities with smoothing.
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Create a function to generate recipe step continuations.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Build a CLI for users to input partial steps and get suggestions.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")
