"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest
from app.core import (
    milestone_1,
    milestone_2,
    milestone_3,
    NGramModel,
    generate_continuation,
    predict_next_word_candidates,
    tokenize,
    clean_text,
    build_corpus,
    get_vocabulary,
    START_TOKEN,
    END_TOKEN,
)


# Milestone 1: Compile a corpus of cooking recipe instructions.
def test_milestone_1():
    result = milestone_1()
    assert isinstance(result, dict)
    assert "corpus" in result
    assert "vocabulary" in result
    assert result["num_instructions"] > 0
    assert result["vocab_size"] > 0

    corpus = result["corpus"]
    first_sentence = corpus[0]
    assert first_sentence[0] == START_TOKEN
    assert first_sentence[-1] == END_TOKEN

    custom_recipes = [
        "Preheat the oven to 375 degrees.",
        "Whisk the eggs gently."
    ]
    custom_result = milestone_1(custom_recipes, n_gram_order=2)
    assert custom_result["num_instructions"] == 2
    assert "preheat" in custom_result["vocabulary"]
    assert "whisk" in custom_result["vocabulary"]

    assert clean_text("  Heat   olive   oil.  ") == "Heat olive oil."
    tokens = tokenize("Chop 2 onions, finely!", n_gram_order=3, add_boundaries=False)
    assert tokens == ["chop", "2", "onions", "finely"]


# Milestone 2: Implement n-gram counting and maximum-likelihood probabilities with smoothing.
def test_milestone_2():
    sentences = [
        ["<s>", "<s>", "heat", "the", "oil", "</s>"],
        ["<s>", "<s>", "heat", "the", "butter", "</s>"],
    ]
    model = NGramModel(n=3, k=1.0)
    model.train(sentences)

    assert model.get_ngram_count(("<s>", "<s>", "heat")) == 2
    assert model.get_ngram_count(("heat", "the", "oil")) == 1
    assert model.get_ngram_count(("heat", "the", "butter")) == 1
    assert model.get_context_count(("heat", "the")) == 2

    prob_oil_mle = model.mle_probability("oil", ("heat", "the"))
    prob_butter_mle = model.mle_probability("butter", ("heat", "the"))
    prob_unseen_mle = model.mle_probability("garlic", ("heat", "the"))
    assert prob_oil_mle == 0.5
    assert prob_butter_mle == 0.5
    assert prob_unseen_mle == 0.0

    v_size = model.vocab_size
    assert v_size == 6
    prob_oil_smooth = model.probability("oil", ("heat", "the"), k=1.0)
    assert prob_oil_smooth == pytest.approx(2.0 / (2.0 + 1.0 * v_size))

    prob_unseen_smooth = model.probability("unseen_word", ("heat", "the"), k=1.0)
    assert prob_unseen_smooth > 0.0

    total_prob = sum(model.probability(w, ("heat", "the"), k=1.0) for w in model.vocabulary)
    assert total_prob == pytest.approx(1.0)

    candidates = model.get_candidates(("heat", "the"), top_n=2)
    assert len(candidates) == 2
    candidate_words = [word for word, _ in candidates]
    assert "oil" in candidate_words
    assert "butter" in candidate_words

    res = milestone_2()
    assert isinstance(res, dict)
    assert res["total_unique_ngrams"] > 50
    assert len(res["sample_candidates"]) > 0


# Milestone 3: Create a function to generate recipe step continuations.
def test_milestone_3():
    # 1. Test next word prediction
    top_candidates = predict_next_word_candidates("Preheat the oven to", top_k=3)
    assert len(top_candidates) > 0
    words = [word for word, _ in top_candidates]
    # In cooking corpus, '350' or '400' follow 'oven to'
    assert any(num in words for num in ["350", "400", "degrees"])

    # 2. Test continuation generation
    continuation = generate_continuation("Preheat the oven to", max_words=10)
    assert len(continuation) > 0
    assert "degrees" in continuation

    # 3. Test season continuation
    season_continuation = generate_continuation("Season generously with", max_words=5)
    assert any(seasoning in season_continuation for seasoning in ["kosher", "salt", "black", "pepper"])

    # 4. Test max_words limits length
    short_cont = generate_continuation("Heat olive oil in", max_words=2)
    assert len(short_cont.split()) <= 2

    # 5. Test milestone_3 wrapper
    m3_res = milestone_3()
    assert "completions" in m3_res
    assert len(m3_res["completions"]) >= 4
    for prompt, data in m3_res["completions"].items():
        assert "continuation" in data
        assert "full_step" in data
        assert data["full_step"].startswith(prompt)


# Milestone 4: Build a CLI for users to input partial steps and get suggestions.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")
