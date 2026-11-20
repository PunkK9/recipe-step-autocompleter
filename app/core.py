"""Core implementation for Recipe Step Autocompleter."""
from collections import Counter, defaultdict
import math
import re
from typing import Any
from app.corpus import DEFAULT_RECIPE_INSTRUCTIONS

START_TOKEN = "<s>"
END_TOKEN = "</s>"


def clean_text(text: str) -> str:
    """Normalize whitespace and strip unnecessary characters."""
    return re.sub(r"\s+", " ", text).strip()


def tokenize(sentence: str, n_gram_order: int = 3, add_boundaries: bool = True) -> list[str]:
    """Tokenize a recipe instruction into lowercase tokens with optional n-gram boundary tokens.

    Args:
        sentence: Raw recipe instruction text.
        n_gram_order: Order of the n-gram model (used for start padding count = n - 1).
        add_boundaries: Whether to add start (<s>) and end (</s>) tokens.
    """
    cleaned = clean_text(sentence)
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", cleaned.lower())
    if not tokens:
        return []
    if add_boundaries:
        start_padding = [START_TOKEN] * max(1, n_gram_order - 1)
        return start_padding + tokens + [END_TOKEN]
    return tokens


def build_corpus(instructions: list[str] | None = None, n_gram_order: int = 3) -> list[list[str]]:
    """Compile and preprocess a corpus of recipe instructions."""
    raw_list = instructions if instructions is not None else DEFAULT_RECIPE_INSTRUCTIONS
    tokenized_corpus = []
    for line in raw_list:
        tokens = tokenize(line, n_gram_order=n_gram_order, add_boundaries=True)
        if tokens:
            tokenized_corpus.append(tokens)
    return tokenized_corpus


def get_vocabulary(tokenized_corpus: list[list[str]]) -> list[str]:
    """Extract sorted unique vocabulary from the preprocessed corpus."""
    vocab = set()
    for sentence in tokenized_corpus:
        vocab.update(sentence)
    return sorted(vocab)


# --- Milestone 1: Compile a corpus of cooking recipe instructions.
# Hint: Corpus creation and preprocessing.
def milestone_1(instructions: list[str] | None = None, n_gram_order: int = 3) -> dict[str, Any]:
    """Milestone 1: Compile and preprocess cooking recipe instructions."""
    corpus = build_corpus(instructions=instructions, n_gram_order=n_gram_order)
    vocab = get_vocabulary(corpus)
    return {
        "corpus": corpus,
        "vocabulary": vocab,
        "num_instructions": len(corpus),
        "total_tokens": sum(len(s) for s in corpus),
        "vocab_size": len(vocab),
    }


class NGramModel:
    """N-gram language model with Maximum Likelihood Estimation, Add-k smoothing, and backoff."""

    def __init__(self, n: int = 3, k: float = 1.0):
        self.n = max(1, n)
        self.k = float(k)
        # Store counts for all orders from 1 up to n for backoff
        self.all_ngram_counts: dict[int, dict[tuple[str, ...], int]] = {
            order: Counter() for order in range(1, self.n + 1)
        }
        self.all_context_counts: dict[int, dict[tuple[str, ...], int]] = {
            order: Counter() for order in range(1, self.n + 1)
        }
        self.vocabulary: set[str] = set()

    @property
    def ngram_counts(self) -> dict[tuple[str, ...], int]:
        return self.all_ngram_counts[self.n]

    @property
    def context_counts(self) -> dict[tuple[str, ...], int]:
        return self.all_context_counts[self.n]

    def train(self, corpus: list[list[str]]) -> None:
        """Extract and count n-grams and contexts from tokenized sentences across all orders."""
        for order in range(1, self.n + 1):
            self.all_ngram_counts[order].clear()
            self.all_context_counts[order].clear()
        self.vocabulary.clear()

        for sentence in corpus:
            self.vocabulary.update(sentence)
            for order in range(1, self.n + 1):
                if len(sentence) < order:
                    continue
                for i in range(len(sentence) - order + 1):
                    ngram = tuple(sentence[i : i + order])
                    context = ngram[:-1]
                    self.all_ngram_counts[order][ngram] += 1
                    self.all_context_counts[order][context] += 1

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    def get_ngram_count(self, ngram: tuple[str, ...]) -> int:
        """Return raw frequency of the n-gram for its specific order."""
        order = len(ngram)
        if order in self.all_ngram_counts:
            return self.all_ngram_counts[order].get(ngram, 0)
        return 0

    def get_context_count(self, context: tuple[str, ...]) -> int:
        """Return raw frequency of the context prefix."""
        order = len(context) + 1
        if order in self.all_context_counts:
            return self.all_context_counts[order].get(context, 0)
        return 0

    def mle_probability(self, word: str, context: tuple[str, ...]) -> float:
        """Calculate unsmoothed Maximum Likelihood Estimation probability: C(context, word) / C(context)."""
        ctx_count = self.get_context_count(context)
        if ctx_count == 0:
            return 0.0
        ngram = context + (word,)
        return self.get_ngram_count(ngram) / ctx_count

    def probability(self, word: str, context: tuple[str, ...], k: float | None = None) -> float:
        """Calculate smoothed probability using Add-k (Laplace) smoothing."""
        smooth_k = self.k if k is None else float(k)
        ctx_count = self.get_context_count(context)
        v_size = max(1, self.vocab_size)

        if smooth_k == 0.0:
            return self.mle_probability(word, context)

        ngram = context + (word,)
        ngram_count = self.get_ngram_count(ngram)
        return (ngram_count + smooth_k) / (ctx_count + smooth_k * v_size)

    def get_candidates(
        self,
        context: tuple[str, ...],
        top_n: int = 5,
        exclude_tokens: set[str] | None = None,
        use_backoff: bool = True,
    ) -> list[tuple[str, float]]:
        """Return top_n candidate words with their probabilities given context, with backoff support."""
        if exclude_tokens is None:
            exclude_tokens = {START_TOKEN}

        expected_len = self.n - 1
        # Truncate if context is longer than expected
        if len(context) > expected_len:
            curr_context = context[-expected_len:]
        elif len(context) < expected_len:
            curr_context = (START_TOKEN,) * (expected_len - len(context)) + context
        else:
            curr_context = context

        # Check if context was observed; if not and backoff is enabled, back off to smaller context
        if use_backoff and self.n > 1:
            while len(curr_context) > 0 and self.get_context_count(curr_context) == 0:
                curr_context = curr_context[1:]

        scores: list[tuple[str, float]] = []
        for word in self.vocabulary:
            if word in exclude_tokens:
                continue
            prob = self.probability(word, curr_context)
            scores.append((word, prob))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_n]


def generate_continuation(
    prompt: str,
    model: NGramModel | None = None,
    max_words: int = 15,
    stop_at_end: bool = True,
) -> str:
    """Generate autocompletion continuation for a partial recipe step using the N-gram model."""
    if model is None:
        model = milestone_2()["model"]

    # Tokenize the user's prompt without adding boundaries
    tokens = tokenize(prompt, n_gram_order=model.n, add_boundaries=False)

    expected_context_len = model.n - 1
    if len(tokens) < expected_context_len:
        current_context = [START_TOKEN] * (expected_context_len - len(tokens)) + tokens
    else:
        current_context = tokens[-expected_context_len:]

    generated_words: list[str] = []

    for _ in range(max_words):
        ctx_tuple = tuple(current_context[-expected_context_len:])
        candidates = model.get_candidates(ctx_tuple, top_n=5, exclude_tokens={START_TOKEN})
        if not candidates:
            break

        best_word = candidates[0][0]
        if stop_at_end and best_word == END_TOKEN:
            break

        generated_words.append(best_word)
        current_context.append(best_word)

    return " ".join(generated_words)


def predict_next_word_candidates(
    prompt: str, model: NGramModel | None = None, top_k: int = 5
) -> list[tuple[str, float]]:
    """Predict the top-k next words and their probabilities for a prompt."""
    if model is None:
        model = milestone_2()["model"]

    tokens = tokenize(prompt, n_gram_order=model.n, add_boundaries=False)
    expected_context_len = model.n - 1
    if len(tokens) < expected_context_len:
        ctx_tuple = (START_TOKEN,) * (expected_context_len - len(tokens)) + tuple(tokens)
    else:
        ctx_tuple = tuple(tokens[-expected_context_len:])

    return model.get_candidates(ctx_tuple, top_n=top_k, exclude_tokens={START_TOKEN, END_TOKEN})


# --- Milestone 2: Implement n-gram counting and maximum-likelihood probabilities with smoothing.
# Hint: N-gram counting and probability calculation with smoothing.
def milestone_2(
    instructions: list[str] | None = None, n: int = 3, k: float = 1.0
) -> dict[str, Any]:
    """Milestone 2: Train N-gram model, compute frequencies, and evaluate smoothed probabilities."""
    corpus = build_corpus(instructions=instructions, n_gram_order=n)
    model = NGramModel(n=n, k=k)
    model.train(corpus)

    sample_context = (START_TOKEN,) * (n - 1)
    sample_candidates = model.get_candidates(sample_context, top_n=5)

    return {
        "model": model,
        "n": n,
        "k": k,
        "total_unique_ngrams": len(model.ngram_counts),
        "total_unique_contexts": len(model.context_counts),
        "sample_context": sample_context,
        "sample_candidates": sample_candidates,
    }


# --- Milestone 3: Create a function to generate recipe step continuations.
# Hint: Text generation using the N-gram model.
def milestone_3(
    test_prompts: list[str] | None = None, model: NGramModel | None = None
) -> dict[str, Any]:
    """Milestone 3: Generate continuations for recipe instruction prefixes."""
    if model is None:
        m2_res = milestone_2()
        model = m2_res["model"]

    prompts = test_prompts or [
        "Preheat the oven to",
        "Heat olive oil in",
        "Season generously with",
        "Bring a large pot of",
    ]

    completions = {}
    for p in prompts:
        continuation = generate_continuation(p, model=model)
        candidates = predict_next_word_candidates(p, model=model, top_k=3)
        completions[p] = {
            "continuation": continuation,
            "full_step": f"{p} {continuation}".strip(),
            "top_candidates": candidates,
        }

    return {
        "model": model,
        "completions": completions,
    }


# --- Milestone 4: Build a CLI for users to input partial steps and get suggestions.
# Hint: Command-line interface (CLI) development.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")
