"""Command-line entry point: `python -m app.cli --help`."""
import argparse
import sys
from app.core import (
    milestone_2,
    generate_continuation,
    predict_next_word_candidates,
    tokenize,
    START_TOKEN,
    END_TOKEN,
)
from app.storage import Store

_MODEL = None


def get_shared_model():
    """Lazily load or train the shared N-gram model."""
    global _MODEL
    if _MODEL is None:
        # Check if user added extra custom recipe instructions via storage
        store = Store()
        custom_recipes = store.get("custom_recipes", [])
        from app.corpus import DEFAULT_RECIPE_INSTRUCTIONS

        all_instructions = DEFAULT_RECIPE_INSTRUCTIONS + custom_recipes
        _MODEL = milestone_2(instructions=all_instructions)["model"]
    return _MODEL


def cmd_autocomplete(args: argparse.Namespace) -> None:
    """Handle autocomplete command."""
    prompt = args.prompt
    model = get_shared_model()

    print(f"\n🍳 Recipe Step: '{prompt}'")
    print("-" * 50)

    # 1. Top N Next Word Predictions
    candidates = predict_next_word_candidates(prompt, model=model, top_k=args.top_n)
    print(f"Top {len(candidates)} Next-Word Suggestions:")
    for rank, (word, prob) in enumerate(candidates, 1):
        percentage = prob * 100
        print(f"  [{rank}] {word:<15} ({percentage:5.2f}%)")

    # 2. Full Completed Step
    continuation = generate_continuation(prompt, model=model, max_words=args.max_words)
    full_step = f"{prompt} {continuation}".strip()
    print("-" * 50)
    print(f"Suggested Completion:\n  -> {full_step}\n")


def cmd_interactive(args: argparse.Namespace) -> None:
    """Interactive step builder for home cooks."""
    model = get_shared_model()
    print("\n👩‍🍳 Welcome to the Interactive Recipe Step Builder!")
    print("Type words or pick a suggested number to build your cooking instruction.")
    print("Type 'exit' to quit or 'done' when your step is finished.\n")

    current_tokens: list[str] = []

    while True:
        current_text = " ".join(current_tokens)
        print(f"\nCurrent Step: '{current_text or '[empty]'}'")

        candidates = predict_next_word_candidates(current_text, model=model, top_k=args.top_n)
        print("Suggestions:")
        for idx, (word, prob) in enumerate(candidates, 1):
            print(f"  ({idx}) {word} [{prob * 100:.1f}%]")

        try:
            choice = input("\nType a word, choose a number (1-5), 'done', or 'exit': ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not choice or choice.lower() in ("exit", "quit"):
            break
        elif choice.lower() == "done":
            print(f"\nFinal Recipe Step:\n  ✨ {current_text}\n")
            current_tokens = []
            continue

        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            selected_word = candidates[int(choice) - 1][0]
            current_tokens.append(selected_word)
        else:
            # Tokenize user typed word(s) and append
            new_words = tokenize(choice, n_gram_order=model.n, add_boundaries=False)
            current_tokens.extend(new_words)


def cmd_add_instruction(args: argparse.Namespace) -> None:
    """Add a new instruction to the corpus to update the model dynamically."""
    instruction = args.instruction.strip()
    if not instruction:
        print("Error: Empty instruction cannot be added.")
        return

    global _MODEL
    store = Store()
    custom_recipes = store.get("custom_recipes", [])
    custom_recipes.append(instruction)
    store.put("custom_recipes", custom_recipes)
    _MODEL = None  # Invalidate cached model to retrain

    print(f"Added instruction to corpus: '{instruction}'")
    print(f"Total custom instructions stored: {len(custom_recipes)}")


def cmd_stats(args: argparse.Namespace) -> None:
    """Show model statistics."""
    model = get_shared_model()
    store = Store()
    custom_recipes = store.get("custom_recipes", [])
    print("\n📊 Model Statistics:")
    print(f"  - N-gram Order (N): {model.n}")
    print(f"  - Smoothing Parameter (k): {model.k}")
    print(f"  - Unique Vocabulary Size: {model.vocab_size} words")
    print(f"  - Unique N-grams Counted: {len(model.ngram_counts)}")
    print(f"  - Unique Contexts Counted: {len(model.context_counts)}")
    print(f"  - Custom Instructions in Store: {len(custom_recipes)}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="recipe-step-autocompleter",
        description="Home cooks use this CLI tool to autocomplete incomplete recipe instructions, ensuring they don't miss steps and can follow recipes smoothly.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Initial hello command
    sub.add_parser("hello", help="check that everything is wired up")

    # Autocomplete command
    p_auto = sub.add_parser("autocomplete", help="autocomplete a partial recipe step")
    p_auto.add_argument("prompt", type=str, help="partial recipe instruction text")
    p_auto.add_argument("--top-n", type=int, default=5, help="number of top word predictions")
    p_auto.add_argument("--max-words", type=int, default=10, help="maximum words to generate")

    # Interactive builder
    p_inter = sub.add_parser("interactive", help="interactively build a recipe step")
    p_inter.add_argument("--top-n", type=int, default=5, help="number of suggestions")

    # Add custom instruction to update corpus
    p_add = sub.add_parser("add-instruction", help="add a new recipe instruction to corpus")
    p_add.add_argument("instruction", type=str, help="new recipe instruction to add")

    # Model statistics
    sub.add_parser("stats", help="display model statistics")

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Recipe Step Autocompleter is ready. Open README.md for your milestones.")
    elif args.command == "autocomplete":
        cmd_autocomplete(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    elif args.command == "add-instruction":
        cmd_add_instruction(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
