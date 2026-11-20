# Recipe Step Autocompleter

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/xwtgp8ven3/badge.svg)](https://ask.thili.ai/b/xwtgp8ven3)

This CLI tool assists home cooks by providing intelligent autocompletion for recipe instructions. It leverages next-word prediction and text generation capabilities to suggest logical continuations, ensuring no crucial steps are missed during cooking.

## What you are building

**Goal.** You will build a Command Line Interface (CLI) tool that helps home cooks autocomplete incomplete recipe instructions, ensuring they follow recipes smoothly.

**Done when.** The tool successfully takes a partial recipe step as input via the CLI and suggests a logical next word or phrase based on its trained model, helping users complete the instruction.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Compile a corpus of cooking recipe instructions.**  
      Hint: Corpus creation and preprocessing.
- [x] **2. Implement n-gram counting and maximum-likelihood probabilities with smoothing.**  
      Hint: N-gram counting and probability calculation with smoothing.
- [x] **3. Create a function to generate recipe step continuations.**  
      Hint: Text generation using the N-gram model.
- [ ] **4. Build a CLI for users to input partial steps and get suggestions.**  
      Hint: Command-line interface (CLI) development.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Display multiple top-N word predictions instead of just one.
- Allow the user to select from the suggested continuations to build out the full step interactively.
- Implement a way to easily update the recipe instruction corpus with new data.
