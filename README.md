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
- [x] **4. Build a CLI for users to input partial steps and get suggestions.**  
      Hint: Command-line interface (CLI) development.

## Run it

You need Python 3.12+ and git.

```bash
# 1. Create virtual environment and install requirements
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Copy settings file
cp .env.example .env

# 3. Check installation
python -m app.cli hello

# 4. Run tests
pytest
```

## CLI Usage

The tool provides several command-line features for home cooks:

### 1. Autocomplete a Step
Input a partial instruction to see the top-N next-word candidate probabilities and a full suggested step:
```bash
python -m app.cli autocomplete "Heat olive oil in a" --top-n 5 --max-words 10
```

Output:
```
🍳 Recipe Step: 'Heat olive oil in a'
--------------------------------------------------
Top 5 Next-Word Suggestions:
  [1] large           ( 0.82%)
  [2] medium          ( 0.54%)
  [3] smoking         ( 0.54%)
  [4] separate        ( 0.54%)
  [5] heavy           ( 0.54%)
--------------------------------------------------
Suggested Completion:
  -> Heat olive oil in a large skillet over medium heat.
```

### 2. Interactive Recipe Step Builder (Stretch Idea 2)
Interactively build recipe steps word-by-word with autocomplete suggestions:
```bash
python -m app.cli interactive
```

### 3. Add Custom Recipe Instructions (Stretch Idea 3)
Dynamically update the training corpus with new recipes:
```bash
python -m app.cli add-instruction "Whisk the matcha powder with hot water until frothy."
```

### 4. View Model Statistics
Inspect vocabulary size, n-gram frequencies, and model parameters:
```bash
python -m app.cli stats
```

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas Implemented

- [x] **Display multiple top-N word predictions**: Ranked next-word predictions with smoothed probabilities.
- [x] **Interactive recipe step builder**: Home cooks can pick suggestions iteratively to construct full steps.
- [x] **Corpus updates**: Support dynamically adding new cooking instructions to the corpus via storage.
