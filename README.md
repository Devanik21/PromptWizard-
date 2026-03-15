# Promptwizard

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square) ![Stars](https://img.shields.io/github/stars/Devanik21/PromptWizard-?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/Devanik21/PromptWizard-?style=flat-square&color=blue) ![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> Master prompt engineering — a comprehensive toolkit for designing, testing, optimising, and managing prompts for LLM applications.

---

**Topics:** `ai-tools` · `deep-learning` · `generative-ai` · `large-language-models` · `llm` · `meta-prompting` · `neural-networks` · `prompt-engineering` · `prompt-optimization` · `automated-prompt-design`

## Overview

PromptWizard is a professional prompt engineering workbench that provides a structured environment
for designing, iterating, evaluating, and managing prompts across multiple LLM backends. It addresses
a genuine gap in the LLM application development workflow: most developers write prompts in ad hoc
string variables, test them manually, and lack systematic ways to compare prompt variants or measure
prompt quality objectively.

The workbench provides a side-by-side prompt comparison interface where multiple prompt variants
can be run against the same set of test inputs simultaneously, with responses displayed in columns
and quantitative evaluation metrics computed automatically. The evaluation framework supports both
reference-based metrics (ROUGE, BLEU for when a ground truth exists) and reference-free metrics
(LLM-as-judge scoring, semantic similarity, format compliance checking).

A prompt template library stores reusable prompt patterns with metadata: the task type, the model
family it was designed for, performance benchmarks, version history, and usage notes. Templates
are parameterised with Jinja2-style variables and can be combined with a prompt composition system
that chains multiple templates for complex multi-step tasks.

---

## Motivation

Prompt engineering is transitioning from an art to an engineering discipline. Production LLM applications
require prompts that are reliable, measurable, and maintainable — not artisanal one-off strings.
PromptWizard was built to bring software engineering rigour to prompt development: version control,
systematic testing, quantitative evaluation, and structured iteration rather than intuitive tweaking.

---

## Architecture

```
Prompt Design Interface
        │
  Parameterised Template (Jinja2)
        │
  ┌──────────────────────────────────────────────┐
  │  Multi-Model Runner                          │
  │  GPT-4o | Gemini | Claude | Ollama           │
  └──────────────────────────────────────────────┘
        │
  ┌──────────────────────────────────────────────┐
  │  Evaluation Framework                       │
  │  ├── Reference-based: ROUGE, BLEU, F1       │
  │  ├── LLM-as-judge: quality scores           │
  │  ├── Format compliance checking             │
  │  └── Semantic similarity (cosine)           │
  └──────────────────────────────────────────────┘
        │
  Prompt Library (versioned, searchable)
        │
  A/B test results dashboard
```

---

## Features

### Multi-Model Side-by-Side Comparison
Run the same prompt across GPT-4o, Gemini, Claude, and locally-running Ollama models simultaneously, with responses displayed in columns for direct comparison.

### Parameterised Prompt Templates
Jinja2-style variable injection in prompt templates ({topic}, {language}, {format}) with a form-based variable filler for quick test case generation.

### Automated Evaluation Framework
Quantitative prompt quality measurement: ROUGE-L for summarisation, BLEU for translation, exact-match for classification, semantic similarity for open-ended tasks, and LLM-as-judge (GPT-4o scoring on a 1–10 rubric) for subjective tasks.

### A/B Prompt Testing
Run two prompt variants across the same evaluation set and receive a statistical comparison (t-test, effect size) of their performance metrics — making prompt changes evidence-based.

### Prompt Version Control
Git-like version history for prompts: save, diff, and rollback prompt versions with performance metadata attached to each version.

### Chain-of-Thought Inspector
For chain-of-thought prompts, display the reasoning trace separately from the final answer, with per-step confidence score estimation.

### Token Usage Profiler
Per-prompt token count breakdown: system prompt, few-shot examples, user input — with cost estimation across model pricing tiers.

### Export to Code
One-click export of any prompt configuration as Python code (OpenAI SDK, LangChain, LlamaIndex) or JSON for immediate integration into production applications.

---

## Tech Stack

| Library / Tool | Role | Why This Choice |
|---|---|---|
| **OpenAI / Anthropic / Gemini SDK** | LLM backends | Multi-model API calls with unified interface |
| **Streamlit** | Workbench UI | Side-by-side comparison, prompt editor, metrics display |
| **Jinja2** | Template engine | Variable injection and prompt composition |
| **NLTK / rouge-score** | Evaluation metrics | BLEU, ROUGE-L computation for text generation quality |
| **sentence-transformers** | Semantic similarity | Cosine similarity between expected and generated outputs |
| **pandas** | Results management | Evaluation results storage and statistical comparison |
| **SQLite** | Prompt library | Versioned prompt storage with metadata |

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Node.js 18+ for TypeScript/JavaScript projects)
- A virtual environment manager (`venv`, `conda`, or equivalent)
- API keys as listed in the Configuration section

### Installation

```bash
git clone https://github.com/Devanik21/PromptWizard-.git
cd PromptWizard-
python -m venv venv && source venv/bin/activate
pip install streamlit openai anthropic google-generativeai jinja2 \
            rouge-score nltk sentence-transformers pandas python-dotenv
echo 'OPENAI_API_KEY=sk-...' > .env
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env
streamlit run app.py
```

---

## Usage

```bash
# Launch workbench
streamlit run app.py

# Run evaluation suite on a prompt
python evaluate_prompt.py \
  --prompt templates/summarisation_v3.j2 \
  --eval_set data/news_articles.jsonl \
  --metrics rouge,semantic

# A/B test two prompts
python ab_test.py \
  --prompt_a templates/v2.j2 \
  --prompt_b templates/v3.j2 \
  --eval_set data/test_cases.jsonl

# Export best prompt to Python
python export_prompt.py --prompt_id 42 --format openai_sdk
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | `(required)` | OpenAI API key |
| `ANTHROPIC_API_KEY` | `(optional)` | Anthropic Claude API key |
| `GOOGLE_API_KEY` | `(optional)` | Google Gemini API key |
| `DEFAULT_JUDGE_MODEL` | `gpt-4o-mini` | LLM-as-judge evaluation model |
| `MAX_TOKENS_RESPONSE` | `1000` | Maximum tokens per model response during evaluation |

> Copy `.env.example` to `.env` and populate required values before running.

---

## Project Structure

```
PromptWizard/
├── README.md
├── requirements.txt
├── app.py
└── ...
```

---

## Roadmap

- [ ] Automatic prompt optimisation: DSPy-style few-shot example selection and instruction refinement
- [ ] Evaluation dataset generator: automatically create test cases from a task description
- [ ] Prompt injection vulnerability scanner for security-sensitive deployments
- [ ] Team collaboration mode with shared prompt libraries and review workflows
- [ ] CI/CD integration: GitHub Actions plugin for automated prompt regression testing on code push

---

## Contributing

Contributions, issues, and suggestions are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-idea`
3. Commit your changes: `git commit -m 'feat: add your idea'`
4. Push to your branch: `git push origin feature/your-idea`
5. Open a Pull Request with a clear description

Please follow conventional commit messages and add documentation for new features.

---

## Notes

LLM-as-judge evaluation costs additional API tokens — a full evaluation suite on GPT-4o can accumulate meaningful costs. Use gpt-4o-mini as the judge model for cost-efficient evaluation at moderate quality, reserving gpt-4o for final validation of production prompts.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with curiosity, depth, and care — because good projects deserve good documentation.*
