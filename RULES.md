# Rules

### Team Size
- **3–4 members** per team
- Every member must be able to explain the code individually during viva

### What You Get and When

| Time | What's Released | Where |
|------|----------------|-------|
| **12 hrs before event** | Problem statements (PROBLEM.md for both A and B), feature schemas, dummy data (dummy.csv for both), submission sheet template, this rules doc | This repo |
| **Event start (T+0:00)** | Training datasets (train.csv for your chosen problem) | This repo |
| **T+4:30** | Final test dataset (test.csv for your chosen problem) | This repo |

**Pick one problem.** You will only submit predictions for Problem A or Problem B, not both.

### What You Submit

By **T+5:00** (hard deadline), submit a Pull Request containing:

```
submissions/your_team_name/
├── predictions.csv
├── submission_sheet.pdf
└── code/
    └── (all notebooks/scripts)
```

See [SUBMISSIONS.md](SUBMISSIONS.md) for exact format and steps.

### predictions.csv Format

**Problem A:**
```csv
id,prediction
1,late
2,on_time
```
Only two values allowed: `late` or `on_time`

**Problem B:**
```csv
id,prediction
1,price_sensitive
2,found_alternative
```
Only four values allowed: `price_sensitive`, `bad_experience`, `found_alternative`, `lost_interest`

All lowercase. Exact spelling. Every test row must have a prediction. No missing values.

### Evaluation

| Component | Weight |
|-----------|--------|
| Model performance (Macro F1) | **50%** |
| Professor viva | **50%** |

**Model scoring is relative, not absolute.** Your score is calculated as:

```
Model Score = (Your Macro F1 / Best Macro F1 in your problem) × 50
```

Teams choosing Problem A are scored against other Problem A teams. Same for Problem B. No cross-problem comparison.

**Viva** covers 5 areas (10 points each, total 50):
- Data understanding and cleaning
- Feature engineering and analysis
- Model choice and reasoning
- Honesty and self-awareness (what didn't work and why)
- Individual understanding and teamwork

### External Data

- **Allowed.** Any publicly available dataset you can find online.
- **Must be relevant** to your chosen problem.
- **Must be cited** in your submission sheet — source name, URL, and why you used it.
- **Not required.** You can score well using only the provided training data if you clean it properly.
- **Warning:** The test set follows the same distribution as the provided training data. External data from a completely different domain or distribution may hurt your model rather than help it.

### Allowed Tools and Libraries

- **Any programming language** — Python recommended
- **Any ML library** — scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch, TensorFlow, etc.
- **Any IDE or notebook** — Jupyter, Colab, VS Code, etc.
- **Internet access** — for documentation, StackOverflow, external data. Not for copying someone else's solution to this exact problem (there isn't one).

### Not Allowed

- **AI coding assistants are banned.** No Cursor, no GitHub Copilot, no Claude Code, no Windsurf, no Codeium, no Amazon CodeWhisperer, no Tabnine, no AI autocomplete of any kind. Turn them off. If we find AI-generated code that you cannot explain line by line in the viva, your submission is disqualified.
- **No ChatGPT / Claude / Gemini to write your code.** You may use LLMs to understand a concept (e.g., "what is Macro F1?") or debug an error message. You may NOT paste your dataset or problem into an LLM and ask it to build your pipeline. The viva will expose this.
- **Pre-trained models on the exact task** — no fine-tuning a model that already solves this problem
- **Sharing data or code between teams** during the event
- **Submitting predictions you didn't generate** — your code must produce your predictions.csv
- **Using the dummy.csv labels as test predictions** — dummy is a sample from training data, not from the test set

### Deadlines

| What | When | Strict? |
|------|------|---------|
| Pick your problem | Before event start | Yes |
| Submit PR | T+5:00 | **Yes. PR timestamp is final. No exceptions.** |
| Viva | T+5:00 to T+5:45 | Scheduled by organizers |

### Tie-Breaking

If two teams have the same total score (model + viva), the team with the higher viva score wins. If still tied, the team that submitted earlier wins.

### Questions?

Ask during the T+0:00 kickoff Q&A, or reach out to organizers on Discord/WhatsApp.

---

*SPEC2MODEL Challenge — GDGOC Silicon University — Zygon x Neosis Annual Fest*
