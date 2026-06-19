# Project 3 — "What Are You Working On?" HN Project-Topic Classifier

A 3-class text classifier that reads a single Hacker News "Ask HN: What are you working on?" comment and labels the **topic** of the project being described: `ai_ml`, `developer_tools`, or `end_user_app`.

## Community

I chose **Hacker News**, specifically the recurring monthly **"Ask HN: What are you working on?"** threads (e.g. [item 48528779](https://news.ycombinator.com/item?id=48528779), June 2026 — 1,100+ comments). In these threads, members post a short description of a project they're currently building. The unit I classify is a **single top-level comment** (one person's project description).

**Why this is a good fit for a classification task:** the discourse is varied but bounded. Every comment is "here's what I'm building," yet _what kind of thing_ people build splits into recognizable topics — an AI/ML system, a developer tool, or an app for ordinary users. The vocabulary differs sharply by topic ("LLM/agent/fine-tune/model" vs. "CLI/library/framework/Postgres" vs. "app/tracker/budget/recipe"), so a classifier has strong lexical signal. The interesting fuzz is the AI-pervasiveness problem: in 2026 almost every comment _mentions_ AI, so the task isn't keyword-trivial — the model must distinguish projects whose **core is AI** from ordinary apps that merely _use_ AI as a feature.

**Why the data is abundant:** the thread recurs every month and is archived for years. I collect via the **HN Algolia API** (`https://hn.algolia.com/api/v1/search_by_date?tags=comment,story_<id>`), which returns every comment with no auth and no blocking, and lets me filter to top-level comments by `parent_id`.

## Labels

I use **three** labels. Each classifies a comment by the project's **topic** — what kind of thing it is. (I originally drafted intent-based labels — commercial vs. hobby vs. open-source — but those depend on _why_ someone built something, which is subtle even for humans; topic-based labels have far stronger signal and are the better-posed task.)

### 1. ai_ml

**Definition:** The project's core **is** an AI/ML system — a model, agent, LLM application, fine-tuning effort, computer-vision/NLP system, RAG pipeline, or AI research. The intelligence is the product, not just a feature.

**Typical signs:** "LLM," "agent," "model," "fine-tune," "neural," "RAG," "inference," "embeddings," "training," AI research.

**Examples:**

- "Building a local-first autonomous agent — experimenting with how planning compares to step-by-step tool-calling, and fine-tuning a small model for it."
- "An open-source Text2SQL tool that turns natural language into SQL using a graph-powered schema understanding."

**Boundary rule:** Use this only when AI/ML is the **central deliverable**. A normal app that merely uses AI as one feature is `end_user_app`. An AI-centric tool aimed squarely at developers (e.g. an AI coding agent) goes here only when the AI _is_ the whole point; otherwise `developer_tools`.

### 2. developer_tools

**Definition:** A library, framework, CLI, database, infrastructure, or developer-facing tool whose main audience is other developers/programmers.

**Typical signs:** GitHub links, "open source," "library," "framework," "CLI," "SDK," "self-hosted," "Postgres," "Kubernetes," "compiler," "API."

**Examples:**

- "An open-source Postgres migration CLI; just cut the 1.0 release on GitHub."
- "A TUI framework so I can build my local agent harness on top of it."

**Boundary rule:** Use this when the users are **developers/technical users**. If the same kind of tool is aimed at everyday people or businesses, it's `end_user_app`. If it's open source but fundamentally an AI model/agent, it's `ai_ml`.

### 3. end_user_app

**Definition:** An app, website, SaaS, game, or product for **non-developers** (consumers or businesses) — productivity, health, finance, social, e-commerce, games, content. Counts even if it happens to use AI as a feature.

**Typical signs:** "app," "iOS/Android," "website," "SaaS," "game," "platform," "tracker," "for users/customers," a landing page.

**Examples:**

- "A budgeting app for iOS and Android with the checking account built in."
- "A daily etymology puzzle game — six clues to reveal a hidden word."

**Boundary rule:** Use this when the audience is everyday people or businesses. If the project's core is an AI/ML system, use `ai_ml`; if it's fundamentally a tool for developers, use `developer_tools`.

## Data collection

**Source.** Hacker News — the recurring monthly **"Ask HN: What are you working on?"** threads. The unit of classification is a single **top-level comment** (one person's project description). I collect via the **HN Algolia API** (`search_by_date?tags=comment,story_<id>`): public, no auth, no rate-blocking. Top-level comments are isolated by filtering `parent_id == story_id`. I pulled from **many recent months** (June 2026 back through 2025) rather than a single thread, so class balance isn't skewed by whatever trended in one month. HTML is stripped, and deleted/empty/very-short comments are dropped.

**Labeling process.** Labels are **topic-based** — _what kind of thing the project is_, not why it was built. Each comment is **AI-pre-labeled** (the `label_source` column is `ai_prelabel` for all 273 rows) and then **human-reviewed** against the boundary rules in the taxonomy above. Comments that aren't software-project descriptions (job-seeker "who wants to be hired" posts, a novel, meta-chatter about the thread) are **dropped**, not forced into a label, since they have no software topic.

**Label distribution (273 comments):**

| Label             | Count | Share |
| ----------------- | ----- | ----- |
| `end_user_app`    | 129   | 47%   |
| `developer_tools` | 113   | 41%   |
| `ai_ml`           | 31    | 11%   |

`ai_ml` is the natural minority — in 2026 almost every comment _mentions_ AI, but few have AI as their _core_. I deliberately down-sampled the majority `end_user_app` and mined extra months for AI-core projects to lift `ai_ml` past 30, then handle the residual imbalance with **class weights** during training (see Fine-tuning) and **per-class F1** in evaluation rather than a single accuracy number.

**3 difficult-to-label examples and my decisions** (one per boundary):

1. **`ai_ml` vs `end_user_app`** — _Vox Labs_: "an AI-powered outbound sales platform... Claude as the primary AI layer for orchestration." → **`end_user_app`**. The AI is _infrastructure_; the product is a sales platform for businesses. Rule: label `ai_ml` only when the intelligence _is_ the deliverable, not when AI is a feature.
2. **`ai_ml` vs `developer_tools`** — _"a TUI framework... so I can use it for my local agent harness."_ → **`developer_tools`**. The shippable deliverable is a framework; the agent is just the use case. Rule: if the product is the tool (not the model), it's `developer_tools`.
3. **`developer_tools` vs `end_user_app`** — _Burst_: "opinionated backup software written in Rust... ported it to use Tokio... cross platform for Windows or macOS users." → **`developer_tools`**. Despite "backup my files" sounding consumer, the GitHub/Rust/runtime framing targets a technical audience. Rule: decide by audience, not by whether ordinary people _could_ use it.

## Fine-tuning approach

**Base model & platform.** **`distilbert-base-uncased`**, fine-tuned for 3-class sequence classification via HuggingFace `AutoModelForSequenceClassification` (labels: `ai_ml`, `developer_tools`, `end_user_app`). Trained on **Google Colab** with a **T4 GPU**.

**Training setup.** A **70 / 15 / 15 stratified split** — 191 train / 41 validation / 41 test — stratified on `label_id` so every split keeps the same class ratio, with `random_state=42` for full reproducibility. Text is tokenized with the DistilBERT tokenizer (`max_length=256`, truncation on). Training: **12 epochs**, **learning rate 2e-5**, **train batch size 8** / eval batch size 32, **weight decay 0.01**, AdamW optimizer (HuggingFace `Trainer` default).

**Hyperparameter / training decisions (with reasoning).**

- **Class weights.** Because `ai_ml` is only ~11% of the data, I pass **inverse-frequency class weights** into the loss so the rare class is penalized proportionally and isn't ignored. Without this, the model can post high overall accuracy while almost never predicting `ai_ml` — exactly the failure mode the imbalance invites.
- **12 epochs (vs. the usual 3).** With only 191 training rows, 3 epochs underfit — the model hadn't converged. I raised it to 12 and watched **validation macro-F1** to stop before overfitting; the small dataset tolerates (and needs) more passes than a typical fine-tune.
- **Model selection by macro-F1, not accuracy**, on the validation set — for the same imbalance reason as the class weights.

## Baseline

**Approach.** Before fine-tuning, I ran a **zero-shot LLM baseline** using **Groq's `llama-3.3-70b-versatile`** (temperature 0, `max_tokens=20`). Each test comment is sent with a system prompt containing the three label definitions and one example per label, and the model must return exactly one label string. No training, no in-context examples beyond the definitions — so it measures how much signal the labels carry on their own and gives the fine-tuned model something to beat.

**Prompt used** (the notebook builds a ~1.8k-character system prompt from the label definitions; representative version):

```
You are classifying a Hacker News "what are you working on" comment by the
TOPIC of the project it describes. Choose exactly one label:

- ai_ml: the project's CORE is an AI/ML system (a model, agent, LLM app,
  fine-tuning, RAG, CV/NLP). The intelligence is the product, not a feature.
- developer_tools: a library, framework, CLI, database, or infra whose main
  audience is other developers.
- end_user_app: an app, website, SaaS, or game for non-developers (consumers
  or businesses), even if it uses AI as a feature.

Rules: if AI is only a feature of a product, it's end_user_app. If a tool is
for developers, it's developer_tools. Reply with ONLY the label.

Comment:
"""
{comment_text}
"""
```

**How results were collected.** I ran the prompt over the same 41-row held-out test split (identical `random_state=42` split as the fine-tuned model, so the two are directly comparable), parsed the single-label reply (all **41/41** responses were parseable), and scored it with the same accuracy / macro-F1 / per-class metrics. Results are in the comparison table below: the baseline reaches **accuracy 0.83 / macro-F1 0.78**, edging out the fine-tuned model.

## Evaluation report

I report **accuracy + per-class precision/recall/F1** rather than accuracy alone, because the classes are imbalanced (`ai_ml` ≈ 11%) and a model can look good on accuracy while ignoring the rare class. **Macro-F1** is the headline metric since it weights all three classes equally.

> **Note:** the fine-tuned numbers are derived from the per-example test results (41-row test set: 20 `end_user_app` / 17 `developer_tools` / 4 `ai_ml`, with the 8 misclassifications enumerated in the error analysis); the baseline numbers are from the notebook's `classification_report` (Groq `llama-3.3-70b-versatile`). Because the model is non-deterministic without a fixed seed, accuracy wanders by ~1–2 examples between runs — for a final submission, re-run the whole notebook once with `set_seed(42)` so the baseline and fine-tuned numbers come from a single coherent run.

### Both models — headline metrics

| Metric               | Zero-shot Llama-70B baseline | Fine-tuned DistilBERT |
| -------------------- | ---------------------------- | --------------------- |
| Accuracy             | **0.83**                     | 0.80                  |
| Macro-F1             | **0.78**                     | 0.71                  |
| F1 `end_user_app`    | 0.86                         | 0.87                  |
| F1 `developer_tools` | 0.90                         | 0.85                  |
| F1 `ai_ml`           | 0.57                         | 0.40                  |

**Headline result: the zero-shot LLM baseline _beats_ the fine-tuned model** (macro-F1 0.78 vs. 0.71, accuracy 0.83 vs. 0.80) — driven mostly by the rare class, where Llama-70B catches all 4 `ai_ml` comments (recall 1.00, F1 0.57) while the fine-tuned model manages F1 0.40. Fine-tuning DistilBERT on only 191 examples did **not** improve over prompting a large general model cold. This is the clearest evidence for the data-scarcity conclusion in the Reflection: with so few `ai_ml` examples, the small model can't learn what a 70B model already knows.

### Fine-tuned model — per-class

| Label             | Precision | Recall | F1   | Support |
| ----------------- | --------- | ------ | ---- | ------- |
| `end_user_app`    | 0.89      | 0.85   | 0.87 | 20      |
| `developer_tools` | 0.88      | 0.82   | 0.85 | 17      |
| `ai_ml`           | 0.33      | 0.50   | 0.40 | 4       |
| **Macro avg**     | 0.70      | 0.72   | 0.71 | 41      |
| **Accuracy**      |           |        | 0.80 | 41      |

`ai_ml` is still the weak class (F1 0.40), but the failure mode has **flipped** from the earlier run: the model now gets 2 of 4 right (recall 0.50, up from 0.25) yet its precision collapses to 0.33 — it **over-predicts `ai_ml`**, sweeping in `developer_tools` comments that merely mention LLMs/agents. So the error moved from "ignores the rare class" to "too eager to fire it on AI vocabulary." `end_user_app` and `developer_tools` are both strong (F1 0.87 / 0.85).

### Confusion matrix (fine-tuned, markdown — rows = true, columns = predicted)

| True ↓ \ Pred →       | `end_user_app` | `developer_tools` | `ai_ml` | Total |
| --------------------- | -------------- | ----------------- | ------- | ----- |
| **`end_user_app`**    | 17             | 2                 | 1       | 20    |
| **`developer_tools`** | 0              | 14                | 3       | 17    |
| **`ai_ml`**           | 2              | 0                 | 2       | 4     |
| **Total**             | 19             | 16                | 6       | 41    |

The error mass concentrates on the `ai_ml` **column**: the model predicts `ai_ml` 6 times but is right only twice (precision 0.33), pulling in 3 true `developer_tools` comments that mention LLMs/agents. The opposite leak persists too — 2 of 4 true `ai_ml` comments are read as `end_user_app` (AI core hidden behind app packaging). The remaining 2 errors are `end_user_app` → `developer_tools` (dev-audience vocabulary). `ai_ml` sits on one side of **6 of the 8** errors — the AI-core boundary I pre-registered, though the dominant _direction_ shifted to `developer_tools` → `ai_ml`. (A rendered version is committed as `confusion_matrix.png`.)

## Sample classifications

Five representative test comments with the model's predicted label and confidence (a mix of correct and incorrect; full prose analysis in the next section):

| #   | Comment (excerpt)                                                | Predicted         | Conf. | True              | Correct? |
| --- | ---------------------------------------------------------------- | ----------------- | ----- | ----------------- | -------- |
| 1   | "I study German and decided to write **app** to simplify tasks…" | `end_user_app`    | 0.98  | `end_user_app`    | ✅       |
| 2   | "launched **IterOps**… heat mapping… simple A/B testing tool"    | `end_user_app`    | 0.93  | `end_user_app`    | ✅       |
| 3   | "Building **Vox Labs**, an AI-powered outbound sales platform…"  | `ai_ml`           | 0.67  | `end_user_app`    | ❌       |
| 4   | "obsessed with **game** theory… large language models…"          | `ai_ml`           | 0.50  | `developer_tools` | ❌       |
| 5   | "**MyLLM**, a private AI assistant for iPhone, runs on-device…"  | `end_user_app`    | 0.98  | `ai_ml`           | ❌       |
| 6   | "Working on **Burst**… backup software written in Rust… Tokio…"  | `developer_tools` | 0.98  | `developer_tools` | ✅       |

**One correct example explained (#2, IterOps).** The framing is product-first — "launched IterOps," "turn it into a product," "micro-saas" — which points squarely at `end_user_app` (a SaaS analytics product for businesses running websites, same family as Hotjar). The audience is site owners/marketers, not developers consuming a library. This one even had a trap the model avoided: the word "tool" plus a technical author shipping `.dev` projects could have pulled it to `developer_tools`, but the SaaS-product framing and business audience won out — the opposite of the "game"/dev-vocab traps that sink the fail cases below.

## Error analysis — sample records

All 8 misclassifications on the held-out test set, straight from the notebook (true label, predicted label, and confidence per example):

![Fine-tuned model — all 8 wrong predictions on the test set, with true vs. predicted labels and confidences](wrong_predictions.png)

The hand-written analysis below works through representative success and fail cases from this output.

**Success cases**

1. Comment: "I study German and decided to write app to simplify some tasks: 1) generate anki cards from text 2) extract highlighted words from paper and generate anki cards 3) collection of texts/dialogs on random topics 4) and so on… not sure if someone needs it, but very helpful for me."
   - **Predicted / True:** `end_user_app` / `end_user_app` ✅
   - **Why it worked:** The head noun is "app to simplify some tasks," and the features (Anki cards, study tasks) are end-user features for a learner. Crucially, none of the failure-mode traps are present — no AI vocabulary, no dev-audience words, no "game" token — so the lexical signal points cleanly one direction.
2. Comment: "This week I launched IterOps (iterops.com), a heat mapping, rage click, dead click, scroll mapping, and simple A/B testing tool… built it to understand my other micro-saas projects… turned it into a product too."
   - **Predicted / True:** `end_user_app` / `end_user_app` ✅
   - **Why it worked:** Product-first framing ("launched," "turn it into a product," "micro-saas") and a business audience override the lone "tool" token. (See full explanation above.)
3. Comment: "Working on Burst (github.com/fred1268/burst), my opinionated backup software written in Rust… ported it to use Tokio… cross platform for Windows or macOS users."
   - **Predicted / True:** `developer_tools` / `developer_tools` ✅
   - **Why it worked:** GitHub link + "written in Rust" + "ported it to use Tokio" (an implementation detail only developers care about) dominate the consumer-sounding "backup my files." The mirror of the IterOps case: audience + framing beat a single misleading token.

**Fail cases** (representative wrong predictions, with analysis — all five are in the screenshot above)

4. Comment: "YouBrokeProd — a fun simulation **game** of prod breaking… race to troubleshoot and fix the issue. Great for SREs, devs, devops, founders and teams…"
   - **Predicted / True:** `developer_tools` (0.97) / `end_user_app` ❌
   - **Why it went wrong:** It's a game (an `end_user_app`), but it's drenched in dev-audience vocabulary — prod, troubleshoot, SRE, devops. The model keyed on _who it's for_ over _what it is_. Confidently wrong at **0.97** — a clean case of dev-audience words overriding product type.
5. Comment: "Building Vox Labs, an AI-powered outbound sales platform for founders, small sales teams, and agencies… Claude as the primary AI layer for orchestration… getting an agent to handle multi-step outbound workflows…"
   - **Predicted / True:** `ai_ml` (0.67) / `end_user_app` ❌
   - **Why it went wrong:** AI-as-feature read as AI-core. Claude is the orchestration _layer_ — infrastructure — but the product is a sales platform for businesses. The model saw `AI-powered`, `Claude`, `agent`, `orchestration` and collapsed to `ai_ml`. The mid 0.67 confidence shows it was only half-sure; this is also a genuinely borderline gold label.
6. Comment: "I've been obsessed with game theory for years… got annoyed at how bad the tooling was, wrote my own… bridging computational game theory and large language models."
   - **Predicted / True:** `ai_ml` (0.50) / `developer_tools` ❌
   - **Why it went wrong:** The phrase "**large language models**" pulled it to `ai_ml`, even though the only concrete deliverable is "wrote my own **tooling**" (a `developer_tools` signal). _(Notably, an earlier run misread this same comment as `end_user_app` on the "game" token — the example is so mixed that which bias wins flips between runs. At 0.50 confidence the model is essentially guessing.)_
7. Comment: "I'm building MyLLM, a (another) private AI assistant for iPhone, it runs models on-device (llama.cpp + Metal) or against your own server…"
   - **Predicted / True:** `end_user_app` (0.98) / `ai_ml` ❌
   - **Why it went wrong:** Delivery format beat core purpose. The model saw "iPhone / assistant / app" and missed that the on-device LLM (llama.cpp + Metal) _is_ the point. Confidently wrong at **0.98** — the packaging hijacked the prediction.
8. Comment: "We've been building Genesis X-1, a layer for independently verifiable economic continuity across organizations."
   - **Predicted / True:** `ai_ml` (0.71) / `developer_tools` ❌
   - **Why it went wrong:** Pure buzzwords with no concrete anchor — no product noun, no named audience, no tech stack, not even AI vocabulary. With nothing lexical to grab, the model falls back to a guess, and the abstract systems language ("layer," "verifiable," "continuity," "across organizations") drifts toward `ai_ml`. The gold label `developer_tools` rests on "a layer… across organizations" reading as infrastructure for technical teams — but this is genuinely ambiguous even for a human, so it belongs in the inherent-ambiguity bucket rather than the clean-miss bucket.

### Recurring pitfalls

After hand-reviewing all **8 of 41** wrong predictions (≈0.80 accuracy), the errors form clear patterns rather than random noise:

1. **AI/agent vocabulary over-triggers `ai_ml` — the dominant failure (3 of 8, all `developer_tools` → `ai_ml`).** Comments that merely _mention_ LLMs or agents get pulled into `ai_ml` even when the deliverable is a tool: the game-theory tooling ("large language models"), Genesis X-1 (abstract systems layer), the tokamak TUI framework ("local agent harness"). The model now leans too eager to fire `ai_ml` on AI vocabulary — which is why `ai_ml` precision is only 0.33.
2. **Dev-audience vocabulary over-triggers `developer_tools` — and very confidently (2 of 8, both at 0.97).** Products _for_ technical people get labeled `developer_tools` even when they're apps: YouBrokeProd (SRE/devops) and BiamOS (automation engineer + GitHub link). Both are `end_user_app` products read as dev tools at 0.97 confidence — audience words override product type.
3. **Delivery format hides the AI core (`ai_ml` → `end_user_app`, 2 of 8).** True `ai_ml` projects shipped _as an app or game_ get misread as `end_user_app`: the prompt-injection game and MyLLM (iPhone app). The model classifies by how the thing is _packaged_, not what it fundamentally _is_ — the `ai_ml ↔ end_user_app` confusion I pre-registered.
4. **AI-as-feature read as AI-core (`end_user_app` → `ai_ml`, 1 of 8).** Vox Labs is a sales platform that _uses_ Claude; the model treated the infrastructure as the product.
5. **The "game" token is a _weak, unstable_ signal — not the reliable trap it first looked like.** "Game" appears in three errors and pulls a different direction each time: YouBrokeProd → `developer_tools`, game-theory → `ai_ml`, prompt-injection game → `end_user_app`. Whichever _other_ strong token is present (dev-audience, LLM, packaging) wins. (An earlier run made "game" look like a consistent `end_user_app` magnet; across runs it isn't — a useful correction.)
6. **Some "errors" are debatable gold labels, not model failures.** The genuinely ambiguous cases carry the lowest confidence (game-theory 0.50, tokamak 0.52, prompt-injection 0.60), while the systematic-bias misses are high-confidence (0.97–0.98) — so confidence is only _partially_ calibrated (see Stretch learning).

### Takeaways

- The dominant _actionable_ error is the **`ai_ml` boundary, in both directions**: the model over-fires `ai_ml` on any AI/LLM/agent mention (pulling in real `developer_tools`), yet still misses AI-core projects packaged as apps. More `developer_tools` and `end_user_app` training examples that are _loud about AI_, plus more `ai_ml` examples _delivered as apps_, would teach the model that AI vocabulary appears across all three classes and that the deliverable — not the buzzwords — decides the label.
- A chunk of the remaining error is **inherent label ambiguity**, which caps achievable accuracy — clearer annotation guidelines on hybrid projects would help more than a bigger model.
- The small test set (41 rows, only 4 `ai_ml`) means these 8 errors are illustrative, not statistically precise; the **patterns** matter more than the exact 0.80, and the `ai_ml` F1 in particular should be read as a noisy range (it swings between runs — see the calibration note).

## Reflection — what the model learned vs. what I intended

**What I intended.** A classifier that decides by what a project _fundamentally is_ — its core deliverable and audience — so that "AI-powered sales platform" lands in `end_user_app` and "private on-device LLM assistant" lands in `ai_ml`, regardless of surface wording.

**What it actually learned.** The model classifies by **surface vocabulary and packaging**, not underlying purpose. It treats high-signal tokens ("game," "agent," "AI-powered," "iPhone," "SRE") as near-deterministic votes, so it flips on whichever token is loudest even when that token contradicts the true label.

**Pre-registered hypothesis, mostly confirmed (with a twist).** Before fine-tuning I predicted the dominant confusion would be `ai_ml ↔ end_user_app` (AI-core vs. AI-feature). `ai_ml` is indeed the error hot-spot — it sits on one side of **6 of the 8** errors. But the dominant _direction_ wasn't the one I guessed: instead of AI-core projects mostly slipping to `end_user_app`, the model **over-predicts `ai_ml`**, dragging in `developer_tools` comments that mention LLMs/agents (`ai_ml` precision just 0.33). The binding constraint is still **data for the rare class** (only 22 `ai_ml` training examples): too few to teach the model when AI vocabulary _is_ vs. _isn't_ the product, so it latches onto the keywords instead.

**Stretch learning — does higher confidence mean higher accuracy? (Calibration)**

Short answer from the evidence: **not reliably.** The softmax "confidence" measures how strongly the input matches the model's learned lexical patterns — not the true probability that the prediction is correct.

- The 8 misclassifications carry an **average confidence of ≈0.74**, and — most tellingly — **3 of the 8 are at ≥0.97**: YouBrokeProd (0.97), BiamOS (0.97), and MyLLM (0.98). The model's _most confident_ predictions include three flat-out wrong ones, each a case where a biased pattern fired hard (dev-audience → `developer_tools`, "iPhone app" → `end_user_app`). It is confident **because** the pattern matched strongly, even though the pattern is wrong.
- There is a weak signal at the **low** end: the genuinely ambiguous comments do cluster at the bottom (game-theory 0.50, tokamak 0.52, prompt-injection 0.60). So low confidence loosely flags "this one is hard," but high confidence does **not** certify "this one is right."
- **Conclusion:** confidence is only **partially calibrated**. It's useful as a _triage_ signal — review the lowest-confidence predictions first — but useless as a _correctness guarantee_: a 0.98 prediction can be flat wrong. This matches the intuition that confidence reflects what the model's features say about the input, not ground truth. The systematic lexical biases ("game", AI-vocab, packaging) produce **confidently-wrong** predictions, which is exactly what a well-calibrated model wouldn't do.

_Caveat / how to verify properly:_ the numbers above use the 8 error confidences only. A full **reliability diagram** bins all 41 predictions by confidence and compares per-bin accuracy (a calibrated model sits on the diagonal — e.g. predictions made at ~0.8 confidence should be right ~80% of the time). The notebook computes this from `ft_probs`; the takeaway held: the high-confidence bins are _not_ proportionally more accurate, because the confident errors drag them down.

## Spec reflection

**One way the spec helped.** Writing explicit boundary rules up front (especially "AI-core vs. AI-feature") turned a vague worry — "everything mentions AI in 2026" — into a **testable, pre-registered hypothesis** about which cells of the confusion matrix would light up. When the results came in, I could confirm the prediction instead of discovering the failure mode by surprise, which made the error analysis sharp rather than anecdotal.

**One way the implementation diverged, and why.** The spec planned to neutralize class imbalance with **down-sampling + class weights**, implying that would be enough to make `ai_ml` competitive. In implementation it wasn't: even with class weights, `ai_ml` lands at F1 0.40 — and the zero-shot Llama-70B baseline still beats it (`ai_ml` F1 0.57) — because the real bottleneck is **absolute example count** (22 train / 4 test), not loss weighting. The divergence taught me the spec conflated two different problems — _relative_ imbalance (fixable with weights) and _absolute_ scarcity (only fixable with more data) — and that the next iteration's effort belongs in collecting more `ai_ml`-core examples, not in tuning the loss.

## AI usage

**Instance 1 — AI pre-labeling (annotation assistance, disclosed).** I used an LLM to pre-label all 273 comments against the three label definitions (recorded as `label_source = ai_prelabel`), then human-reviewed every label. **What I overrode:** borderline AI-core-vs-feature cases — e.g., the LLM tended to mark any "AI-powered" product as `ai_ml`; I corrected those to `end_user_app`/`developer_tools` when the AI was only infrastructure (Vox Labs), and dropped non-project comments (job-seeker posts, meta-chatter) the LLM had tried to force into a label.

**Instance 2 — Error-analysis tabulation.** I directed an LLM to reconstruct the confusion matrix and per-class precision/recall/F1 from my enumerated per-example test results, and to group the 8 misclassifications into recurring failure patterns. **What I revised:** I verified the reconstructed counts against the known test distribution (20/17/4) and the 8 listed errors before trusting them, and kept the framing tied to my own pre-registered hypothesis rather than accepting generic "the model struggles with ambiguity" boilerplate.

**Instance 3 — Notebook debugging.** I used an AI assistant to recover which samples were in the held-out test split and to build the correct/incorrect-prediction extraction cells (aligning predictions to `test_df` order). **What I directed/overrode:** I insisted predictions be generated in `test_df` order (no shuffled loader) so the text-to-prediction alignment was correct, and moved the extraction logic out of the data-split cell into its own post-training cell.
