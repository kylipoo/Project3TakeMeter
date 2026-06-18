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

**Base model.** `[FILL: e.g. bert-base-uncased / distilbert-base-uncased]`, fine-tuned for 3-class sequence classification via HuggingFace `AutoModelForSequenceClassification` (labels: `ai_ml`, `developer_tools`, `end_user_app`).

**Training setup.** A **70 / 15 / 15 stratified split** — 191 train / 41 validation / 41 test — stratified on `label_id` so every split keeps the same class ratio, with `random_state=42` for full reproducibility. Text is tokenized with the model's tokenizer (`max_length=[FILL]`, truncation on). `[FILL: epochs, batch size, learning rate, optimizer — pull from your TrainingArguments/Trainer cell]`.

**Hyperparameter / training decision (≥1).** **Class weights.** Because `ai_ml` is only ~11% of the data, I pass **inverse-frequency class weights** into the loss, so the rare class is penalized proportionally and isn't ignored. Without this, the model can post high overall accuracy while almost never predicting `ai_ml` — exactly the failure mode the imbalance invites. _(Optional second decision: select the checkpoint by **macro-F1 on the validation set**, not accuracy, for the same imbalance reason.)_

> _To finalize: paste your `TrainingArguments` / `Trainer` (or training loop) cell and I'll replace every `[FILL]` with the real values._

## Reflection

1. Reflect briefly: where did the baseline struggle? Are there specific labels it consistently confuses? Write down your hypothesis — you'll test it after fine-tuning.
   1. Baseline seems to especially struggle at the commercial_venture and open_source_devtool. The main confusion is probably that the model treats `commercial_venture` as a catch-all default: it had high recall (0.79) but low precision (0.61), meaning it over-predicts CV and sweeps borderline comments into it. The mirror image is `open_source_devtool`, which had perfect precision (1.00) but only 0.50 recall — when the model commits to OSD it's right, but it misses half of them, most likely re-labeling those dev tools (and some `side_project_hobby` comments) as `commercial_venture`. This lines up with my Option B label definition, which deliberately made CV the default for "clearly a product, but no stated personal/fun/learning reason" — so a zero-shot model with no training naturally collapses ambiguous, productized-looking comments into CV.

      **Hypothesis:** the dominant error direction is `open_source_devtool` → `commercial_venture` and `side_project_hobby` → `commercial_venture` (everything leaks toward the majority class). `research_or_career` looked strong (F1 0.80) but rests on only 5 test examples, so I don't trust that number yet.

      **What I'll test after fine-tuning:** whether training on labeled examples tightens the CV boundary — I expect CV precision to rise and OSD recall to rise, with the confusion matrix showing fewer OSD/SPH cells bleeding into the CV column. I'll also watch whether the tiny `research_or_career` support makes its score swing, since the 35-row test set means any single misclassification moves a per-class F1 a lot.

2. Results of my first attempt:
   1. With ~150 labeled examples, a fine-tuned BERT did not beat a zero-shot LLM baseline — more data is needed
   2. ![alt text](<Screenshot 2026-06-18 at 2.00.27 PM.jpg>)

## Pitfalls — error analysis of the fine-tuned model

After switching to the 3 topic labels (`ai_ml`, `developer_tools`, `end_user_app`) and retraining with class weights, the fine-tuned model got **9 / 41 test examples wrong (≈0.78 accuracy)**. Reviewing each wrong prediction by hand surfaced clear, recurring failure patterns rather than random noise.

## Sample records:

1. Comment body: "I study German and decided to write app to simplify some tasks: 1) generate anki cards from text 2) extract highlighted words from paper and generate anki cards 3) collection of texts/dialogs on random topics 4) and so on... ) not sure if someone needs it, but very helpful for me )"
   1. Prediction: End user app
   2. True: End user app
   3. Why it was successful: Mentions "decided to write app to simplify some tasks.". In addition, the description of this accomplishment sounds like part of the end user experience, with the user being expected to take the iniative by entering text.
2. Comment body: "This week I launched IterOps https://iterops.com, a heat mapping, rage click, dead click, scroll mapping, and simple A/B testing tool. I originally built it to have a better idea of what people are doing on my other micro-saas projects like https://securitybot.dev and https://contributoriq.com. Already finding it so useful that I figured I'd just turn it into a product too."
   1. Prediction: End user app
   2. True: End user app
   3. Why it was successful: The framing is product-first — "launched IterOps", "turn it into a product", "micro-saas" — which points squarely at end_user_app (a SaaS product for businesses running websites, in the same family as Hotjar). The audience is site owners / marketers analyzing visitor behavior (heat maps, rage clicks, A/B testing), not developers consuming a library. This one had a real trap the model avoided: the word "tool" plus a technical author shipping ".dev" projects could have pulled it toward developer_tools, but the model correctly weighed the SaaS-product framing and business audience over the lone "tool" token — the opposite of the "game"/dev-vocab traps that sink the fail cases.

3. Comment body: "Working on Burst (https://github.com/fred1268/burst) my opinionated backup software written in Rust, non only to weekly backup my files but also to learn Rust. I recently ported it to use Tokio. Ah, and it's cross platform for the Windows or macOS users."
   1. Prediction: Developer tools
   2. True: Developer tools
   3. Why it was successful: The developer_tools signals dominate — a GitHub repo link, "written in Rust", and "ported it to use Tokio" (an implementation detail only other developers care about). It reads as an open-source project shown to a technical audience. There was a real trap here too: "backup my files... for the Windows or macOS users" could have read as a consumer backup app (end_user_app), but the model correctly weighed the open-source / language / runtime framing over the consumer-sounding phrase — the mirror of the IterOps case, where audience+framing beat a single misleading token.

4. (Fail case) Comment body: "Building Vox Labs, an AI-powered outbound sales platform for founders, small sales teams, and agencies. The idea is you work with it the way you'd direct an SDR: give it guidance and it handles research, enrichment, personalization, and sequencing. Stack is React frontend, Node.js backend, Claude as the primary AI layer for orchestration. The most interesting engineering problem has been the orchestration layer. Getting an agent to handle multi-step outbound workflows with real judgment, not just automation, takes a lot of iteration on the prompting and state management side. Also feedback from users on the best workflows to collaborate with agents."
   1. Prediction: AI_ML (with a confidence of 0.58)
   2. True: End user app.
   3. Possibilities: The agent missed the bigger picture. Yes it's true that Claude is an orchestration layer and that Vox Labs is AI powered, but the key detail is that is only infrastructure, Vox is its own platform that just happens to utilize AI.
5. (Fail case) Comment body: "I've been obsessed with game theory for years. Started by reimplementing papers, got annoyed at how bad the tooling was, wrote my own, and somewhere along the way realized the field's core assumptions about competition are wrong. Helps that many of my friends are from the competitive gaming scene -- multi-R1 players, MLG winners, etc. So I get to have a view on strategy from both the FANG MLE/SWE side but also the actually competing at the highest levels side. Decided to found a company with some of those friends, bridging computational game theory and large language models :) https://mieza.ai/"
   1. Prediction: End user app
   2. True: Developer tools.
   3. Possibilities: Why the model said end_user_app instead: the "game" lexical trap, and it's dense here — "game theory," "competitive gaming scene," "MLG winners." Four game/gaming tokens. The model reads "game" → consumer/games → end_user_app, ignoring that "game theory" is a math/research field, not a video game. That 0.83 confidence is the trap firing confidently.

6.

### Recurring pitfalls

1. **The word "game" is a lexical trap.** It appears in 3 of 7 errors (#1, #2, #5) and pulls the prediction in whatever direction "game" suggests — even when the true label is `developer_tools` (#2) or `ai_ml` (#5). The model over-trusts a single high-signal token.

2. **Delivery format beats core purpose — this confirms my pre-registered hypothesis.** AI-core projects shipped _as an app or game_ (#5, #7) get misread as `end_user_app`. The model classifies by how the thing is _packaged_ ("iPhone app", "a game you can play") instead of what it fundamentally _is_ (an LLM). This is exactly the `ai_ml ↔ end_user_app` confusion I predicted — the AI-core-vs-AI-feature boundary.

3. **Dev-audience vocabulary pulls toward `developer_tools`.** Products _for_ or _by_ technical people — SREs (#1), automation engineers (#3) — get labeled `developer_tools` even when they're games or workspace apps. Audience words override product type.

4. **Some "errors" are debatable gold labels, not model failures.** #2, #3, #4, and #6 are genuinely ambiguous even for a human annotator, and the model's prediction is defensible. Notably the two most ambiguous cases carry the **lowest confidence** (#6 = 0.55, #4 = 0.74), while clear misses are high-confidence — so confidence is partially calibrated, except where the "game" trap makes it confidently wrong (#5 = 0.98, #1 = 0.99).

### Takeaways

- The dominant _actionable_ error is the **AI-core-vs-AI-feature** boundary (`ai_ml` → `end_user_app`). More `ai_ml` training examples that are _delivered as apps_ (assistants, AI games, AI iPhone apps) would teach the model to look past the packaging.
- A chunk of the remaining error is **inherent label ambiguity**, which caps achievable accuracy — clearer annotation guidelines on hybrid projects would help more than a bigger model.
- The small test set (41 rows) means these 7 errors are illustrative, not statistically precise; the patterns matter more than the exact 0.83.
