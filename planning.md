# Planning: "What Are You Working On?" Project-Topic Classifier

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

## Hard edge cases

Some comments sit between labels. These are the tie-break rules I apply during annotation.

**ai_ml vs. end_user_app (the AI-pervasiveness problem)**
Almost everything mentions AI. Decide by whether the AI _is_ the product or just a feature.

- "ChatGPT for kids with parental controls" → ai_ml (the AI chat _is_ the product)
- "A recipe app that has AI suggestions" → end_user_app (AI is one feature)

**ai_ml vs. developer_tools (AI dev tools)**
An AI-centric tool aimed at developers.

- "A coding agent that writes and applies code for you" → ai_ml (the agent/AI is the whole point)
- "A testing/eval framework you point at your AI agents" → developer_tools (the deliverable is the tool, not the intelligence)

**developer_tools vs. end_user_app (who is the user?)**
Decide by audience, not by whether there's code involved.

- "An open-source color-separation engine for screen printers" → end_user_app (users are printers, not developers)
- "A self-hosted CI runner" → developer_tools (users are developers)

**Handling unclassifiable comments during annotation:**
A topic scheme only works for actual software-project descriptions. Comments that aren't projects — job-seeker resumes ("Who wants to be hired?" posts), a novel, a physics PhD, a conference talk, meta-chatter — are **dropped**, not forced into a label, since they have no software topic.

## Data collection plan

**Source:** HN Algolia API. For each "Ask HN: What are you working on?" thread, fetch all comments, filter to top-level (`parent_id == story_id`), strip HTML, and drop deleted/empty/very-short comments.

**Threads:** I pull from **many recent months** (June 2026 back through 2025), not a single month, so class balance isn't skewed by whatever trended in one month.

**Fields collected:** `comment_id, thread_month, author, created_at, text, label`. The `label` column is AI-pre-labeled and then human-reviewed (see AI Tool Plan).

**Volume:** **273 labeled comments** after dropping non-project comments — well above the 200 minimum. Distribution: `end_user_app` 129, `developer_tools` 113, `ai_ml` 31.

**Handling the underrepresented class:** `ai_ml` (projects whose _core_ is AI) is genuinely the rarest, because most "AI" mentions are apps that merely use AI. I mined additional months specifically for AI-core projects to lift it past 30, and I deliberately **down-sampled `end_user_app`** (the natural majority) so the three classes stay closer in size. `ai_ml` remains the minority, so I handle it with **class weights** during training and report **per-class F1** rather than hiding it in an accuracy average.

## Evaluation metrics

**Accuracy alone is not enough** — the classes are imbalanced (`ai_ml` ≈ 11%), and a model could post decent accuracy while ignoring the rare class. So I evaluate with:

1. **Macro-averaged F1** as the headline metric — it weights all three classes equally, so failing `ai_ml` is penalized as hard as failing the common classes.
2. **Per-class precision, recall, and F1**, to see which topics the model handles well.
3. **A confusion matrix**, to see _where_ errors go. I expect the main confusion to be `ai_ml` ↔ `end_user_app` (AI-core vs. AI-as-feature), the exact boundary I flagged above.

Evaluation is on a held-out, label-stratified test split; I report all three together. Because the test set is small (~41 rows, ~5–6 `ai_ml`), the `ai_ml` score is noisy — I read it as a range, not a precise number.

## Definition of success

This classifier would be useful to **auto-categorize and filter** the giant "what are you working on" threads (e.g. "show me only the dev-tool projects").

- **Good enough to be useful:** **macro-F1 ≥ 0.75**, with **no class below 0.60 F1**.
- **Strong:** **macro-F1 ≥ 0.85** with every class ≥ 0.70, and a confusion matrix whose remaining errors fall on the genuinely ambiguous AI-core-vs-AI-feature cases rather than random misfires.

These are objective: I read the held-out macro-F1 and per-class F1 and state plainly whether each bar was cleared.

## AI Tool Plan

This project produces a labeled dataset and an evaluation, so AI tools help in three places.

**1. Label stress-testing (before annotating).**
Give an AI the three definitions and ask it to generate boundary cases — e.g. an app that heavily uses AI but isn't AI-core, a dev tool that's also an AI agent. If I can't classify its output cleanly with my own rules, the definitions are too loose and I tighten them first.

**2. Annotation assistance.**
I use an LLM to **pre-label** comments, then review and correct each one myself. Every row carries a `label_source` column (`ai_prelabel`) for disclosure; I remain the final annotator.

**3. Failure analysis.**
After evaluation I hand the AI my list of wrong predictions (text, true label, predicted label) and ask it to find patterns (e.g. "AI-feature apps systematically read as ai_ml"). I treat its output as hypotheses and verify each one myself against the actual misclassified examples and the confusion matrix before writing it up.
