# Dataset plan

Community: Hacker News / Ask HN

Thread: Ask HN: What are you working on? (June 2026)
Source page: https://news.ycombinator.com/item?id=48528779
Reason: public, no login, lots of comments, and the community has clear norms around building, technical projects, feedback, and practical discussion.

Recommended label set for this thread:

1. project_showcase
   - The comment describes the author’s own project, product, research, or build.

2. feedback_or_question
   - The comment asks a question, gives advice, offers help, or responds to someone else’s project.

3. meta_discussion
   - The comment is about the HN thread/community, process, launch strategy, business model, or discussion norms rather than directly showing a project.

Hard-edge rule:
- If a comment both describes a project and asks for feedback, label it project_showcase if the main purpose is presenting the project.
- If a comment is mostly a reply to someone else, label it feedback_or_question.
- If it discusses HN, launch strategy, monetization in general, or the thread itself, label it meta_discussion.

Collection command:
python3 collect_hn_200.py

The script writes:
hn_ask_working_on_june_2026_200.csv
