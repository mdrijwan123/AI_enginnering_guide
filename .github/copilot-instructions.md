# FAANG LLM/AI Engineer Study Plan — Workspace Instructions

This workspace is a **personal self-study plan** (not a code project).  
Goal: MLOps Engineer → FAANG LLM/AI Engineer | April 2026 → April 2027  
Current date context: always check `README.md` for the current week/phase.

---

## Workspace Structure

```
Phase_1_Foundation_Apr_Jun_year_01/   ← Apr–Jun 2026 (active)
Phase_2_Advanced_Systems_Jul_Sep_year_01/
Phase_3_Deep_Specialization_Oct_Dec_year_01/
Phase_4_Interview_Ready_Jan_Mar_2027_year_02/
README.md                             ← current week, overview
PROGRESS_TRACKER.md                   ← daily status, mark completions here
AI_INTERVIEW_MASTER_CHECKLIST.md      ← full topic checklist for interview readiness
```

Each folder follows the pattern: `Month_NN_MonthName/Week_N_Topic/study_guide.md`  
Supplementary deep-dives (e.g. `Vector_Databases_Deep_Dive.md`) live alongside the monthly `study_guide.md`.

---

## Study Guide Format

All `study_guide.md` files follow this structure — preserve it when editing or extending:

1. **Header** — Phase / Month / Week / dates, daily schedule
2. **Learning Objectives** — bullet list of concrete outcomes
3. **Theory sections** — explanation, code snippets, tables (time/space complexity), interview traps
4. **LeetCode problems** — problem name, pattern, Python solution, complexity analysis, follow-up questions
5. **Interview Q&A** — numbered Q&A pairs mimicking real FAANG verbal rounds
6. **Summary / cheat-sheet** — at the end

Key style rules:
- Python code blocks always include time and space complexity comments
- Complexity tables use `| Operation | Time | Notes |` format
- Interview traps are highlighted with `> **Interview trap:**`
- Emojis are used sparingly for headers only (🎯, 📊, etc.)

---

## Content Conventions

- **Language**: Python 3.10+ for all code examples
- **DSA problems**: always include brute-force → optimised solution progression
- **AI/LLM content**: link to papers/docs when possible rather than paraphrasing
- **Progress tracking**: update `PROGRESS_TRACKER.md` statuses (⏳ → 🔄 → ✅) — never auto-complete items the user hasn't confirmed
- **No hallucinated LeetCode solutions**: if unsure of a problem constraint, note it explicitly

---

## How to Help

When asked to **expand a study guide**, follow the existing section structure and maintain the difficulty ramp (easy → medium → hard).

When asked to **add LeetCode problems**, include: problem name, LeetCode number, pattern tag, Python solution with comments, O(n)/O(1) analysis, and one follow-up question.

When asked to **generate interview Q&As**, write questions as a FAANG interviewer would ask them: concise, open-ended, with expected depth in the answer.

When asked to **review a solution**, check: correctness, edge cases, time/space complexity, Pythonic style, and whether a better pattern exists.

When asked to **update progress**, edit `PROGRESS_TRACKER.md` only — never modify study guides to mark things done.

---

## Key Reference Files

- Current week / overview → [README.md](../README.md)
- Daily progress → [PROGRESS_TRACKER.md](../PROGRESS_TRACKER.md)
- Full interview topic map → [AI_INTERVIEW_MASTER_CHECKLIST.md](../AI_INTERVIEW_MASTER_CHECKLIST.md)
- Behavioral prep → [Phase_4_Interview_Ready_Jan_Mar_2027_year_02/BEHAVIORAL_STAR_GUIDE.md](../Phase_4_Interview_Ready_Jan_Mar_2027_year_02/BEHAVIORAL_STAR_GUIDE.md)
