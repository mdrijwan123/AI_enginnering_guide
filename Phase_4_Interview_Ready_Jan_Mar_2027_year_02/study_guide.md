# Phase 4: Interview Ready — Jan–Mar 2027
### Final Preparation for FAANG LLM/AI Engineer Roles

---

## Overview

By January 2027, you've spent 9 months building deep knowledge. This phase is about **converting knowledge into offers**. The focus shifts from learning to performing under pressure.

| Month | Focus | Key Activities |
|---|---|---|
| January (Month 10) | Company-specific DSA + Behavioral prep | LeetCode grind, STAR stories, mock coding |
| February (Month 11) | Full mock interview loops | Coding + System Design + Behavioral loops |
| March (Month 12) | Applications + Live interviews | Apply, negotiate, receive offer |

---

# MONTH 10 (January 2027): DSA Company-Specific + Behavioral

> 📖 **What changes in Phase 4:** For 9 months you’ve been building knowledge. Now the focus shifts from *learning* to *performing*. The difference is real: you can know the solution to a problem but fail to communicate it well under pressure, or know the architecture but freeze when an interviewer asks "what would you monitor?"
>
> **Phase 4 is about deliberate practice under test conditions:** Company-specific problem sets (different companies favour different patterns), STAR story preparation, mock loops, and application strategy. Every hour in this phase should simulate an actual interview.

## Week 1–2: LeetCode by Company

### Google LeetCode Patterns

Google values: **correctness over speed**, clean code, edge cases, time/space complexity analysis.

**Top Google interview patterns:**
- Arrays/Strings with multiple pointers
- Dynamic programming (especially on sequences)
- BFS/DFS on matrices and graphs
- Design questions (LRU Cache, Data Stream)

**Google top problems:**

| # | Problem | Pattern | Difficulty |
|---|---|---|---|
| 1 | Two Sum | HashTable | Easy |
| 15 | 3Sum | Two Pointer | Med |
| 42 | Trapping Rain Water | Two Pointer / Stack | Hard |
| 56 | Merge Intervals | Sorting | Med |
| 57 | Insert Interval | Intervals | Med |
| 128 | Longest Consecutive Sequence | HashSet | Med |
| 200 | Number of Islands | BFS/DFS | Med |
| 207 | Course Schedule | Topo Sort | Med |
| 238 | Product of Array Except Self | Prefix | Med |
| 295 | Find Median from Stream | Two Heaps | Hard |
| 297 | Serialize/Deserialize BTree | BFS | Hard |
| 322 | Coin Change | DP | Med |
| 347 | Top K Frequent Elements | Heap/Bucket | Med |
| 394 | Decode String | Stack/Recursion | Med |
| 416 | Partition Equal Subset Sum | DP | Med |
| 424 | Longest Repeating Char Replacement | Sliding Window | Med |
| 647 | Palindromic Substrings | DP/Expand | Med |
| 743 | Network Delay Time | Dijkstra | Med |
| 778 | Swim in Rising Water | Binary Search + BFS | Hard |
| 1235 | Max Profit in Job Scheduling | DP + Binary Search | Hard |

**Be able to discuss:** "What if the input doesn't fit in memory?" (external sort, streaming)

---

### Meta (Facebook) LeetCode Patterns

Meta values: **practical, optimised solutions**, attention to scale, clean APIs.

**Meta top patterns:**
- Arrays / Two Pointers — very common!
- Trees (especially binary trees, without recursion)
- Graphs (BFS for shortest path)
- String manipulation
- Dynamic programming (simpler than Google's)

**Meta top problems:**

| # | Problem | Pattern | Notes |
|---|---|---|---|
| 1 | Two Sum | HashMap | Classic warm-up |
| 2 | Add Two Numbers | Linked List | |
| 21 | Merge Two Sorted Lists | Linked List | |
| 88 | Merge Sorted Array | Two Pointer | In-place |
| 121 | Best Time to Buy/Sell Stock | Greedy | |
| 125 | Valid Palindrome | Two Pointer | |
| 146 | LRU Cache | HashMap + DLL | Very common at Meta |
| 200 | Number of Islands | BFS | |
| 236 | LCA of Binary Tree | DFS | |
| 252 | Meeting Rooms | Sorting | |
| 253 | Meeting Rooms II | Heap | |
| 270 | Closest BST Value | BST | |
| 301 | Remove Invalid Parentheses | BFS | Hard |
| 314 | Binary Tree Vertical Order | BFS + HashMap | |
| 339 | Nested List Weight Sum | BFS/DFS | |
| 398 | Random Pick Index | Reservoir Sampling | |
| 543 | Diameter of Binary Tree | DFS | |
| 560 | Subarray Sum Equals K | Prefix Sum | |
| 739 | Daily Temperatures | Monotonic Stack | |
| 1650 | LCA of BT III (parent pointer) | HashSet | |

---

### Amazon LeetCode Patterns

Amazon values: **Leadership Principles expressed through code**, efficient solutions, scalability thinking.

**Amazon top patterns:**
- Arrays, Two Sum variants
- Trees
- Dynamic programming
- Design questions (very common at Amazon)
- Graphs (connectivity problems)

**Amazon top problems:**

| # | Problem | Pattern | Notes |
|---|---|---|---|
| 1 | Two Sum | HashMap | |
| 3 | Longest Substring Without Repeating | Sliding Window | |
| 15 | 3Sum | Two Pointer | |
| 23 | Merge K Sorted Lists | Heap | |
| 41 | First Missing Positive | Array Trick | Hard |
| 49 | Group Anagrams | Sorting/HashMap | |
| 76 | Minimum Window Substring | Sliding Window | Hard |
| 127 | Word Ladder | BFS | |
| 139 | Word Break | DP | |
| 146 | LRU Cache | Design | |
| 200 | Number of Islands | BFS/DFS | |
| 207 | Course Schedule | Topo Sort | |
| 347 | Top K Frequent Elements | Heap | |
| 438 | Find All Anagrams | Sliding Window | |
| 763 | Partition Labels | Greedy | |
| 973 | K Closest Points to Origin | Heap | |
| 1041 | Robot Bounded In Circle | Simulation | |
| 1152 | Analyze User Website Visit | HashMap/Sort | |
| 2131 | Longest Palindrome by Concat | HashMap | |

---

## Week 3: Behavioral Interview — STAR Stories

> 📖 **Why behavioral prep matters as much as technical:** At FAANG L5+, everyone who makes it to interviews can code. Behavioral rounds differentiate candidates by leadership, ownership, and impact. A weak behavioral performance with a perfect technical round can still lead to rejection.
>
> **STAR stories aren’t memorised scripts — they’re structured recall.** The goal is to have 8-10 clear stories from your career that can be adapted to different questions. The same story about a production incident might answer "tell me about a time you failed", "tell me about working under pressure", and "tell me about a time you made a technical decision others disagreed with."

### The STAR Framework

```
S = Situation: Context, team, project, timeframe
T = Task: What you were responsible for, the challenge
A = Action: Specifically what YOU did (use "I", not "we")
R = Result: Quantified outcome, what you learned
```

**Time allocation:** Situation (15%) → Task (10%) → Action (55%) → Result (20%)

### 8 Pre-Built STAR Stories for MLOps → AI Engineer Transition

---

**Story 1: Delivered High-Impact System (Shows Ownership)**

*Question:* "Tell me about a time you had significant ownership of a project."

```
S: At Dunnhumby, our ML model serving infrastructure had no monitoring.
   Models deployed to prod with zero visibility into prediction quality.
   (Jan 2025)

T: I identified this gap and took ownership of building a complete
   observability stack for 12 production ML models.

A: - Designed and implemented LangSmith-inspired tracing for all model requests
   - Built custom Prometheus metrics (prediction latency, drift, error rate)
   - Created Grafana dashboards for each model's health
   - Set up automated alerts (PagerDuty) for performance degradation
   - Ran a 2-week shadow period to validate metrics before going live
   - Presented findings to ML teams, trained 15 engineers on the system

R: - P99 latency alerts caught 3 model degradations before they reached users
   - Mean time to detect model issues: 4 hours → 12 minutes
   - Became the standard for all new model deployments at Dunnhumby
   - Presented at an internal tech talk, adopted by 3 other teams
```

---

**Story 2: Handled Technical Disagreement (Shows Leadership)**

*Question:* "Tell me about a time you disagreed with a colleague."

```
S: In a sprint planning session, a senior engineer proposed migrating our
   feature store from on-prem Redis to a new internal solution that the
   infrastructure team was building. The migration would take 3 months
   and leave us without real-time features for 6 models.

T: I believed the migration timeline was too optimistic and the risk to
   production models was too high. I needed to make this case constructively.

A: - Collected data: queried 3 months of on-call records, showing Redis
     had 99.97% uptime vs. the new solution's 99.1% in testing
   - Built a cost/risk framework: outlined 3 migration scenarios (big-bang,
     side-by-side, phased) with timeline, risk, and rollback complexity
   - Presented this in the next design review with the data backing
   - Proposed a phased approach: migrate lowest-stakes model first,
     monitor for 4 weeks, then scale
   - The senior engineer agreed after seeing the data

R: - We adopted the phased migration
   - First model migrated without incident; 2 minor bugs found and fixed
     before migrating high-stakes models
   - The migration took 5 months total (not 3 as originally planned)
   - No production incidents during migration
```

---

**Story 3: Learned New Technology Fast (Shows Adaptability)**

*Question:* "Tell me about a time you had to learn something quickly."

```
S: Our team was asked to deliver a production RAG system for our retail
   analytics knowledge base in 6 weeks. Neither I nor my teammates had
   production RAG experience — only academic knowledge.

T: I led the technical design and implementation, which meant rapidly
   upskilling in: LangChain, vector databases, RAGAS evaluation, and
   LangSmith observability.

A: - Spent first week doing intensive learning: read LangChain docs,
     built 3 prototype RAG pipelines, studied RAGAS metrics
   - Set up evaluation first (golden Q&A set of 100 questions) before
     building the pipeline — this became the decision framework
   - Ran 4 chunking strategies and 3 embedding models in parallel,
     measured all with RAGAS: selected best combo (semantic chunking +
     bge-large-en)
   - Applied LangSmith tracing from day 1 — had full visibility into
     retrieval quality throughout development
   - Delivered functioning system in 5 weeks with documented evaluation

R: - Production RAG system serving 200 analysts daily since July 2025
   - RAGAS faithfulness: 0.87, AnswerRelevancy: 0.83
   - Analyst satisfaction: 78% → 91% NPS vs. keyword search
   - I documented the whole process as an internal guide, now the
     standard RAG implementation template at Dunnhumby
```

---

**Story 4: Handled Ambiguity (Shows Judgment)**

*Question:* "Tell me about a time you worked in an ambiguous situation."

```
S: Product team asked: "Can we use AI to reduce customer support volume?"
   No specifications, no constraints, no timeline. I was the only ML
   person assigned.

T: Needed to scope and define the project from scratch, then deliver
   something valuable without getting stuck in analysis paralysis.

A: - Week 1: Talked to 8 customer support agents to understand top 20
     question categories — found 40% of queries were about order status
     (fully answerable from structured data)
   - Defined: "Build a phase 1 system for order-status questions with
     structured data; evaluate before expanding to open-ended queries"
   - Built an LLM-powered chatbot (GPT-4o-mini + SQL generation for
     order data) in 3 weeks, got it in front of 50 beta users
   - Presented data to stakeholders: 73% of beta users got correct answers
     without agent escalation
   - Used beta results to scope Phase 2 (broader Q&A with RAG)

R: - Phase 1 deployed, handling 35% of order-status queries without agent
   - Agent time freed: 2 FTE-equivalent hours/day
   - Stakeholder alignment achieved: clear Phase 2 plan approved with budget
```

---

**Story 5: Failed and Recovered (Shows Self-Awareness)**

*Question:* "Tell me about a failure."

```
S: I deployed a retrained sales forecasting model that I had only validated
   on the training distribution. It went to production on the week before
   Black Friday 2024 — the highest-traffic week of the year.

T: The model was responsible for inventory recommendations for 500+ products
   across 3 retailers.

A: - Model performed well on normal weeks but Black Friday has a completely
     different demand distribution
   - 4 hours after deployment, retailers reported inventory mismatches
   - I immediately rolled back to the previous model version (had defined
     rollback procedure, which paid off)
   - Root cause: I hadn't included any Black Friday historical data in the
     validation set and hadn't explicitly reviewed data distribution
   - Implemented: timestamp-aware validation split, seasonality checks in
     CI pipeline, distribution shift monitoring in production

R: - Rollback completed in 8 minutes (pre-planned runbook)
   - No inventory SLA breaches
   - Post-mortem shared with team; new validation checklist adopted
   - The model was retrained with proper temporal validation, deployed
     successfully for Christmas 2024 with +5% forecast accuracy
```

---

**Story 6: Improved a Process (Shows Innovation)**

*Question:* "Give me an example of when you improved a process or system."

```
S: Our ML team was manually managing model versions in a shared Google Sheet.
   Every deployment required slack pings and manual log updates.
   We had 12 models in production and it was chaos.

T: Build a lightweight model registry / deployment tracking that the team
   would actually use.

A: - Evaluated MLflow, W&B, and a custom solution
   - Chose MLflow (free, self-hosted, minimal ops overhead)
   - Implemented in 2 weeks: logged all model runs with params, metrics,
     artifacts; defined staging → production promotion workflow
   - Wrote a deployment runbook & trained 8 engineers
   - Added automated comparison: "new model must beat prod model on
     held-out evaluation set before promotion"

R: - Time to find "what's in production?" reduced: 30 minutes → 30 seconds
   - 3 regressions caught before production deployment (automated comparison)
   - Team adopted it for all 12 models within 1 month
   - Zero confusion about model versions in the 8 months since
```

---

**Story 7: Cross-Functional Collaboration (Shows Teamwork)**

*Question:* "Tell me about working with non-technical stakeholders."

```
S: Data science team built a churn prediction model but the business team
   (Commercial Managers) didn't trust or use its output.

T: Bridge the gap between technical model output and business decision making.

A: - Sat with 3 Commercial Managers for 2 hours to understand their mental
     model of customer risk ("I just know from experience")
   - Reframed output: not "0.73 churn probability" but "High Risk (likely
     to lapse in 30 days)" with 3 categories (High/Medium/Low)
   - Built a Tableau dashboard surfacing top 50 at-risk customers per manager
   - Added: "Why is this customer at risk?" (SHAP feature importance in
     plain English: "No purchases in 45 days, down from 12/year")
   - Ran 2 training sessions: 90 minutes each with Q&A

R: - Dashboard active usage: 0% → 78% of targeted managers weekly
   - 3 managers ran targeted outreach campaigns using the predictions
   - Campaign conversion: 23% (vs. 7% baseline for non-targeted)
   - Model is now referenced in quarterly business reviews
```

---

**Story 8: Technical Leadership (Shows Seniority)**

*Question:* "Tell me about a time you led a technical initiative."

```
S: Dunnhumby had no standard for how ML models should be deployed. Each
   squad had their own approach: some used Docker, some Lambda, some bare
   VMs. This created maintenance burden, inconsistent monitoring, and
   security gaps.

T: Define a standard MLflow → Cloud Run deployment pattern that all squads
   could adopt. Lead adoption across 5 squads.

A: - Researched: evaluated GKE, Cloud Run, Vertex AI endpoints for fit
   - Designed: MLflow → Cloud Build → Docker → Cloud Run pipeline
   - Built reference implementation with: auto-scaling, Prometheus metrics,
     health checks, rollback procedure, and security (IAM, no hardcoded keys)
   - Reviewed with platform, security, and ML teams for feedback
   - Ran 2 workshops to walk squads through the pattern
   - Created a Cookiecutter template — new project scaffold takes 10 minutes
   - Supported 3 squads in their first migration (pair programming sessions)

R: - 4 of 5 squads now use the standard pattern (1 grandfathered in)
   - Deployment time: 3-4 days → 4 hours for a new model
   - Security team: zero findings on reviewed deployments (vs. 3 findings
     previously per review)
   - Pattern documented as engineering standard in company wiki
```

---

## Week 4: Resume, Portfolio, and Applications

### Resume Checklist for FAANG AI Engineer

**Contact & Header:**
- [ ] Professional email
- [ ] LinkedIn URL
- [ ] GitHub URL with projects visible
- [ ] No photo (FAANG blind review)

**Summary (2-3 lines):**
```
AI/ML Engineer with X years optimising and deploying production ML systems.
Specialised in LLM application development: RAG, fine-tuning, and inference.
Delivered [specific measurable impact] at [company]. Seeking AI Engineer role.
```

**Technical Skills (scannable section):**
```
LLMs/GenAI: LangChain, LangGraph, RAG, Fine-tuning (LoRA/QLoRA), vLLM, RLHF
Frameworks: PyTorch, HuggingFace Transformers, PEFT, TRL, OpenAI API, Anthropic
Infrastructure: GCP (Vertex AI, Cloud Run, GKE), Docker, Kubernetes, Terraform
Observability: LangSmith, Prometheus, Grafana, W&B, Arize Phoenix
DSA/Coding: Python (expert), SQL (expert), LeetCode Medium+
```

**Experience bullets — must be:**
- Action + Metric + Impact
- "Built X using Y that achieved Z" format
- Examples:
  - ❌ "Worked on RAG system"
  - ✅ "Designed and implemented production RAG pipeline (LangChain + Pinecone + GPT-4o) serving 200 daily users, achieving 0.87 RAGAS faithfulness and reducing analyst resolution time by 45%"

### 5 Portfolio Projects to Push to GitHub

```
Project 1: Production RAG System
  Repo: github.com/yourname/production-rag
  Stack: LangChain + Pinecone + GPT-4o-mini + FastAPI + LangSmith
  README: architecture diagram, RAGAS scores, how to run
  Bonus: add Streamlit demo UI

Project 2: ReAct Agent with Tools
  Repo: github.com/yourname/react-agent
  Stack: LangGraph + Tools (search, code interpreter, SQL) + FastAPI
  Demo: video of agent completing multi-step research task

Project 3: QLoRA Fine-tuning Pipeline
  Repo: github.com/yourname/qlora-finetuning
  Content: notebook showing end-to-end QLoRA on LLaMA 3-8B
  Evaluation: before/after comparison on domain benchmark

Project 4: MCP Server
  Repo: github.com/yourname/mcp-server
  Stack: Python MCP SDK + BigQuery/GCS tools + Resources
  Demo: Claude Desktop using your server to query data

Project 5: LLM Evaluation Dashboard
  Repo: github.com/yourname/llm-eval-dashboard
  Stack: RAGAS + LangSmith + Streamlit/Gradio
  Shows: you think about quality, not just building
```

---

# MONTH 11 (February 2027): Full Mock Interview Loops

> 📖 **The most important month:** Nothing builds interview muscle like full mock loops under time pressure. By February, your knowledge is complete. The bottleneck is now execution: explaining your thinking while coding, structuring system design answers, keeping STAR answers under 3 minutes.
>
> **The rule of 10:** You need at least 10 full mock loops before your first real interview. Each one should include all three rounds (coding + system design + behavioral), followed by honest self-assessment. If you’re not failing some mocks, the mocks are too easy.

## Mock Interview Schedule

**Each week: 2 full loops (Coding + System Design + Behavioral)**

### Coding Round Framework (45 minutes)

```
Minutes 0-2:   Read problem. Clarify edge cases (blank input? overflow? sorted?)
Minutes 2-5:   Think aloud — state approach, time/space complexity
Minutes 5-35:  Code. Say what you're doing. Don't code silently.
Minutes 35-45: Test with examples. Consider edge cases. Optimise if time.
```

**Things to say during coding:**
- "I'll start with a brute force — O(n²) — then optimise"
- "This feels like a two-pointer problem because..."
- "I'm using a hash map here for O(1) lookup"
- "Let me trace through with this example: [input] → [expected]"
- "Edge cases I should handle: empty input, single element, all duplicates"

### System Design Round Framework (45 minutes)

```
Minutes 0-3:   Clarify requirements (scale, latency, consistency)
Minutes 3-8:   High-level design (boxes and arrows)
Minutes 8-20:  Deep dive on critical components (DB choice, model, serving)
Minutes 20-30: Scale and fault tolerance
Minutes 30-40: Monitoring and evaluation
Minutes 40-45: Discuss trade-offs and alternatives
```

**Say out loud:**
- "I'll start with requirements: functional and non-functional"
- "At [scale], I'd need [X] because..."
- "The trade-off here is [consistency vs. availability]. For this use case, I'd choose..."
- "If I had more time, I'd also address..."

### Resources for Mock Interviews

- **Interviewing.io** (paid, real FAANG interviewers): https://interviewing.io
- **Pramp** (free peer mock): https://www.pramp.com
- **Exponent** (ML system design mocks): https://www.tryexponent.com
- **LeetCode**: company-specific premium problem sets
- **Glassdoor**: recent interview questions by company + role

---

# MONTH 12 (March 2027): Applications, Live Interviews, Offer

## Application Strategy

### Target Companies — Tier 1 (apply first)
```
Google / DeepMind     — prioritise if interested in research-oriented work
Meta AI               — strong on LLM infrastructure, Llama team
OpenAI                — selective, strong AI engineering culture
Anthropic             — values alignment and safety depth
Amazon Alexa AI / AWS — strong on LLM services (Bedrock)
```

### Target Companies — Tier 2 (core big tech)
```
Microsoft / GitHub Copilot — strong AI integration roles
Apple (Siri, MLR)          — competitive pay, privacy-focused
Databricks                 — LLM+data engineering crossover
Snowflake                  — ML platform roles
Stripe / Cloudflare        — ML Engineering in financial/infra tech
```

### Application Tips
- Apply early Monday morning (first batch reviewed)
- Referrals: 40% higher callback rate — use LinkedIn connections
- Personalise cover letter: 3 sentences on why THIS team/role
- Apply to 5-7 companies in same week to generate competing offers

---

## Offer Evaluation & Negotiation

### Total Compensation (TC) Components
```
Base salary:       Fixed annual salary
RSU (stock):       Restricted Stock Units vesting over 4 years (cliff at year 1)
Sign-on bonus:     One-time joining bonus
Performance bonus: Annual bonus (typically 10-20% of base)

Total Comp = Base + (RSU / 4) + Bonus

Example FAANG AI Engineer L5/E5:
  Base: $180K
  RSU: $600K over 4 years = $150K/year
  Bonus: ~$36K (20% base)
  Total: $366K/year
```

### Research Compensation
- **levels.fyi** — verified TC data by level and company
- **Glassdoor** — less accurate, useful for base range
- **Blind** — insider compensation discussions

### Negotiation Script
```
"Thank you for the offer — I'm genuinely excited about [specific team/project].
Based on my research on levels.fyi and conversations with engineers in similar
roles, I was expecting a total compensation closer to [$X]. Is there flexibility
on [RSU / sign-on]?"

Key rules:
  - Always negotiate (recruiters expect it, 80% of offers have room)
  - Never give a number first ("What's your expected salary?")
    → "I'm focused on the right opportunity; I trust you'll make a fair offer"
  - Have competing offers: "I have an offer from [Company Y] for $X, 
    [Company] is my first choice but I need to be close on TC"
  - Negotiate in writing after verbal: easier record, time to think
```

---

## Final Checklist Before First Interview

**DSA:**
- [ ] Solved 200+ LeetCode problems (80% Med, 20% Hard)
- [ ] Comfortable with all major patterns (two pointer, sliding window, BFS/DFS, DP, union-find)
- [ ] Can code LCA, Dijkstra, topological sort from scratch

**ML Fundamentals:**
- [ ] Can explain attention mechanism mathematically
- [ ] Know Chinchilla scaling law implications
- [ ] Can compare BERT vs GPT architectures confidently

**System Design:**
- [ ] Practised 8+ design questions using 7-step framework
- [ ] Have concrete numbers memorised (latency, storage, throughput)
- [ ] Know how to size GPU cluster for LLM serving

**Behavioral:**
- [ ] 8 STAR stories ready (can adapt to any behavioral question)
- [ ] Know your "Why FAANG?" answer cold
- [ ] Know your "Why this specific company/team?" for each target

**Portfolio:**
- [ ] GitHub profile public, 3+ repos with READMEs
- [ ] RAG project deployed and accessible
- [ ] LinkedIn updated with measurable impact bullets

**Mental prep:**
- [ ] Sleep schedule locked (8 hours, consistent)
- [ ] Have water and notes ready for phone screens
- [ ] Know: "I don't know immediately" is better than wrong answer; think out loud

---

## 📚 Final Resources

**Interview prep:**
- **"Cracking the Coding Interview" by Gayle Laakmann McDowell** — DSA fundamentals
- **"System Design Interview" (Vol. 1 & 2) by Alex Xu** — HLD blueprints
- NeetCode.io roadmap: https://neetcode.io/roadmap
- AlgoMonster patterns: https://algo.monster

**Compensation:**
- levels.fyi: https://www.levels.fyi
- Negotiation guide: https://haseebq.com/my-ten-rules-for-negotiating-a-job-offer

**Community:**
- r/cscareerquestions — general FAANG prep advice
- r/MachineLearning — research and industry
- Blind (app) — insider FAANG compensation and interview intel
- LeetCode Discuss — company-specific interview reports

---

> **You've completed all 12 months. You are ready.**

---

## Transition Guide: MLOps Engineer → AI/LLM Engineer (Day-to-Day)

### What Changes in Your Daily Work

```
WHAT STAYS THE SAME:
  ✓ CI/CD pipelines (you already know this)
  ✓ Docker, Kubernetes, Cloud services
  ✓ Monitoring, alerting, observability
  ✓ Python, SQL, data processing
  ✓ Working with data scientists and product managers

WHAT'S NEW:
  + Prompt engineering (writing and iterating on prompts daily)
  + LLM API usage (OpenAI, Anthropic, local models)
  + RAG pipeline development and maintenance
  + Agent development (LangGraph, CrewAI)
  + Evaluation (RAGAS, LLM-as-judge, human eval)
  + Vector databases (Pinecone, pgvector, Weaviate)
  + Fine-tuning workflows (LoRA, QLoRA, data curation)
  + LLM serving (vLLM, TGI, cost optimization)
  + New tools: LangChain, LangSmith, Ollama, HuggingFace

WHAT YOU STOP DOING:
  - Less focus on traditional ML model training (XGBoost, sklearn)
  - Less feature engineering (LLMs do this via prompts)
  - Less model retraining pipelines (LLMs are pre-trained)
```

### Your First 90 Days as an AI/LLM Engineer

```
WEEK 1-2: RAMP UP
  □ Learn the team's LLM tech stack (which models, frameworks, infrastructure)
  □ Get access to all LLM APIs, vector databases, monitoring tools
  □ Read existing RAG/agent code — understand architecture and design decisions
  □ Run the application locally, trace a request end-to-end
  □ Identify the evaluation strategy — how does the team measure "good"?

WEEK 3-4: FIRST CONTRIBUTION
  □ Pick a well-scoped issue (improve a prompt, add a metric, fix a retrieval bug)
  □ Ship to production with proper testing and monitoring
  □ Learn the deployment process for your team

MONTH 2: BUILD CREDIBILITY  
  □ Take ownership of a component (e.g., retrieval pipeline, evaluation suite)
  □ Propose and implement one measurable improvement
  □ Start contributing to system design discussions
  □ Build relationship with ML team (you bridge ML and engineering)

MONTH 3: DRIVE IMPACT
  □ Own a feature end-to-end (from design to production)
  □ Establish a new best practice (evaluation framework, prompt testing CI, etc.)
  □ Present a tech talk or write an internal blog post
  □ Identify the next big improvement opportunity
```

### Salary Negotiation Script

```
WHEN YOU GET THE OFFER:

1. Express enthusiasm (genuine): 
   "I'm really excited about this role — the work on [specific project] 
   is exactly what I want to be doing."

2. Ask for the full package:
   "Could you walk me through the complete compensation package — 
   base, equity, signing bonus, and any other components?"

3. Take time:
   "Thank you for this. I'd like to take a day to review the full package 
   before we discuss further. When would be a good time to reconnect?"

4. Counter (be specific):
   "Based on my research of AI Engineer compensation at [company level], 
   and the skills I bring in LLM systems, production RAG, and fine-tuning, 
   I was hoping for [X base / Y equity]. Is there flexibility on [base/equity]?"

5. If they push back:
   "I understand the bands. Would you consider [signing bonus/equity refresh/
   level bump/earlier review] to help bridge the gap?"

COMPENSATION DATA POINTS (2026 AI Engineer):
  Google L5:   $200-250K base + $200-400K equity/yr = $400-650K TC
  Meta E5:     $200-250K base + $200-400K equity/yr = $400-650K TC  
  Amazon L6:   $180-220K base + $100-250K equity/yr = $280-470K TC
  Microsoft 63: $180-220K base + $150-300K equity/yr = $330-520K TC

  Note: AI/LLM specialists command 10-20% premium over general SWE
```

### Interview Day Checklist

```
NIGHT BEFORE:
  □ Review your STAR stories (read them aloud once)
  □ Review top 5 DSA patterns (template code, not specific problems)
  □ Review ML system design framework (7 steps on one page)
  □ Prepare questions for each interviewer (specific to their work)
  □ Charge laptop, test camera/mic, quiet room
  □ Sleep 7+ hours

MORNING OF:
  □ Light breakfast, water
  □ Review your "one-pager" (key numbers: latency, throughput, cost savings)
  □ Open VS Code / IDE, have a blank file ready
  □ Have a notebook for System Design whiteboarding

DURING INTERVIEW:
  □ Think out loud — silence is the enemy
  □ Ask clarifying questions before coding
  □ State time/space complexity before and after implementation
  □ Test your code with examples
  □ If stuck: say "Let me think about this differently" and try a new approach
  □ For behavioral: follow STAR, emphasize YOUR actions and quantified results
  □ Ask thoughtful questions at the end (shows genuine interest)

AFTER INTERVIEW:  
  □ Send thank-you email to recruiter within 24 hours
  □ Note down every question asked (for future prep)
  □ Identify gaps in your knowledge → update study materials
```

---

> 
> Good luck. You've got this. 🎯
