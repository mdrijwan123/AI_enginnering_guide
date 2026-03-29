# Month 4: Dynamic Programming + Multi-Agent Systems
### Phase 2 | July 2026

---

## Week 1–2: Dynamic Programming

### The DP Mindset

DP = "memoised recursion" or "tabulation". Use when:
1. Problem has **optimal substructure** (optimal solution built from optimal sub-solutions)
2. Problem has **overlapping subproblems** (same sub-problems solved repeatedly)

**2 approaches:**
- **Top-down (memoisation):** Recursion + cache results
- **Bottom-up (tabulation):** Iterative, fill a table from smallest subproblems up

---

### 1D DP Problems

#### Climbing Stairs (LC #70)
```python
def climbStairs(n):
    # dp[i] = ways to reach step i
    # dp[i] = dp[i-1] + dp[i-2]   (come from 1 step or 2 steps behind)
    if n <= 2: return n
    prev2, prev1 = 1, 2
    for _ in range(3, n+1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    return prev1
# O(n) time, O(1) space
```

#### House Robber (LC #198)
```python
def rob(nums):
    # dp[i] = max money robbing houses 0..i
    # dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    prev2, prev1 = 0, 0
    for n in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + n)
    return prev1
```

#### Coin Change (LC #322)
```python
def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i-coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
# O(amount × len(coins)) time, O(amount) space
```

#### Longest Increasing Subsequence (LC #300)
```python
def lengthOfLIS(nums):
    # O(n²) DP
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

# O(n log n) with patience sorting + binary search
import bisect
def lengthOfLIS_fast(nums):
    tails = []
    for n in nums:
        pos = bisect.bisect_left(tails, n)
        if pos == len(tails): tails.append(n)
        else: tails[pos] = n
    return len(tails)
```

#### Word Break (LC #139)
```python
def wordBreak(s, wordDict):
    word_set = set(wordDict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    for i in range(1, len(s)+1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[-1]
```

---

### 2D DP Problems

#### Unique Paths (LC #62)
```python
def uniquePaths(m, n):
    dp = [[1]*n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]
# Space optimisation: use single row
```

#### Longest Common Subsequence (LC #1143)
```python
def longestCommonSubsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```

#### 0/1 Knapsack Pattern (Subset Sum)
```python
def canPartition(nums):  # LC #416
    total = sum(nums)
    if total % 2: return False
    target = total // 2
    dp = {0}
    for n in nums:
        dp = {s + n for s in dp} | dp
    return target in dp
```

#### Edit Distance (LC #72) — Classic 2D DP
```python
def minDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],    # delete
                                    dp[i][j-1],    # insert
                                    dp[i-1][j-1])  # replace
    return dp[m][n]
```

### DP Interview Q&A

**Q: How do you recognise a DP problem?**
> Keywords: "minimum/maximum", "how many ways", "is it possible", "longest/shortest". Check: does solving sub-problems help? Can I define the answer at position i in terms of previous positions?

**Q: What's the space optimisation for most DP problems?**
> Most 2D DP only needs the previous row: `dp = [dp_curr, dp_prev]` rolling array. For 1D DP only needs last 1–2 values.

**Q: Bottom-up vs top-down — which to prefer in interviews?**
> Bottom-up (tabulation) in interviews: cleaner code, no recursion stack overflow, easier to optimise space. Top-down (memoization) can be faster to write initially.

---

### Practice Tracker — DP

| Problem | Pattern | Solved? |
|---|---|---|
| LC #70 Climbing Stairs | 1D DP | ⬜ |
| LC #198 House Robber | 1D DP | ⬜ |
| LC #213 House Robber II | 1D DP Circular | ⬜ |
| LC #322 Coin Change | Unbounded Knapsack | ⬜ |
| LC #300 LIS | Patience Sort | ⬜ |
| LC #139 Word Break | 1D DP + Hash | ⬜ |
| LC #62 Unique Paths | 2D DP | ⬜ |
| LC #1143 LCS | 2D DP | ⬜ |
| LC #416 Partition Equal Sum | 0/1 Knapsack | ⬜ |
| LC #72 Edit Distance | 2D DP Classic | ⬜ |
| LC #115 Distinct Subsequences | 2D DP | ⬜ |
| LC #312 Burst Balloons | Interval DP | ⬜ |
| LC #309 Best Time Stock Cooldown | State Machine DP | ⬜ |

---

## Week 3–4: Multi-Agent Systems

### Why Multi-Agent?

Single agent limitations:
- Context window gets full for very long tasks
- One LLM making all decisions reduces quality (mixing roles)
- Specialisation: a "coding" agent is better at code than a generalist

Multi-agent systems divide:
- **Roles:** Planner, Researcher, Coder, Critic, Executor
- **Parallelism:** Multiple agents working simultaneously
- **Specialisation:** Each agent has a focused system prompt

### CrewAI Framework

```python
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# Define specialist agents
researcher = Agent(
    role='Research Analyst',
    goal='Find accurate and up-to-date information on the given topic',
    backstory='Expert at finding and synthesising information from web research',
    tools=[search_tool, scrape_tool],
    llm=llm,
    verbose=True
)

writer = Agent(
    role='Technical Writer',
    goal='Write clear, accurate technical reports based on research findings',
    backstory='Expert at communicating complex technical topics clearly',
    tools=[],  # no tools needed for writing
    llm=llm
)

critic = Agent(
    role='Quality Reviewer',
    goal='Review content for accuracy, completeness, and clarity',
    backstory='Senior editor with expertise in fact-checking',
    tools=[search_tool],
    llm=llm
)

# Define tasks
research_task = Task(
    description='Research the latest developments in {topic}. Include key papers, companies, and statistics.',
    agent=researcher,
    expected_output='A detailed research brief with facts, citations, and key names'
)

write_task = Task(
    description='Write a 500-word technical blog post based on the research brief',
    agent=writer,
    expected_output='A polished technical blog post in markdown format'
)

review_task = Task(
    description='Review the blog post for accuracy. Flag any claims that need verification.',
    agent=critic,
    expected_output='Reviewed post with corrections and a quality score (1-10)'
)

# Create crew
crew = Crew(
    agents=[researcher, writer, critic],
    tasks=[research_task, write_task, review_task],
    process=Process.sequential,  # or Process.hierarchical
    verbose=True
)

result = crew.kickoff(inputs={"topic": "RAG systems in production 2025"})
```

### LangGraph Multi-Agent

```python
# Supervisor pattern: one LLM routes to specialist sub-agents
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

def supervisor_node(state):
    """Route to the right agent based on task type"""
    last_msg = state["messages"][-1].content
    response = supervisor_llm.invoke([
        SystemMessage("You route tasks to: 'researcher', 'coder', 'writer', or FINISH"),
        HumanMessage(last_msg)
    ])
    return {"next": response.content}  # "researcher" or "coder" etc.

workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("coder", coder_node)
workflow.add_node("writer", writer_node)

# Supervisor routes to specialists; specialists report back to supervisor
for agent in ["researcher", "coder", "writer"]:
    workflow.add_edge(agent, "supervisor")

workflow.add_conditional_edges("supervisor", 
    lambda s: s["next"],
    {"researcher": "researcher", "coder": "coder", "writer": "writer", "FINISH": END}
)
```

### Multi-Agent Patterns

| Pattern | Description | Use When |
|---|---|---|
| Sequential | Agent A → Agent B → Agent C | Pipeline: clear hand-off order |
| Supervisor-Worker | Central router → specialist agents | Complex tasks, unknown routing |
| Peer-to-Peer | Agents communicate directly | Collaborative, emergent tasks |
| Hierarchical | Multi-level: manager → team → worker | Large, decomposable tasks |
| Parallel | Multiple agents run simultaneously | Independent subtasks |

### Interview Q&A — Multi-Agent

**Q1: When should you use a multi-agent system vs a single agent?**
> Use multi-agent when: task is too long for one context window, task has clearly distinct sub-roles, you need parallel execution, quality improves from specialisation + critique. Don't use if: simple task, latency is critical (multi-agent adds overhead), debugging complexity not worth it.

**Q2: How do you handle failures in a multi-agent system?**
> 1. Retry with exponential backoff at tool level
> 2. Fallback: if agent A fails, route to agent B with different approach
> 3. Timeout: max time per agent, supervisor escalates if exceeded
> 4. Human escalation: if all agents fail, notify human operator
> 5. Partial results: return what was completed, flag incomplete parts

**Q3: How do you share state between agents?**
> Shared state in LangGraph (all agents read/write same state dict), message passing, shared database (Redis, Postgres), or explicit hand-off format (Agent A's output is Agent B's input in a defined schema).

---

## MCP Client Implementation

### What Is an MCP Client?

An MCP (Model Context Protocol) client connects to MCP servers to use their tools. While MCP servers expose capabilities, the client is what your LLM application uses to discover and invoke those tools.

### Building an MCP Client

```python
import asyncio
import json
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

async def run_mcp_client():
    # Connect to an MCP server (stdio transport)
    server_params = StdioServerParameters(
        command="python",
        args=["my_mcp_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize connection
            await session.initialize()
            
            # 1. Discover available tools
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
                print(f"    Input schema: {json.dumps(tool.inputSchema, indent=2)}")
            
            # 2. Call a tool
            result = await session.call_tool(
                "search_documents",
                arguments={"query": "RAG best practices", "top_k": 5}
            )
            print(f"Result: {result.content}")
            
            # 3. List resources (if server supports)
            resources = await session.list_resources()
            for resource in resources.resources:
                print(f"Resource: {resource.uri} - {resource.name}")
            
            # 4. Read a resource
            content = await session.read_resource("file:///docs/guide.md")
            print(content)

asyncio.run(run_mcp_client())
```

### MCP Client with LLM Integration

```python
from openai import OpenAI

class MCPLLMClient:
    """Integrate MCP tools with an LLM for automatic tool calling."""
    
    def __init__(self, mcp_session, llm_client):
        self.session = mcp_session
        self.llm = llm_client
    
    async def get_tools_as_openai_format(self):
        """Convert MCP tools to OpenAI function calling format."""
        mcp_tools = await self.session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            for tool in mcp_tools.tools
        ]
    
    async def chat(self, messages: list[dict]) -> str:
        tools = await self.get_tools_as_openai_format()
        
        response = self.llm.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        msg = response.choices[0].message
        
        # If LLM wants to call a tool
        if msg.tool_calls:
            messages.append(msg)
            
            for tool_call in msg.tool_calls:
                # Execute via MCP
                result = await self.session.call_tool(
                    tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments)
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result.content)
                })
            
            # Get final response with tool results
            final = self.llm.chat.completions.create(
                model="gpt-4", messages=messages
            )
            return final.choices[0].message.content
        
        return msg.content
```

### SSE Transport (Remote MCP Servers)

```python
from mcp.client.sse import sse_client

async def connect_remote_mcp():
    async with sse_client("https://mcp-server.example.com/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            # Use tools...
```

**Interview Key**: "MCP separates tool providers (servers) from tool consumers (clients + LLMs). The client discovers tools dynamically, converts them to function-calling format, and orchestrates the LLM↔tool loop. This makes LLM applications composable — swap tools without changing application code."

---

## 📚 Further Resources

- **DeepLearning.AI: Multi AI Agent Systems with crewAI** — https://learn.deeplearning.ai/courses/multi-ai-agent-systems-with-crewai
- **Berkeley LLM Agents Course** — https://llmagents-learning.org/f24
- **CrewAI GitHub & Docs** — https://docs.crewai.com/
- **LangGraph Multi-Agent Tutorial** — https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/
- **"Designing LLM Applications" Book (Chip Huyen)** — production multi-agent patterns

> This month's project: **Multi-Agent Research System** (Portfolio Project #4) — Planner + Researcher + Writer agents in LangGraph.

---

## Day-to-Day Work: DP & Multi-Agent at Work

### DP Patterns in Production Code

```
Where you'll use DP thinking as an AI engineer:

1. TEXT SIMILARITY / EDIT DISTANCE
   - Fuzzy matching product names (Levenshtein distance = edit distance)
   - "Heinz Ketchup 400g" vs "HEINZ TOMATO KETCHUP 400G" → edit distance = 14
   - Use for deduplication, entity matching, data cleaning

2. SEQUENCE ALIGNMENT (LCS variant)
   - Matching user journeys / clickstreams to templates
   - Finding common patterns in customer basket sequences

3. OPTIMAL RESOURCE ALLOCATION (Knapsack)
   - Budget allocation across marketing campaigns
   - Selecting features for a model within compute budget
   - Which experiments to run given GPU-hour constraints

4. DYNAMIC TOKEN BUDGETING
   - Distributing context window tokens across multiple RAG sources
   - max_tokens_for_source_i given priority_i and total budget

5. CACHING STRATEGIES (related to DP's memoisation)
   - LRU cache for LLM API calls
   - Which model responses to cache and for how long
```

```python
# Real production example: fuzzy product matching
from functools import lru_cache

@lru_cache(maxsize=10000)
def edit_distance(s1: str, s2: str) -> int:
    """Memoised edit distance — same DP as LC #72."""
    if not s1: return len(s2)
    if not s2: return len(s1)
    if s1[0] == s2[0]:
        return edit_distance(s1[1:], s2[1:])
    return 1 + min(
        edit_distance(s1[1:], s2),      # delete
        edit_distance(s1, s2[1:]),       # insert
        edit_distance(s1[1:], s2[1:])   # replace
    )

def find_best_match(query: str, candidates: list[str], threshold=5) -> str:
    """Find the closest product name match."""
    query_lower = query.lower().strip()
    best, best_dist = None, float('inf')
    for c in candidates:
        d = edit_distance(query_lower, c.lower().strip())
        if d < best_dist:
            best, best_dist = c, d
    return best if best_dist <= threshold else None
```

### Multi-Agent Production Patterns

```python
# Pattern 1: SUPERVISOR AGENT (most common at work)
# Central orchestrator delegates to specialists

class SupervisorAgent:
    """
    Real production pattern: supervisor routes tasks to specialists.
    Used for: complex customer queries, multi-step automation, report generation
    """
    def __init__(self):
        self.agents = {
            "data_analyst": DataAnalystAgent(),    # SQL queries, data pulls
            "report_writer": ReportWriterAgent(),  # Generates reports
            "chart_maker": ChartMakerAgent(),      # Creates visualisations
            "qa_checker": QACheckerAgent(),         # Validates outputs
        }
    
    def route(self, task: str) -> str:
        """LLM-based routing to the right specialist."""
        routing_prompt = f"""Given this task, which specialist should handle it?
        Options: {list(self.agents.keys())}
        Task: {task}
        Respond with just the agent name."""
        return llm.invoke(routing_prompt).strip()
    
    def execute(self, task: str) -> str:
        agent_name = self.route(task)
        return self.agents[agent_name].execute(task)

# Pattern 2: PIPELINE AGENT (sequential processing)
# Each agent processes and passes result to next
# Used for: ETL with AI steps, content pipeline, data enrichment

pipeline = [
    ("extractor", "Extract entities from the document"),
    ("validator", "Validate extracted entities against our database"),
    ("enricher", "Add metadata and classifications"),
    ("formatter", "Format as the final output JSON")
]

# Pattern 3: DEBATE/CONSENSUS AGENT
# Multiple agents generate answers → vote/debate → best answer selected
# Used for: high-stakes decisions, reducing hallucination
```

### Multi-Agent Cost Management

```python
# Multi-agent = multiple LLM calls = costs multiply fast
# Track and budget per-agent

class AgentCostTracker:
    def __init__(self, budget_per_request: float = 0.50):
        self.budget = budget_per_request
        self.spent = 0.0
        self.agent_costs = {}
    
    def log_call(self, agent_name: str, tokens: int, model: str):
        cost = self._estimate_cost(tokens, model)
        self.spent += cost
        self.agent_costs[agent_name] = self.agent_costs.get(agent_name, 0) + cost
        
        if self.spent > self.budget:
            raise BudgetExceededError(
                f"Budget ${self.budget} exceeded. Spent: ${self.spent:.4f}. "
                f"Breakdown: {self.agent_costs}"
            )
    
    def _estimate_cost(self, tokens, model):
        rates = {"gpt-4o": 7.5e-6, "gpt-4o-mini": 0.375e-6}
        return tokens * rates.get(model, 1e-6)
```
