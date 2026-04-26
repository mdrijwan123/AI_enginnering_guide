# MCP — Model Context Protocol
### Phase 2 | August 2026 | Week 3

> **What is MCP?** The Model Context Protocol (MCP) is an open standard introduced by Anthropic (November 2024) that defines how AI assistants connect to external tools, data sources, and services. Think of it as USB-C for AI: one standard connector that works across all LLM applications and providers.

> 💡 **ELI5 (Explain Like I'm 5):**
> Imagine your AI assistant is a very smart person locked in a room. MCP is the standardised postal system that slides documents, tools, and data under the door. Without MCP, every AI app had to invent its own letter format. With MCP, one format works everywhere.

---

## Why MCP Matters in 2026

Before MCP, every AI application (ChatGPT, Claude, Gemini) had its own way of calling tools. Developers had to write custom integrations for each. MCP changes this:

```
Before MCP:                    After MCP:
  App A ─── custom ─── DB        App A ─┐
  App A ─── custom ─── API      App B ─┤── MCP ── DB
  App B ─── custom ─── DB       App C ─┤── MCP ── API
  App C ─── custom ─── API      Claude─┘── MCP ── Filesystem
```

**Current MCP support (2026):** Claude Desktop, Claude.ai, Cursor, VS Code Copilot, Zed, Windsurf, many open-source frameworks.

---

## Core Architecture

### MCP Components

```
┌─────────────────────────────────────────────────────┐
│                   MCP Architecture                   │
│                                                       │
│  ┌─────────────┐   MCP Protocol   ┌───────────────┐  │
│  │  MCP Client │◄────────────────►│   MCP Server  │  │
│  │  (Claude,   │                  │  (Your tool,  │  │
│  │   Cursor,   │                  │  database,    │  │
│  │   your app) │                  │  API, etc.)   │  │
│  └─────────────┘                  └───────────────┘  │
│                                                       │
│  Transport: stdio (local) | SSE (remote/HTTP)         │
└─────────────────────────────────────────────────────┘
```

### The 3 MCP Primitives

| Primitive | Controlled by | What it is |
|---|---|---|
| **Tools** | Client (LLM decides when to call) | Functions the LLM can invoke (like function calling) |
| **Resources** | Client (LLM reads) | Data sources the LLM can read (files, DB rows, API responses) |
| **Prompts** | User | Pre-built prompt templates exposed by the server |

---

## Part 1 — Building Your First MCP Server

### 1.1 Installation
```bash
pip install mcp
# or with uvx (recommended):
pip install uv
```

### 1.2 Minimal MCP Server
```python
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server.stdio import stdio_server

# Create server instance
server = Server("my-first-mcp-server")

# ── Tools ─────────────────────────────────────────────────────
@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_weather",
            description="Get current weather for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "get_weather":
        city = arguments["city"]
        # In production: call real weather API
        return [types.TextContent(
            type="text",
            text=f"Weather in {city}: 22°C, Partly cloudy"
        )]
    raise ValueError(f"Unknown tool: {name}")

# ── Resources ─────────────────────────────────────────────────
@server.list_resources()
async def list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri="file:///data/report.md",
            name="Monthly Report",
            description="Latest monthly analysis report",
            mimeType="text/markdown"
        )
    ]

@server.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "file:///data/report.md":
        with open("/data/report.md") as f:
            return f.read()
    raise ValueError(f"Unknown resource: {uri}")

# ── Run ───────────────────────────────────────────────────────
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="my-first-mcp-server",
                server_version="1.0.0"
            )
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 1.3 Claude Desktop Configuration (claude_desktop_config.json)
```json
{
  "mcpServers": {
    "my-first-server": {
      "command": "python",
      "args": ["/path/to/your/server.py"]
    },
    "sqlite-server": {
      "command": "uvx",
      "args": ["mcp-server-sqlite", "--db-path", "/path/to/database.db"]
    }
  }
}
```

---

## Part 2 — Production MCP Server (BigQuery + GCP)

### 2.1 GCP Data MCP Server
```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions
import mcp.types as types
from google.cloud import bigquery
import json
import asyncio

server = Server("gcp-data-server")
bq_client = bigquery.Client()

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="run_bigquery",
            description="Run a BigQuery SQL query and return results",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to run"},
                    "limit": {"type": "integer", "description": "Max rows (default 100)", "default": 100}
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="list_datasets",
            description="List all BigQuery datasets in the project",
            inputSchema={"type": "object", "properties": {}}
        ),
        types.Tool(
            name="get_table_schema",
            description="Get schema of a BigQuery table",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset": {"type": "string"},
                    "table": {"type": "string"}
                },
                "required": ["dataset", "table"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "run_bigquery":
        query = arguments["query"]
        limit = arguments.get("limit", 100)
        
        # Safety: prevent DDL/DML (read-only)
        forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER"]
        if any(kw in query.upper() for kw in forbidden):
            return [types.TextContent(type="text", 
                text="Error: Only SELECT queries are allowed.")]
        
        # Add LIMIT if missing
        if "LIMIT" not in query.upper():
            query = f"{query} LIMIT {limit}"
        
        try:
            job = bq_client.query(query)
            results = job.result()
            rows = [dict(row) for row in results]
            return [types.TextContent(
                type="text",
                text=json.dumps(rows, indent=2, default=str)
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Query error: {str(e)}")]
    
    elif name == "list_datasets":
        datasets = [d.dataset_id for d in bq_client.list_datasets()]
        return [types.TextContent(type="text", text="\n".join(datasets))]
    
    elif name == "get_table_schema":
        table_ref = f"{bq_client.project}.{arguments['dataset']}.{arguments['table']}"
        table = bq_client.get_table(table_ref)
        schema = [{"name": f.name, "type": f.field_type} for f in table.schema]
        return [types.TextContent(type="text", text=json.dumps(schema, indent=2))]
    
    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="gcp-data-server",
                server_version="1.0.0"
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Part 3 — Remote MCP Server (SSE Transport)

For sharing MCP servers across teams or over HTTP (not just local stdio):

```python
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount
import uvicorn

# Create SSE transport
sse = SseServerTransport("/messages")

async def handle_sse(request):
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await server.run(
            streams[0], streams[1],
            InitializationOptions(server_name="remote-server", server_version="1.0.0")
        )

async def handle_messages(request):
    await sse.handle_post_message(request.scope, request.receive, request._send)

app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages", app=handle_messages),
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

**Client connection to remote MCP (claude_desktop_config.json):**
```json
{
  "mcpServers": {
    "remote-server": {
      "url": "http://your-server:8080/sse"
    }
  }
}
```

---

## Part 4 — Using MCP from Your Own LLM Application

```python
import anthropic
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_agent_with_mcp():
    client = anthropic.Anthropic()
    
    # Connect to MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["./my_mcp_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Get available tools from MCP server
            tools_result = await session.list_tools()
            tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
                for tool in tools_result.tools
            ]
            
            messages = [{"role": "user", "content": "What's the weather in London?"}]
            
            # Agentic loop
            while True:
                response = client.messages.create(
                    model="claude-opus-4-5",
                    max_tokens=1024,
                    tools=tools,
                    messages=messages
                )
                
                if response.stop_reason == "end_turn":
                    print(response.content[0].text)
                    break
                
                # Handle tool calls
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                
                for block in response.content:
                    if block.type == "tool_use":
                        result = await session.call_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result.content[0].text
                        })
                
                messages.append({"role": "user", "content": tool_results})

asyncio.run(run_agent_with_mcp())
```

---

## Part 5 — MCP Security Best Practices

```python
# 1. Input validation — never trust tool arguments
def validate_sql(query: str) -> tuple[bool, str]:
    import re
    # Block dangerous operations
    dangerous = r'\b(DROP|INSERT|UPDATE|DELETE|ALTER|TRUNCATE|EXEC)\b'
    if re.search(dangerous, query, re.IGNORECASE):
        return False, "Dangerous SQL operation blocked"
    return True, ""

# 2. Rate limiting per client
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_calls: int, period: int):
        self.max_calls = max_calls
        self.period = period
        self.calls = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        self.calls[client_id] = [t for t in self.calls[client_id] if now - t < self.period]
        if len(self.calls[client_id]) >= self.max_calls:
            return False
        self.calls[client_id].append(now)
        return True

# 3. Secret management — never hardcode
import os
api_key = os.environ.get("API_KEY")  # ✅ from environment
# api_key = "sk-abc123"              # ❌ never hardcode

# 4. Audit logging
import logging
logger = logging.getLogger("mcp_audit")

async def call_tool_with_audit(name: str, arguments: dict, client_id: str):
    logger.info(f"Tool call: {name}, client: {client_id}, args: {arguments}")
    result = await actual_call_tool(name, arguments)
    logger.info(f"Tool result: {name}, success: True")
    return result
```

---

## Part 6 — Ready-to-Use MCP Servers (Official)

```bash
# Official servers you can use immediately:
uvx mcp-server-sqlite --db-path mydb.db        # SQLite
uvx mcp-server-filesystem /path/to/files        # File system
uvx mcp-server-git --repository /path/to/repo   # Git
uvx mcp-server-github                            # GitHub API
uvx mcp-server-postgres postgresql://...         # PostgreSQL
uvx mcp-server-brave-search                      # Brave Search API
uvx mcp-server-puppeteer                         # Web browser automation
```

**Community servers:** https://github.com/punkpeye/awesome-mcp-servers

---

## Interview Q&A

### Q1: What problem does MCP solve?
**A:** The "M×N integration problem." Before MCP, connecting M AI applications to N tools required M×N custom integrations. MCP reduces this to M+N: each client implements MCP once, each server implements MCP once. It's the USB standard for AI tool connectivity.

### Q2: What are the three MCP primitives and who controls them?
**A:** (1) **Tools** — LLM-controlled functions the model decides to invoke (like function calling). (2) **Resources** — LLM-readable data sources (files, DB entries). Client decides when to expose/read. (3) **Prompts** — user-controlled prompt templates from the server. The key distinction is who initiates each interaction.

### Q3: How does MCP differ from OpenAI function calling?
**A:** Function calling is vendor-specific and client-side only (schemas defined by the app developer). MCP is an open protocol where the *server* self-describes its capabilities. MCP servers can run anywhere, be shared across teams, and work with any MCP-compatible client. MCP also adds resources and prompts, which function calling has no equivalent for.

### Q4: When would you choose stdio vs SSE transport?
**A:** **stdio** for local servers on the same machine (Claude Desktop, Cursor). Simple, no networking needed. **SSE (HTTP/Server-Sent Events)** for remote servers accessible over the network — shared team servers, cloud-hosted tools, multi-user scenarios. SSE servers need to handle auth and security.

### Q5: How would you expose your company's internal analytics database via MCP?
**A:** Build an MCP server with: (1) `run_query` tool with SQL validation (SELECT only), rate limiting, and audit logging. (2) Resources for schema documentation. (3) stdio transport for local dev, SSE for shared team access. Auth via API keys in environment variables. Deploy as Docker container on internal network.

---

## Further Resources

- **Official MCP Docs** — https://modelcontextprotocol.io/docs
- **MCP Python SDK** — https://github.com/modelcontextprotocol/python-sdk
- **Awesome MCP Servers** — https://github.com/punkpeye/awesome-mcp-servers
- **Anthropic MCP Announcement** — https://www.anthropic.com/news/model-context-protocol
