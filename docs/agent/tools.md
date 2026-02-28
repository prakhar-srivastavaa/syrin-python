# Tools

Tools are functions the agent can call during execution. They extend the agent with search, computation, APIs, and other external actions.

## Defining Tools

Use the `@syrin.tool` decorator:

```python
from syrin.tool import tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Use only +, -, *, /, **."""
    return str(eval(expression))
```

The docstring and parameter types become the tool schema sent to the model.

## Adding Tools to an Agent

Pass tools in the constructor or as a class attribute:

```python
agent = Agent(
    model=model,
    tools=[search, calculate],
)
```

## Tool Execution Flow

1. Model returns tool calls (names and arguments).
2. Agent looks up the tool by name.
3. Agent calls `tool.func(**arguments)`.
4. Result is appended to the conversation.
5. Loop continues until no more tool calls.

## ToolSpec

Import from `syrin.tool`: `from syrin.tool import ToolSpec` (or `from syrin import ToolSpec`). Each tool is represented as a `ToolSpec` with:

- `name` — Tool name (from function name).
- `parameters` — JSON schema for arguments.
- `func` — The underlying callable.

The `@tool` decorator builds `ToolSpec` from the function signature and docstring.

## Parameter Types

Supported types: `str`, `int`, `float`, `bool`, `list`, `dict`, and `Optional[T]`.

```python
@tool
def create_task(title: str, priority: int = 1, tags: list[str] | None = None) -> str:
    """Create a task with title, optional priority and tags."""
    return f"Created: {title} (priority={priority})"
```

## Error Handling

Tool failures raise `ToolExecutionError`:

```python
from syrin.exceptions import ToolExecutionError

try:
    response = agent.response("Search for X")
except ToolExecutionError as e:
    print(f"Tool failed: {e}")
```

## execute_tool (Custom Loops)

Custom loops can call tools via `execute_tool`:

```python
result = await agent.execute_tool("search", {"query": "hello"})
```

## Grouping Tools in MCP

You can group related tools in an MCP server and use the MCP as an agent tool source.

Define an MCP with `@tool` (same as Agent):

```python
from syrin import MCP, Agent, tool

class ProductMCP(MCP):
    name = "product-mcp"
    description = "Product catalog tools"

    @tool
    def search_products(self, query: str, limit: int = 10) -> str:
        """Search the product catalog."""
        return f"Results for: {query}"

    @tool
    def get_product(self, product_id: str) -> str:
        """Get product by ID."""
        return f"Product {product_id} details"
```

## Using MCP Inside Agent Tools

Add the MCP instance to your agent's `tools=[]`. The agent can call all MCP tools:

```python
product_mcp = ProductMCP()

class ProductAgent(Agent):
    model = almock
    tools = [product_mcp]  # MCP tools become agent tools

# Or pick specific tools: tools=[product_mcp.select("search_products")]
```

When serving, if MCP is in `tools`, `/mcp` is auto-mounted alongside `/chat`. See [MCP](mcp.md) for details.

## See Also

- [MCP](../mcp.md) — Group tools, use MCP in agents, co-location
- [Use Case 2: Research Agent with Tools](../research-agent-with-tools.md)
- [Loop Strategies](loop-strategies.md) — How tools integrate with loops
