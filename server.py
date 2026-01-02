"""
MemoryGate - Persistent Memory-as-a-Service for AI Agents
MCP Server with PostgreSQL + pgvector backend
"""

import os
import uvicorn
from fastapi import FastAPI
from fastmcp import FastMCP

# =============================================================================
# Configuration
# =============================================================================

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:password@memorygate-db.internal:5432/postgres"
)

# =============================================================================
# FastMCP Server
# =============================================================================

mcp_app = FastMCP(
    "MemoryGate",
    stateless_http=True,
    json_response=True,
)

# =============================================================================
# MCP Tools (placeholder - will expand)
# =============================================================================

@mcp_app.tool()
def memory_ping() -> str:
    """Test that MemoryGate is operational."""
    return "MemoryGate is alive!"

# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(title="MemoryGate", redirect_slashes=False)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "memorygate"}

@app.get("/")
async def root():
    return {
        "service": "MemoryGate",
        "description": "Persistent Memory-as-a-Service for AI Agents",
        "mcp_endpoint": "/mcp/",
        "health": "/health"
    }

# Mount MCP app
app.mount("/mcp/", mcp_app.get_asgi_app())

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("MemoryGate starting...")
    print(f"Database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'not configured'}")
    uvicorn.run(app, host="0.0.0.0", port=8080)
