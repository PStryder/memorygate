#!/usr/bin/env python3
"""Test MemoryGate MCP protocol"""
import requests
import json

BASE_URL = "http://localhost:8080"

def mcp_call(method, params=None):
    """Call MCP tool via JSON-RPC"""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": f"tools/call",
        "params": {
            "name": method,
            "arguments": params or {}
        }
    }
    
    resp = requests.post(f"{BASE_URL}/mcp", json=payload)
    resp.raise_for_status()
    return resp.json()

# Test 1: Store observation
print("==> Test 1: Storing observation via MCP...")
result = mcp_call("memory_store", {
    "observation": "Test memory via MCP protocol",
    "domain": "mcp_test",
    "confidence": 0.9
})
print(f"Result: {json.dumps(result, indent=2)}")

# Test 2: Recall observation
print("\n==> Test 2: Recalling observations...")
result = mcp_call("memory_recall", {
    "domain": "mcp_test",
    "limit": 10
})
print(f"Result: {json.dumps(result, indent=2)}")

# Test 3: Stats
print("\n==> Test 3: Memory stats...")
result = mcp_call("memory_stats", {})
print(f"Result: {json.dumps(result, indent=2)}")

print("\n✅ MCP protocol working!")
