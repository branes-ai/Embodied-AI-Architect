"""MCP server for LLM agent integration with the MOO engine.

Exposes optimization as tools for the ArchitectAgent via MCP protocol.
This is the interface layer, NOT the compute layer.

Usage:
    # Start MCP server
    from embodied_ai_architect.mcp.server import create_mcp_server
    server = create_mcp_server()
"""
