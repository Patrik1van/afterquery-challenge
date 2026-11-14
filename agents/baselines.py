"""Baselines for Harbor beamline task."""

from __future__ import annotations

import json
from typing import override

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class StopEarlyAgent(BaseAgent):
    """Harbor baseline agent: takes 2 small exploratory shots then calls submit."""

    capabilities = AgentCapabilities(windows=True)

    @staticmethod
    @override
    def name() -> str:
        return "stop-early"

    @override
    def version(self) -> str:
        return "1.0.0"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Execute 2 shots against the MCP server from inside the container, then submit."""
        script = """
import json, urllib.request

URL = "http://beamline:8000/mcp"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
req_id = 1
session_id = None

def call_mcp(method, params):
    global req_id, session_id
    req_id += 1
    req_body = json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}).encode("utf-8")
    hdrs = dict(HEADERS)
    if session_id:
        hdrs["mcp-session-id"] = session_id
    req = urllib.request.Request(URL, data=req_body, headers=hdrs)
    with urllib.request.urlopen(req) as resp:
        if not session_id:
            session_id = resp.headers.get("mcp-session-id")
        body = resp.read().decode("utf-8")
        for line in body.splitlines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                return data.get("result", {})
    return {}

# 1. Initialize
call_mcp("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "stop-early", "version": "1.0"}})

# 2. Shot 1: zero setting
call_mcp("tools/call", {"name": "set_correctors", "arguments": {"h": [0.0]*10, "v": [0.0]*12}})

# 3. Shot 2: slight kick
call_mcp("tools/call", {"name": "set_correctors", "arguments": {"h": [1e-4]*10, "v": [-1e-4]*12}})

# 4. Submit
call_mcp("tools/call", {"name": "submit", "arguments": {}})
print("StopEarlyAgent completed 2 shots and submitted.")
"""
        cmd = f"python3 -c {json.dumps(script)}"
        await self.exec_as_agent(environment, command=cmd)
