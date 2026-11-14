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

    capabilities = AgentCapabilities(windows=True, mcp_servers=True)

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
import json, time, urllib.request

URL = "http://beamline:8000/mcp"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

session_id = None
req_id = 1

for _ in range(15):
    try:
        init_body = json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "stop-early", "version": "1.0"},
            },
        }).encode("utf-8")
        req = urllib.request.Request(URL, data=init_body, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=5) as resp:
            session_id = resp.headers.get("mcp-session-id")
            break
    except Exception as e:
        print(f"Waiting for beamline MCP server: {e}")
        time.sleep(1)

def send_call(tool_name, args):
    global req_id, session_id
    req_id += 1
    call_body = json.dumps({
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": args},
    }).encode("utf-8")
    hdrs = dict(HEADERS)
    if session_id:
        hdrs["mcp-session-id"] = session_id
    req = urllib.request.Request(URL, data=call_body, headers=hdrs)
    with urllib.request.urlopen(req, timeout=10) as resp:
        text = resp.read().decode("utf-8")
        for line in text.splitlines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                res = data.get("result", {})
                content = res.get("content", [])
                if content and "text" in content[0]:
                    return json.loads(content[0]["text"])
                return res
    return {}

# 1. Shot 1: zero setting
s1 = send_call("set_correctors", {"h": [0.0]*10, "v": [0.0]*12})
print(f"Shot 1 response: {s1}")

# 2. Shot 2: small setting
s2 = send_call("set_correctors", {"h": [1e-4]*10, "v": [-1e-4]*12})
print(f"Shot 2 response: {s2}")

# 3. Submit
sub = send_call("submit", {})
print(f"Submit response: {sub}")
"""
        cmd = f"""cat << 'EOF' > /tmp/run_baseline.py
{script}
EOF
python3 /tmp/run_baseline.py
"""
        res = await environment.exec(command=cmd)
        if res.return_code != 0:
            raise RuntimeError(f"StopEarlyAgent failed with return code {res.return_code}:\nstdout: {res.stdout}\nstderr: {res.stderr}")
        print("StopEarlyAgent completed:\n", res.stdout)

