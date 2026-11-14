import json
import urllib.request
from bayes import BayesAgent


class SyncMCPBeamlineClient:
    def __init__(self, url="http://beamline:8000/mcp"):
        self.url = url
        self.headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        self.id = 1
        self.session_id = None
        self._init()

    def _init(self):
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": self.id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "bayes-oracle", "version": "1.0"},
            },
        }).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, headers=self.headers)
        with urllib.request.urlopen(req) as resp:
            self.session_id = resp.headers.get("mcp-session-id")

    def _send(self, method: str, params: dict) -> dict:
        self.id += 1
        body = json.dumps({"jsonrpc": "2.0", "id": self.id, "method": method, "params": params}).encode("utf-8")
        hdrs = dict(self.headers)
        if self.session_id:
            hdrs["mcp-session-id"] = self.session_id
        req = urllib.request.Request(self.url, data=body, headers=hdrs)
        with urllib.request.urlopen(req) as resp:
            text = resp.read().decode("utf-8")
            for line in text.splitlines():
                if line.startswith("data: "):
                    return json.loads(line[6:]).get("result", {})
        return {}

    def get_machine_info(self) -> dict:
        res = self._send("tools/call", {"name": "get_machine_info", "arguments": {}})
        return json.loads(res["content"][0]["text"])

    def set_correctors(self, h, v) -> dict:
        h_list = [float(x) for x in h]
        v_list = [float(x) for x in v]
        res = self._send("tools/call", {"name": "set_correctors", "arguments": {"h": h_list, "v": v_list}})
        return json.loads(res["content"][0]["text"])

    def get_history(self) -> list:
        res = self._send("tools/call", {"name": "get_history", "arguments": {}})
        return json.loads(res["content"][0]["text"])

    def submit(self) -> dict:
        res = self._send("tools/call", {"name": "submit", "arguments": {}})
        return json.loads(res["content"][0]["text"])


def main():
    client = SyncMCPBeamlineClient()
    agent = BayesAgent(n_particles=25_000, roughening=0.02, seed=0)
    agent.solve(client)
    print("Bayes Oracle finished episode.")


if __name__ == "__main__":
    main()
