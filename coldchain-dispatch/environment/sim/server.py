"""Simulator sidecar. Runs in its own container so the agent cannot read future orders or the scorer.

Agent-facing endpoints (via the `dispatch` CLI in the main container):
  GET  /state            POST /route {vehicle, stops}     POST /wait {minutes}     POST /finish
Verifier-only endpoint (needs the token that lives in tests/, uploaded after the agent finishes):
  POST /verify  header X-Verify-Token
"""
import json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from coldchain import ColdChainEnv

LEVEL = os.environ.get("LEVEL", "medium")
SEED = int(os.environ.get("SEED", "0"))
TOKEN = os.environ.get("VERIFY_TOKEN", "change-me")
PORT = int(os.environ.get("PORT", "8000"))

env = ColdChainEnv(LEVEL, SEED)
lock = Lock()

class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return None

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True})
        if self.path == "/state":
            with lock:
                return self._send(200, env.step("get_state"))
        self._send(404, {"error": "not found"})

    def do_POST(self):
        args = self._body()
        if args is None:
            return self._send(400, {"error": "invalid JSON"})
        if self.path == "/verify":
            if self.headers.get("X-Verify-Token") != TOKEN:
                return self._send(403, {"error": "forbidden"})
            with lock:
                if not env.done:            # agent stopped early: the shift still plays out
                    env._advance(env.horizon)
                r = env.reward()
                r.update(level=LEVEL, seed=SEED, tool_calls=env.calls, truth=env.truth(), log=env.log)
                return self._send(200, r)
        tool = {"/route": "set_route", "/wait": "wait", "/finish": "finish"}.get(self.path)
        if tool is None:
            return self._send(404, {"error": "not found"})
        with lock:
            self._send(200, env.step(tool, args))

if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
