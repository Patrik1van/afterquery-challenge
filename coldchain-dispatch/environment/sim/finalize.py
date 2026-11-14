"""Run inside the sim container by Harbor's [[verifier.collect]] hook, after the agent has stopped.
Plays out the rest of the shift and writes the score to /sim/out/result.json, which Harbor
copies into the separate verifier container as an artifact. The agent never touches this path."""
import json, os, pathlib, urllib.request

req = urllib.request.Request("http://localhost:8000/verify", data=b"{}", method="POST",
                             headers={"X-Verify-Token": os.environ["VERIFY_TOKEN"]})
result = json.load(urllib.request.urlopen(req, timeout=60))
out = pathlib.Path("/sim/out"); out.mkdir(parents=True, exist_ok=True)
(out / "result.json").write_text(json.dumps(result, indent=1))
print("score", result["score"])
