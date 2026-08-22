import assert from "node:assert/strict";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  findRepoRoot,
  isOurRuntime,
  parseEnvFile,
  resolveRepoRoot,
} from "../dsh-plugin/src/runtime.js";

const REPO = dirname(fileURLToPath(new URL(".", import.meta.url)));

const env = parseEnvFile(`
# comment
export RECA_PLANNER_API_KEY="sk-test"
RECA_GATEWAY_PORT=8799
`);
assert.equal(env.RECA_PLANNER_API_KEY, "sk-test");
assert.equal(env.RECA_GATEWAY_PORT, "8799");

assert.equal(
  isOurRuntime({ ok: true, service: "reca-gateway", repo_root: REPO }, REPO),
  true,
);
assert.equal(
  isOurRuntime({ ok: true, service: "reca-gateway" }, REPO),
  false,
);

assert.equal(findRepoRoot([REPO]), REPO);
assert.equal(resolveRepoRoot({ cwd: REPO, pluginRoot: join(REPO, "dsh-plugin") }), REPO);

console.log("test_plugin_runtime ok");
