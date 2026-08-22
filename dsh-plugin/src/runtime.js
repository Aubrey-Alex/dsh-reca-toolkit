import { spawn } from "node:child_process";
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, realpathSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const SRC_DIR = dirname(fileURLToPath(import.meta.url));
const PLUGIN_ROOT = dirname(SRC_DIR);
const DEFAULT_HOST = "127.0.0.1";
const DEFAULT_PORT = 8787;
const PORT_TRIES = 20;
const WAIT_MS = 45_000;

export function parseEnvFile(text) {
  const env = {};
  for (const raw of String(text || "").split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const body = line.startsWith("export ") ? line.slice(7).trim() : line;
    const eq = body.indexOf("=");
    if (eq <= 0) continue;
    const key = body.slice(0, eq).trim();
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) continue;
    let value = body.slice(eq + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    env[key] = value;
  }
  return env;
}

export function samePath(left, right) {
  if (!left || !right) return false;
  try {
    return realpathSync(left) === realpathSync(right);
  } catch {
    return join(left) === join(right);
  }
}

export function isOurRuntime(payload, repoRoot) {
  return Boolean(
    payload
    && payload.ok
    && payload.service === "reca-gateway"
    && samePath(payload.repo_root, repoRoot),
  );
}

function looksLikeRepo(root) {
  return existsSync(join(root, "gateway", "server.py")) && existsSync(join(root, "dsh-plugin", "package.json"));
}

export function findRepoRoot(starts = []) {
  const seen = new Set();
  for (const start of starts) {
    let dir = start;
    for (let i = 0; i < 8 && dir && !seen.has(dir); i += 1) {
      seen.add(dir);
      if (looksLikeRepo(dir)) return dir;
      const parent = dirname(dir);
      if (parent === dir) break;
      dir = parent;
    }
  }
  return "";
}

export function resolveRepoRoot({ env = process.env, cwd = process.cwd(), pluginRoot = PLUGIN_ROOT } = {}) {
  if (env.RECA_ROOT && looksLikeRepo(env.RECA_ROOT)) return env.RECA_ROOT;
  const marker = join(pluginRoot, ".repo-root");
  if (existsSync(marker)) {
    const listed = readFileSync(marker, "utf8").trim();
    if (listed && looksLikeRepo(listed)) return listed;
  }
  return findRepoRoot([cwd, pluginRoot, dirname(pluginRoot)]);
}

async function probeHealth(url) {
  try {
    const response = await fetch(url, { method: "GET", signal: AbortSignal.timeout(2000) });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

function loadDotEnv(root) {
  const path = join(root, ".env");
  if (!existsSync(path)) return {};
  return parseEnvFile(readFileSync(path, "utf8"));
}

export class RecaRuntime {
  constructor(options = {}) {
    this.env = options.env || process.env;
    this.pluginRoot = options.pluginRoot || PLUGIN_ROOT;
    this.python = options.python || this.env.PYTHON || "python3";
    this.host = options.host || this.env.RECA_GATEWAY_HOST || DEFAULT_HOST;
    this.preferredPort = Number(options.port || this.env.RECA_GATEWAY_PORT || DEFAULT_PORT);
    this.waitMs = options.waitMs || WAIT_MS;
    this.child = null;
    this.baseUrl = "";
    this.logFd = null;
  }

  get root() {
    const root = resolveRepoRoot({ env: this.env, cwd: process.cwd(), pluginRoot: this.pluginRoot });
    if (!root) {
      throw new Error(
        "reca-director could not find this repository. Run `dsh web` from the repo root after `dsh plugin --profile web add \"file:$PWD/dsh-plugin\"`.",
      );
    }
    return root;
  }

  healthUrl(port) {
    return `http://${this.host}:${port}/health`;
  }

  async ensure() {
    if (this.baseUrl) {
      const current = await probeHealth(`${this.baseUrl}/health`);
      if (isOurRuntime(current, this.root)) return this.baseUrl;
    }
    const root = this.root;
    const fileEnv = loadDotEnv(root);
    const host = fileEnv.RECA_GATEWAY_HOST || this.host;
    this.host = host;
    let port = Number(fileEnv.RECA_GATEWAY_PORT || this.preferredPort || DEFAULT_PORT);
    for (let i = 0; i < PORT_TRIES; i += 1) {
      const payload = await probeHealth(this.healthUrl(port));
      if (isOurRuntime(payload, root)) {
        this.baseUrl = `http://${host}:${port}`;
        return this.baseUrl;
      }
      if (!payload) {
        await this.#spawn(root, host, port, fileEnv);
        this.baseUrl = `http://${host}:${port}`;
        return this.baseUrl;
      }
      port += 1;
    }
    throw new Error(`reca-director could not bind a local runtime port starting at ${this.preferredPort}`);
  }

  async #spawn(root, host, port, fileEnv) {
    const logPath = join(root, ".dsh_gateway.log");
    mkdirSync(dirname(logPath), { recursive: true });
    this.logFd = openSync(logPath, "a");
    const childEnv = {
      ...this.env,
      ...fileEnv,
      RECA_GATEWAY_HOST: host,
      RECA_GATEWAY_PORT: String(port),
      PYTHONUNBUFFERED: "1",
    };
    const child = spawn(this.python, ["-m", "gateway.server", "--host", host, "--port", String(port)], {
      cwd: root,
      env: childEnv,
      stdio: ["ignore", this.logFd, this.logFd],
    });
    this.child = child;
    const started = Date.now();
    while (Date.now() - started < this.waitMs) {
      if (child.exitCode != null) {
        throw new Error(`reca-director runtime exited before it was healthy. See ${logPath}`);
      }
      const payload = await probeHealth(this.healthUrl(port));
      if (isOurRuntime(payload, root)) return;
      await new Promise((resolve) => setTimeout(resolve, 400));
    }
    this.stopOwned();
    throw new Error(`reca-director runtime did not become healthy. See ${logPath}`);
  }

  stopOwned() {
    if (this.child && this.child.exitCode == null) {
      try {
        this.child.kill("SIGTERM");
      } catch {
        // ignore
      }
    }
    this.child = null;
    if (this.logFd != null) {
      try {
        closeSync(this.logFd);
      } catch {
        // ignore
      }
      this.logFd = null;
    }
  }
}
