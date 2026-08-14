const DEFAULT_GATEWAY_URL = "http://127.0.0.1:8787";

export class RecaClient {
  constructor(baseUrl = process.env.RECA_GATEWAY_URL || DEFAULT_GATEWAY_URL) {
    this.baseUrl = String(baseUrl).replace(/\/+$/, "");
  }

  async request(path, options = {}) {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...options,
      headers: {
        "content-type": "application/json",
        ...(options.headers || {}),
      },
    });
    const text = await response.text();
    let payload;
    try {
      payload = text ? JSON.parse(text) : {};
    } catch {
      payload = { raw: text };
    }
    if (!response.ok) {
      const message = payload?.error || `Gateway returned HTTP ${response.status}`;
      throw new Error(message);
    }
    return payload;
  }

  start(input) {
    return this.request("/v1/runs", {
      method: "POST",
      body: JSON.stringify(input),
    });
  }

  createVideo(input) {
    return this.start(input);
  }

  status(runId) {
    return this.request(`/v1/runs/${encodeURIComponent(runId)}`);
  }

  cancel(runId) {
    return this.request(`/v1/runs/${encodeURIComponent(runId)}/cancel`, {
      method: "POST",
      body: "{}",
    });
  }

  resume(runId) {
    return this.request(`/v1/runs/${encodeURIComponent(runId)}/resume`, {
      method: "POST",
      body: "{}",
    });
  }

  listRuns() {
    return this.request("/v1/runs");
  }

  getArtifact(runId, relativePath = "") {
    return this.request(`/v1/runs/${encodeURIComponent(runId)}/artifacts`).then((manifest) => {
      if (!relativePath) return manifest;
      const wanted = relativePath.replace(/^\/+/, "");
      const item = (manifest.artifacts || []).find((entry) => entry.path === wanted);
      return item || { error: "artifact not found", path: wanted };
    });
  }
}
