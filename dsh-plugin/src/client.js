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

  status(runId) {
    return this.request(`/v1/runs/${encodeURIComponent(runId)}`);
  }

  cancel(runId) {
    return this.request(`/v1/runs/${encodeURIComponent(runId)}/cancel`, {
      method: "POST",
      body: "{}",
    });
  }
}
