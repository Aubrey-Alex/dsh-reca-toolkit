import { defineTool } from "@deepseek-ai/dsh-tools";
import { renderJson } from "../renderers/json.js";

export function registerListRuns(ctx, client) {
  return ctx.tools.register(defineTool({
    name: "reca_list_runs",
    description: "List ReCA Director runs and their current lifecycle state.",
    parameters: {},
    output: { schema: { type: "object", additionalProperties: true }, render: (_args, value) => renderJson(value) },
    async execute() {
      return client.listRuns();
    },
  }));
}
