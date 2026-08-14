import { defineTool } from "@deepseek-ai/dsh-tools";
import { renderJson } from "../renderers/json.js";

export function registerGetStatus(ctx, client, name = "reca_get_status") {
  return ctx.tools.register(defineTool({
    name,
    description: "Query a ReCA Director run, including ReCA stage, audit state, logs, and artifacts.",
    parameters: { run_id: { type: "string", required: true, description: "The ReCA run id." } },
    output: { schema: { type: "object" }, render: (_args, value) => renderJson(value) },
    async execute(args) {
      return client.status(args.run_id);
    },
  }));
}
