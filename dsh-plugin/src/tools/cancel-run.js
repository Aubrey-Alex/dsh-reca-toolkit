import { defineTool } from "@deepseek-ai/dsh-tools";
import { renderJson } from "../renderers/json.js";

export function registerCancelRun(ctx, client, name = "reca_cancel") {
  return ctx.tools.register(defineTool({
    name,
    description: "Cancel an active ReCA Director run.",
    parameters: { run_id: { type: "string", required: true, description: "The ReCA run id." } },
    output: { schema: { type: "object" }, render: (_args, value) => renderJson(value) },
    async execute(args) {
      return client.cancel(args.run_id);
    },
  }));
}
