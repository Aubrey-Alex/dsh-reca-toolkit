import { defineTool } from "@deepseek-ai/dsh-tools";
import { renderJson } from "../renderers/json.js";

export function registerResumeRun(ctx, client) {
  return ctx.tools.register(defineTool({
    name: "reca_resume",
    description: "Resume a failed, cancelled, or interrupted ReCA Director run from its persisted run directory.",
    parameters: { run_id: { type: "string", required: true, description: "The run id to resume." } },
    output: { schema: { type: "object" }, render: (_args, value) => renderJson(value) },
    async execute(args) {
      return client.resume(args.run_id);
    },
  }));
}
