import { defineTool } from "@deepseek-ai/dsh-tools";
import { renderJson } from "../renderers/json.js";

export function registerGetArtifact(ctx, client) {
  return ctx.tools.register(defineTool({
    name: "reca_get_artifact",
    description: "Get the published artifact manifest or a named ReCA artifact URL.",
    parameters: {
      run_id: { type: "string", required: true, description: "The ReCA run id." },
      path: { type: "string", required: false, description: "Optional manifest path such as run/final.mp4." },
    },
    output: { schema: { type: "object" }, render: (_args, value) => renderJson(value) },
    async execute(args) {
      return client.getArtifact(args.run_id, args.path || "");
    },
  }));
}
