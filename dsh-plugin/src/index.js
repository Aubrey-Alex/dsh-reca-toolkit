import { defineTool } from "@deepseek-ai/dsh-tools";
import { RecaClient } from "./client.js";

export const name = "dsh-reca-toolkit";
export const inject = ["tools"];

function asText(value) {
  return [{ type: "text", text: JSON.stringify(value, null, 2) }];
}

export async function apply(ctx, config = {}) {
  if (!ctx?.tools || typeof ctx.tools.register !== "function") {
    throw new Error("dsh-reca-toolkit requires the DSH tools service");
  }
  const client = new RecaClient(config.gatewayUrl);
  const disposers = [];

  disposers.push(ctx.tools.register(defineTool({
    name: "reca_start",
    description:
      "Start a ReCA long-form video generation run. The run is asynchronous; " +
      "return the run_id to the user and query reca_status for progress.",
    parameters: {
      story: {
        type: "string",
        required: true,
        description: "The narrative or shot idea to turn into a coherent video.",
      },
      backend: {
        type: "string",
        required: false,
        description: "ReCA video backend branch, normally wan.",
      },
      resolution: {
        type: "string",
        required: false,
        description: "Target video resolution, for example 1280x720.",
      },
      seed: {
        type: "number",
        required: false,
        description: "Render seed used for reproducible runs.",
      },
      validate: {
        type: "boolean",
        required: false,
        description: "Enable ReCA anchor validation and repair.",
      },
      validate_segments: {
        type: "boolean",
        required: false,
        description: "Enable ReCA segment-level visual validation and repair.",
      },
      resume_run_id: {
        type: "string",
        required: false,
        description: "Reuse a previous failed or cancelled run directory when resuming.",
      },
    },
    output: {
      schema: { type: "object" },
      render: (_args, value) => asText(value),
    },
    async execute(args) {
      return client.start({
        story: args.story,
        options: {
          backend: args.backend || "wan",
          resolution: args.resolution || "1280x720",
          seed: Number.isFinite(args.seed) ? args.seed : 0,
          validate: args.validate ?? true,
          validate_segments: args.validate_segments ?? false,
          ...(args.resume_run_id ? { resume_run_id: args.resume_run_id } : {}),
        },
      });
    },
  })));

  disposers.push(ctx.tools.register(defineTool({
    name: "reca_status",
    description: "Query the progress, logs, stages, and final artifact of a ReCA run.",
    parameters: {
      run_id: { type: "string", required: true, description: "The run_id returned by reca_start." },
    },
    output: {
      schema: { type: "object" },
      render: (_args, value) => asText(value),
    },
    async execute(args) {
      return client.status(args.run_id);
    },
  })));

  disposers.push(ctx.tools.register(defineTool({
    name: "reca_cancel",
    description: "Cancel an active ReCA run and stop its child process.",
    parameters: {
      run_id: { type: "string", required: true, description: "The run_id to cancel." },
    },
    output: {
      schema: { type: "object" },
      render: (_args, value) => asText(value),
    },
    async execute(args) {
      return client.cancel(args.run_id);
    },
  })));

  if (ctx.logger?.info) ctx.logger.info("%s ready: ReCA tools are available", name);
  return () => disposers.forEach((dispose) => {
    if (typeof dispose === "function") dispose();
  });
}
