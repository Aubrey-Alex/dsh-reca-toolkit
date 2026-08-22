import { defineTool } from "@deepseek-ai/dsh-tools";
import { RecaClient } from "./client.js";
import { RecaRuntime } from "./runtime.js";
import { renderJson } from "./renderers/json.js";
import { registerDirectorSkill } from "./skill.js";
import { registerCancelRun } from "./tools/cancel-run.js";
import { registerCreateVideo } from "./tools/create-video.js";
import { registerGetArtifact } from "./tools/get-artifact.js";
import { registerGetStatus } from "./tools/get-status.js";
import { registerListRuns } from "./tools/list-runs.js";
import { registerResumeRun } from "./tools/resume-run.js";

export const name = "dsh-reca-toolkit";
export const inject = ["tools", "skills"];

function registerLegacyStart(ctx, client) {
  return ctx.tools.register(defineTool({
    name: "reca_start",
    description: "Compatibility alias for reca_create_video.",
    parameters: {
      story: { type: "string", required: true },
      backend: { type: "string" },
      resolution: { type: "string" },
      seed: { type: "number" },
      validate: { type: "boolean" },
      validate_segments: { type: "boolean" },
      resume_run_id: { type: "string" },
    },
    output: { schema: { type: "object", additionalProperties: true }, render: (_args, value) => renderJson(value) },
    async execute(args) {
      return client.start({
        story: args.story,
        options: {
          backend: args.backend || "wan",
          resolution: args.resolution || "1280x720",
          seed: Number.isFinite(args.seed) ? args.seed : 0,
          enable_audit: args.validate ?? true,
          validate_segments: args.validate_segments ?? false,
          ...(args.resume_run_id ? { resume_run_id: args.resume_run_id } : {}),
        },
      });
    },
  }));
}

export async function apply(ctx, config = {}) {
  if (!ctx?.tools || typeof ctx.tools.register !== "function") {
    throw new Error("dsh-reca-toolkit requires the DSH tools service");
  }
  if (!ctx?.skills || typeof ctx.skills.register !== "function") {
    throw new Error("dsh-reca-toolkit requires the DSH skills service");
  }
  const runtime = new RecaRuntime({
    host: config.gatewayHost,
    port: config.gatewayPort,
  });
  const client = new RecaClient(config.gatewayUrl, runtime);
  runtime.ensure().catch((err) => {
    if (ctx.logger?.warn) ctx.logger.warn("%s runtime: %s", name, err.message);
  });
  const disposers = [
    registerCreateVideo(ctx, client),
    registerGetStatus(ctx, client),
    registerCancelRun(ctx, client),
    registerResumeRun(ctx, client),
    registerListRuns(ctx, client),
    registerGetArtifact(ctx, client),
    // Compatibility aliases kept until the Director UI migration is complete.
    registerLegacyStart(ctx, client),
    registerGetStatus(ctx, client, "reca_status"),
  ];
  disposers.push(ctx.skills.register(registerDirectorSkill()));
  if (ctx.logger?.info) ctx.logger.info("%s ready: ReCA Director tools are available", name);
  return () => {
    runtime.stopOwned();
    disposers.forEach((dispose) => {
      if (typeof dispose === "function") dispose();
    });
  };
}
