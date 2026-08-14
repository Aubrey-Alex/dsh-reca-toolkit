import { defineTool } from "@deepseek-ai/dsh-tools";
import { renderJson } from "../renderers/json.js";

export function registerCreateVideo(ctx, client) {
  return ctx.tools.register(defineTool({
    name: "reca_create_video",
    description:
      "Create a coherent long-form video from a natural-language story. " +
      "ReCA handles planning, rendering, audit, repair, resume, and delivery asynchronously.",
    parameters: {
      story: { type: "string", required: true, description: "The story or video idea." },
      duration: { type: "number", required: false, description: "Target duration in seconds." },
      resolution: { type: "string", required: false, description: "Target resolution, for example 1280x720." },
      style: { type: "string", required: false, description: "Overall visual style." },
      aspect_ratio: { type: "string", required: false, description: "Target aspect ratio, for example 16:9." },
      backend: { type: "string", required: false, description: "Video backend, normally wan." },
      enable_audit: { type: "boolean", required: false, description: "Run visual audit and repair." },
      validate_segments: { type: "boolean", required: false, description: "Run segment-level validation." },
      seed: { type: "number", required: false, description: "Reproducible render seed." },
    },
    output: { schema: { type: "object" }, render: (_args, value) => renderJson(value) },
    async execute(args) {
      return client.createVideo({
        story: args.story,
        options: {
          duration: args.duration,
          resolution: args.resolution || "1280x720",
          style: args.style || "cinematic",
          aspect_ratio: args.aspect_ratio || "16:9",
          backend: args.backend || "wan",
          enable_audit: args.enable_audit ?? true,
          validate_segments: args.validate_segments ?? false,
          seed: Number.isFinite(args.seed) ? args.seed : 0,
        },
      });
    },
  }));
}
