# Tools

The plugin registers these DSH tools:

- `reca_create_video`: submit a natural-language story and RunConfig.
- `reca_get_status`: return Gateway state plus ReCA stage, video state, audit state, and manifest.
- `reca_cancel`: request SIGTERM, then SIGKILL after the configured grace period.
- `reca_resume`: resume an interrupted, cancelled, or failed run from its run directory.
- `reca_list_runs`: list persisted runs.
- `reca_get_artifact`: return the manifest or a complete artifact URL.

`reca_create_video` accepts optional visual inputs in addition to the story:

```json
{
  "story": "...",
  "first_frame": "/path/to/start.png",
  "reference_images": [
    {"path": "/path/to/hero.png", "role": "character", "name": "hero"},
    {"url": "https://example.com/scene.png", "role": "scene", "name": "battlefield"}
  ]
}
```

The ReCA-compatible aliases `first_url` and `reference_image_urls` are also
accepted. If both names are present, the product names (`first_frame` and
`reference_images`) take precedence.

The Gateway copies local images into the isolated run directory and records an
`input_manifest.json`. It never puts image bytes or provider credentials into
the DSH model context. `first_frame` is optional; without it, ReCA generates
the first anchor as before. Reference-image forwarding follows the ReCA
planner and serial segment contract. The Wan3.0 adapter maps I2V to one
reference image. For R2V it sends the current frame as `reference_image[0]`,
then appends up to three planner-selected references and prefixes the prompt to
make the first image's temporal role explicit. This is a soft start constraint;
the deployed Wan3.0 API does not accept a hard `first_frame` together with
additional `reference_image` entries. Bridge requests still use the real
first/last-frame pair.

`reca_start` and `reca_status` remain compatibility aliases.
