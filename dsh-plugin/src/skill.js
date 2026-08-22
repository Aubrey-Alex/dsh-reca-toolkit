export const directorSkill = {
  name: "reca-director",
  description: "Professional long-form video creation through the ReCA engine.",
  whenToUse: "Use when a user wants to create, revise, monitor, cancel, resume, or retrieve a long-form video.",
  source: "runtime",
  content: `# ReCA Director

Use reca_create_video for a natural-language video request. Ask only for
missing information that materially changes the result; otherwise use sensible
cinematic defaults and start the asynchronous run.

Return the run_id for long jobs and use reca_get_status for progress. Present
the user-facing stage and progress without exposing provider implementation
details. Report video_state and audit_state separately: a generated video is
not automatically an audited video.

Use reca_resume for failed, cancelled, or interrupted runs, reca_cancel for
stop requests, reca_list_runs to find prior runs, and reca_get_artifact for the
final video, plan, audit report, contact sheet, and run report.

Keep ReCA planning, segment decomposition, provider calls, repair, retry, and
concatenation inside ReCA. Long-form duration is supported; pass duration when
the user asks for a longer film. Do not tell the user to start a Gateway or a
helper script. Do not manually split shots or call providers from the DSH
agent loop.`,
};

export function registerDirectorSkill() {
  return directorSkill;
}
