export function renderJson(value) {
  return [{ type: "text", text: JSON.stringify(value, null, 2) }];
}
