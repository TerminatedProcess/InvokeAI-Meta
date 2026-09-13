const hasOpenCloseCurlyBracesRegex = /.*\{[\s\S]*\}.*/;
// `__name__` pulls a line from a wildcard file. Without this check a prompt using only wildcards is
// never sent to the backend for expansion, so the references survive into the generated prompt as
// literal text. A false positive (prose containing `foo__bar__baz`) only costs one request that
// returns the prompt unchanged, so this errs on the side of matching.
const hasWildcardRegex = /__[^\n]+__/;

export const getShouldProcessPrompt = (prompt: string): boolean =>
  hasOpenCloseCurlyBracesRegex.test(prompt) || hasWildcardRegex.test(prompt);
