from pathlib import Path

from dynamicprompts.generators import CombinatorialPromptGenerator, RandomPromptGenerator
from dynamicprompts.wildcards import WildcardManager

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import InputField, UIComponent
from invokeai.app.invocations.primitives import StringCollectionOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.dynamicprompts import find_missing_wildcards


def _resolve_wildcard_dir(context: InvocationContext, wildcard_dir: str) -> Path:
    """Resolve `wildcard_dir`, treating a relative path as relative to the InvokeAI root.

    An empty value falls back to the app's configured `wildcards_dir`, which is also what the linear UI's
    prompt box uses — so the two stay in sync when that setting is changed in `invokeai.yaml`.
    """
    config = context.config.get()
    if not wildcard_dir.strip():
        return config.wildcards_path
    path = Path(wildcard_dir).expanduser()
    if not path.is_absolute():
        path = config.root_path / path
    return path.resolve()


def _available_wildcards(wildcard_manager: WildcardManager, limit: int = 30) -> str:
    """A short, human-readable list of the wildcard names the manager can see."""
    names = sorted(wildcard_manager.get_collection_names())
    if not names:
        return "none"
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown}, ... ({len(names)} total)"


@invocation(
    "wildcard_prompt",
    title="Wildcard Prompt",
    tags=["prompt", "collection", "wildcard", "dynamicprompts"],
    category="prompt",
    version="1.0.0",
    use_cache=False,
)
class WildcardPromptInvocation(BaseInvocation):
    """Parses a prompt with dynamicprompts, resolving `__wildcard__` references against a folder of .txt files.

    This is the stock Dynamic Prompt node plus a wildcard folder: a reference like `__poses__` is replaced with a
    line from `<wildcard_dir>/poses.txt`. Subfolders work too (`__clothing/dresses__`). Variant syntax such as
    `{a|b|c}` behaves exactly as it does in the stock node.
    """

    prompt: str = InputField(
        description="The prompt to parse. Use __name__ to pull a line from <wildcard_dir>/name.txt",
        ui_component=UIComponent.Textarea,
    )
    wildcard_dir: str = InputField(
        default="",
        description="Folder of .txt wildcard files. Blank uses the configured wildcards_dir. "
        "Relative paths resolve against the InvokeAI root.",
    )
    max_prompts: int = InputField(default=1, ge=1, description="The number of prompts to generate")
    combinatorial: bool = InputField(
        default=False,
        description="Generate every combination in order instead of sampling randomly",
    )
    seed: int = InputField(
        default=0,
        description="Seed for random generation. 0 uses a fresh random seed each run. Ignored if combinatorial.",
    )

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        wildcard_path = _resolve_wildcard_dir(context, self.wildcard_dir)
        if not wildcard_path.is_dir():
            raise ValueError(f"Wildcard directory not found: {wildcard_path}")

        wildcard_manager = WildcardManager(wildcard_path)

        if self.combinatorial:
            # An unknown wildcard used as a variant value sends the combinatorial generator into an infinite
            # loop, so fail fast with a clear message instead of hanging the invocation. The random generator
            # handles unknown wildcards gracefully and needs no guard.
            missing_wildcards = find_missing_wildcards(self.prompt, wildcard_manager)
            if missing_wildcards:
                raise ValueError(
                    f"No values found for wildcard(s): {', '.join(missing_wildcards)}. "
                    f"Searched {wildcard_path}; available: {_available_wildcards(wildcard_manager)}"
                )
            prompts = CombinatorialPromptGenerator(wildcard_manager=wildcard_manager).generate(
                self.prompt, max_prompts=self.max_prompts
            )
        else:
            generator = RandomPromptGenerator(wildcard_manager=wildcard_manager, seed=self.seed or None)
            prompts = generator.generate(self.prompt, num_images=self.max_prompts)

        return StringCollectionOutput(collection=prompts)
