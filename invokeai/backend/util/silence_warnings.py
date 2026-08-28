import warnings
from contextlib import ContextDecorator

from diffusers.utils import logging as diffusers_logging
from transformers import logging as transformers_logging


# Inherit from ContextDecorator to allow using SilenceWarnings as both a context manager and a decorator.
class SilenceWarnings(ContextDecorator):
    """A context manager that disables warnings from transformers & diffusers modules while active.

    As context manager:
    ```
    with SilenceWarnings():
        # do something
    ```

    As decorator:
    ```
    @SilenceWarnings()
    def some_function():
        # do something
    ```
    """

    def __enter__(self) -> None:
        self._transformers_verbosity = transformers_logging.get_verbosity()
        self._diffusers_verbosity = diffusers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        diffusers_logging.set_verbosity_error()
        # catch_warnings snapshots the filter list so __exit__ can put back exactly what was
        # there. A bare simplefilter("default") on exit would not restore state — it prepends a
        # catch-all "default" entry, un-suppressing every DeprecationWarning for the rest of the
        # process (and overriding any -W / PYTHONWARNINGS the user set).
        self._warnings_ctx = warnings.catch_warnings()
        self._warnings_ctx.__enter__()
        warnings.simplefilter("ignore")

    def __exit__(self, *args) -> None:
        transformers_logging.set_verbosity(self._transformers_verbosity)
        diffusers_logging.set_verbosity(self._diffusers_verbosity)
        self._warnings_ctx.__exit__(*args)
