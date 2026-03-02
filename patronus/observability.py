from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator

from dotenv import load_dotenv

_langfuse_client: Any | None = None
_initialized: bool = False


def _get_langfuse() -> Any | None:
    global _langfuse_client, _initialized
    if _initialized:
        return _langfuse_client
    _initialized = True
    load_dotenv()
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import Langfuse
        _langfuse_client = Langfuse()
    except ImportError:
        pass
    return _langfuse_client


class _NoopObs:
    def update(self, **kwargs: Any) -> None:
        pass


@contextmanager
def pipeline_run(
    name: str,
    input: dict[str, Any],
) -> Generator[_NoopObs | Any, None, None]:
    lf = _get_langfuse()
    if lf is None:
        yield _NoopObs()
        return
    with lf.start_as_current_observation(name=name, as_type="trace", input=input) as obs:
        try:
            yield obs
        finally:
            lf.flush()


@contextmanager
def agent_run(
    name: str,
    input: dict[str, Any],
) -> Generator[_NoopObs | Any, None, None]:
    lf = _get_langfuse()
    if lf is None:
        yield _NoopObs()
        return
    with lf.start_as_current_observation(name=name, as_type="agent", input=input) as obs:
        try:
            yield obs
        finally:
            lf.flush()


@contextmanager
def iteration_span(
    name: str,
    input: dict[str, Any] | None = None,
) -> Generator[_NoopObs | Any, None, None]:
    lf = _get_langfuse()
    if lf is None:
        yield _NoopObs()
        return
    with lf.start_as_current_observation(
        name=name,
        as_type="span",
        input=input,
    ) as obs:
        yield obs


@contextmanager
def llm_generation(
    name: str,
    model: str,
    input: Any,
) -> Generator[_NoopObs | Any, None, None]:
    lf = _get_langfuse()
    if lf is None:
        yield _NoopObs()
        return
    with lf.start_as_current_observation(
        name=name,
        as_type="generation",
        model=model,
        input=input,
    ) as obs:
        yield obs


@contextmanager
def planning_generation(
    name: str,
    model: str,
    input: Any,
) -> Generator[_NoopObs | Any, None, None]:
    lf = _get_langfuse()
    if lf is None:
        yield _NoopObs()
        return
    with lf.start_as_current_observation(
        name=name,
        as_type="generation",
        model=model,
        input=input,
        metadata={"phase": "planning"},
    ) as obs:
        yield obs


@contextmanager
def tool_call(
    name: str,
    input: dict[str, Any],
) -> Generator[_NoopObs | Any, None, None]:
    lf = _get_langfuse()
    if lf is None:
        yield _NoopObs()
        return
    with lf.start_as_current_observation(
        name=name,
        as_type="tool",
        input=input,
    ) as obs:
        yield obs
