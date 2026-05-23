"""Small async helpers used by tools and workflows."""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


async def to_async(fn: Callable[..., T], *args, **kwargs) -> T:
    """Run a sync function in a thread; pass through a coroutine if given."""
    result = fn(*args, **kwargs)
    if inspect.iscoroutine(result):
        return await result  # type: ignore[no-any-return]
    if isinstance(result, asyncio.Future):
        return await result  # type: ignore[no-any-return]
    if callable(getattr(fn, "__await__", None)):
        return await result  # type: ignore[no-any-return]
    return result


def maybe_await(x: Any) -> Awaitable[Any]:
    """Wrap a value or coroutine in something awaitable."""
    if inspect.isawaitable(x):
        return x

    async def _wrap():
        return x

    return _wrap()