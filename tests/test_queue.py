import asyncio

import pytest

from coralapi.runtime.queue import QueueFull, TPUExecutor


def test_reserve_enforces_capacity():
    executor = TPUExecutor(queue_depth=1)
    executor.reserve()  # running
    executor.reserve()  # waiting
    with pytest.raises(QueueFull):
        executor.reserve()


def test_submit_serializes_and_releases_slots():
    async def main():
        executor = TPUExecutor(queue_depth=2)
        running = 0
        peak = 0

        def work():
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            running -= 1
            return "done"

        results = await asyncio.gather(*(executor.submit(work) for _ in range(3)))
        assert results == ["done"] * 3
        assert peak == 1  # the TPU lock never admits two at once
        assert executor.pending == 0

    asyncio.run(main())
