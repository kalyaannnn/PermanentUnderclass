from dataclasses import replace

import pytest

from lagrl.sandbox.gvisor import GVisorExecutor
from lagrl.sandbox.records import IsolationUnavailable
from lagrl.testing.fixtures import RecordedSandbox

pytestmark = pytest.mark.isolation


async def test_unavailable_isolation_fails_explicitly() -> None:
    fixture = RecordedSandbox()
    executor = GVisorExecutor(runtime_path="/intentionally-unavailable/runsc")
    with pytest.raises(IsolationUnavailable):
        await executor.run(fixture.session, fixture.execute_request)


async def test_foreign_workspace_cannot_be_released() -> None:
    fixture = RecordedSandbox()
    session = replace(fixture.session, workspace_ref="unowned://workspace")
    with pytest.raises(ValueError):
        await GVisorExecutor().release(session)
