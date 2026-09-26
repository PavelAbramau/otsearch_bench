import pytest

from otsearch_bench.env import Mode, OTAdapter
from otsearch_bench.search import Runner, Trajectory
from otsearch_bench.search.stubs import stub_agent

pytestmark = pytest.mark.integration


def test_stub_agent_against_live_api(tmp_path):
    with OTAdapter(tmp_path / "ot.sqlite", mode=Mode.LIVE) as ot:
        traj = Runner(ot).run(stub_agent(), "What evidence links EGFR to NSCLC?")
    assert traj.stopping.stop and len(traj.steps) == 4
    assert traj.total_usage.calls == 3 and not traj.answer.no_evidence
    assert Trajectory.from_jsonl(traj.to_jsonl(tmp_path)) == traj
