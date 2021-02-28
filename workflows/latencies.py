from pipe import *


def workflow(wf):
    wf.name = "Example Latencies"
    wf.new_subgraph("Some Work").serial(
        ("OK Step1", NormalGenerator(10, 2)),
        ("OK Step2", NormalGenerator(8, 2)),
    )
    wf.new_subgraph("Safely Random").serial(
        ("Roll 1", UniformGenerator(10, 20)),
        ("Roll 2", UniformGenerator(20, 40)),
    )
    wf.new_subgraph("Flaky Latency").serial(("Erratic", BernoulliGenerator(0.1, 100)))
    wf.new_subgraph("Troublesome").serial(
        ("Failing Step", FailingGenerator(failure_rate=0.1))
    )
    return wf
