from pipe import *


def workflow(wf):
    wf.name = "Project 1"

    userwf = wf.new_subgraph("alias1@")

    userwf.serial(
        ("DesignDoc", PERTTask(4, complete=True)),
        ("IDL", PERTTask(4, 5, 8)),
    )
    userwf.new_subgraph("Development").parallel(
        {
            "API1": PERTTask(3),
            "API2": PERTTask(3),
            "API3": PERTTask(5),
        }
    )
    userwf.new_subgraph("Release").serial(
        ("Metrics", PERTTask(3)),
        ("Alarms", PERTTask(3)),
        ("Runbooks", PERTTask(2)),
    )

    wf.new_subgraph("alias2@").serial(("Documentation", FixedTask(7)))
    return wf