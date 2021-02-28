# PyWorkflowGraphSimulator

TODO

See _pert.py_ and _tasks.py_ for example usage for now...

## Input

```python

# File: tasks.py
################################################################################
def main():
    workflow = WorkflowGraph(name="Example Workflow", run_count=100000)
    workflow.new_subgraph("Some Work").serial(
        ("OK Step1", NormalGenerator(10, 2)),
        ("OK Step2", NormalGenerator(8, 2)),
    )
    workflow.new_subgraph("Safely Random").serial(
        ("Roll 1", UniformGenerator(10, 20)),
        ("Roll 2", UniformGenerator(20, 40)),
    )
    workflow.new_subgraph("Flaky Latency").serial(
        ("Erratic", BernoulliGenerator(0.1, 100))
    )
    workflow.new_subgraph("Troublesome").serial(
        ("Failing Step", FailingGenerator(failure_rate=0.1))
    )

    # Stitch everything together
    workflow.resolve()
    workflow.graph.evaluate(mode="linear")
    graphviz_output = workflow.as_dot()

```

## Execution

### PERT Schedule

```shell
> python pert.py --input=workflows.project --output=svg
```

### Latency Schedule

```shell
> python tasks.py --input=workflows.latencies --output=svg
```

## Output

![Example Workflow](https://raw.githubusercontent.com/mweagle/PyWorkflowGraphSimulator/master/site/Example_Latencies.gv.svg)
