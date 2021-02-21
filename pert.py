from pipe import *

################################################################################
def main():
    workflow = WorkflowGraph(name="Example Work", show_end_date=True)
    workflow.new_subgraph("Some Work").serial(
        ("OK Step1", PERTTask(4, complete=True)),
        ("OK Step2", PERTTask(4)),
    )
    workflow.new_subgraph("Another Stream").serial(
        ("Item 1", PERTTask(4)),
        ("Item 2", PERTTask(4)),
    )

    # Stitch everything together
    workflow.resolve()
    workflow.graph.evaluate(mode="linear")
    graphviz_output = workflow.as_dot()


################################################################################
# MAIN
################################################################################
if __name__ == "__main__":
    main()