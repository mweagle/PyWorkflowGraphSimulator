from pipe import *
import sys, getopt
import importlib
import os

################################################################################
def main(tasks_filepath, output_format):
    tasks_module = importlib.import_module(tasks_filepath)
    file_base_name = os.path.basename(tasks_filepath)

    # Stitch everything together
    wf = WorkflowGraph(
        name=file_base_name,
        show_end_date=False,
        percentiles_to_compute=JOIN_PERCENTILES,
    )
    tasks_graph = tasks_module.workflow(wf)
    tasks_graph.resolve()
    tasks_graph.graph.evaluate(mode="linear")
    tasks_graph.as_dot(output_format=output_format)


################################################################################
# MAIN
################################################################################

################################################################################
# MAIN
################################################################################
if __name__ == "__main__":
    output_format = "jpeg"
    task_file = ""

    # Options
    short_options = "i:o:"
    # Long options
    long_options = ["input=", "output="]

    # checking each argument
    argv = sys.argv[1:]
    arguments, values = getopt.getopt(argv, short_options, long_options)
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-o", "--output"):
            output_format = currentValue
        elif currentArgument in ("-i", "--input"):
            task_file = currentValue

    ############################################################################
    main(task_file, output_format)
