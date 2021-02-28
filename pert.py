from pipe import *
import sys, getopt
import importlib
import os

################################################################################
def main(pert_filepath, output_format):
    pert_module = importlib.import_module(pert_file)
    file_base_name = os.path.basename(pert_file)
    # Stitch everything together
    wf = WorkflowGraph(
        name=file_base_name,
        show_end_date=True,
        percentiles_to_compute=PERT_JOIN_PERCENTILES,
    )
    pert_graph = pert_module.workflow(wf)
    pert_graph.resolve()
    pert_graph.graph.evaluate(mode="linear")
    pert_graph.as_dot(output_format=output_format)


################################################################################
# MAIN
################################################################################
if __name__ == "__main__":

    output_format = "jpeg"
    pert_file = "workflows.project"

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
            pert_file = currentValue

    ############################################################################
    main(pert_file, output_format)
