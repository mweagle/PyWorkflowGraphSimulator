from flowpipe import Graph, INode, Node, InputPlug, OutputPlug
import numpy
from networkdays import networkdays
from numpy.random import Generator, PCG64
import matplotlib.pyplot as plt
import sys
import uuid
import logging
from abc import ABC, abstractmethod
from graphviz import Digraph
import base64
from networkdays import networkdays
import datetime
import math

LATENCIES = "latencies"
LATENCY_DICT = "latencydict"
SIMULATED_RESULTS_OUTPUT_FILENAME = "results.png"

# Color scheme to use
DOT_COLOR_SCHEME = "spectral"
DOT_COLOR_SCHEME_COUNT = 11

OPERATION_FAILED_LATENCY = sys.maxsize
JOIN_PERCENTILES = [50, 99, 99.5, 100]

logging.basicConfig(format=logging.BASIC_FORMAT)
logger = logging.getLogger("flowpipe")
logger.setLevel(logging.INFO)


def max_latency(dict_of_latency):
    currentUpper = None
    for eachKey in dict_of_latency.keys():
        eachLatency = dict_of_latency[eachKey]
        if currentUpper is not None:
            currentUpper = numpy.maximum(currentUpper, eachLatency)
        else:
            currentUpper = eachLatency
    return currentUpper


def classname_for_object(some_object):
    classname = str(type(some_object)).split("'")[1]
    barename = classname.split(".").pop()
    return barename


def duration_percentiles(array_of_durations):
    def split(arr, cond):
        return [arr[cond], arr[~cond]]

    # Only compute percentiles for successful operations
    valid, failed = split(
        array_of_durations, array_of_durations < OPERATION_FAILED_LATENCY
    )
    return numpy.percentile(valid, JOIN_PERCENTILES)


def formatted_float(in_value):
    float_value = "{0:.3f}".format(in_value)
    if in_value == math.floor(in_value):
        float_value = "{0:.0f}".format(in_value)
    return float_value


def graphviz_percentiles_label(percentile_values, show_end_dates):

    percentile_stats = []

    for num, percent_val in enumerate(JOIN_PERCENTILES):
        # Handle the case where it's already a floor
        p_value = formatted_float(percentile_values[num])
        percentile_entry = "p{0}={1}".format(percent_val, p_value)

        if show_end_dates:
            try:
                if math.floor(percentile_values[num]) > 0:
                    end_project_date = networkdays.JobSchedule(
                        math.floor(percentile_values[num]),
                        1,
                        datetime.date.today(),
                        networkdays=None,
                    )
                    percentile_entry = "ECD: {0} ({1})".format(
                        end_project_date.prj_ends, percentile_entry
                    )
                else:
                    percentile_entry = "Done: {0} ({1})".format(
                        datetime.date.today(), percentile_entry
                    )

            except Exception as ex:
                logger.error(
                    "Checking percentile: {0} - {1}".format(ex, percentile_values[num])
                )
                percentile_entry = "p{0}={1}".format(percent_val, p_value)

        percentile_stats.append(
            percentile_entry
            # "p{0}={1}".format(percent_val, p_value),
        )

    return "\\n".join(percentile_stats)


################################################################################
#
# GENERATORS
#
################################################################################


################################################################################
# DurationGenerator
# ABC duration generator
################################################################################
class DurationGenerator(ABC):
    def __init__(self):
        self.only_integers = False
        super().__init__()

    def integral(self, only_integers):
        self.only_integers = only_integers
        return self

    def transform_values(self, durations):
        if self.only_integers:
            durations = numpy.ceil(durations)

        return durations

    @abstractmethod
    def generate(self, count):
        pass

    def graphviz_node_attrs(self):
        return self.__dict__

    def graphviz_node(self, node_id, step_name, parent_graph):
        this_barename = classname_for_object(self)
        generator_params = []
        this_generator_attrs = self.graphviz_node_attrs()
        for attr, value in this_generator_attrs.items():
            if attr == "only_integers":
                if value:
                    generator_params.append("{0}={1}".format(attr, value))
            else:
                generator_params.append("{0}={1}".format(attr, value))

        parent_graph.node(
            node_id,
            "{{{0} ({1})|{2}}}".format(
                step_name, this_barename, "\\n".join(generator_params)
            ),
            {"fontsize": "8"},
        )


################################################################################
# CompletableTask
# A completable task is something that has already completed
################################################################################
class CompletableTask(DurationGenerator):
    def __init__(self, generator, complete=False):
        super(CompletableTask, self).__init__()
        self.generator = generator
        self.complete = complete
        self.notes = None

    def graphviz_node_attrs(self):
        this_task_attrs = {
            "complete": "✅" if self.complete else "🔜",
            "generator": classname_for_object(self.generator),
        } | self.generator.__dict__

        return this_task_attrs

    def generate(self, count):
        if self.complete:
            return numpy.zeros(count)

        return self.generator.generate(count)


################################################################################
# PERTTask
# returns a CompletableTask with the given TriangularGenerator and using integral
# values only
################################################################################


class PERTTask(CompletableTask):
    def __init__(self, lower_bound, mode=None, upper_bound=None, complete=False):
        if upper_bound is None:
            upper_bound = 2 * lower_bound
        if mode is None:
            mode = math.ceil(lower_bound * 1.66)

        super(PERTTask, self).__init__(
            generator=TriangularGenerator(lower_bound, mode, upper_bound).integral(
                True
            ),
            complete=complete,
        )
        self.notes = None

    def graphviz_node_attrs(self):
        this_task_attrs = {
            "complete": "✅" if self.complete else "🔜",
            "pert": "{0}:{1}:{2}".format(
                self.generator.lower_bound,
                self.generator.mode,
                self.generator.upper_bound,
            ),
        }
        return this_task_attrs


################################################################################
# TriangularGenerator
# Generates a triangular distribution
################################################################################
class TriangularGenerator(DurationGenerator):
    def __init__(self, lower_bound, mode, upper_bound):
        super(TriangularGenerator, self).__init__()
        self.lower_bound = lower_bound
        self.mode = mode
        self.upper_bound = upper_bound

    def generate(self, count):
        dist = numpy.random.default_rng().triangular(
            self.lower_bound, self.mode, self.upper_bound, count
        )
        dist[dist < 0] = 0
        return self.transform_values(dist)


################################################################################
# NormalGenerator
# Normal distribution generator, clamped above zero
################################################################################
class NormalGenerator(DurationGenerator):
    def __init__(self, mean, std_dev):
        super(NormalGenerator, self).__init__()
        self.mean = mean
        self.std_dev = std_dev

    def generate(self, count):
        dist = numpy.random.default_rng().normal(self.mean, self.std_dev, count)
        dist[dist < 0] = 0
        return dist


################################################################################
# Failing Generator
# Uses a uniform distribution and a supplied probability threshold to produce
# a failing outcome
################################################################################
class FailingGenerator(DurationGenerator):
    def __init__(self, failure_rate):
        super(FailingGenerator, self).__init__()
        self.failure_rate = failure_rate

    def generate(self, count):
        uniform_results = numpy.random.uniform(low=0, high=1, size=count)
        uniform_results[uniform_results > self.failure_rate] = 0
        uniform_results[uniform_results != 0] = OPERATION_FAILED_LATENCY
        return uniform_results


################################################################################
# BernoulliGenerator
# Creates a cost penalty with the given probability.
################################################################################
class BernoulliGenerator(DurationGenerator):
    def __init__(self, prob, cost):
        super(BernoulliGenerator, self).__init__()
        self.prob = prob
        self.cost = cost

    def generate(self, count):
        vals = numpy.random.binomial(size=count, n=1, p=self.prob)
        cost_vals = self.cost * vals
        return cost_vals


################################################################################
# BetaGenerator
# Duration generator that uses a combination of a lower bound plus
# a coefficient times a Beta distribution
################################################################################
class BetaGenerator(DurationGenerator):
    def __init__(self, alpha, beta, lower_bound=0, coef=0):
        super(BetaGenerator, self).__init__()
        self.lower_bound = lower_bound
        self.coef = lower_bound if coef == 0 else 1
        self.alpha = alpha
        self.beta = beta

    def generate(self, count):
        return self.lower_bound + (
            self.coef * numpy.random.beta(a=self.alpha, b=self.beta, size=count)
        )


################################################################################
# UniformGenerator
# Duration generator that uses uniform distribution
################################################################################
class UniformGenerator(DurationGenerator):
    def __init__(self, lower_bound, upper_bound):
        super(UniformGenerator, self).__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def generate(self, count):
        return numpy.random.uniform(
            low=self.lower_bound, high=self.upper_bound, size=count
        )


################################################################################
#
# Workflow
# Workflow is the root of all PyFlow graphs and contains subgraphs
# for parallel or serial steps
################################################################################
class Workflow(INode):
    def __init__(self, run_count, **kwargs):
        logger.debug("Workflow::__init__")

        super(Workflow, self).__init__(**kwargs)
        self.run_count = run_count
        OutputPlug(LATENCIES, self)

    def compute(self):
        logger.debug("Workflow::compute")
        zeros_vals = numpy.zeros(self.run_count)
        return {LATENCIES: zeros_vals}


################################################################################
# WorkflowNode
# A node that exists in a WorkflowSubgraph and delegates values
# to the supplied DurationGenerator
################################################################################
class WorkflowNode(INode):
    def __init__(self, generator, **kwargs):
        super(WorkflowNode, self).__init__(**kwargs)
        logger.debug("InputNode::__init__")
        self.generator = generator
        InputPlug(LATENCIES, self)
        OutputPlug(LATENCIES, self)
        logger.debug("InputNode::__init__, OUTPUTS", self.outputs)

    def compute(self, latencies):
        logger.debug("InputNode::compute")
        operationLatencies = self.generator.generate(len(latencies))
        output_latencies = numpy.add(latencies, operationLatencies)
        return {LATENCIES: output_latencies}


################################################################################
# InputNode
# Stable input node
################################################################################
@Node(outputs=[LATENCIES])
def InputNode(latencies):
    logger.debug("InputNode. Dict len: %d", len(latencies))
    return {LATENCIES: latencies}


################################################################################
# JoinUpper
# Ensure reduce the set of input latencies to their highest pairwise values
################################################################################
class JoinUpper(INode):
    def __init__(self, **kwargs):
        super(JoinUpper, self).__init__(**kwargs)
        max_value = 0
        min_value = 0
        percentiles = None

        logger.debug("JoinUpper::__init__")
        InputPlug(LATENCY_DICT, self)
        OutputPlug(LATENCIES, self)
        logger.debug("JoinUpper::__init__, OUTPUTS", self.outputs)

    def compute(self, latencydict):
        logger.debug("JoinUpper::__init__")
        logger.debug("JoinUpper::compute. Dict len: %d", len(latencydict))
        computed_values = max_latency(latencydict)

        self.max_value = numpy.amax(
            computed_values,
            where=computed_values != OPERATION_FAILED_LATENCY,
            initial=0,
        )
        self.min_value = numpy.amin(
            computed_values,
            where=computed_values >= 0,
            initial=OPERATION_FAILED_LATENCY,
        )
        # Only compute percentiles for successful operations
        self.percentiles = duration_percentiles(computed_values)
        return {LATENCIES: computed_values}


################################################################################
# Done
# Done is the node that produces the matplotlib for all the inputs
################################################################################
@Node()
def Done(latencies):
    # To support subgraphs we're going to get a dict of latencies
    latency = latencies  # max_latency(latencies)

    def split(arr, cond):
        return [arr[cond], arr[~cond]]

    # Split it
    # plt.xkcd()
    plt.rcParams["font.family"] = "Avenir"

    # Success/failure pie chart
    valid, failed = split(latency, latency < OPERATION_FAILED_LATENCY)
    has_failures = True if len(failed) != 0 else False

    fig, axs = plt.subplots(1, 2 if not has_failures else 3)
    fig.suptitle("Results".format(len(latencies)))

    # Distribution
    axs[0].hist(valid, density=True, histtype="stepfilled", bins=100, cumulative=True)
    axs[0].set_xlabel("Duration")
    axs[0].set_ylabel("Probability")
    axs[0].set_title("Cumulative")

    # Distribution
    axs[1].hist(valid, density=True, histtype="step", bins=100, cumulative=False)
    axs[1].set_xlabel("Duration")
    axs[1].set_ylabel("Frequency")
    axs[1].set_title("Distribution")

    # Is there a failure rate?
    if has_failures:
        labels = "Success", "Failure"
        sizes = [len(valid), len(failed)]
        explode = (0, 0.1)
        axs[2].pie(
            sizes,
            explode=explode,
            labels=labels,
            autopct="%1.1f%%",
            shadow=True,
            startangle=90,
        )
        axs[2].axis("equal")
        axs[2].set_title("Success Rate")

    # Save it
    fig.tight_layout()
    plt.savefig(SIMULATED_RESULTS_OUTPUT_FILENAME)


################################################################################
# WorkflowSubgraph
# Subgraphs contain either serial or parallel steps
################################################################################
class WorkflowSubgraph:
    def __init__(self, name, parent_graph=None, create_input_node=True, **kwargs):
        self.subgraphs = []
        self.parallel_steps = dict()
        self.serial_steps = []
        self.parent_graph = parent_graph
        self.graph = Graph(name=name)

        # All inputs come through here...
        if create_input_node:
            input_node_name = "Latency"
            self.input_node = InputNode(graph=self.graph, name=input_node_name)
            self.input_node.inputs[LATENCIES].promote_to_graph(name=LATENCIES)

        # Everything in this subgraph will end up feeding into the Join
        join_output_name = "Upper Bound"
        self.join_output = JoinUpper(graph=self.graph, name=join_output_name)
        self.join_output.outputs[LATENCIES].promote_to_graph(name=LATENCIES)

    def parallel(self, dict_nodes):
        logger.debug("parallel_steps: %s", self.graph.name)
        for each_name, each_generator in dict_nodes.items():
            step = WorkflowNode(
                graph=self.graph, name=each_name, generator=each_generator
            )
            self.input_node.outputs[LATENCIES] >> step.inputs[LATENCIES]
            step.outputs[LATENCIES] >> self.join_output.inputs[LATENCY_DICT][each_name]
            logger.debug("Mapping step latency to join dict key: %s", each_name)

            # Parallel steps
            self.parallel_steps[each_name] = step

        return self

    def serial(self, *args_of_tuples):
        head_node = self.input_node
        for each_tuple in args_of_tuples:
            step = WorkflowNode(
                graph=self.graph, name=each_tuple[0], generator=each_tuple[1]
            )
            head_node.outputs[LATENCIES] >> step.inputs[LATENCIES]
            head_node = step

            # Save it...
            self.serial_steps.append(step)

        head_node.outputs[LATENCIES] >> self.join_output.inputs[LATENCY_DICT][
            head_node.name
        ]
        return self

    def new_subgraph(self, subgraph_name):
        subgraph = WorkflowSubgraph(name=subgraph_name)
        self.subgraphs.append(subgraph)
        return subgraph


################################################################################
# WorkflowGraph
# WorkflowGraph is the root of the entire layout
################################################################################
class WorkflowGraph(WorkflowSubgraph):
    def __init__(self, name, run_count=100000, show_end_date=False, **kwargs):
        super().__init__(name, create_input_node=False, **kwargs)
        self.start_node = Workflow(
            name="{0} StartNode".format(name), run_count=run_count
        )
        self.show_end_date = show_end_date
        self.done_node = Done(graph=self.graph, name="{0} DoneNode".format(name))

        # Hook them all up...
        self.join_output.outputs[LATENCIES] >> self.done_node.inputs[LATENCIES]

    # All this does is allow subgraphs...we'll then stitch those together
    def resolve(self):
        def resolve_subgraph(input_source_node, each_subgraph, parent_join_node):
            if len(each_subgraph.subgraphs) <= 0:
                each_subgraph.join_output.outputs[LATENCIES] >> parent_join_node.inputs[
                    LATENCY_DICT
                ][each_subgraph.graph.name]
            else:
                for each_child_graph in each_subgraph.subgraphs:
                    resolve_subgraph(
                        each_subgraph.join_output,
                        each_child_graph,
                        parent_join_node,
                    )

            # What do do with this subgraph?
            if input_source_node is not None:
                input_source_node.outputs[LATENCIES] >> each_subgraph.input_node.inputs[
                    LATENCIES
                ]

        # Stitch them....
        for each_subgraph in self.subgraphs:
            resolve_subgraph(self.start_node, each_subgraph, self.join_output)

    def as_dot(self):
        # Accumulate any nested nodes that need to be written at the end...
        nested_orphaned_nodes = []

        ########################################################################
        # render_subgraph() - START
        # Local function to recursively output subgraphs
        def render_subgraph(
            input_node,
            each_subgraph,
            parent_graph,
            output_node,
            root_output_node,
            root_graph,
            color_index=1,
        ):

            node_color_index = DOT_COLOR_SCHEME_COUNT - color_index

            subgraph_name = "cluster_{0}_{1}".format(
                uuid.uuid4(), each_subgraph.graph.name
            )
            logger.debug(
                "Adding child graph: %s (node: %s)",
                each_subgraph.graph.name,
                subgraph_name,
            )
            # Add the subgraph nodes
            color_attrs = {
                "colorscheme": "{0}{1}".format(
                    DOT_COLOR_SCHEME, DOT_COLOR_SCHEME_COUNT
                ),
                "color": "{0}".format(node_color_index),
            }
            dot_subgraph = Digraph(
                name=subgraph_name,
                node_attr=color_attrs,
                graph_attr={
                    "label": "<<B>{0}</B>>".format(each_subgraph.graph.name),
                    "fontsize": "10",
                }
                | color_attrs,
                edge_attr=color_attrs,
            )
            subgraph_node_start = "start_subgraph_{0}_{1}".format(
                uuid.uuid4(), each_subgraph.graph.name
            )
            subgraph_node_end = "end_subgraph_{0}_{1}".format(
                uuid.uuid4(), each_subgraph.graph.name
            )
            dot_subgraph.node(
                subgraph_node_start, "{0} ⬇️".format(each_subgraph.graph.name)
            )
            node_end_label = "{0} ⬆️".format(each_subgraph.graph.name)
            if each_subgraph.join_output.percentiles is not None:
                node_end_label = "{0}|{1}".format(
                    node_end_label,
                    graphviz_percentiles_label(
                        each_subgraph.join_output.percentiles, self.show_end_date
                    ),
                )

            if (
                each_subgraph.join_output.min_value
                <= each_subgraph.join_output.max_value
            ):
                node_end_label = "{0}|Range: [{1} - {2}]".format(
                    node_end_label,
                    formatted_float(each_subgraph.join_output.min_value),
                    formatted_float(each_subgraph.join_output.max_value),
                )

            node_end_label = "{{{0}}}".format(node_end_label)
            dot_subgraph.node(
                subgraph_node_end,
                node_end_label,
            )

            def append_subgraph_step_node(a_step):
                node_id = "node{0}".format(uuid.uuid4())
                try:
                    a_step.generator.graphviz_node(node_id, a_step.name, dot_subgraph)
                except Exception as ex:
                    logger.error("Graphviz node doesn't support self-render: %s", ex)
                    dot_subgraph.node(node_id, label=a_step.name)
                return node_id

            # Parallel
            for each_name, each_step in each_subgraph.parallel_steps.items():
                node_id = append_subgraph_step_node(each_step)
                dot_subgraph.edge(subgraph_node_start, node_id)
                dot_subgraph.edge(node_id, subgraph_node_end)

            # Serial
            previous_node = ""
            for each_step in each_subgraph.serial_steps:
                current_node_id = append_subgraph_step_node(each_step)

                if len(previous_node) <= 0:
                    previous_node = subgraph_node_start

                dot_subgraph.edge(previous_node, current_node_id)
                previous_node = current_node_id

            if len(previous_node) != 0:
                dot_subgraph.edge(previous_node, subgraph_node_end)

            # Workflows start when our last item is complete, and they need to terminate
            # where we were going to terminate...but that means we will need a list of workflows that don't have
            # edge links to the end...
            if len(each_subgraph.subgraphs) != 0:
                for each_child in each_subgraph.subgraphs:
                    logger.debug("Adding child graph: %s", each_subgraph.graph.name)
                    render_subgraph(
                        subgraph_node_end,
                        each_child,
                        dot_subgraph,
                        output_node,
                        root_output_node,
                        root_graph,
                        color_index=((color_index + 3) % DOT_COLOR_SCHEME_COUNT),
                    )

            # Then add at the end
            # https://github.com/xflr6/graphviz/issues/84
            parent_graph.subgraph(dot_subgraph)
            parent_graph.edge(input_node, subgraph_node_start)

            # Nested subgraphs that need to be stiched to the Done node
            # need to be written in the root graph context
            if len(each_subgraph.subgraphs) <= 0:
                nested_orphaned_nodes.append(subgraph_node_end)

        # render_subgraph() - END
        ########################################################################

        ########################################################################
        # Primary Workflow graph
        ########################################################################
        color_attrs = {
            "colorscheme": "{0}{1}".format(DOT_COLOR_SCHEME, DOT_COLOR_SCHEME_COUNT),
            "color": "{0}".format(3),
        }

        digraph = Digraph(
            "workflow",
            filename="{0}.gv".format(self.graph.name),
            format="svg",
            node_attr={
                "shape": "record",
                "fontname": "Avenir",
                "fontsize": "10",
            }
            | color_attrs,
            graph_attr={
                "compound": "true",
                "rankdir": "TD",
                "fontname": "Avenir",
                "fontsize": "20",
                "splines": "ortho",
                "labelloc": "t",
                "ranksep": "0",
                "nodesep": "0.1",
                "labelfontsize": "20",
                "label": "<<B>{0}</B>>".format(self.graph.name),
            }
            | color_attrs,
            edge_attr=color_attrs,
        )

        start_node_name = "node{0}".format(uuid.uuid4())
        end_node_name = "node{0}".format(uuid.uuid4())
        results_target_node = "node{0}".format(uuid.uuid4())

        # Add the start node...
        digraph.node(start_node_name, "START")

        # Include ECDs in the percentiles if we have self.show_end_date
        percentiles_labels = graphviz_percentiles_label(
            self.join_output.percentiles, self.show_end_date
        )

        # And the SIMULATION results node
        # And the results node...
        simulation_label = "Simulation (n={0})|{1}|Range: [{2} - {3}]".format(
            self.start_node.run_count,
            percentiles_labels,
            formatted_float(self.join_output.min_value),
            formatted_float(self.join_output.max_value),
        )
        if self.show_end_date:
            job_min_end = networkdays.JobSchedule(
                self.join_output.min_value, 1, datetime.date.today(), networkdays=None
            )
            job_max_end = networkdays.JobSchedule(
                self.join_output.max_value, 1, datetime.date.today(), networkdays=None
            )
            simulation_label = "{0} | ECD: {1} - {2}".format(
                simulation_label, job_min_end.prj_ends, job_max_end.prj_ends
            )
        digraph.node(
            results_target_node,
            "{{{0}}}".format(
                simulation_label,
            ),
            {"penwidth": "2"},
        )

        # Add the virtual node that will link to to the Matplot lib image
        digraph.node(
            end_node_name,
            "",
            {
                "shape": "none",
                "image": SIMULATED_RESULTS_OUTPUT_FILENAME,
            },
        )
        digraph.edge(results_target_node, end_node_name, None, {"penwidth": "5"})

        # Recurse into every subgraph
        for each_subgraph in self.subgraphs:
            render_subgraph(
                start_node_name,
                each_subgraph,
                digraph,
                results_target_node,
                results_target_node,
                digraph,
            )
        # And wrap up any node definitions that need to exist in the root
        for each_nested_node_id in nested_orphaned_nodes:
            digraph.edge(each_nested_node_id, results_target_node)

        digraph.view()
