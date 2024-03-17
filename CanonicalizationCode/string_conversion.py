"""
Author: Bruno Salomão Leão
Description: This file contains all functions related to the conversion from state machine into BigSMILES
"""


# External imports -----------------------------------------------------------------------------------------------------
import networkx as nx
import copy
from polymersearch.graphs import RDKit_to_networkx_graph
from rdkit import Chem
from rdkit.Chem import Descriptors
import re


# Internal imports -----------------------------------------------------------------------------------------------------
from parser import BigSMILES_BigSmilesObj
import tree_automata
import canon_tools


# Functions ------------------------------------------------------------------------------------------------------------
def get_self_loop_nodes(nx_graph, end_states):
    """
    This function gets a list of self-loop nodes. A node has a self loop if it is part of a cycle whose nodes
    do not connect to nodes that are not in the cycle. A self loop cannot have an end state.
    Args:
        nx_graph: networkx graph.
        end_states: list of end states

    Returns: list of nodes that are in cycles, list of self-loop nodes

    """

    # Get simple cycles
    initial_cycles = list(nx.simple_cycles(nx_graph))
    nodes_in_cycles = list({n for c in initial_cycles for n in c})

    # Remove cycles with end states
    cycles = [c for c in initial_cycles if all([n not in end_states for n in c])]

    # Get nodes in cycles
    nodes = list({n for c in cycles for n in c})

    # Number of bonds
    number_of_bonds = []
    for cycle in cycles:
        number_of_bonds_cycle = []
        for node in cycle:
            edges = {e[:2] for e in list(nx_graph.in_edges(node)) + list(nx_graph.out_edges(node))}
            number_of_edges = len(edges)
            number_of_bonds_cycle.append(number_of_edges)
        number_of_bonds.append(number_of_bonds_cycle)

    # Initialize the list of self-loop nodes
    self_loop_nodes = []
    # For each node, check if it has a self-loop
    for i, node in enumerate(nodes):
        for j, bond_number_cycle in enumerate(number_of_bonds):
            cycle = cycles[j]
            # Only check the cycles that contain the node
            if node not in cycle:
                continue
            # Get nodes that are connected to more than 2 nodes. These nodes must represent states, not branch points
            other_node_bonds = [n for k, n in enumerate(bond_number_cycle) if (cycle[k] != node) and (n != 2)
                                and (not nx_graph.nodes[cycle[k]]["is_branch"])]
            # If there are nodes connected to more than 2 nodes, it is not a self loop
            if other_node_bonds:
                pass
            else:
                self_loop_nodes.append(node)
                break

    # # For each node, check if it has a self-loop
    # for node in nodes:
    #     for cycle in cycles:
    #         # Only check the cycles that contain the node
    #         if node not in cycle:
    #             continue
    #         # If any of the nodes have more than 2 edges, it means they are connected to other nodes -> it is not a self-loop
    #         self_loop = True
    #         other_nodes = [x for x in cycle if x != node]
    #         for other_node in other_nodes:
    #             edges = {e[:2] for e in nx_graph.edges[other_node]}
    #             number_of_edges = len(edges)
    #             if number_of_edges > 2:
    #                 self_loop = False
    #                 break
    #         if self_loop:
    #             self_loop_nodes.append(node)
    #             break

    return nodes_in_cycles, self_loop_nodes


def choose_states_to_split(tree, nx_graph):
    """
    This function sets the list of states that must be split. It prioritizes states with 1 input and many outputs,
    that are in fewer paths (from start to end), are not a starting state and are within a cycle
    Args:
        tree: tree automaton
        nx_graph: networkx graph of the tree automaton

    Returns: list of states

    """

    # Get non-ending states
    nonending_states = [x for x in tree.states if x not in tree.end_states]

    # Get start states
    starting_transitions = tree.get_starting_transitions()
    starting_states = [tr.output for tr in starting_transitions]

    # States with self loops
    states_in_loops, self_loop_states = get_self_loop_nodes(nx_graph, tree.end_states)

    # Remove states with self loops
    nonending_states = [x for x in nonending_states if x not in self_loop_states]

    # Rank each state
    state_rank = dict({})
    for s in nonending_states:
        rank = [0, 0, 0, 0]
        # Get number of inputs and outputs
        inputs = len(tree.transition_map[s]["in"])
        outputs = len(tree.transition_map[s]["out"])
        # Give priority to states with one input and many outputs
        if inputs == 1 and outputs > 1:
            rank[0] = 30
        elif inputs > 1 and outputs == 1:
            rank[0] = 20
        elif inputs == 1 and outputs == 1:
            continue    # If there is one input and one output, do not split it
        else:
            rank[0] = 10

        # If there are many paths from starts, decrease priority
        _paths = 0
        for start in starting_states:
            _paths -= len(list(nx.all_simple_paths(nx_graph, source=start, target=s)))
        rank[1] = _paths

        # If a state is not a starting state, increase priority
        if s not in starting_states:
            rank[2] = 1

        # If a state is in a cycle, increase priority
        if s in states_in_loops:
            rank[3] = 1

        # Add to state rank
        state_rank[s] = rank

    # Create list of states according to rank
    try:
        to_split = [list(sorted(state_rank.keys(), key=lambda x: state_rank[x], reverse=True))[0]]
    except:
        return []

    # # States that are not starts with one input and many outputs
    # one_input_many_outputs_not_starts = [x for x in nonending_states
    #                           if (len(tree.transition_map[x]["in"]) == 1) and (len(tree.transition_map[x]["out"]) > 1)
    #                           and (x not in starting_states)]    # Leave start states at the end
    # if one_input_many_outputs_not_starts:
    #     return [one_input_many_outputs_not_starts[0]]
    #
    # # States with many inputs and one outputs
    # many_inputs_one_output = [x for x in nonending_states if (len(tree.transition_map[x]["in"]) > 1 and len(tree.transition_map[x]["out"]) == 1)]
    # if many_inputs_one_output:
    #     return [many_inputs_one_output[0]]
    #
    # # States with many inputs and one outputs
    # many_inputs_many_outputs = [x for x in nonending_states if (len(tree.transition_map[x]["in"]) > 1 and len(tree.transition_map[x]["out"]) > 1)]
    # if many_inputs_many_outputs:
    #     return [many_inputs_many_outputs[0]]
    #
    # # Start states with one input and many outputs
    # one_input_many_outputs_starts = [x for x in nonending_states
    #                           if (len(tree.transition_map[x]["in"]) == 1) and (len(tree.transition_map[x]["out"]) > 1)
    #                           and (x in starting_states)]    # Leave start states at the end
    # if one_input_many_outputs_starts:
    #     return [one_input_many_outputs_starts[0]]
    #
    # to_split = one_input_many_outputs_not_starts + many_inputs_one_output + many_inputs_many_outputs + one_input_many_outputs_starts

    return to_split


def unfold_cycles(dfta, output_folder=None):
    """
    This function unfolds cycles. This means that it separates cycles that overlap.

    Procedure:
        1) List all states but ending states and states with self loop
        2) Sort the states by the following criteria: states with 1 input and many outputs, states with many inputs
        and 1 output, and states with many inputs and many outputs.
        3) If the state has 1 input and n outputs, replicate the state n times with one output at a time. Preserve the
        input in all of them.
            3.i) Get the output states of each output transition, named q
            3.ii) Get the transitions that go into q
            3.iii) Get the input state of each transition, named p
            3.iiii) Merge p with the states resulted from the split that have the same output to q
        4) If the state has 1 or many outputs and many inputs, replicate the state for each combination of input and
        output.

    Args:
        dfta: tree automaton
        output_folder: if specified, saves the state machine in a file

    Returns: dfta with separated cycles

    """

    # Initialize new state
    new_state = max(list(dfta.transition_map.keys()))

    # Generate networkx graph
    nx_graph = dfta.DFTA_to_networkx()

    # Generate empty tree automaton
    new_dfta = tree_automata.TreeAutomata(transitions=dfta.transitions, states=dfta.states, end_states=dfta.end_states)

    # Get non-ending states
    states_to_split = choose_states_to_split(dfta, nx_graph)

    # Automaton index. Only needed to save the automaton
    tree_index = 0

    # Do this until no changes can be made
    loop = True
    while loop:

        # Plot if required
        if output_folder:
            new_dfta.plot(tree_name=f"tree_{tree_index}", output_folder=output_folder)
            # Update index
            tree_index += 1

        loop = False
        # Check each state and apply the rules
        for state in states_to_split:

            # Number of input transitions
            input_transitions = new_dfta.transition_map[state]["in"]
            number_inputs = len(input_transitions)
            # Number of output transitions
            output_transitions = new_dfta.transition_map[state]["out"]
            number_outputs = len(output_transitions)

            # If it has many inputs and one output
            if number_inputs > 1 and number_outputs >= 1:
                loop = True

                # Set transition map to empty
                new_dfta.transition_map[state]["in"] = []
                new_dfta.transition_map[state]["out"] = []

                # For each input, replicate the state with only 1 input and 1 output
                for tr_in in input_transitions:
                    for tr_out in output_transitions:

                        # Create new state
                        new_state += 1
                        # Add to non-ending states list
                        new_dfta.states.append(new_state)

                        # Set transition map to empty
                        new_dfta.transition_map[new_state] = {"in": [], "out": []}

                        # Edit output transition
                        out_transition = copy.deepcopy(tr_out)
                        out_transition.input = [x if x != state else new_state for x in out_transition.input]
                        # Add new output transition
                        new_dfta.add_transition(out_transition)
                        # Remove old output transition
                        new_dfta.remove_transition(tr_out)

                        # Edit input transition
                        in_transition = copy.deepcopy(tr_in)
                        in_transition.output = new_state
                        # Add new input transition
                        new_dfta.add_transition(in_transition)
                        # Remove old input transition
                        new_dfta.remove_transition(tr_in)


            # If it has one input and many outputs
            elif number_inputs == 1 and number_outputs > 1:
                loop = True

                # Set transition map to empty
                new_dfta.transition_map[state]["in"] = []
                new_dfta.transition_map[state]["out"] = []

                # Input transition
                input_transition = input_transitions[0]
                # Remove old input transition
                new_dfta.remove_transition(input_transition)

                # Find the states that will be merged
                transitions_already_handled = []
                states_to_merge = []
                for tr_out in output_transitions:

                    # Get the output state
                    q = tr_out.output

                    # Get the transitions coming into q
                    transitions_into_q = new_dfta.transition_map[q]["in"]
                    # Compare the transitions into q with the ones from the state that will be split to get the states
                    # that we are only going to merge
                    for tr_into_q in transitions_into_q:
                        # Get the states that precede q
                        states_before_q = {x for x in tr_into_q.input if x != state}
                        for p in states_before_q:
                            # Get the transitions from p
                            transitions_from_p = new_dfta.transition_map[p]["out"]
                            merge = True
                            state_transitions = []
                            for tr_from_p in transitions_from_p:
                                for tr_out_state in output_transitions:
                                    equivalent = False
                                    # Check if they have the same alphabet to the same output
                                    if_same_alphabet = (tr_from_p.alphabet == tr_out_state.alphabet)
                                    if_same_output = (tr_from_p.output == tr_out_state.output)
                                    # Replace state and p by None in both inputs to check if they are interchangeable
                                    if_same_input = ([None if x in [state, p] else x for x in tr_from_p.input] == [None if x in [state, p] else x for x in tr_out_state.input])
                                    # Check if they are not the same transition
                                    if_different_transitions = tr_from_p != tr_out_state
                                    if if_same_alphabet and if_same_output and if_same_input and if_different_transitions:
                                        equivalent = True
                                        state_transitions.append(tr_out_state)
                                        break
                                # If they are not equivalent, do not merge
                                if not equivalent:
                                    merge = False
                                    break
                            # If they are equivalent, merge
                            if merge:
                                states_to_merge.append(p)
                                transitions_already_handled += state_transitions

                # Redirect input transition to states that need to be merged (they are not actually merged)
                transitions_created = []    # Keeps track of the transitions that were created
                for s in set(states_to_merge):
                    # Copy the input transition
                    new_input_transition = copy.deepcopy(input_transition)
                    # Redirect it to the s
                    new_input_transition.output = s
                    # Add to transition list and transition map
                    new_dfta.add_transition(new_input_transition)
                    # if not [new_input_transition.same_transition(x) for x in transitions_created]:
                    #     new_dfta.add_transition(new_input_transition)
                    #     transitions_created.append(new_input_transition)

                # Delete the transitions that were already handled
                for already_handled in transitions_already_handled:
                    new_dfta.remove_transition(already_handled)

                # Deal with the remaining transitions. Combine the input with each output
                for tr_out in [x for x in output_transitions if x not in transitions_already_handled]:
                    # Create new state
                    new_state += 1
                    # Add to non-ending states list
                    new_dfta.states.append(new_state)

                    # Set transition map to empty
                    new_dfta.transition_map[new_state] = {"in": [], "out": []}

                    # Edit input transitions
                    in_transition = copy.deepcopy(input_transition)
                    in_transition.output = new_state
                    # Add input transition
                    new_dfta.add_transition(in_transition)

                    # Edit output transition
                    out_transition = copy.deepcopy(tr_out)
                    out_transition.input = [x if x != state else new_state for x in out_transition.input]
                    # Remove old output transition
                    new_dfta.remove_transition(tr_out)
                    # Add output transition
                    new_dfta.add_transition(out_transition)

        # Remove states that were split
        for state in states_to_split:
            new_dfta.remove_state(state)

        # Generate networkx graph
        nx_graph = new_dfta.DFTA_to_networkx()

        states_to_split = choose_states_to_split(new_dfta, nx_graph)


    # Remove duplicates of list of transition
    list_of_transitions = list(set(new_dfta.transitions))
    new_dfta.transitions = list_of_transitions

    return new_dfta


def transitions_to_merge(dfta):
    """
    This function groups the transitions that can be merged pairwise. Transitions can only be merged if
        1) There is a state between them
        2) One goes into the state and the other goes out of it
        3) There is no other transition into or out of the state
        4) The state is not an ending state
    Args:
        dfta: deterministic finite tree automaton

    Returns: list of pairs of transitions that can be merged
    """

    _merge = []

    # Loop over all states
    for state in dfta.transition_map.keys():
        # If the state has only one transition in and one out, they should be merged
        qtt_in = len(dfta.transition_map[state]["in"])
        qtt_out = len(dfta.transition_map[state]["out"])
        is_ending_state = state in dfta.end_states
        if (qtt_in == qtt_out == 1) and not is_ending_state:
            _merge.append([dfta.transition_map[state]["in"][0], dfta.transition_map[state]["out"][0]])
    return _merge


def merge_transitions(dfta, merge):
    """
    This function takes a list of pairs of transitions that must be merged and merge them. For example, it converts
    transitions A([0]) -> 1 and B([1]) -> 2 into AB([0]) -> 2
    Args:
        dfta: deterministic finite tree automaton
        merge: list of pairs of transitions that must be merged

    Returns: dfta with merged transitions

    """

    # Dictionary that maps old transitions to new transitions
    transition_map = {tr: tr for tr in dfta.transitions}

    # New state machine
    new_dfta = tree_automata.TreeAutomata(transitions=[], states=[], end_states=[])

    # Transitions to add to new state machine
    transitions_to_add = dfta.transitions

    # List that keeps track of the pairs of transitions that were already computed
    computed_transitions = []

    # Loop over each pair of transitions to merge
    for tr_in, tr_out in merge:

        # If both transitions have already been merged, skip them
        if (tr_in, tr_out) in computed_transitions:
            continue

        # If tr_in has already been merged, take the new one
        tr_in = transition_map[tr_in]
        # Do the same for tr_out
        tr_out = transition_map[tr_out]

        # Get the state that will be collapsed
        collapsed_state = tr_in.output

        #### GENERATE NEW LIST OF INPUT STATES AND OUTPUT STATE
        # Generate list of new input states in the right order
        input_cnx_point_map = {1: [1]}    # List that maps the old connection ids to the new ones in the input graph
        output_cnx_point_map = {1: [1]}    # List that maps the old connection ids to the new ones in the output graph
        new_input_states = []    # New input states
        count = 2
        # For each input state of the output transition, check if it is the collapsed state
        for output_old_cnx_id, s in enumerate(tr_out.input):
            # New connection id, based on the position in the new_input_states
            output_old_cnx_id = output_old_cnx_id + 2
            # If it is the collapsed state, add the inputs of the input transition to new_input_states
            if s == collapsed_state:
                # Update output_cnx_point_map
                if output_old_cnx_id in output_cnx_point_map.keys():
                    output_cnx_point_map[output_old_cnx_id].append(output_old_cnx_id)  # Update mapping dict
                else:
                    output_cnx_point_map[output_old_cnx_id] = [output_old_cnx_id]
                # Add inputs to new input
                new_input_states += tr_in.input
                # Loop over the input transition inputs to create new connection ids
                for input_old_cnx_id, _s in enumerate(tr_in.input):
                    new_cnx_symbol = count    # Old connection id
                    input_old_cnx_id = input_old_cnx_id + 2    # New connection id, based on the position in the new_input_states
                    if input_old_cnx_id in input_cnx_point_map.keys():
                        input_cnx_point_map[input_old_cnx_id].append(new_cnx_symbol)    # Update mapping dict
                    else:
                        input_cnx_point_map[input_old_cnx_id] = [new_cnx_symbol]    # If old cnx symbol is not yet in the dict, add it
                    # Update output_cnx_point_map
                    if output_old_cnx_id in output_cnx_point_map.keys():
                        output_cnx_point_map[output_old_cnx_id].append(new_cnx_symbol)    # Update mapping dict
                    else:
                        output_cnx_point_map[output_old_cnx_id] = [new_cnx_symbol]
                    count += 1
            # If it is not the collapsed state, add it to new_input_states
            else:
                new_input_states.append(s)
                new_cnx_symbol = count    # Old connection id
                # output_cnx_point_map[output_old_cnx_symbol] = new_cnx_symbol    # Update mapping dict
                if output_old_cnx_id in output_cnx_point_map.keys():
                    output_cnx_point_map[output_old_cnx_id].append(new_cnx_symbol)  # Update mapping dict
                else:
                    output_cnx_point_map[output_old_cnx_id] = [new_cnx_symbol]
                count += 1
        # New output
        new_output_state = tr_out.output

        #### GENERATE NEW ALPHABET
        # Get connection index from incoming transition. It is always 1 when it is the output of a transition
        index_in = "*:1"
        # Get connection index from output transition
        count = 2
        indices_out = []
        for s in tr_out.input:
            if s == collapsed_state:
                indices_out.append(f"*:{count}")
            # else:
            count += 1

        # Generate graph from the output alphabet
        rdkit_graph_out = Chem.MolFromSmiles(tr_out.smiles)
        graph_out = RDKit_to_networkx_graph(mol=rdkit_graph_out, is_bigsmarts=False, level=0)

        # Add connecting indices back to graph
        for node in graph_out.nodes():
            if "*" in graph_out.nodes[node]["symbol"]:
                # Replace it by the new connection symbol
                new_symbol = f"*:{output_cnx_point_map[graph_out.nodes[node]['map_num']][0]}"
                # Add symbol to graph
                graph_out.nodes[node]["symbol"] = new_symbol
                # Update mapping number
                graph_out.nodes[node]["map_num"] = output_cnx_point_map[graph_out.nodes[node]['map_num']][0]

        # New graph will be composed at each iteration
        new_graph = graph_out

        # For each input of the output transition that will be edited
        for i, index_out in enumerate(indices_out):

            # Add connecting indices back to graph
            for node in new_graph.nodes():
                if "*" in new_graph.nodes[node]["symbol"]:
                    # Get the connection symbol
                    symbol = f"*:{new_graph.nodes[node]['map_num']}"
                    # If the symbol is the one that will be merged, save the node index
                    if symbol == index_out:
                        node_out = node
                        break


            # Generate the graph from the input alphabet
            rdkit_graph_in = Chem.MolFromSmiles(tr_in.smiles)
            graph_in = RDKit_to_networkx_graph(mol=rdkit_graph_in, is_bigsmarts=False, level=0)
            # Relabel nodes of the input alphabet graph starting from the highest node id from the output graph. Do this to merge the graphs
            # node_shift = max(list(graph_out.nodes())) + 1
            node_shift = max(list(new_graph.nodes())) + 1
            graph_in = nx.relabel_nodes(graph_in, mapping={n: n+node_shift for n in graph_in.nodes()})
            # Add connecting indices back to graph
            for node in graph_in.nodes():
                if "*" in graph_in.nodes[node]["symbol"]:
                    # Get the connection symbol
                    symbol = f"*:{graph_in.nodes[node]['map_num']}"
                    # If the symbol is the one that will be merged, save the node index
                    if symbol == index_in:
                        node_in = node
                    else:
                        # pass
                        # Replace it by the new connection symbol
                        new_symbol = f"*:{input_cnx_point_map[graph_in.nodes[node]['map_num']][i]}"
                        # Add symbol to graph
                        graph_in.nodes[node]["symbol"] = new_symbol
                        # Update mapping number
                        graph_in.nodes[node]["map_num"] = input_cnx_point_map[graph_in.nodes[node]['map_num']][i]

            # Compose both input and output graphs
            # new_graph = nx.compose(graph_in, graph_out)
            new_graph = nx.compose(graph_in, new_graph)

            # Get the neighbors of node_in and node_out
            neighbor_in = canon_tools.find_neighbors(new_graph, node_in)[0]
            neighbor_out = canon_tools.find_neighbors(new_graph, node_out)[0]
            # Replace node_in and node_out by a connection between their neighbors
            bond_type = nx.get_edge_attributes(new_graph, "bond_type")[(neighbor_in, node_in)]
            bond_type_object = nx.get_edge_attributes(new_graph, "bond_type_object")[(neighbor_in, node_in)]
            new_graph.add_edge(neighbor_in, neighbor_out, bond_type=bond_type, bond_type_object=bond_type_object)
            # Remove node_in and node_out
            new_graph.remove_node(node_in)
            new_graph.remove_node(node_out)
            # Remove any Es at ends
            new_graph_nodes = copy.deepcopy(new_graph.nodes())
            for node in new_graph_nodes:
                # If there is an Es at the end, remove the node
                if (new_graph.degree(node) == 1) and ("Es" in new_graph.nodes[node]["symbol"]):
                    new_graph.remove_node(node)

        #### CREATE RDKIT MOLECULE
        # Get node and edge attributes
        symbols = nx.get_node_attributes(new_graph, "symbol")
        formal_charge = nx.get_node_attributes(new_graph, "formal_charge")
        is_aromatic = nx.get_node_attributes(new_graph, "is_aromatic")
        bond_object = nx.get_edge_attributes(new_graph, "bond_type_object")
        # Create molecule
        mol = Chem.RWMol()
        # Add atoms
        node_index_dict = {}
        for node in new_graph.nodes():
            # If it is any connecting point, create a Bk atom with the same mapping number
            if "*:" in symbols[node]:
                atom = Chem.Atom("Bk")
                atom.SetAtomMapNum(new_graph.nodes[node]["map_num"])
            # If it is a regular atom, create the atom
            else:
                atom = Chem.Atom(symbols[node])
                atom.SetFormalCharge(formal_charge[node])
                atom.SetIsAromatic(is_aromatic[node])
            # Add atom to molecule
            idx = mol.AddAtom(atom)
            # Save atom index
            node_index_dict[node] = idx
        # Add bonds
        for edge in new_graph.edges():
            # Get nodes
            node1, node2 = edge
            # Get indices
            idx1, idx2 = node_index_dict[node1], node_index_dict[node2]
            # Add bond
            mol.AddBond(idx1, idx2, bond_object[edge])
        # Sanitize molecule
        Chem.SanitizeMol(mol)

        #### GENERATE SMILES
        new_smiles = Chem.MolToSmiles(mol)
        # Replace heavy atoms by *
        new_smiles = new_smiles.replace("[Bk:", "[*:")
        # new_smiles = new_smiles.replace("[Cf:", "[*:")

        #### ADD NEW TRANSITION TO STATE MACHINE
        new_transition = tree_automata.Transitions(input=new_input_states,
                                                   output=new_output_state,
                                                   smiles=new_smiles,
                                                   alphabet=f"{tr_in.alphabet}{tr_out.alphabet}")    # The alphabet will not matter here
        new_dfta.transitions.append(new_transition)

        # Remove tr_in and tr_out from new_dfta
        if tr_in in new_dfta.transitions:
            new_dfta.transitions.remove(tr_in)
        if tr_out in new_dfta.transitions:
            new_dfta.transitions.remove(tr_out)

        # Update transition map
        for tr_before, tr_after in transition_map.items():
            if (tr_after == tr_in) or (tr_after == tr_out):
                transition_map[tr_before] = new_transition
        # transition_map = {tr_old: tr_new if tr_new not in (transition_map[tr_in], transition_map[tr_in])
        #                     else new_transition for tr_old, tr_new in transition_map.items()}
        transition_map[tr_in] = new_transition
        transition_map[tr_out] = new_transition

        # Remove tr_in and tr_out from list of transitions to be added
        if tr_in in transitions_to_add:
            transitions_to_add.remove(tr_in)
        if tr_out in transitions_to_add:
            transitions_to_add.remove(tr_out)

        # Point out that the transitions have already been computed
        computed_transitions.append([tr_in, tr_out])

    #### FINISH ADDING ATTRIBUTES TO TREE AUTOMATON
    # Add remaining transitions
    for tr in transitions_to_add:
        new_dfta.transitions.append(tr)
    # Add ending states
    new_dfta.end_states = dfta.end_states
    # Create list of states and the transition map
    new_dfta.states = new_dfta.get_states()
    new_dfta.transition_map = new_dfta.generate_transition_map()

    return new_dfta


def collapse_linkers_and_RUs(dfta):
    """
    This function collapses linkers and repeat units. If 2 accepting states are connected by a transition
    that is also a RU of the resulting state, the linker can be replaced by an e-transition. Otherwise, if they are not
    ending states but both of them have the same transition rule to the same state, the same can be done.
    Args:
        dfta: deterministic finite tree automaton

    Returns: deterministic finite tree automaton

    """
    # List that contains the transitions that have already been created. Its elements will be [input_state, output_state]
    transitions_created = []

    for si in dfta.states:
        # Check if it is an accepting state
        if_si_accepting = si in dfta.end_states

        # List of transitions to be deleted and added
        transitions_to_delete = []
        transitions_to_add = []

        # For each transition from si, get the output
        for si_output_tr in dfta.transition_map[si]["out"]:
            # Get the output state
            sj = si_output_tr.output
            if si == sj:
                continue

            # Check if the linker between them is equal to a loop over sj
            has_loop = False
            for sj_output_tr in dfta.transition_map[sj]["out"]:
                # Check if sj_output_tr is a self loop
                if_loop = sj_output_tr.output == sj
                # Check if its alphabet and si_output_tr alphabet are the same
                if_same_alphabet = sj_output_tr.alphabet == si_output_tr.alphabet
                # Replace sj and si by None to check if the two transitions have the same input
                if_same_input = [None if x == sj else x for x in sj_output_tr.input] == [None if x == si else x for x in si_output_tr.input]
                # If all conditions are fulfilled, break loop
                if if_loop and if_same_alphabet and if_same_input:
                    has_loop = True
                    break
            if not has_loop:
                continue

            # Check if the output state is an accepting state
            if_sj_accepting = sj in dfta.end_states
            # Check if both states are accepting
            collapse_linker = if_si_accepting and if_sj_accepting

            # If they are not both accepting states, check if si and sj have the same transition to the same state
            if not collapse_linker:
                for _si_output_tr in dfta.transition_map[si]["out"]:
                    for _sj_output_tr in dfta.transition_map[sj]["out"]:
                        if (_si_output_tr.output == _sj_output_tr.output != sj) and (_si_output_tr.alphabet == _sj_output_tr.alphabet):
                            collapse_linker = True
                            transitions_to_delete.append(_si_output_tr)
                            continue

            # If the linker must be replaced by an e-transition
            if collapse_linker:
                # Remove from input transitions in the transition map
                try:
                    dfta.transition_map[sj]["in"].remove(si_output_tr)
                except:
                    pass
                # Remove from list of transitions
                try:
                    dfta.transitions.remove(si_output_tr)
                except:
                    pass
                # Remove from output transitions in the transition map
                transitions_to_delete.append(si_output_tr)
                # If the transition has not been created already, create it
                if not [si, sj] in transitions_created:
                    # Create empty transition
                    new_transition = tree_automata.Transitions([si], "[Es]", "[*:1][Es][*:2]", sj)
                    # Add empty transition to list of transitions
                    dfta.transitions.append(new_transition)
                    # Add empty transition to input in the transition map
                    dfta.transition_map[sj]["in"].append(new_transition)
                    # Add empty transition to output in the transition map
                    transitions_to_add.append(new_transition)
                    # Add to list of created transitions
                    transitions_created.append([si, sj])

        # Delete transitions
        for tr in transitions_to_delete:
            try:
                dfta.transition_map[si]["out"].remove(tr)
            except:
                pass
            try:
                dfta.transitions.remove(tr)
            except:
                pass
        # Add transitions
        for tr in transitions_to_add:
            dfta.transition_map[si]["out"].append(tr)

    return dfta


def format_smiles(smiles):
    """
    This function formats the SMILES by placing the [*:1] at the beginning of the string. If there is no [*:1], which
    is the case where there is only [*:2], [*:2] must be at the end. It also replaces [Es] by ''.
    This function creates a BigSMILES object from BigSMILES_BigSmilesObj.py
    Args:
        smiles: SMILES string of the alphabet

    Returns: formatted SMILES

    """

    # If the smiles is a hydrogen atom, the function that converts p into SMILES fails to format it. So, we hard-coded
    # all cases that can be a [H] by itself
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    if smiles == Chem.MolToSmiles(Chem.MolFromSmiles("[*:1][H]")):
        return "[*:1][H]"
    elif smiles == Chem.MolToSmiles(Chem.MolFromSmiles("[*:2][H]")):
        return "[H][*:2]"

    # Create a BigSMILES parser object
    p = BigSMILES_BigSmilesObj.BigSMILES(smiles)

    # Candidate to be the first bonding descriptor: [*:1]
    sources = [node for node in p.G.nodes if ("[*:1]" in p.G.nodes[node]["rawStr"])]

    # Candidates to be the last bonding descritors: anything but [*:1]
    targets = [node for node in p.G.nodes if ("*" in p.G.nodes[node]["rawStr"]) and ("[*:1]" not in p.G.nodes[node]["rawStr"])]

    # If there is only one bonding descriptor, get the longest, heaviest path
    # If there is only an output bonding descriptor
    if not sources:
        # Find the heaviest, longest path between the bonding descriptor and an atom
        _paths = canon_tools.find_all_paths(list(p.G.nodes), targets, p.G)
        # Choose the longest path
        _paths = sorted(_paths, key=lambda x: len(x), reverse=True)
        size_longest_path = len(_paths[0])
        longest_paths = [x for x in _paths if len(x) == size_longest_path]
        # Among the longest paths, choose the heaviest one and the one with the heaviest atoms by the end
        periodic_table = Chem.GetPeriodicTable()
        path_masses = []
        for path in longest_paths:
            mass = 0
            weighted_mass = 0
            # Calculate the mass and weighted mass of the path
            for i, node in enumerate(path):
                # If it is not an atom, continue
                if p.G.nodes[node]["_type"] == "BigSMILES_Bond":
                    continue
                # If it is an atom, add its mass
                else:
                    atom_symbol = p.G.nodes[node]["atom"]
                    neighbors = p.G.nodes[node]["neighList"]
                    if "*" in atom_symbol:
                        number_of_hs = 0
                        _mass = periodic_table.GetAtomicWeight(atom_symbol) + number_of_hs
                    # I did this because GetDefaultValence does not work with aromatic atoms. So it takes the aliphatic representation and subtract 1 (due to the = bond)
                    elif atom_symbol.lower() == atom_symbol:
                        number_of_hs = periodic_table.GetDefaultValence(atom_symbol.upper()) - len(neighbors) - 1
                        _mass = periodic_table.GetAtomicWeight(atom_symbol.upper()) + number_of_hs
                    else:
                        number_of_hs = periodic_table.GetDefaultValence(atom_symbol) - len(neighbors)
                        _mass = periodic_table.GetAtomicWeight(atom_symbol) + number_of_hs
                    # Increment mass
                    mass += _mass
                    # Increment weighted mass
                    weighted_mass += _mass*(i+1)**2
            # Add to path_masses
            path_masses.append([path, mass, weighted_mass])
        # Select the heaviest, longest path
        path_masses = sorted(path_masses, key=lambda x: [x[1], x[2]], reverse=True)
        longest_path = path_masses[0][0]
        # Get the first atom of the path
        source = longest_path[0]
        target = targets[0]
    # If there is only an input bonding descriptor
    elif not targets:
        # Find the heaviest, longest path between the bonding descriptor and an atom
        _paths = canon_tools.find_all_paths(sources, list(p.G.nodes), p.G)
        # Choose the longest path
        _paths = sorted(_paths, key=lambda x: len(x), reverse=True)
        size_longest_path = len(_paths[0])
        longest_paths = [x for x in _paths if len(x) == size_longest_path]
        # Among the longest paths, choose the heaviest one and the one with the heaviest atoms by the end
        periodic_table = Chem.GetPeriodicTable()
        path_masses = []
        for path in longest_paths:
            mass = 0
            weighted_mass = 0
            # Calculate the mass and weighted mass of the path
            for i, node in enumerate(path):
                # If it is not an atom, continue
                if p.G.nodes[node]["_type"] == "BigSMILES_Bond":
                    continue
                # If it is an atom, add its mass
                else:
                    atom_symbol = p.G.nodes[node]["atom"]
                    neighbors = p.G.nodes[node]["neighList"]
                    if "*" in atom_symbol:
                        number_of_hs = 0
                        _mass = periodic_table.GetAtomicWeight(atom_symbol) + number_of_hs
                    # I did this because GetDefaultValence does not work with aromatic atoms. So it takes the aliphatic representation and subtract 1 (due to the = bond)
                    elif atom_symbol.lower() == atom_symbol:
                        number_of_hs = periodic_table.GetDefaultValence(atom_symbol.upper()) - len(neighbors) - 1
                        _mass = periodic_table.GetAtomicWeight(atom_symbol.upper()) + number_of_hs
                    else:
                        number_of_hs = periodic_table.GetDefaultValence(atom_symbol) - len(neighbors)
                        _mass = periodic_table.GetAtomicWeight(atom_symbol) + number_of_hs
                    # Increment mass
                    mass += _mass
                    # Increment weighted mass
                    weighted_mass += _mass*i**2
            # Add to path_masses
            path_masses.append([path, mass, weighted_mass])
        # Select the heaviest, longest path
        path_masses = sorted(path_masses, key=lambda x: [x[1], x[2]], reverse=True)
        longest_path = path_masses[0][0]
        # Get the first atom of the path
        target = longest_path[-1]
        source = sources[0]
    # Choose a source and a targets
    else:
        # Pick a target and a source != target. This is because when only $ are present, sources = targets
        target = targets[0]
        source = sources[0]

    # Place bonding descriptors at the ends
    smiles = p.place_bonding_descriptors_at_end(source=source, target=target)

    # Forms of empty alphabets
    empty_alphabet_forms = ["[*:1][Es][*:2]", "[*:1][Es]", "[Es][*:2]"]
    # If it is an empty alphabet, remove Es from string
    if smiles in empty_alphabet_forms:
        smiles = smiles.replace("[Es]", "")

    return smiles


def remove_redundant_empty_transitions(tree, nx_tree, forms_of_empty_transitions):
    """
    This function removes redundant empty transitions. For example, consider this automaton: e(0)->1, e(1)->2, e(2)->3,
    e(0)->3. The path (0, 3) is redundant because it is already contained within the path (0, 1, 2, 3), so it must be
    removed. This is done so that it reduces the number of empty transitions.

    This function takes all simple paths that only take empty transitions. Then, it removes paths that are contained by
    others, until no more changes can be made. For example, (0, 1, 3) contains (0, 3) but does not contain (3, 0).

    Args:
        tree: tree automaton
        nx_tree: networkx representation of the tree automaton
        forms_of_empty_transitions: possible forms of empty transitions

    Returns: None

    """

    # Find the states we can start the traversal from. Those will be the ones with empty output transitions
    possible_starts = []
    for s in tree.states:
        empty_tr = [tr for tr in tree.transition_map[s]["out"] if tr.smiles in forms_of_empty_transitions]
        if empty_tr:
            possible_starts.append(s)

    # Find the possible ends. Those will be the states with empty input transitions (that are not starting transitions)
    possible_ends = []
    for s in tree.states:
        empty_tr = [tr for tr in tree.transition_map[s]["in"] if (tr.smiles in forms_of_empty_transitions) and (tr.input != [])]
        if empty_tr:
            possible_ends.append(s)

    # Combine each input with each output
    starts_ends = []
    for si in possible_starts:
        for sj in possible_ends:
            if si != sj:
                starts_ends.append([si, sj])

    # Find all paths between starts and ends that only have empty transitions
    path_edges = []
    for start, end in starts_ends:
        for path in nx.all_simple_edge_paths(G=nx_tree, source=start, target=end):
            # Find non-empty edges
            non_empty_edges = [e for e in path if not (nx_tree.edges[e]["smiles"] in forms_of_empty_transitions)]
            # If there are not non-empty edges, keep it
            if not non_empty_edges:
                path_edges.append(tuple(path))

    path_nodes = []
    for path in path_edges:
        len_path = len(path)
        _path_nodes = []
        # Flatten the list
        for i, edge in enumerate(path):
            _path_nodes.append(edge[0])
            if i == len_path - 1:
                _path_nodes.append(edge[1])
        # Add to path_nodes
        path_nodes.append(_path_nodes)

    # Find transitions that are not redundant
    loop = True
    initial_path_edges = copy.deepcopy(path_edges)
    _dict = dict(zip(path_edges, path_nodes))
    _dict_old = copy.deepcopy(_dict)
    while loop:

        old_path_edges = list(_dict_old.keys())
        old_path_nodes = list(_dict_old.values())

        for i in range(len(old_path_edges[:-1])):

            index_path1 = i
            path1 = old_path_edges[index_path1]

            # Get size of first path
            len_path1 = len(path1)

            for j in range(len(old_path_edges[i+1:])):

                index_path2 = i + j + 1
                path2 = old_path_edges[index_path2]

                # Get size of second path
                len_path2 = len(path2)

                # Separate the longest from shortest path
                if len_path1 > len_path2:
                    long_nodes = old_path_nodes[index_path1]
                    short_edges = path2
                    short_nodes = old_path_nodes[index_path2]
                elif len_path2 > len_path1:
                    long_nodes = old_path_nodes[index_path2]
                    short_edges = path1
                    short_nodes = old_path_nodes[index_path1]
                # If they are the same size, it does not matter who is first
                else:
                    long_edges = path1
                    long_nodes = old_path_nodes[index_path1]
                    short_edges = path2
                    short_nodes = old_path_nodes[index_path2]

                # Check if the short path is a subpath of the long path
                is_subpath = [s for s in long_nodes if s in short_nodes] == short_nodes
                if is_subpath:
                    # Remove short path from list of paths
                    if short_edges in _dict.keys():
                        del _dict[short_edges]

        if _dict != _dict_old:
            _dict_old = _dict
        else:
            loop = False

    # Get list of edges to remove
    edges_to_keep = [edge for path in _dict.keys() for edge in path]
    all_edges = {edge for path in initial_path_edges for edge in path}
    edges_to_remove = [p for p in all_edges if not (p in edges_to_keep)]

    # Remove redundant path from state machine and from networkx graph
    for edge in edges_to_remove:
        # Find transition to remove
        transition_to_remove = [tr for tr in tree.transition_map[edge[0]]["out"] if (tr.output == edge[1]) and
                                (tr.smiles in forms_of_empty_transitions)][:1]
        for tr in transition_to_remove:
            # Remove transition
            tree.remove_transition(tr)
        # Remove edge from networkx graph
        nx_tree.remove_edge(*edge)

    # Update transition map
    tree.generate_transition_map()


def find_adjacent_cycles(simple_cycles):
    """
    This function groups cycles that share at least one atom
    Args:
        simple_cycles: list of simple cycles

    Returns: list of adjacent cycles

    """
    # Set initial list of cycles
    old_adjacent_cycles = [set(x) for x in simple_cycles]

    # Group cycles that share nodes in common until no more changes can be done
    loop = True
    while loop:
        # Initialize the list of adjacent cycles
        adjacent_cycles = []
        # Loop over cycles
        for i, ci in enumerate(old_adjacent_cycles):
            cycle = ci
            for j, cj in enumerate(old_adjacent_cycles[i::]):
                # If they are not disjoint, merge them
                if not cycle.isdisjoint(cj):
                    cycle = cycle.union(cj)
            # Add union to list of adjacent cycles
            adjacent_cycles.append(cycle)
        # If changes were made, keep repeating the process
        if old_adjacent_cycles != adjacent_cycles:
            old_adjacent_cycles = adjacent_cycles
        else:
            loop = False

    return [list(x) for x in adjacent_cycles]


def define_backbone(nx_dfta, adjacent_cycles, starts, ends):
    """
    This function defines the backbone of a tree automaton. Criteria:
        1) Pick the heaviest, longest path with the largest number of accepting states
        2) If a branch point is not within a cycle, it may be on the backbone
        3) If a branch point is within a cycle and loops back to the same cycle (path), it may be on the backbone
        4) If a branch point is within a cycle and does not loop back to the same cycle (path), it may not be on the backbone

    Args:
        nx_dfta: networkx graph that represents the state machine
        adjacent_cycles: cycles of the graph. Note that adjacent cycles must be represented as a single cycle
        starts: possible starts
        ends: possible ends

    Returns: list of backbone nodes

    """
    # Find simple paths from each start to each end
    shortest_paths = []
    for s in starts:
        for e in ends:
            _paths = nx.all_simple_paths(G=nx_dfta, source=s, target=e)
            shortest_paths += list(_paths)

    # Check the branch points of the paths. If a branch point is not within a cycle, it is ok. If it is within a cycle
    # and it loops back to the path, it is also ok. If it is within a cycle but does not loop back along the path, then
    # it cannot be the backbone (this case is a side chain)
    paths_to_remove = []
    for path in shortest_paths:
        # Get branch points
        branch_points = [n for n in path if nx_dfta.nodes[n]["is_branch"]]
        # Get branch points within cycles
        cycle_branch_points = [n for n in branch_points if any([n in _cycle for _cycle in adjacent_cycles])]
        # If at least one branch is within a cycle, check if it loops back along the path
        for branch in cycle_branch_points:
            # Get the position of the branch along the path
            branch_position_in_path = path.index(branch)
            # Get predecessor node
            predecessor_node = path[branch_position_in_path - 1]
            # Get sucessor node
            sucessor_node = path[branch_position_in_path + 1]
            # If precedessor and sucessor are in different cycles, it does not loop back. Remove it from list of backbones
            predecessor_cycle = [_c for _c in adjacent_cycles if predecessor_node in _c]
            sucessor_cycle = [_c for _c in adjacent_cycles if sucessor_node in _c]
            if predecessor_cycle != sucessor_cycle:
                paths_to_remove.append(path)
                break

    # Remove paths
    shortest_paths = [p for p in shortest_paths if p not in paths_to_remove]

    # Choose the longest, heaviest path with the greatest number of accepting states
    path_rank = []
    for path in shortest_paths:
        number_of_end_states = len([n for n in path if n in ends])
        mass = 0
        weighted_mass = 0
        # Calculate the mass and weighted mass of the path
        for i, node in enumerate(path):
            # If it is an atom add its mass
            if nx_dfta.nodes[node]["is_state"] or nx_dfta.nodes[node]["is_branch"]:
                smiles = nx_dfta.nodes[node]["smiles"]
                Mol = Chem.MolFromSmiles(smiles)
                _mass = Descriptors.ExactMolWt(Mol)
                # Increment mass
                mass += _mass
                # Increment weighted mass
                weighted_mass += _mass * (i+1) ** 2
        # Add to path_masses
        path_rank.append([path, number_of_end_states, mass, weighted_mass])

    # Sort paths
    path_rank = sorted(path_rank, key=lambda x: [x[1], x[2], x[3]], reverse=True)

    # Get the first path
    backbone = path_rank[0][0]

    return backbone


def eliminate_states(dfta, states_to_eliminate, start_state, end_state):
    """
    This function recreates the BigSMILES string by eliminating states. The method is similar to the algorithm for
    converting state machines into regular expressions. It elimintates all states until there is one transition from the
    start state to the end state

    Args:
        dfta: tree automaton
        states_to_eliminate: states to eliminate

    Returns: BigSMILES string

    """
    bigsmiles = ""

    for state_to_eliminate in states_to_eliminate:

        transitions_to_remove = []

        # List of inputs that are starting transitions
        input_starting_transitions = []

        # Get input and output transitions
        inputs = dfta.transition_map[state_to_eliminate]["in"]
        outputs = dfta.transition_map[state_to_eliminate]["out"]
        # Check if it has a self loop and find them
        if_has_self_loop = False
        self_loops = []
        for tr in inputs:
            _in = tr.input
            _out = tr.output
            # If output is in input, it is a self loop. This accounts for branches that loop back
            if _out in _in:
                self_loops.append(tr)
                if_has_self_loop = True

        # Self loop string
        self_loop_string = ""
        remove_indices = False    # If True, bonding descriptors of the nested object will not have indices, i.e., will be [<], [>] or [$]
        list_endgroup = False     # If True, list end groups after semicolon
        # If it has a self loop, create a stochastic object
        if if_has_self_loop:

            # Find input starting transitions that are not empty
            input_starting_transitions = [tr for tr in inputs if (start_state in tr.input) and not tr.smiles in tree_automata.FORMS_OF_EMPTY_ALPHABET]

            # If all input and output states are the same, do not add indices to bonding descriptors
            bonding_descriptor_indices = set({})
            for tr in self_loops:
                if tr.output != end_state:    # The connection to the end state is purely theoretical, not chemical
                    bonding_descriptor_indices.add(tr.output)
                for _in in tr.input:
                    if _in != start_state:    # The connection to the start state is purely theoretical, not chemical
                        bonding_descriptor_indices.add(_in)
            if len(bonding_descriptor_indices) == 1:
                remove_indices = True

            # If not all input transitions are starting transitions or if there is only one input, leave it empty so that it does not list starting alphabets after ;
            if (input_starting_transitions != [x for x in inputs if x not in self_loops]) or len(input_starting_transitions) == 1:
                input_starting_transitions = []
            else:
                list_endgroup = True
            # Add to transitions to be removed
            transitions_to_remove += input_starting_transitions

            # For each self loop, create a repeating unit
            nested_repeating_units = []
            for tr in self_loops:
                transitions_to_remove.append(tr)
                ru = tr.smiles
                # Replace output by > and inputs by <
                if remove_indices:
                    ru = ru.replace("[*:1]", f"[>]")
                else:
                    ru = ru.replace("[*:1]", f"[>{state_to_eliminate}]")
                for i, _in_state in enumerate(tr.input):
                    if remove_indices:
                        ru = ru.replace(f"[*:{i+2}]", f"[<]")
                    else:
                        ru = ru.replace(f"[*:{i + 2}]", f"[<{_in_state}]")
                # Add to list of repeating units
                nested_repeating_units.append(ru)

            # Join all repeating units into one string, separated by comma
            self_loop_string = ",".join(nested_repeating_units)

        # For each combination of input and output transitions, write: output + self_loop_string + input
        for tr_input in inputs:

            # If it is a self-loop, skip. If the end groups have to be listes and tr_input is not the first element of the
            # list, skip as well so it only processes these end groups once (does not replicate it).
            if (tr_input in self_loops) or (list_endgroup and tr_input in input_starting_transitions and tr_input != input_starting_transitions[0]):
                continue

            transitions_to_remove.append(tr_input)

            # Replace output by > and inputs by < in the SMILES of the input transition
            input_string = tr_input.smiles#.replace("[*:1]", f"[{input.output}>]")
            # for i, _in_state in enumerate(input.input):
            #     input_string = input_string.replace(f"[*:{i + 1}]", f"[{_in_state}<]")

            for tr_output in outputs:

                if tr_output in self_loops:
                    continue

                transitions_to_remove.append(tr_output)

                #### GENERATE NEW LIST OF INPUT STATES AND OUTPUT STATE
                # Generate list of new input states in the right order
                input_cnx_point_map = {1: [1]}  # List that maps the old connection ids to the new ones in the input graph
                output_cnx_point_map = {1: [1]}  # List that maps the old connection ids to the new ones in the output graph
                new_input_states = []  # New input states
                count = 2
                # For each input state of the output transition, check if it is the collapsed state
                for output_old_cnx_id, s in enumerate(tr_output.input):
                    # New connection id, based on the position in the new_input_states
                    output_old_cnx_id = output_old_cnx_id + 2
                    # If it is the collapsed state, add the inputs of the input transition to new_input_states
                    if s == state_to_eliminate:
                        # Update output_cnx_point_map
                        if output_old_cnx_id in output_cnx_point_map.keys():
                            output_cnx_point_map[output_old_cnx_id].append(output_old_cnx_id)  # Update mapping dict
                        else:
                            output_cnx_point_map[output_old_cnx_id] = [output_old_cnx_id]
                        # Add inputs to new input
                        new_input_states += tr_input.input
                        # Loop over the input transition inputs to create new connection ids
                        for input_old_cnx_id, _s in enumerate(tr_input.input):
                            new_cnx_symbol = count  # Old connection id
                            input_old_cnx_id = input_old_cnx_id + 2  # New connection id, based on the position in the new_input_states
                            if input_old_cnx_id in input_cnx_point_map.keys():
                                input_cnx_point_map[input_old_cnx_id].append(new_cnx_symbol)  # Update mapping dict
                            else:
                                input_cnx_point_map[input_old_cnx_id] = [new_cnx_symbol]  # If old cnx symbol is not yet in the dict, add it
                            # Update output_cnx_point_map
                            if output_old_cnx_id in output_cnx_point_map.keys():
                                output_cnx_point_map[output_old_cnx_id].append(new_cnx_symbol)  # Update mapping dict
                            else:
                                output_cnx_point_map[output_old_cnx_id] = [new_cnx_symbol]
                            count += 1
                    # If it is not the collapsed state, add it to new_input_states
                    else:
                        new_input_states.append(s)
                        new_cnx_symbol = count  # Old connection id
                        # output_cnx_point_map[output_old_cnx_symbol] = new_cnx_symbol    # Update mapping dict
                        if output_old_cnx_id in output_cnx_point_map.keys():
                            output_cnx_point_map[output_old_cnx_id].append(new_cnx_symbol)  # Update mapping dict
                        else:
                            output_cnx_point_map[output_old_cnx_id] = [new_cnx_symbol]
                        count += 1
                # New output
                new_output_state = tr_output.output

                # Replace output by > and inputs by < in the SMILES of the output transition
                output_string = tr_output.smiles

                # Points in the output alphabet where the input alphabet will be connected
                cnx_points = [i+2 for i, s in enumerate(tr_output.input) if s == tr_input.output]
                cnx_points = [f"[*:{output_cnx_point_map[x][0]}]" for x in cnx_points]

                # Replace old connection points by new
                for old, new in input_cnx_point_map.items():
                    input_string = input_string.replace(f"[*:{old}]", f"[*:{new[0]}]")
                # Replace old connection points by new
                for old, new in output_cnx_point_map.items():
                    output_string = output_string.replace(f"[*:{old}]", f"[*:{new[0]}]")

                # If input transition comes from starting state, output transition goes to end state and both have
                # an empty alphabet, do not add end group bonding descriptors to the self_loop_string. Otherwise, add.
                empty_ending_smiles = ["[*:1]", "[*:2]"]
                is_input_empty_start = (tr_input.input == [start_state] and tr_input.smiles in empty_ending_smiles)
                is_output_empty_end = (tr_output.output == end_state and tr_output.smiles in empty_ending_smiles)
                if if_has_self_loop:
                    if is_output_empty_end:
                        self_loop_string = "{[]" + ",".join(nested_repeating_units)
                    else:
                        if remove_indices:
                            self_loop_string = "{[<]" + ",".join(nested_repeating_units)
                        else:
                            self_loop_string = "{" + f"[<{state_to_eliminate}]" + ",".join(nested_repeating_units)
                    if is_input_empty_start:
                        self_loop_string += "[]}"
                    else:
                        # If end groups have to be listed
                        if list_endgroup and (tr_input in input_starting_transitions):
                            if remove_indices:
                                self_loop_string += f";{','.join([tr.smiles.replace('[*:1]', '[>]') for tr in input_starting_transitions])}" + "[]}"
                            else:
                                self_loop_string += f";{','.join([tr.smiles.replace('[*:1]', f'[>{state_to_eliminate}]') for tr in input_starting_transitions])}" + "[]}"
                        # If they do not have to be listed
                        else:
                            if remove_indices:
                                self_loop_string += "[>]}"
                            else:
                                self_loop_string += f"[>{state_to_eliminate}]" + "}"

                # if is_input_empty_start and is_output_empty_end:
                #     self_loop_string = "{[]" + ",".join(nested_repeating_units) + "[]}"
                # else:
                #     if remove_indices:
                #         self_loop_string = "{[<]" + ",".join(nested_repeating_units) + "[>]}"
                #     else:
                #         self_loop_string = "{" + f"[<{states_to_eliminate}]" + ",".join(nested_repeating_units) \
                #                        + f"[>{states_to_eliminate}]" + "}"

                # If the end groups have to be listed, set input_string as self_loop_string
                if list_endgroup and (tr_input in input_starting_transitions):
                    input_string = self_loop_string
                # Remove [*:1] from input transition
                else:
                    input_string = self_loop_string + input_string.replace("([*:1])", "").replace("[*:1]", "")

                # Combine output, self loop and input
                combination = output_string
                for cnx_point in cnx_points:
                    combination = combination.replace(f"({cnx_point})", input_string).replace(cnx_point, input_string)

                # Create new transition
                new_transition = tree_automata.Transitions(input=new_input_states, output=new_output_state,
                                                           smiles=combination, alphabet=tr_input.alphabet+tr_output.alphabet)

                # Add new transition to transition list
                dfta.transitions.append(new_transition)

                # Add new transition to transition map
                dfta.transition_map[new_transition.output]["in"].append(new_transition)
                for s in new_transition.input:
                    dfta.transition_map[s]["out"].append(new_transition)

        # Remove transitions that have to be removed
        for tr in set(transitions_to_remove):
            dfta.remove_transition(tr)

        # Remove state
        dfta.states.remove(state_to_eliminate)

    # If there is only one transition from start to end state, it will be the BigSMILES. Otherwise, all transitions
    # will become a repeating unit of a stochastic object
    if len(dfta.transitions) == 2:    # One is the starting transition and the other is the actual BigSMILES
        bigsmiles = [tr.smiles for tr in dfta.transitions if tr.smiles != ""][0]    # Get the one that is not empty
    else:
        list_of_rus = []
        # Create a RU for each transition
        for tr in dfta.transitions:
            ru = f"[<{start_state}]{tr.smiles}[>{end_state}]"
            list_of_rus.append(ru)
        # Condense all of them into one object
        bigsmiles = "{" + f"[>{start_state}]" + ",".join(list_of_rus) + f"[<{end_state}]" + "}"

    return bigsmiles


def clear_starting_transitions(tree):
    """
    This function removes duplicates of starting transitions with the same SMILES to the same state
    Args:
        tree: tree automaton

    Returns:

    """
    # Get starting transitions
    starting_transitions = tree.get_starting_transitions()

    # Find the ones to be removed
    to_remove = set([])
    for i, tr_i in enumerate(starting_transitions[:-1]):
        for j, tr_j in enumerate(starting_transitions[i+1:]):
            if (tr_i.smiles == tr_j.smiles) and (tr_i.output == tr_j.output):
                to_remove.add(tr_j)

    # Remove transitions
    for tr in to_remove:
        tree.remove_transition(tr)

    return tree


# IDEA: do not eliminate accepting states ------------------------------------------------------------------------------
def to_bigsmiles(dfta, tree_name, output_folder, draw_alphabets):
    """
    This functions converts a tree automaton into BigSMILES. It uses an algorithm similar to the one that converts
    deterministic finite automata into regular expressions.

    Here are the steps it follows:
        1) Sort the state machine
        2) Separate (unfold) cycles
        3) Merge transitions
        4) Replace linkers by empty transitions when necessary
            - When a linker is just like the repeat units of the resulting state, and (both states are accepting or
             both have the same transition to the same state)
        5) Make [*:1] the root of every alphabet
            - This makes it way easier to convert it back to string later → does not require string reversing
        6) Create one single start state
        7) Create one single end state
        8) Convert tree automaton into networkx graph
        9) Remove redundant empty transitions
        10) Detect cycles and adjacent cycles
        11) Define the backbone
            . Heaviest, longest path with the largest number of accepting states
            . Branch points
                . If it is not within a cycle, may be on backbone
                . If it is within a cycle and loops back to the path, may be on backbone
                . If it is within a cycle and does not loop back to the path (it is a side chain), may
                not be on backbone
        12) Define which nodes will be eliminated first
            . Prioritize nodes not along the backbone → this makes sure all side chains are collapsed first
            . Decrease priority of nodes in cycles with backbone nodes → prioritize nested and side chains
            . Increase priority of nodes with self loops → dealing with them first makes sure the nested stochastic
            objects will be treated first
        13) Eliminate nodes (convert to BigSMILES)

    Args:
        dfta: tree automaton
        tree_name: name of the tree, which will appear in the output file name
        output_folder: output folder
        draw_alphabets: function that will draw the alphabets

    Returns: BigSMILES

    """

    # Treat the state machine
    tree, nx_graph, new_start_state, new_end_state = treat_automaton(tree=dfta,
                                                                     tree_name=tree_name,
                                                                     output_folder=output_folder,
                                                                     draw_alphabets=draw_alphabets)

    # Identify cycles
    cycles = list(nx.simple_cycles(nx_graph))
    # Find adjacent cycles
    adjacent_cycles = find_adjacent_cycles(cycles)

    # Identify backbone
    starts = [new_start_state]
    ends = [new_end_state]
    backbone = define_backbone(nx_dfta=nx_graph, adjacent_cycles=adjacent_cycles, starts=starts, ends=ends)

    # Define states to be removed and old ending states
    states_to_eliminate, old_end_states = define_states_to_remove(tree=tree, backbone=backbone, cycles=cycles,
                                                                  adjacent_cycles=adjacent_cycles,
                                                                  new_start_state=new_start_state,
                                                                  new_end_state=new_end_state)

    # Convert the treated state machine into BigSMILES
    bigsmiles = _to_bigsmiles(tree=tree,
                              states_to_eliminate=states_to_eliminate,
                              new_start_state=new_start_state,
                              new_end_state=new_end_state,
                              old_end_states=old_end_states,
                              tree_name=tree_name,
                              output_folder=output_folder)

    return bigsmiles


def _to_bigsmiles(tree, states_to_eliminate, new_start_state, new_end_state, old_end_states, tree_name, output_folder):
    """
    This function converts a treated state machine into BigSMILES through a state elimination, akin the regex
    generation from finite state machines

    Args:
        tree: state machine
        states_to_eliminate: list of states that will be eliminated
        new_start_state: single start state
        new_end_state: single end state
        old_end_states: states that were previously end states
        tree_name: name of the tree
        output_folder: output folder

    Returns: BigSMILES

    """
    # Make a copy of the input state machine
    original_tree = copy.deepcopy(tree)

    # Eliminate states
    bigsmiles = eliminate_states(tree, states_to_eliminate, start_state=new_start_state, end_state=new_end_state)
    # Plot
    tree.plot(tree_name=f"States_eliminated_{tree_name}", output_folder=output_folder)

    # When sidechains, like in grafts, have their states eliminated, the state elimination algorithm kept the transitions associated with the start
    # state. The block of code below removes their unnecessary connection to the start state and fixes the wildcard indices in smiles
    _to_remove = []
    for tr in tree.transition_map[new_start_state]["out"]:
        if len(tr.input) > 1:
            replacing_dict = {}
            actual = 2
            to_be = 2
            for state in tr.input:
                if state != new_start_state:
                    replacing_dict[actual] = to_be
                    to_be += 1
                actual += 1
            for old, new in replacing_dict.items():
                tr.smiles = tr.smiles.replace(f"[*:{old}]", f"[*:{new}]")

            # Remove start state from input list
            tr.input = [s for s in tr.input if s != new_start_state]
            # Save to remove later
            _to_remove.append(tr)
    # Remove the transitions from the transition map out of the start state
    tree.transition_map[new_start_state]["out"] = [tr for tr in tree.transition_map[new_start_state]["out"]
                                                       if tr not in _to_remove]


    # Get list of former start states
    start_transitions = [tr for tr in tree.transition_map[new_start_state]["out"]]

    # Non empty start transitions
    non_empty_starts = [tr for tr in tree.transition_map[new_start_state]["out"]
                        if (tr.smiles not in tree_automata.FORMS_OF_EMPTY_ALPHABET)
                        and (set(tr.input) == {new_start_state})]   # Added this condition so that we only keep branch points that can be end groups, not the ones that can be repeat units. For example, a grafted chain should not be seen as an end group.

    # Check if it has cycles
    nx_graph = tree.DFTA_to_networkx()
    if_has_cycles = list(nx.simple_cycles(nx_graph))

    # If there is only one end state, eliminate it normally
    if len(old_end_states) == 1 and len(set(non_empty_starts)) <= 1:

        bigsmiles = eliminate_states(tree, old_end_states, start_state=new_start_state, end_state=new_end_state)

    # If it has only one start but many end states, add the start as the right end group
    elif len(set(start_transitions)) == 1:

        # Get the starting transition
        start_transition = tree.transition_map[new_start_state]["out"][0]

        # List of transitions that will be converted into repeating units: all of them but the ones to the end state and
        # the start_transition
        transitions_to_convert = [tr for tr in tree.transitions if (tr != start_transition) and
                                  (tr.output != new_end_state) and (tr.output != new_start_state)]

        # Generate list of the other repeating units
        repeat_units = generate_list_of_rus(transitions_to_convert)

        # Create BigSMILES
        bigsmiles = "{[]" + ",".join(repeat_units) + f"[>{start_transition.output}]" + \
                    "}" + f"{start_transition.smiles.replace('[*:1]', '')}"

    # If there are cycles (i.e. repeat units), list transitions are repeat units, and list end groups after ;.
    elif if_has_cycles:
        # Get the starting transition
        start_transitions = tree.transition_map[new_start_state]["out"]

        # List of transitions that will be converted into repeating units: all of them but the ones to the end state and
        # the start_transition
        transitions_to_convert = [tr for tr in tree.transitions if not (tr in start_transitions) and
                                  (tr.output != new_end_state) and (tr.output != new_start_state)]

        # Generate list of the other repeating units
        repeat_units = generate_list_of_rus(transitions_to_convert)

        # Generate list of end groups with non-empty transitions
        end_group_transitions = [x for x in start_transitions if not (x.smiles in tree_automata.FORMS_OF_STARTING_EMPTY_ALPHABET)]
        end_groups = generate_list_of_rus(end_group_transitions)

        # Create BigSMILES
        if end_groups:
            bigsmiles = "{[]" + ",".join(repeat_units) + f";{','.join(end_groups)}" + "[]}"
        else:
            bigsmiles = "{[]" + ",".join(repeat_units) + "[]}"

    # If there are no cycles, take the initial state machine, eliminate all states but starts and ends, make all starting transitions and transitions to end states end groups
    else:
        # Start states
        start_states = [tr.output for tr in original_tree.transition_map[new_start_state]["out"]]

        # From list of states to eliminate, remove start states and end states
        states_to_eliminate = [s for s in states_to_eliminate if (not s in start_states) and (not s in old_end_states)]

        # Eliminate states
        bigsmiles = eliminate_states(original_tree, states_to_eliminate, start_state=new_start_state, end_state=new_end_state)

        # Remove input from original input transitions
        for tr in original_tree.transition_map[new_start_state]["out"]:
            tr.input = []
        # Remove unique start transition
        for tr in original_tree.transition_map[new_start_state]["in"]:
            original_tree.remove_transition(tr)
        # Remove transitions into unique end state
        for tr in original_tree.transition_map[new_end_state]["in"]:
            original_tree.remove_transition(tr)
        # Remove unique initial and end states
        original_tree.remove_state(new_start_state)
        original_tree.remove_state(new_end_state)
        # Update end states
        original_tree.end_states = old_end_states

        # Eliminate empty transitions
        original_tree.eliminate_epsilon_transitions(canonicalize=False)

        # List of starting transitions that are not empty
        start_transitions = [tr for tr in original_tree.get_starting_transitions() if tr.alphabet not in tree_automata.FORMS_OF_STARTING_EMPTY_ALPHABET]
        # List of starting end groups
        start_end_groups = generate_list_of_rus(start_transitions)
        # List of ending transitions that are not self-loops
        ending_transitions = [tr for tr in original_tree.get_ending_transitions() if tr.output not in tr.input]
        # List of ending end groups
        ending_end_groups = generate_list_of_rus(ending_transitions)
        # List of transitions that are neither starts nor ends
        repeat_unit_transitions = [tr for tr in original_tree.transitions if (tr not in start_transitions) and (tr not in ending_transitions)]
        # List of repeat units
        repeat_units = generate_list_of_rus(repeat_unit_transitions)

        # Create BigSMILES
        bigsmiles = "{[]" + ",".join(repeat_units) + f";{','.join(ending_end_groups + start_end_groups)}" + "[]}"

    return bigsmiles


def treat_automaton(tree, tree_name, output_folder, draw_alphabets):
    """
    This function treats the state machine before it is converted into BigSMILES.

    Here are the steps it follows:
        1) Sort the state machine
        2) Separate (unfold) cycles
        3) Merge transitions
        4) Replace linkers by empty transitions when necessary
            - When a linker is just like the repeat units of the resulting state, and (both states are accepting or both
            have the same transition to the same state)
        5) Make [*:1] the root of every alphabet
            - This makes it way easier to convert it back to string later → does not require string reversing
        6) Create one single start state
        7) Create one single end state
        8) Convert tree automaton into networkx graph
        9) Remove redundant empty transitions

    Args:
        tree: tree automaton
        tree_name: name of the tree
        output_folder: output folder
        draw_alphabets: function that draws the alphabets

    Returns: the tree, the networkx graph of the tree

    """
    # Remove duplicates of starting transitions with the same alphabet
    tree = clear_starting_transitions(tree)

    # Sort the state machine and relabel states so that all steps beyond this point result in the same output
    tree.sort()

    # Unfold cycles
    tree = unfold_cycles(tree, output_folder=output_folder + "\\Tree_Unfolding")
    tree.plot(tree_name=f"Unfolded_{tree_name}", draw_alphabet_function=draw_alphabets, output_folder=output_folder)

    # Get the transtions that have to be merged
    merge = transitions_to_merge(tree)
    # Merge transitions
    tree = merge_transitions(tree, merge)
    tree.plot(tree_name=f"Collapsed_tree_{tree_name}", draw_alphabet_function=draw_alphabets, output_folder=output_folder)

    # Relabel the states again
    tree.sort()

    # Merge e-transitions with repeat units
    tree = collapse_linkers_and_RUs(tree)
    tree.plot(tree_name=f"RU_and_Linkers_{tree_name}", output_folder=output_folder)

    # Make [*:1] the root of each alphabet
    for tr in tree.transitions:
        tr.smiles = format_smiles(tr.smiles)

    # Add one single start and one single end
    tree, new_start_state, new_end_state = add_single_start_single_end(tree=tree)
    tree.plot(tree_name=f"Single_StartAndEnd_{tree_name}", output_folder=output_folder)

    # Generate networkx graph
    nx_graph = tree.DFTA_to_networkx()

    # Remove redundant e-transitions
    forms_of_empty_transitions = ["[*:1]", "[*:1][*:2]", "[*:2]", "[*:1][Es][*:2]", "[*:1][Es]", "[Es][*:2]"]
    remove_redundant_empty_transitions(tree, nx_graph, forms_of_empty_transitions)
    tree.plot(tree_name=f"No_Redundant_Empty_{tree_name}", output_folder=output_folder)

    # Return the state machine and the graph
    return tree, nx_graph, new_start_state, new_end_state


def add_single_start_single_end(tree):
    """
    This function create one single start state and one single end state.
    Args:
        tree: tree automaton

    Returns: tree, new start state and new end state

    """
    # Create one single start state
    new_start_state = max(tree.states) + 1
    tree.states.append(new_start_state)
    tree.transition_map[new_start_state] = {"in": [], "out": []}

    # Make all initial transitions come from this state
    for tr in tree.transitions:
        if not tr.input:
            tr.input = [new_start_state]
            tree.transition_map[new_start_state]["out"].append(tr)

    # Add an empty transition to the start state
    single_start_transition = tree_automata.Transitions(input=[], output=new_start_state, alphabet="START", smiles="")
    tree.transitions.append(single_start_transition)
    tree.transition_map[new_start_state]["in"].append(single_start_transition)

    # Create new end state
    new_end_state = new_start_state + 1

    # Add e-transitions from end states to new ending state
    tree.transition_map[new_end_state] = {"in": [], "out": []}
    for s in tree.end_states:
        _tr = tree_automata.Transitions(input=[s], output=new_end_state, alphabet="END", smiles="[*:2]")
        tree.transitions.append(_tr)
        tree.transition_map[s]["out"].append(_tr)
        tree.transition_map[new_end_state]["in"].append(_tr)
    tree.end_states = [new_end_state]
    tree.states.append(new_end_state)

    return tree, new_start_state, new_end_state


def generate_list_of_rus(list_of_transitions):
    """
    This function converts a list of transitions into a list of repeat units
    Args:
        list_of_transitions: list of transitions

    Returns: list of repeat units

    """
    # Generate list of the other repeating units
    repeat_units = []
    for tr in list_of_transitions:
        # Convert transition into repeat unit
        ru = transition_to_ru(tr)
        # Add to list
        repeat_units.append(ru)

    return repeat_units


def transition_to_ru(transition):
    """
    This function converts a transition rule into a repeat unit
    Args:
        transition: transition

    Returns: repeat unit

    """
    # Take the smiles
    ru = transition.smiles

    count = 2
    # Replace * by heavy atoms. For input, use Bk
    for node in transition.input:
        ru = ru.replace(f"[*:{count}]", f"[Bk:{node}]")
        count += 1
    # For output, use Cf
    ru = ru.replace("[*:1]", f"[Cf:{transition.output}]")
    # Replace Bk by < and Cf by >
    ru = ru.replace("[Bk:", "[<").replace("[Cf:", "[>")

    return ru


def define_states_to_remove(tree, backbone, cycles, adjacent_cycles, new_start_state, new_end_state):
    """
    This function defines the order by which states will be eliminated during string conversion. Here are the rules
    it follows:
        . Prioritize nodes not along the backbone → this makes sure all side chains are collapsed first
        . Decrease priority of nodes in cycles with backbone nodes → prioritize nested and side chains
        . Increase priority of nodes with self loops → dealing with them first makes sure the nested stochastic objects
        will be treated first

    End states are not selected. This function also defines the list of end states

    Args:
        tree: tree automaton
        backbone: list of backbone states
        cycles: list of simple cycles
        adjacent_cycles: list of adjacent cycles
        new_start_state: new single starting state
        new_end_state: new single ending state
        ends: single ending state

    Returns: list of states to be removed and list of ending states

    """


    # Initialize backbone states with ranking -10, while the others are 0
    state_rank = {n: 0 if n not in backbone else -10 for n in tree.states}

    # Get the simple and adjacent cycles with backbone atoms. If a node is in a cycle with a
    # backbone node, decrease priority
    simple_cycles_backbone = [c for c in cycles if not set(c).isdisjoint(set(backbone))]    # Simples cycles with backbone atoms
    adjacent_cycles_backbone = [c for c in adjacent_cycles if not set(c).isdisjoint(set(backbone))]  # Adjacent cycles with backbone atoms

    # Check simple cycles and decrease priority (this gives priority to nested objects along backbone). Reduce 1
    for node in tree.states:
        if node not in backbone:
            for c in simple_cycles_backbone:
                if node in c:
                    state_rank[node] += -1
                    break

    # Check adjacent cycles and decrease priority (this gives priority to side chains). Reduce 1
    for node in tree.states:
        if node not in backbone:
            for c in adjacent_cycles_backbone:
                if node in c:
                    state_rank[node] += -1
                    break

    # If it has a self loop, increase priority + 1
    self_loop_states = []
    for state in tree.states:
        for tr in tree.transition_map[state]["in"]:
            _in = set(tr.input)
            _out = set({tr.output})
            # If input and output are the same, add 1 and move to other state
            if _in == _out:
                state_rank[state] += 1
                self_loop_states.append(state)
                break

    # Sort and convert into a list
    states_to_eliminate = sorted(state_rank.keys(), key=lambda x: state_rank[x], reverse=True)

    # Remove end states and the starting state
    states_to_eliminate.remove(new_start_state)
    states_to_eliminate.remove(new_end_state)

    # Get former end states in the sequence that they appear in states_to_eliminate
    old_end_states = [tr.input[0] for tr in tree.transition_map[new_end_state]["in"]]
    old_end_states = [s for s in states_to_eliminate if s in old_end_states]
    # Remove former end states
    for s in old_end_states:
        states_to_eliminate.remove(s)

    return states_to_eliminate, old_end_states


# Main -----------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass