"""
Author: Bruno Salomão Leão
Description: This file contains all functions related to the conversion from BigSMILES into state machine and the
             function that orchestrates the canonicalization.
"""

# External imports -----------------------------------------------------------------------------------------------------
import pydot
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import re
import pandas as pd
import numpy as np
import networkx as nx
import copy
import string
import itertools
import os


# Internal imports -----------------------------------------------------------------------------------------------------
from polymersearch.search_tools import generate_all_possible_graphs
from polymersearch.graphs import get_comp
import tree_automata
import string_conversion


# Functions ------------------------------------------------------------------------------------------------------------
def find_atomistic_cycles(atomistic_graph):
    """
    Finds all cycles in an atomistic graph
    Args:
        atomistic_graph: atomistic graph

    Returns: list of lists of nodes within the cycle
    """
    cycles = nx.cycle_basis(atomistic_graph)
    return cycles


def find_topology_cycles(topology_graph):
    """
    Finds all cycles in a topology graph
    Args:
        topology_graph: topology graph

    Returns: list of lists of nodes within the cycle
    """
    cycles = nx.simple_cycles(topology_graph)
    return cycles


def visualize_atomistic_graph(atomistic_graph, output_folder, filename="Atomistic_Graph", show_state_index=True):
    """
    This function saves the atomistic graph as a figure in the specified folder
    Args:
        atomistic_graph: networkx graph
        output_folder: path name
        filename: file name
        show_state_index: if True, labels the states as: "{atom_symbol}_{node_index}"

    Returns: None
    """
    # Create pydot graph
    _g = pydot.Dot(graph_type='graph', rankdir="TD")

    # Add nodes
    for n in atomistic_graph.nodes():
        symbol = atomistic_graph.nodes[n]['symbol']
        if show_state_index:
            node = pydot.Node(n, label=f"{symbol}_{n}", shape="circle")
        else:
            node = pydot.Node(n, label=f"{symbol}", shape="circle")
        _g.add_node(node)

    # Add edges
    for e1, e2 in atomistic_graph.edges():
        bond_type = atomistic_graph.edges[(e1, e2)]["bond_type"]
        edge = pydot.Edge(e1, e2, label=bond_type)
        _g.add_edge(edge)

    # Ouput folder
    output_folder = os.path.join(os.getcwd(), output_folder)
    # If it does not exist, create it
    try:
        os.makedirs(output_folder)
    except:
        pass

    # If the filename does not end in ".pgn", add it
    if filename[-4:] != ".svg":
        filename += ".svg"
    # Save as .svg
    _g.write_svg(os.path.join(output_folder, filename))


def draw_alphabets(**kwargs):
    """
    This function establishes how the output file that lists alphabets should be. Their SMILES representation
    will be listed above the alphabet letter.
    Args:
        alphabet_dictionary: dictionary whose keys are SMILES and values are alphabet letters
        filename: output file name

    Returns:

    """
    # Get dictionary of alphabets
    alphabet_dictionary = kwargs.get("alphabet_dictionary")
    # Sort alphabet dictionary
    alphabet_dictionary = dict(sorted(alphabet_dictionary.items()))
    # Get file name
    filename = kwargs.get("filename")

    # Plot molecules
    visual = []
    legend = []
    for alphabet, smiles in alphabet_dictionary.items():
        # If it is an empty transition, do not plot its alphabet
        if not smiles in tree_automata.FORMS_OF_EMPTY_ALPHABET:
            visual.append(Chem.MolFromSmiles(smiles))
            legend.append(alphabet)
    img = Draw.MolsToGridImage(visual, molsPerRow=3, subImgSize=(200, 250), useSVG=True, legends=legend)

    # Save file
    with open(filename, 'w') as f:
        f.write(img)
    f.close()


def find_endgroups_and_linkers_ids(topology_graph):
    """
    Finds the IDs of end group and linker atoms. End groups from nested objects are also identified.
    Args:
        topology_graph: topology graph

    Returns: lists of IDs of end groups, linkers and nested end groups

    """
    # List of linker IDs
    linker_list = []
    # List of end group IDs
    endgroup_list = []
    # List of nested end group IDs
    nested_endgroup_list = []

    # Loop over all nodes that have "explicit_atom_ids" as attributes (i.e., atom nodes)
    for node in [x for x in topology_graph.nodes() if ("explicit_atom_ids" in topology_graph.nodes[x].keys())]:
        # Select the ones with explicit_atom_ids == True, meaning they are not within stochastic objects
        if topology_graph.nodes[node]["explicit_atom_ids"]:
            # If it only has one connection, it is an end group
            if topology_graph.degree(node) == 1:
                endgroup_list.append(topology_graph.nodes[node]["ids"])
            # Otherwise, it is a linker
            else:
                linker_list.append(topology_graph.nodes[node]["ids"])
        else:
            # If explicit_atom_ids != True and it only has one connection, it is a nested end group
            if topology_graph.degree(node) == 1:
                nested_endgroup_list.append(topology_graph.nodes[node]["ids"])

    return endgroup_list, linker_list, nested_endgroup_list


def find_bonding_descriptor_ids(topology_graph):
    """
    Finds the IDs of bonding descriptor nodes. Separates bonding descriptor nodes into 2 categories:
        -   Non-nested: bonding descriptors associated with the base (non-nested) stochastic objects
        -   Nested: bonding descriptors of nested stochastic objects that are associated with the end of side chains.
            For example: bonding descriptors of end of graft chains, but not bonding descriptors of segmented polymers.
    Args:
        topology_graph: topology graph

    Returns: lists of IDs of non-nested and nested bonding descriptor nodes

    """
    # List of non-nested bonding descriptor IDs
    non_nested_bd_id_list = []

    # List of nested bonding descriptor IDs
    nested_bd_id_list = []

    # Bonding descriptor characters
    bd_chars = ["$", "<", ">"]

    # Loop over all nodes that do not have "explicit_atom_ids" as attribute
    for node in [x for x in topology_graph.nodes() if not ("explicit_atom_ids" in topology_graph.nodes[x].keys())]:
        # If it is a bonding descriptor, add to list
        is_bd = any([x for x in bd_chars if x in topology_graph.nodes[node]["symbol"]])
        if is_bd:
            # Take the bonding descriptor level (if 0, it is from the base stochastic object; otherwise, it is nested)
            bd_level = topology_graph.nodes[node]["level"]
            if bd_level == 0:
                non_nested_bd_id_list.append(topology_graph.nodes[node]["ids"])
            else:
                # Get neighbors
                neighbors = set(nx.all_neighbors(topology_graph, node))

                # Check the number of neighbors that have lower level
                qtt_lower_level_neighbor = 0
                for neighbor in neighbors:
                    neighbor_level = topology_graph.nodes[neighbor]["level"]
                    if neighbor_level < bd_level:
                        qtt_lower_level_neighbor += 1

                # If it is only connected to one neighbor of lower level, it represents an end of side chain
                if qtt_lower_level_neighbor == 1:
                    nested_bd_id_list.append(topology_graph.nodes[node]["ids"])

    return non_nested_bd_id_list, nested_bd_id_list


def generate_shortest_path(graph, start, end):
    """
    Generates shortest path between 2 nodes in a graph
    Args:
        graph: graph
        start: starting node
        end: ending node

    Returns: list of nodes
    """
    try:
        return nx.shortest_path(graph, start, end)
    except:
        return []


def generate_simple_paths(graph, start, end):
    """
    Generates all simple paths between 2 nodes in a graph
    Args:
        graph: graph
        start: starting node
        end: ending node

    Returns: list of nodes
    """
    try:
        return nx.all_simple_paths(graph, start, end)
    except:
        return []


def group_bonding_descriptors(bd_nodes, topology_graph, lower_level_group):
    """
    This function checks whether a group of bonding descriptors must be tied, i.e., if they represent starting or
    ending states together. This must happen when the connection between them is through repeating units, not
    atoms at a lower level (e.g., end groups).
    Args:
        bd_nodes: list of bonding descriptor nodes
        topology_graph: topology graph
        lower_level_group: list of atoms at a lower level

    Returns: nested list, in which items in the same sublist are tied

    """

    # Check all bonding descriptors that must be tied together
    difference = True
    bd_nodes = [bd_nodes]
    # Partition interval, grouping only the bonding descriptors that are not connected by non-stochastic atoms
    while difference:

        # Keep old value of bd_nodes
        old_bd_nodes = copy.deepcopy(bd_nodes)

        # Loop over each group of bonding descriptors
        for group_index in range(len(bd_nodes)):
            # Set variables that help group the bonding descriptors
            refinements = []
            # Compare the bonding descriptors within a group
            for i in range(len(bd_nodes[group_index])):
                for j in range(len(bd_nodes[group_index])):
                    # Get nodes from indices
                    i_state = bd_nodes[group_index][i]
                    j_state = bd_nodes[group_index][j]
                    must_tie = False
                    if i_state != j_state:
                        # Generate shortest path
                        _path = generate_shortest_path(topology_graph, i_state, j_state)
                        # If the only connections between them are through repeating units, they need to be tied
                        for node in _path:
                            must_tie = True
                            if node in lower_level_group:
                                must_tie = False
                                break
                    if must_tie:
                        # They are assigned the same class
                        refinements.append([i_state, j_state])
                    else:
                        refinements.append([i_state])
                        refinements.append([j_state])
            else:
                # If there are more than 2 lists with the same elements, merge them
                for i in range(len(refinements)):
                    for j in range(len(refinements)):
                        listi = refinements[i]
                        listj = refinements[j]
                        # If they share at least one element
                        if not set(listi).isdisjoint(listj):
                            # Merge them
                            refinements[i] = list(set(listi + listj))
                # Remove duplicates
                refinements = list(set(map(lambda i: tuple(sorted(i)), refinements)))
                refinements = [list(x) for x in refinements]  # Convert tuples to list
                # Replace the initial group by the first item of refinements and then append the others
                if refinements:
                    bd_nodes[group_index] = sorted(refinements[0])
                if len(refinements) > 1:
                    for group in refinements[1:]:
                        bd_nodes.append(sorted(group))
        # Check if equivalent_states changed
        difference = sorted(bd_nodes) != sorted(old_bd_nodes)

    # Reformat list: convert lists of sublists of 1 element into integer and remove empty sublists
    _temp = []
    for x in bd_nodes:
        if len(x) > 1:
            _temp.append(x)
        elif x:
            _temp.append(x[0])
    bd_nodes = _temp
    # bd_nodes = [sorted(x) if len(x) > 1 else x[0] if x for x in bd_nodes]

    return bd_nodes


def choose_bonding_descriptors(bd_nodes, topology_graph, lower_level_group):
    """
    This function chooses bonding descriptors by removing the ones that are along the path between any two
    atoms at a lower level. It also removes the conjugate descriptors from the same level
    Args:
        bd_nodes: list of bonding descriptor nodes
        topology_graph: topology graph
        lower_level_group: list of atoms at a lower level (e.g., end groups)

    Returns: list of chosen bonding descriptors

    """
    # Generate all possible pairs of non-stochastic groups
    removed_nodes = []
    for _start, _end in itertools.combinations(lower_level_group, 2):
        # Generate paths (forward and reverse because it is a DiGraph)
        paths_forward = list(generate_simple_paths(topology_graph, _start, _end))
        paths_reverse = list(generate_simple_paths(topology_graph, _end, _start))
        paths = paths_forward + paths_reverse
        for _path in paths:
            # Eliminate bonding descriptors along _path from bd_nodes
            for node in _path:
                if node in bd_nodes:
                    bd_nodes.remove(node)
                    removed_nodes.append(node)

    # Remove conjugate descriptors from the same level
    for bd1 in removed_nodes:
        for bd2 in bd_nodes:
            # If they are connected by end groups, it should not be removed. This is because they are not in the same
            # stochastic object
            if_same_object = True
            for _path in list(generate_simple_paths(topology_graph, bd1, bd2)) + list(generate_simple_paths(topology_graph, bd2, bd1)):
                for n in _path:
                    if n in lower_level_group:
                        if_same_object = False
                        break
                if if_same_object == False:
                    break
            if_same_node = bd1 == bd2
            if_conjugate = topology_graph.nodes[bd1]["symbol"] == get_comp(topology_graph.nodes[bd2]["symbol"])
            # If both nodes are not the same and are a conjugate pair, remove bd2 from bd_nodes
            if (not if_same_node) and if_conjugate and if_same_object:
                bd_nodes.remove(bd2)


    return bd_nodes


def get_nonstochastic_atoms(topology_graph, atomistic_graph):
    """
    This function returns a list of non-stochastic atoms (nodes), i.e., atoms within end groups or linkers, from
    the topology graph.
    Args:
        topology_graph: topology graph
        atomistic_graph: atomistic graph

    Returns: list of non-stochastic nodes from topology graph and list of non-stochastic nodes from atomistic graph

    """
    # Get explicit atoms
    explicit_atoms = [node for node, att in topology_graph.nodes(data=True) if
                             ("explicit_atom_ids" in att.keys()) and (att["explicit_atom_ids"])]

    # return explicit_atoms

    # # Get cycles from topology graph
    # topology_cycles = find_topology_cycles(topology_graph)
    # # Extract atoms from topology cycles and remove duplicates
    # topology_cycle_atoms = list(set(element for innerList in topology_cycles for element in innerList))
    #
    # # Get atoms from level 0 (the ones associated with the outermost stochastic object) that are not within cycles
    # topology_non_stochastic_atoms = [n for n in topology_graph.nodes if
    #                                  (topology_graph.nodes[n]["level"] == 0) and (n not in topology_cycle_atoms)
    #                                  ]
    #
    # # Get ids of these nodes in the atomistic graph
    # non_stochastic_atom_ids = [topology_graph.nodes[n]["ids"] for n in topology_non_stochastic_atoms]
    # # Get non-stochastic nodes from atomistic graph
    # atomistic_non_stochastic_atoms = [n for n in atomistic_graph.nodes if
    #                                   (atomistic_graph.nodes[n]["ids"] in non_stochastic_atom_ids)
    #                                   ]
    #
    #
    # return topology_non_stochastic_atoms, atomistic_non_stochastic_atoms
    return explicit_atoms, []


def define_starts_and_ends(topology_graph, atomistic_graph):
    """
    This function defines the groups of atoms or bonding descriptors that will lead to the starting transitions and
    accepting states of the tree automata.
    Args:
        topology_graph: topology graph
        atomistic_graph: atomistic graph

    Returns: list of dictionaries. Each element has a key "start" that is a list of the starts and a key "end"
    that is a list of ends. If an element is a list of groups, such groups should be starts or ends at the same time.

    """

    # Get all bonding descriptor nodes
    all_bd_nodes = [node for node, att in topology_graph.nodes(data=True) if
                    any([x in att["symbol"] for x in ("$", "<", ">")])]
    # Get all non-stochastic groups (end groups or linkers)
    non_stochastic_groups, _ = get_nonstochastic_atoms(topology_graph, atomistic_graph)
    # Get all non-stochastic end groups (not linkers)
    end_groups = [node for node in non_stochastic_groups if topology_graph.degree(node) == 1]

    # FIRST STEP: Define starts and ends from atoms and bonding descriptors within non-nested stochastic objects -------
    # Get all bonding descriptors at level 0
    bd_nodes = [x for x in all_bd_nodes if topology_graph.nodes[x]["level"] == 0]
    # Remove bonding descriptors along the path of any 2 atoms from non_stochastic_groups
    bd_nodes = choose_bonding_descriptors(bd_nodes=bd_nodes, topology_graph=topology_graph,
                                          lower_level_group=non_stochastic_groups)
    # Group bonding descriptors that must be tied
    bd_nodes = group_bonding_descriptors(bd_nodes=bd_nodes, topology_graph=topology_graph,
                                         lower_level_group=non_stochastic_groups)
    # List of nodes that can be starts or ends
    starts_and_ends = bd_nodes + end_groups

    # SECOND STEP: Define starts from atoms and bonding descriptors within nested stochastic objects -------------------
    # List of nodes that can only be starts
    starts = []
    # Get all levels
    levels = set(sorted([topology_graph.nodes[x]["level"] for x in topology_graph.nodes()
                         if topology_graph.nodes[x]["level"] != 0]
                        )
                 )
    # Look for starts at each level
    for l in levels:
        # Get all bonding descriptors at the level l
        bd_level = [x for x in all_bd_nodes if topology_graph.nodes[x]["level"] == l]
        # Get all atoms at level l-1
        atoms_l_1 = [x for x in topology_graph.nodes()
                     if x not in all_bd_nodes if topology_graph.nodes[x]["level"] == l - 1]
        # Remove bonding descriptors along the path of any 2 atoms from atom_l_1
        bd_level = choose_bonding_descriptors(bd_nodes=bd_level, topology_graph=topology_graph,
                                              lower_level_group=atoms_l_1)
        # Group bonding descriptors that must be tied
        bd_level = group_bonding_descriptors(bd_nodes=bd_level, topology_graph=topology_graph,
                                             lower_level_group=atoms_l_1)
        # Now get end groups (at level l-1, not in start_and_ends already and non-stochastic)
        end_groups_l_1 = [x for x in topology_graph.nodes()
                          if (topology_graph.nodes[x]["level"] == l - 1 and topology_graph.degree(x) == 1)
                          and (x not in starts_and_ends)
                          and (x not in non_stochastic_groups)]

        # Add to starts
        starts += bd_level + end_groups_l_1

    # Convert nodes to group ids
    starts = [[topology_graph.nodes[x]["ids"]] if type(x) != list
              else sorted([topology_graph.nodes[y]["ids"] for y in x])
              for x in starts]
    starts_and_ends = [[topology_graph.nodes[x]["ids"]] if type(x) != list
                       else sorted([topology_graph.nodes[y]["ids"] for y in x])
                       for x in starts_and_ends]

    # Generate all possible combinations of starts and ends
    start_end_keys = ["start", "end"]
    start_end_list = []
    # Pick one node from starts_and_ends to be the end
    for end in starts_and_ends:
        # All the other nodes are going to be the start
        if len(starts_and_ends) == 1:  # If it only has one element, it must be both the start and the end
            start = list(np.concatenate([x for x in starts_and_ends + starts]))
        else:
            start = list(np.concatenate([x for x in starts_and_ends + starts if x != end]))
        # Add to start_end_list
        start_end_list.append(dict(zip(start_end_keys, [start, end])))

    # Filter out the combinations in which end groups connected to the same bonding descriptor with different
    # bond types are starts (or ends) simultaneously. This is because it is impossible to traverse the graph in this manner
    filtered_start_end_list1 = []
    # Checking starts
    for starts_ends in start_end_list:
        # Get starts
        starts = starts_ends[start_end_keys[0]]
        remove = False

        # Check the starts
        for start1 in starts:
            # Check if it is a bonding descriptor
            group_node_index = [x for x in topology_graph.nodes() if topology_graph.nodes[x]["ids"] == start1][0]
            is_bonding_descriptor = check_if_bonding_descriptor(topology_graph.nodes[group_node_index]["symbol"])
            # If it is a bonding descriptor, skip it
            if is_bonding_descriptor:
                continue
            # Get the bonding descriptor connected to it
            bd1 = [x for x in find_neighbors(topology_graph, group_node_index)
                   if check_if_bonding_descriptor(topology_graph.nodes[x]["symbol"])][0]
            # Get node
            node1 = [x for x in find_neighbors(atomistic_graph, bd1) if atomistic_graph.nodes[x]["ids"] == start1][0]
            # Get the bond type
            bond_type1 = atomistic_graph.edges[node1, bd1]["bond_type"]
            for start2 in starts:
                # Check if it is a bonding descriptor
                group_node_index = [x for x in topology_graph.nodes() if topology_graph.nodes[x]["ids"] == start2][0]
                is_bonding_descriptor = check_if_bonding_descriptor(topology_graph.nodes[group_node_index]["symbol"])
                # If it is a bonding descriptor, skip it
                if is_bonding_descriptor:
                    continue
                # Get the bonding descriptor connected to it
                bd2 = [x for x in find_neighbors(topology_graph, group_node_index)
                       if check_if_bonding_descriptor(topology_graph.nodes[x]["symbol"])][0]
                # Get node
                node2 = [x for x in find_neighbors(atomistic_graph, bd2) if atomistic_graph.nodes[x]["ids"] == start2][0]
                # Get the bond type
                bond_type2 = atomistic_graph.edges[node2, bd2]["bond_type"]
                # If both bonding descriptors are the same and the bond types are different, remove it
                if bd1 == bd2 and bond_type1 != bond_type2:
                    remove = True
                    break
            # If at least one pair is found, it is enough for removing this combination
            if remove == True:
                break

        # If it does not have to be removed, add to filtered_start_end_list
        if not remove:
            filtered_start_end_list1.append(starts_ends)

    # Checking ends
    filtered_start_end_list2 = []
    for starts_ends in filtered_start_end_list1:
        # Get ends
        ends = starts_ends[start_end_keys[1]]
        remove = False

        # Check the starts
        for end1 in ends:
            # Check if it is a bonding descriptor
            group_node_index = [x for x in topology_graph.nodes() if topology_graph.nodes[x]["ids"] == end1][0]
            is_bonding_descriptor = check_if_bonding_descriptor(topology_graph.nodes[group_node_index]["symbol"])
            # If it is a bonding descriptor, skip it
            if is_bonding_descriptor:
                continue
            # Get the bonding descriptor connected to it
            bd1 = [x for x in find_neighbors(topology_graph, group_node_index)
                   if check_if_bonding_descriptor(topology_graph.nodes[x]["symbol"])][0]
            # Get node
            node1 = [x for x in find_neighbors(atomistic_graph, bd1) if atomistic_graph.nodes[x]["ids"] == end1][0]
            # Get the bond type
            bond_type1 = atomistic_graph.edges[node1, bd1]["bond_type"]
            for end2 in ends:
                # Check if it is a bonding descriptor
                group_node_index = [x for x in topology_graph.nodes() if topology_graph.nodes[x]["ids"] == end2][0]
                is_bonding_descriptor = check_if_bonding_descriptor(topology_graph.nodes[group_node_index]["symbol"])
                # If it is a bonding descriptor, skip it
                if is_bonding_descriptor:
                    continue
                # Get the bonding descriptor connected to it
                bd2 = [x for x in find_neighbors(topology_graph, group_node_index)
                       if check_if_bonding_descriptor(topology_graph.nodes[x]["symbol"])][0]
                # Get node
                node2 = [x for x in find_neighbors(atomistic_graph, bd2) if atomistic_graph.nodes[x]["ids"] == end2][0]
                # Get the bond type
                bond_type2 = atomistic_graph.edges[node2, bd2]["bond_type"]
                # If both bonding descriptors are the same and the bond types are different, remove it
                if bd1 == bd2 and bond_type1 != bond_type2:
                    remove = True
                    break
            # If at least one pair is found, it is enough for removing this combination
            if remove == True:
                break

        # If it does not have to be removed, add to filtered_start_end_list
        if not remove:
            filtered_start_end_list2.append(starts_ends)

    start_end_list = filtered_start_end_list2

    return start_end_list


def find_all_paths(start, end, atomistic):
    """
    This function finds all possible paths from starts to ends
    Args:
        start: list of starts
        end: list of ends
        atomistic: atomistic graph

    Returns: list of paths

    """
    _paths = []
    for s in start:
        for e in end:
            for path in nx.all_simple_paths(atomistic, s, e):
                _paths.append(path)

    return _paths


def choose_heaviest_longest_path(start, end, atomistic):
    """
    From possible starts and ends, this function determines the longest. This is used to choose
    the path along an end group that the algorithm will use to generate the alphabets.
    To untie, it chooses the heaviest path with heaviest atoms by the end.
    Args:
        start: list of starts
        end: list of ends
        atomistic: atomistic graph

    Returns: the longest and heaviest path

    """
    # Find all paths between starts and end
    _paths = find_all_paths(start, end, atomistic)
    # If there is no path, return the start
    if not _paths:
        return start

    # Choose the longest paths
    size_longest_path = len(_paths[0])
    for path in _paths:
        size_path = len(path)
        if size_path > size_longest_path:
            size_longest_path = size_path
    longest_path = [x for x in _paths if len(x) == size_longest_path]

    # Among the longest paths, choose the heaviest one and the one with the heaviest atoms by the end
    periodic_table = Chem.GetPeriodicTable()
    heaviest_path = sorted(longest_path,
                           key=lambda p: (
                           sum([periodic_table.GetAtomicWeight(atomistic.nodes[node]["symbol"]) + atomistic.nodes[node]["num_hs"]  for node in p]),
                           sum([(periodic_table.GetAtomicWeight(atomistic.nodes[node]["symbol"]) + atomistic.nodes[node]["num_hs"]) * (i + 1)**2 for i, node in    # The position is squared because we want to give a higher weight to it
                                enumerate(p)])
                           ),
                           reverse=True)[0]

    return heaviest_path


def get_longest_path(start, end, atomistic):
    """
    Returns the longest path between a set of possible starts and ends
    Args:
        start: list of starts
        end: list of ends
        atomistic: atomistic graph

    Returns: longest path. It there are more than 1 longest paths, they are all returned

    """
    # Find all paths between starts and end
    _paths = []
    for s in start:
        for e in end:
            for path in nx.all_simple_paths(atomistic, s, e):
                _paths.append(path)
    # If there is no path, return the start
    if not _paths:
        return [start]

    # Choose the longest paths
    size_longest_path = len(_paths[0])
    for path in _paths:
        size_path = len(path)
        if size_path > size_longest_path:
            size_longest_path = size_path
    longest_path = [x for x in _paths if len(x) == size_longest_path]

    return longest_path


def get_endgroup_end(atom_group, atomistic):
    """
    This function returns the ends of an end group, given its group id. If the end group ends in a cycle, choose the
    longest or heaviest path in the cycle
    Args:
        atom_group:
        atomistic:

    Returns:

    """
    ends = [x for x in atom_group if atomistic.degree(x) == 1]

    # If the ends are cycles, take the atoms connected to bonding descriptors and choose the longest path from it
    if not ends:
        # Choose atom connected to bonding descriptor
        connected_to_bd = []
        for n in atom_group:
            for neighbor in atomistic.neighbors(n):
                if check_if_bonding_descriptor(atomistic.nodes[neighbor]["symbol"]):
                    connected_to_bd.append(n)
                    break
        # Choose longest path
        ends = get_longest_path(start=connected_to_bd, end=atom_group, atomistic=atomistic)
        # Sort paths according to weight
        ends = sort_paths(ends, atomistic)[0]

    return ends


def sort_paths(paths, atomistic):
    """
    Sort paths based on path weight. Heaviest path comes first.
    First, it sorts by weight. To untie, it takes the path whose heaviest elements are by the end
    Args:
        paths: list of paths
        atomistic: atomistic graph

    Returns: sorted list of paths

    """
    # Define periodic table
    periodic_table = Chem.GetPeriodicTable()

    # Sort based on weight
    sort = sorted(paths,
                  key=lambda p: (sum([periodic_table.GetAtomicWeight(atomistic.nodes[node]["symbol"] + atomistic.nodes[node]["num_hs"]) for node in p]),
                                 sum([periodic_table.GetAtomicWeight(atomistic.nodes[node]["symbol"] + atomistic.nodes[node]["num_hs"]) * (i + 1) for
                                      i, node in enumerate(p)])
                                 ),
                  reverse=True)

    return sort


def find_shortest_paths(non_descriptors, atomistic, atomistic_bonds, atomistic_ids):
    """
    This function finds the shortest paths along the repeat units
    Args:
        non_descriptors:
        atomistic:
        atomistic_bonds:
        atomistic_ids:

    Returns:

    """

    # List of shortest paths along the repeat units
    shortest_paths = []

    # Nested list of the ends of the end groups. The first item is the start(s) and the second is the end(s)
    end_group_paths = []

    for id in non_descriptors:
        atom = {"1": [], "2": []}
        for key in atomistic_bonds:
            # This if statements select the non descriptor nodes (atoms) that are in the RU because it must have
            # a 1_SINGLE and a 2_SINGLE in the same group
            if "1" in atomistic_bonds[key] and atomistic_ids[key[0]] == id:
                atom["1"].append(key[0])
            elif "1" in atomistic_bonds[key] and atomistic_ids[key[1]] == id:
                atom["1"].append(key[1])
            elif "2" in atomistic_bonds[key] and atomistic_ids[key[0]] == id:
                atom["2"].append(key[0])
            elif "2" in atomistic_bonds[key] and atomistic_ids[key[1]] == id:
                atom["2"].append(key[1])

        # For repeating units, find the path between starts and ends
        for start in atom["1"]:
            for end in atom["2"]:
                for path in nx.all_simple_paths(atomistic, start, end):
                    found = True
                    for node in path:
                        if atomistic_ids[node] != id:
                            found = False
                            break
                    if found:
                        shortest_paths.append(path)
                        break

        # For end groups (if one of the atom list is empty)
        if atom["1"] == [] or atom["2"] == []:

            # Get atoms in the group
            atom_group = [k for k, v in atomistic_ids.items() if v == id]
            # If atom["1"] is empty, the start will be an atom of the end group and the end will be atom["2"]
            if atom["1"] == []:
                start = atom_group
                end = atom["2"]
            # If atom["2"] is empty, the start will be atom["1"] and the end will be an atom of the end group
            else:
                start = atom["1"]
                end = atom_group

            # Generate all possible paths from start to end
            _paths = find_all_paths(start=start, end=end, atomistic=atomistic)
            # If the start and end are the same, skip
            if _paths == []:
                _paths = [[start[0], end[0]]]
            # Add paths do paths to list of paths along end groups
            end_group_paths.append(_paths)


    return shortest_paths, end_group_paths


def one_path_connection(non_bd_single_bonds, bonding_descriptor_list, atomistic):
    """
    Finds the bonds between atoms that are only connected by 1 path without bonding descriptors
    Args:
        non_bd_single_bonds: list of single bonds that do not have bonding descriptors
        bonding_descriptor_list: list of bonding descriptor
        atomistic: atomistic graph

    Returns: list of bonds

    """
    # Remove bonding descriptors from graph to prevent it from generating paths with bonding descriptors
    _atomistic = copy.deepcopy(atomistic)
    _atomistic.remove_nodes_from(bonding_descriptor_list)

    # Find bonds to be broken
    break_bond = []
    # Only break single bonds that connect atoms
    for key in non_bd_single_bonds:
        # Get all paths between the 2 bonded atoms
        all_paths = nx.all_simple_paths(_atomistic, key[0], key[1])
        # Remove all paths that have bonding descriptors
        filtered_paths = all_paths
        # filtered_paths = filter(lambda path: set(path).isdisjoint(bonding_descriptor_list), all_paths)
        # Count the number of paths found
        number_of_paths = 0
        for _ in filtered_paths:
            number_of_paths += 1
            if number_of_paths > 1:
                break

        # If it only found one path, save bond to be broken later
        if number_of_paths == 1:
            break_bond.append(key)


    return break_bond


def atoms_in_alphabet(extracted, symbols, state_machine):
    """
    This function determines which atoms will be grouped in an alphabet. This will happen if there are no bonding
    descriptors or state nodes along the path between 2 atoms.
    Args:
        extracted: list of nodes/atoms
        symbols: dictionary whose keys are nodes and values are the symbols (<, >, $ or atom symbol)
        state_machine: state machine without edges to bonding descriptors

    Returns: list of groups of nodes

    """
    # Determine the nodes that will be grouped in alphabets
    alphabet_indices = []
    # Check what atoms must be grouped into 1 alphabet. This will happen if there are no bonding descriptors or
    # state nodes along the path between 2 atoms
    for atom in extracted:
        group = [atom]
        for key in symbols:
            if "<" in symbols[key] or ">" in symbols[key] or "$" in symbols[key]:
                continue
            # Check if there is at least one path between atom and key that does not have a bonding descriptor
            # or state node
            for _ in nx.all_simple_paths(state_machine, key, atom):
                group.append(key)
                break
        if atom in symbols and sorted(group) not in alphabet_indices:
            alphabet_indices.append(sorted(group))

    return alphabet_indices


def alphabet_to_SMILES(group, state_machine, symbols, formal_charge, is_aromatic, right, left, bonds_object, bond_stereo_object, bond_dir_object):
    """
    This function converts a group of atoms from an alphabet into a SMILES string
    Args:
        group: group of nodes
        state_machine: state machine
        symbols: dict that maps nodes onto symbols
        formal_charge: dict that maps nodes onto formal charges
        is_aromatic: dict that maps nodes onto aromaticity
        right: list of right states
        left: left state
        bonds_object:

    Returns: smiles of the alphabet and a dict that maps the symbol of the heavy atom to the node

    """
    # Initialize an empty molecule
    mol = Chem.RWMol()
    node_to_idx = {}
    # This will define the index associated to Bk
    count = 2
    # This will define how the input will be sorted
    heavyatom_node_dict = {}

    # First, add the atoms that correspond to the inputs. This guarantees that the indices associated
    # with Bk's will be smaller for the inputs than for the output
    for node in state_machine.nodes():
        if node in group:
            if node in symbols and not (
                    "<" in symbols[node] or ">" in symbols[node] or "$" in symbols[node]):
                atom = Chem.Atom(symbols[node])
                atom.SetFormalCharge(formal_charge[node])
                atom.SetIsAromatic(is_aromatic[node])
                idx = mol.AddAtom(atom)
                node_to_idx[node] = idx
            for r in right:
                if node == r:
                    # Add to node_order so that we can sort the input later
                    heavyatom_node_dict[f"Cf:{count}"] = node
                    end2 = Chem.Atom("Cf")
                    end2.SetAtomMapNum(count)  # Sets the number the Bk atom is associated with
                    count += 1
                    idx2 = mol.AddAtom(end2)
                    mol.AddBond(idx, idx2, rdkit.Chem.rdchem.BondType.SINGLE)
            if node == left:
                # Add to node_order so that we can sort the input later
                heavyatom_node_dict[f"Bk:{1}"] = node
                end1 = Chem.Atom("Bk")
                end1.SetAtomMapNum(1)  # Sets the number the Bk atom is associated with to 1 so it is always the first
                idx2 = mol.AddAtom(end1)
                mol.AddBond(idx, idx2, rdkit.Chem.rdchem.BondType.SINGLE)

    # Add missing bonds to the molecule
    already_added = set()
    for edge in state_machine.edges():
        first, second = edge
        if first in group and second in group:
            ifirst = node_to_idx[first]
            isecond = node_to_idx[second]
            b = bonds_object[first, second]
            if tuple(sorted([ifirst, isecond])) not in already_added:
                # mol.AddBond(ifirst, isecond, b)
                bond_idx = mol.AddBond(ifirst, isecond, b) - 1
                bond = mol.GetBondWithIdx(bond_idx)
                # Add bond stereochemistry and direction
                bond.SetStereo(bond_stereo_object[first, second])
                bond.SetBondDir(bond_dir_object[first, second])
                already_added.add(tuple(sorted([ifirst, isecond])))

    # Generate the SMILES representation
    Chem.SanitizeMol(mol)
    smiles = Chem.MolToSmiles(mol)
    # Replace heavy atoms by *
    smiles = smiles.replace("[Bk:", "[*:")
    smiles = smiles.replace("[Cf:", "[*:")

    return smiles, heavyatom_node_dict


def define_left_right_states(terminals, extracted, reverse_extracted=False):
    """
    Define right and left states. They will later become input and output states, respectively
    Args:
        terminals: list of terminals
        extracted: list of nodes
        reverse_extracted: if True, reverses extracted
    Returns:

    """

    # Sort elements to indicate which will be input and output
    _terminals = copy.deepcopy(terminals)
    for i in range(len(terminals)):
        to_end = -1
        to_beginning = -1
        if len(terminals[i]) == 3 and "1" in terminals[i][2]:
            to_end = terminals[i]
        elif len(terminals[i]) == 3 and "2" in terminals[i][2]:
            to_beginning = terminals[i]
        if to_end != -1:
            _terminals.append(_terminals.pop(_terminals.index(to_end)))
        elif to_beginning != -1:
            _terminals.insert(0, _terminals.pop(_terminals.index(to_beginning)))
    terminals = _terminals

    ends = dict()
    if reverse_extracted:
        reverse_extracted = copy.deepcopy(extracted)
        reverse_extracted.reverse()
        for i in range(len(terminals)):
            if len(terminals[i]) == 3 and i == 0:
                ends[tuple(terminals[i])] = -1
            elif len(terminals[i]) == 3 and i != 0:
                ends[tuple(terminals[i])] = 10000
            else:
                ends[tuple(terminals[i])] = reverse_extracted.index(terminals[i][1])
        ends = sorted(ends.items(), key=lambda item: item[1])
        terminals = []
        for e in ends:
            terminals.append(e[0])
        left = terminals[0][0]
        left_states = terminals[0][1]
        right = []
        right_states = []
        for i in range(1, len(terminals)):
            right.append(terminals[i][0])
            right_states.append(terminals[i][1])

        if len(ends) == 1:
            right = [terminals[0][0]]
            right_states = [terminals[0][1]]
            left = []
            left_states = []

    else:
        for i in range(len(terminals)):
            if len(terminals[i]) == 3 and i == 0:
                ends[tuple(terminals[i])] = -1
            elif len(terminals[i]) == 3 and i != 0:
                ends[tuple(terminals[i])] = 10000
            else:
                ends[tuple(terminals[i])] = extracted.index(terminals[i][1])
        ends = sorted(ends.items(), key=lambda item: item[1])
        terminals = []
        for e in ends:
            terminals.append(e[0])
        left = terminals[0][0]
        left_states = terminals[0][1]
        right = []
        right_states = []
        for i in range(1, len(terminals)):
            right.append(terminals[i][0])
            right_states.append(terminals[i][1])

    return left, left_states, right, right_states


def set_terminals(state_machine, group, symbols, bonds):
    """
    Sets terminals: a list of [node1, node2, bond_type], where node1 and node2 are connected. If one of the nodes is a
    bonding descriptor, add the bond type as a third element
    Args:
        state_machine: state machine
        group: group of nodes
        symbols: dict that maps nodes onto symbols
        bonds: dict that maps bonds onto bond types

    Returns: terminals

    """
    terminals = []
    # Get the terminals of the group (alphabet), which will connect to bonding descriptors or states
    for key in group:
        for neighbor in state_machine[key]:
            if neighbor not in symbols:
                terminals.append([key, neighbor])
            elif "<" in symbols[neighbor] or ">" in symbols[neighbor] or "$" in symbols[neighbor]:
                terminals.append([key, neighbor, bonds[tuple(sorted([neighbor, key]))]])
    return terminals


def direction(graph, extracted, start_atom, symbols):
    """
    This function does a breadth-first search on the graph, starting from start_atoms.
    It does not traverse bonding descriptors.
    Args:
        graph: graph
        extracted: list of nodes that were traversed
        start_atom: start
        symbols: dict that maps nodes onto symbols

    Returns: list of nodes

    """
    if start_atom not in extracted:
        extracted.append(start_atom)
    next_atoms = []
    for n in graph[start_atom]:
        if n in symbols and ("<" in symbols[n] or ">" in symbols[n] or "$" in symbols[n]):
            continue
        if n not in extracted:
            extracted.append(n)
            next_atoms.append(n)
    for n in next_atoms:
        extracted = direction(graph, extracted, n, symbols)
    return extracted


def sort_right_states(state_machine, right_states, smiles, heavyatom_node_dict):
    """
    This function sorts right_states based on the index of the heavy atoms in the SMILES string. Right_states will
    later become the inputs of the state machine. Thus, the input states will be sorted so that their position
    corresponds to the index of the connecting points of the SMILES string.
    Args:
        state_machine: state machine
        right_states: list of right states
        smiles: SMILES representation of the alphabet
        heavyatom_node_dict: dict that maps heavy atoms onto nodes

    Returns: sorted right_states

    """

    # Create a dict whose keys are the heavy atoms and values are their positions in the string
    heavyatom_position = {heavy: smiles.find(heavy) for heavy, _ in heavyatom_node_dict.items()}
    # Sort based on their position
    heavyatom_position = dict(sorted(heavyatom_position.items(), key=lambda item: item[1]))

    # Sort heavyatom_node_dict based on the positions of the heavy atoms in the string
    heavyatom_node_dict = dict(sorted(heavyatom_node_dict.items(),
                                      key=lambda item: heavyatom_position[item[0]],
                                      reverse=True))  # Reverse because I want to assign weights to lowest priority first
    # Dictionary that will define the positions of the inputs
    weights = {}
    # Sort right states because, if all weights are the same, it does not change
    right_states = sorted(right_states)
    # Loop over the heavy atoms and assign weights to inputs based on the positions of the heavy atoms
    count = 0
    for heavy_atom, heavy_node in heavyatom_node_dict.items():
        # Only choose Cf because they represent inputs
        if "Cf" in heavy_atom:
            # Get all neighbors
            _neighbors = find_neighbors(state_machine, heavy_node)
            # If any node in the input is a neighbor, assign weight
            for n in right_states:
                if n in _neighbors:
                    weights[n] = count
                    # sorted_input.append(n)
                    # break
        # Update count
        count += 1

    # Finally sort right states
    right_states = sorted(right_states, key=lambda x: weights[x])

    return right_states


def add_to_transitions(left_states, smiles, right_states, alphabets, transitions, alpha_count, caps, group, atomistic):
    # Sort inputs as they appear in the smiles
    if smiles in alphabets:
        transitions.append([left_states, alphabets[smiles], smiles, right_states])
    else:
        # If it is a starting Es transition, add id to distinguish
        if "Es_id" in atomistic.nodes[group[0]].keys():
            _id = str(atomistic.nodes[group[0]]["Es_id"])
            smiles = smiles.replace("Es", f"Es_{_id}")
        alphabets[smiles] = caps[alpha_count]
        alpha_count += 1
        transitions.append([left_states, alphabets[smiles], smiles, right_states])

    return alpha_count, smiles


def bonds_to_transitions(bonds, state_machine, simplified_state_machine, alphabets, transitions, caps, atomistic,
                         symbols, checked_nodes, alpha_count):
    """
    This function converts bonds into transitions (inputs, alphabets and outputs)
    Args:
        bonds: list of bonds
        state_machine: state machine
        simplified_state_machine: state machine where edges to bonding descriptor and state nodes were removed
        alphabets: list of alphabets
        transitions: list of transitions
        caps: upper case characters
        atomistic: atomistic graph

    Returns: alpha count

    """
    # List of node ids
    node_ids = nx.get_node_attributes(atomistic, "ids")
    id_list = list(set(node_ids.values()))

    # Dict that will keep track of the RU alphabets. Each RU id will be associated with a dict whose keys are SMILES
    # and values are alphabets. This is used for defining the path along end groups later on.
    ru_alphabets = {i: {} for i in id_list}

    # Convert bonds into transitions
    for b in bonds:

        # Only proceed if the bond is type 2
        if "2" in bonds[b]:

            # Define the start
            if "<" in symbols[b[0]] or ">" in symbols[b[0]] or "$" in symbols[b[0]]:
                start = b[1]
            else:
                start = b[0]

            # Get nodes of the repeat unit starting from start
            extracted = direction(state_machine, [], start, symbols)

            # If all atoms in extracted have already been checked, skip
            if all([x in checked_nodes for x in extracted]):
                continue

            # Determine the nodes that will be grouped in alphabets
            alphabet_indices = atoms_in_alphabet(extracted, symbols, simplified_state_machine)

            # For each group of atoms (atoms that will be grouped into one alphabet)
            for group in alphabet_indices:
                symbols = nx.get_node_attributes(state_machine, "symbol")
                formal_charge = nx.get_node_attributes(state_machine, "formal_charge")
                is_aromatic = nx.get_node_attributes(state_machine, "is_aromatic")
                bonds_object = nx.get_edge_attributes(state_machine, "bond_type_object")
                bond_dir_object = nx.get_edge_attributes(state_machine, "bond_dir_object")
                bond_stereo_object = nx.get_edge_attributes(state_machine, "bond_stereo_object")

                # Set the terminals: [node1, node2, bond_type], where node1 and node2 are connected. If one of them
                # is a bonding descriptor, it includes bond_type
                terminals = set_terminals(state_machine, group, symbols, bonds)

                # Get left and right states
                left, left_states, right, right_states = define_left_right_states(terminals, extracted)

                # Generate SMILES representations of the alphabets
                smiles, heavyatom_node_dict = alphabet_to_SMILES(group, state_machine, symbols, formal_charge,
                                                                 is_aromatic, right, left, bonds_object, bond_stereo_object, bond_dir_object)

                # Sort inputs based on the index of the heavy atoms in the string
                right_states = sort_right_states(state_machine, right_states, smiles, heavyatom_node_dict)

                # Add transitions to tree automaton
                alpha_count, smiles = add_to_transitions(left_states, smiles, right_states, alphabets, transitions,
                                                         alpha_count, caps, group, atomistic)

                # Add alphabet to list of repeat unit alphabet
                ru_index = node_ids[group[0]]
                ru_alphabets[ru_index][smiles] = alphabets[smiles]

            # Update list of nodes that were checked
            checked_nodes = checked_nodes + extracted

    for b in bonds:

        # Define the start
        if "<" in symbols[b[0]] or ">" in symbols[b[0]] or "$" in symbols[b[0]]:
            start = b[1]
        else:
            start = b[0]

        # Get nodes of the repeat unit starting from start
        extracted = direction(state_machine, [], start, symbols)

        # If a node has not been checked yet, proceed
        if not all([x in checked_nodes for x in extracted]):

            # Determine the nodes that will be grouped in alphabets
            alphabet_indices = atoms_in_alphabet(extracted, symbols, simplified_state_machine)

            for group in alphabet_indices:
                symbols = nx.get_node_attributes(state_machine, "symbol")
                formal_charge = nx.get_node_attributes(state_machine, "formal_charge")
                is_aromatic = nx.get_node_attributes(state_machine, "is_aromatic")
                bonds_object = nx.get_edge_attributes(state_machine, "bond_type_object")
                bond_dir_object = nx.get_edge_attributes(state_machine, "bond_dir_object")
                bond_stereo_object = nx.get_edge_attributes(state_machine, "bond_stereo_object")

                # Set the terminals: [node1, node2, bond_type], where node1 and node2 are connected. If one of them
                # is a bonding descriptor, it includes bond_type
                terminals = set_terminals(state_machine, group, symbols, bonds)

                # Get left and right states
                left, left_states, right, right_states = define_left_right_states(terminals, extracted,
                                                                                  reverse_extracted=True)

                # Generate SMILES representations of the alphabets
                smiles, heavyatom_node_dict = alphabet_to_SMILES(group, state_machine, symbols, formal_charge,
                                                                 is_aromatic, right, left, bonds_object, bond_stereo_object, bond_dir_object)

                # Add transitions to tree automaton
                alpha_count, smiles = add_to_transitions(left_states, smiles, right_states, alphabets, transitions,
                                                         alpha_count, caps, group, atomistic)

                # Add alphabet to list of repeat unit alphabet
                ru_index = node_ids[group[0]]
                ru_alphabets[ru_index][smiles] = alphabets[smiles]

                # Update list of nodes that were checked
                checked_nodes = checked_nodes + extracted

    return alpha_count, ru_alphabets, transitions


def get_index_max_state(table):
    """
    This function gets the index of the maximum state, given the table of transitions
    Args:
        table: table of transitions

    Returns: index of maximum state

    """
    # Take index of the maximum state
    max_state = 0
    for i, row in table.iterrows():
        # If input is a list, take the maximum value
        if not row["input"]:
            candidate = 0
        elif type(row["input"]) == list:
            candidate = max(row["input"])
        else:
            candidate = row["input"]
        # If output is a list, take the maximum value
        if not row["output"]:
            candidate2 = 0
        elif type(row["output"]) == list:
            candidate2 = max(row["output"])
        else:
            candidate2 = row["output"]
        # If output (candidate2) is grater than input (candidate), candidate will be the output
        if candidate2 > candidate:
            candidate = candidate2
        # If candidate is greater than max_state, max state will get cadidate's value
        if candidate > max_state:
            max_state = candidate
    return max_state


def cap_transitions(table, max_state, final_states):
    """
    This function caps uncapped transitions and defined the accepting states
    Args:
        table: table of transitions
        max_state: maximum state
        final_states: list of accepting states

    Returns: accepting states and table of transitions

    """

    for i, row in table.iterrows():
        if not row["output"]:
            # Add an output state
            table.loc[i, "output"] = max_state
            final_states.append(max_state)
            # In these cases, the alphabet only has one input that has index *:1. We need to replace it by :*2
            table.loc[i, "smiles"] = table.loc[i, "smiles"].replace("*:1", "*:2")
            max_state += 1

    return final_states, table


def simplify_state_machines(state_machine, non_bd_symbol_list):
    """
    This function removes edges from nodes that are bonding descriptors or state nodes.
    This is done so the code never finds a path with such nodes in-between. This saves a lot of time
    Args:
        state_machine: state machine
        non_bd_symbol_list: list of nodes that are not bonding descriptors

    Returns:

    """
    simplified_state_machine = copy.deepcopy(state_machine)
    _nodes = copy.deepcopy(simplified_state_machine.nodes())
    for node in _nodes:
        # If node is not an atom, remove its edges
        if node not in non_bd_symbol_list:
            _edges = copy.deepcopy(simplified_state_machine.edges(node))
            for e in _edges:
                simplified_state_machine.remove_edge(e[0], e[1])
    return simplified_state_machine


def create_state_machine(atomistic, break_bond_modified, start = None):
    """
    This function creates a state machine, given the atomistic graph and the bonds to be replaced by states
    Args:
        atomistic: atomistic graph
        break_bond_modified: bonds to be replaced by states
        start: id of the next created state

    Returns: state machine

    """
    # Create state machine
    state_machine = copy.deepcopy(atomistic)

    # Define id of the next created node
    if start == None:
        state = max(list(nx.get_node_attributes(state_machine, "symbol").keys()))
    else:
        state = start + 1

    for breaking in break_bond_modified:
        state += 1
        state_machine.remove_edge(breaking[0], breaking[1])
        state_machine.add_node(state)
        state_machine.add_edge(breaking[0], state)
        state_machine.add_edge(state, breaking[1])

    return state_machine, state


def get_endgroup_atoms(atomistic, topology):
    """
    Gets a list of atoms in end groups. Such end groups can be explicit atoms or grafted end groups
    Args:
        atomistic: atomistic grap
        topology: topology graph

    Returns: list of nodes

    """
    endgroup_ids = [topology.nodes[n]["ids"] for n in topology.nodes if topology.degree(n) == 1]
    endgroup_atoms = [n for n in atomistic.nodes if atomistic.nodes[n]["ids"] in endgroup_ids]

    return endgroup_atoms


def generate_tree_transitions(atomistic, topology, ending_bonding_descriptors, starting_bonding_descriptors):
    """
    This function generates the transitions of a tree automata that represent a polymer
    Args:
        atomistic: polymer atomistic graph
        topology: polymer topology graph
        ending_bonding_descriptors: list of bonding descriptors that will be ending states

    Returns: DataFrame with the transitions

    """

    # Get atomistic graph
    atomistic_symbols = nx.get_node_attributes(atomistic, "symbol")
    atomistic_bonds = nx.get_edge_attributes(atomistic, "bond_type")
    atomistic_ids = nx.get_node_attributes(atomistic, "ids")

    # Get topology graph
    topology_symbols = nx.get_node_attributes(topology, "symbol")
    topology_ids = nx.get_node_attributes(topology, "ids")

    # Get list of bonds with end group atoms
    endgroup_atoms = get_endgroup_atoms(atomistic, topology)
    endgroup_bonds = []
    for b in atomistic_bonds:
        # if one of the atoms is an end group atom, keep it
        if b[0] in endgroup_atoms or b[1] in endgroup_atoms:
            endgroup_bonds.append(b)

    # List of bonding descriptor nodes
    bonding_descriptor_list = [x for x in atomistic_symbols.keys() if check_if_bonding_descriptor(atomistic_symbols[x])]
    # List of bonds that do not connect bonding descriptors and are not single bonds
    non_bd_single_bonds = {nodes: bond for nodes, bond in atomistic_bonds.items()
                            if not (check_if_bonding_descriptor(atomistic.nodes[nodes[0]]["symbol"]) or    # Remove bonding descriptors
                                    check_if_bonding_descriptor(atomistic.nodes[nodes[1]]["symbol"]))
                            and ("SINGLE" in atomistic.edges[nodes]["bond_type"])    # Keep single bonds
                           }

    # Find bonds to be broken. They should be between atoms connected by a single bond and that only have one path
    break_bond = one_path_connection(non_bd_single_bonds, bonding_descriptor_list, atomistic)

    # Find nodes that are not bonding descriptors
    non_descriptors = []
    for key in topology_symbols:
        if not ("<" in topology_symbols[key] or ">" in topology_symbols[key] or "$" in topology_symbols[key]):
            non_descriptors.append(topology_ids[key])

    # Get list of shortest paths along repeat units and end groups. This will later be used to break the bonds
    shortest_paths, end_group_paths = find_shortest_paths(non_descriptors, atomistic, atomistic_bonds, atomistic_ids)

    # Define the bonds that will be broken
    break_bond_modified = []
    for bond in break_bond:
        found = False
        for path in shortest_paths:  # They will be broken if the nodes are in a path from shortest_paths
            if bond[0] in path and bond[1] in path:
                found = True
                break
        if found:
            break_bond_modified.append(bond)

    # Create state machine
    state_machine, max_node = create_state_machine(atomistic, break_bond_modified)
    # Remove end group atoms
    state_machine.remove_nodes_from(endgroup_atoms)

    # traverse to get directed graph
    symbols = nx.get_node_attributes(state_machine, "symbol")
    # Get all bonds from state machine. Assign 0 to bonds with "1" or "2" and 1 to the rest in order to sort
    bonds = {k: (v, 0 if ("1" in v or "2" in v) else 1)
             for k, v in nx.get_edge_attributes(state_machine, "bond_type").items()}
    # Sort bonds so that bonds with "1" or "2" are always checked first. This is essential
    bonds = sorted(bonds.items(), key=lambda item: item[1][1])
    # Remove numbers used for sorting
    bonds = {b[0]: b[1][0] for b in bonds}

    # List of alphabets
    alphabets = dict()
    caps = list(string.ascii_uppercase)
    alpha_count = 0
    # List of transitions
    transitions = []
    # List of nodes that have already been checked
    checked_nodes = []

    # List of bonding descriptors
    bonding_descriptor_list = [x for x in symbols.keys() if check_if_bonding_descriptor(symbols[x])]
    # List of symbols
    symbol_list = list(symbols.keys())
    # List of symbols that are not bonding descriptors
    non_bd_symbol_list = [x for x in symbol_list if x not in bonding_descriptor_list]

    # Remove edges from nodes that are bonding descriptors or state nodes
    simplified_state_machine = simplify_state_machines(state_machine, non_bd_symbol_list)

    # Convert repeat unit bonds into transitions
    alpha_count, ru_alphabets, transitions = bonds_to_transitions(bonds, state_machine, simplified_state_machine,
                                                                         alphabets, transitions, caps, atomistic,
                                                                         symbols, checked_nodes, alpha_count)


    # For each end group, generate alphabets from paths and select the one whose alphabets most coincide with the RU alphabets
    for end_paths in end_group_paths:

        endgroup_hits = []

        # For each path, only break the bonds of the path
        for path in end_paths:

            # Fix the alpha count
            _alpha_count = alpha_count

            # Get end group atoms
            endgroup_ids = atomistic_ids[path[0]]    # Id of end group
            atoms_in_this_endgroup = [x for x in atomistic.nodes if atomistic.nodes[x]["ids"] == endgroup_ids]

            # Define the bonds that will be broken
            break_bond_modified = []
            for bond in break_bond:
                # They will be broken if the nodes are in a path from shortest_paths
                if bond[0] in path and bond[1] in path:
                    break_bond_modified.append(bond)

            # Create state machine
            _state_machine, max_node = create_state_machine(atomistic, break_bond_modified, max_node+1)
            # Remove all nodes that are not associated with end group atoms
            for edge in _state_machine.edges():
                # If one of the atoms is an end group atom, do nothing
                if edge[0] in atoms_in_this_endgroup or edge[1] in atoms_in_this_endgroup:
                    pass
                # Otherwise, remove the edge
                else:
                    _state_machine.remove_edge(edge[0], edge[1])
            # Remove disconnected nodes
            _state_machine.remove_nodes_from(list(nx.isolates(_state_machine)))


            # Get symbols
            symbols = nx.get_node_attributes(_state_machine, "symbol")
            # Get all end group bonds from state machine. Assign 0 to bonds with "1" or "2" and 1 to the rest in order to sort
            edge_att = {b: _state_machine.edges[b]["bond_type"] for b in _state_machine.edges if b in endgroup_bonds}
            bonds = {k: (v, 0 if ("1" in v or "2" in v) else 1)
                     for k, v in edge_att.items()}
            # Sort bonds so that bonds with "1" or "2" are always checked first. This is essential
            bonds = sorted(bonds.items(), key=lambda item: item[1][1])
            # Remove numbers used for sorting
            bonds = {b[0]: b[1][0] for b in bonds}

            # List of bonding descriptors
            bonding_descriptor_list = [x for x in symbols.keys() if check_if_bonding_descriptor(symbols[x])]
            # List of symbols
            symbol_list = list(symbols.keys())
            # List of symbols that are not bonding descriptors
            non_bd_symbol_list = [x for x in symbol_list if x not in bonding_descriptor_list]

            # Remove edges from nodes that are bonding descriptors or state nodes
            simplified_state_machine = simplify_state_machines(_state_machine, non_bd_symbol_list)

            # Convert end group bonds into transitions
            _alpha_count, endgroup_alphabets, _transitions = bonds_to_transitions(bonds,
                                                                                 copy.deepcopy(_state_machine),
                                                                                 simplified_state_machine,
                                                                                 copy.deepcopy(alphabets),
                                                                                 [],
                                                                                 caps,
                                                                                 atomistic,
                                                                                 symbols,
                                                                                 checked_nodes,
                                                                                 _alpha_count)
            # Filter the alphabets that correspond to the end group we are processing now
            _endgroup_alphabets = {}
            for _id, _map in endgroup_alphabets.items():
                if _id == endgroup_ids:
                    _endgroup_alphabets = _endgroup_alphabets | _map
            endgroup_alphabets = _endgroup_alphabets


            # Choose repeat units that will be used to compare
            # Get the bonding descriptor node
            topology_endgroup_node = [x for x in topology.nodes if topology.nodes[x]["ids"] == endgroup_ids][0]    # Node in the topology graph with same id
            bonding_descriptor_node = [x for x in find_neighbors(topology, topology_endgroup_node)
                                  if check_if_bonding_descriptor(topology.nodes[x]["symbol"])][0]
            atomistic_endgroup_node = [n for n in find_neighbors(atomistic, bonding_descriptor_node) if atomistic.nodes[n]["ids"] == endgroup_ids][0]    # Node in the atomistic graph that is connected to the BD
            # Get bond type
            bond_type_endgroup_bd = atomistic_bonds[tuple(sorted([atomistic_endgroup_node, bonding_descriptor_node]))]
            if "1" in bond_type_endgroup_bd:
                bond_type_endgroup_bd = 1
            else:
                bond_type_endgroup_bd = 2
            # Select RUs to compare
            ru_ids = []
            for node in find_neighbors(atomistic, bonding_descriptor_node):
                # If it is another end group, skip
                if atomistic.nodes[node]["explicit_atom_ids"]:
                    continue
                # Get bond type
                bond_type_endgroup_ru = atomistic_bonds[tuple(sorted([bonding_descriptor_node, node]))]
                if "1" in bond_type_endgroup_ru:
                    bond_type_endgroup_ru = 1
                else:
                    bond_type_endgroup_ru = 2
                # If the bond type of the RU is different than the end group, select it
                if bond_type_endgroup_bd != bond_type_endgroup_ru:
                    ru_ids.append(atomistic_ids[node])

            # Count the number of hits
            hits = []
            for id in ru_ids:
                alphabets_from_ru = ru_alphabets[id]
                for smiles_ru, alphabet_ru in alphabets_from_ru.items():
                    for smiles_endgroup, alphabet_endgroup in endgroup_alphabets.items():
                        if smiles_ru == smiles_endgroup:
                            hits.append(smiles_ru)
            # Remove duplicates
            hits = list(set(hits))
            # Number of hits divided by the total end group alphabets to get the path with the most coincidence
            number_of_hits = len(hits)/len(endgroup_alphabets)

            # Add to list of end groups and hits
            endgroup_hits.append([number_of_hits, copy.deepcopy(_transitions), path])
            _transitions = []

            # Return _alpha_count to original value
            _alpha_count = alpha_count

        # Choose from the set of end group transitions based on the number of hits
        endgroup_hits = sorted(endgroup_hits, key=lambda x: x[0], reverse=True)
        max_hits = endgroup_hits[0][0]
        endgroup_hits = [x[1] for x in endgroup_hits if x[0] == max_hits]
        # To untie, get the one that has the most transitions
        if len(endgroup_hits) == 1:
            end_group_transitions = endgroup_hits[0]
        else:
            end_group_transitions = [[len(x), x] for x in endgroup_hits]
            end_group_transitions = sorted(end_group_transitions, key=lambda x: x[0])[-1][1]

        # Add to RU transitions
        transitions += end_group_transitions

        # Update alpha_count
        endgroup_alphabet_list = list(set([x[1] for x in end_group_transitions]))
        alpha_count += len(endgroup_alphabet_list)

    # Adding initial transitions to bonding descriptors
    es_id = 0
    for bd in starting_bonding_descriptors:
        smiles = f"[Es_{es_id}][*]"
        alphabets[smiles] = caps[alpha_count]
        new_transition = [bd, alphabets[smiles], smiles, []]
        transitions.append(new_transition)
        es_id += 1
        alpha_count += 1

    # Swap left and right to convert into bottom-up tree automata
    table = pd.DataFrame(transitions, columns=['output', 'alphabet', 'smiles', 'input'])

    # Take index of the maximum state
    max_state = get_index_max_state(table) + 1

    # Cap uncapped transitions
    final_states, table = cap_transitions(table, max_state, ending_bonding_descriptors)

    return final_states, table

def check_if_bonding_descriptor(symbol: str) -> bool:
    """
    Checks if a symbol is a bonding descriptor
    Args:
        symbol: symbol

    Returns: True if it is a bonding descriptor

    """
    return any([(x in symbol) for x in ("$", "<", ">")])


def atomistic_to_transitions(bigsmarts_graphs):
    """
    This function converts a polymer into a set of tree automaton transitions. As each polymer may be represented by
    multiple tree automatons, the output is a list of sets of transitions
    Args:
        bigsmarts_graphs: atomistic and topology

    Returns: list of DataFrames of transitions

    """

    # Get atomistic graph
    atomistic_graph = bigsmarts_graphs["atomistic"]
    # Replace empty symbols ("") by Es
    for n in atomistic_graph.nodes:
        if atomistic_graph.nodes[n]["symbol"] == "":
            atomistic_graph.nodes[n]["symbol"] = "Es"

    # Get topology graph
    topology_graph = bigsmarts_graphs["topology"]

    # Define nodes that will lead to starts and ends
    start_end_list = define_starts_and_ends(topology_graph=topology_graph, atomistic_graph=atomistic_graph)

    # Generate a tree for each combination of starts and ends
    transitions_list = []
    for start_end_dict in start_end_list:
        # Get starts and ends
        starts = start_end_dict["start"]
        ends = start_end_dict["end"]
        # Copy atomistic graph
        _atomistic_graph = copy.deepcopy(atomistic_graph)
        # Copy topology graph
        _topology_graph = copy.deepcopy(topology_graph)
        # List of bonding descriptors that will be ends
        ending_bonding_descriptors = []
        # Variable that distinguished Es atoms that are added
        Es_count = 0
        # List of starting bonding descriptors
        start_bd_list = []
        # List of ending bonding descriptors
        end_bd_list = []
        # Variable that will tell whether we need to skip to the next combination
        skip = False

        for group_id in starts:
            # Check if it is a bonding descriptor
            group_node_index = [x for x in _topology_graph.nodes() if _topology_graph.nodes[x]["ids"] == group_id][0]
            is_bonding_descriptor = check_if_bonding_descriptor(_topology_graph.nodes[group_node_index]["symbol"])
            if is_bonding_descriptor:
                # Add to start_bd_list
                start_bd_list.append(group_node_index)
                # First: add an Es to topology graph and connect it to the bonding descriptor
                new_node = max([x for x in _atomistic_graph.nodes()]) + 1
                new_ids = max([_topology_graph.nodes[x]["ids"] for x in _topology_graph.nodes()]) + 1
                level = 0 if _topology_graph.nodes[group_node_index]["level"] == 0 else \
                _topology_graph.nodes[group_node_index]["level"] - 1
                _topology_graph.add_node(node_for_adding=new_node,
                                         symbol="Es",
                                         formal_charge=0,
                                         is_aromatic=False,
                                         chirality="CHI_UNSPECIFIED",
                                         hum_hs=0,
                                         stoch_el=[[], []],
                                         active=False,
                                         level=level,
                                         explicit_atom_ids=True if level == 0 else False,  # It is an end group
                                         ids=new_ids,
                                         Es_id=Es_count)
                # Add edge with index 2 (defines start)
                _topology_graph.add_edge(group_node_index, new_node,
                                         bond_type="")

                # Second: add "" to atomistic graph
                _atomistic_graph.add_node(node_for_adding=new_node,
                                          symbol="Es",
                                          formal_charge=0,
                                          is_aromatic=False,
                                          chirality="CHI_UNSPECIFIED",
                                          hum_hs=0,
                                          stoch_el=[[], []],
                                          active=False,
                                          level=level,
                                          explicit_atom_ids=True if level == 0 else False,  # It is an end group
                                          ids=new_ids,
                                          Es_id=Es_count)
                # Add edge with index 2 (defines start)
                _atomistic_graph.add_edge(group_node_index, new_node, bond_type="2_SINGLE")
                # Update Es count
                Es_count += 1

                # Third: Change indices of edges involving group_node_index
                adj_nodes = [x for x in find_neighbors(_atomistic_graph, group_node_index) if x != new_node]
                # Select the nodes that are not in repeating units
                non_ru_adj = [x for x in adj_nodes
                              if (level == _atomistic_graph.nodes[x]["level"] != 0)
                              or (level == 0 and _atomistic_graph.nodes[x]["explicit_atom_ids"])]
                # Only change indices if all such nodes' edges are index 2
                if_change_ids = all(
                    ["2" in _atomistic_graph.edges[x, group_node_index]["bond_type"] for x in non_ru_adj]) \
                    if non_ru_adj else False
                if if_change_ids:
                    skip = True
                    break
            else:
                # Get adjacent bonding descriptors
                adj_bonding_descriptors = [x for x in find_neighbors(_topology_graph, group_node_index)
                                           if check_if_bonding_descriptor(_topology_graph.nodes[x]["symbol"])]
                for adj in adj_bonding_descriptors:
                    # Adjacent node to bonding descriptor with ids equal to group_id
                    group_node = [x for x in find_neighbors(_atomistic_graph, adj)
                                  if _atomistic_graph.nodes[x]["ids"] == group_id][0]
                    # If the bond between the end group and the bonding descriptor is not index 2, swap indices of
                    # all edges connected to the bonding descriptor
                    if "2" not in _atomistic_graph.edges[adj, group_node]["bond_type"]:
                        skip = True
                        break

        # Skip to next combination of starts and ends if necessary
        if skip:
            continue

        for group_id in ends:
            # Check if it is a bonding descriptor
            group_node_index = [x for x in _topology_graph.nodes() if _topology_graph.nodes[x]["ids"] == group_id][0]
            is_bonding_descriptor = check_if_bonding_descriptor(_topology_graph.nodes[group_node_index]["symbol"])
            if is_bonding_descriptor:
                # Add to end_bd_list
                end_bd_list.append(group_node_index)
                # Change indices of edges involving group_node_index
                # Select the nodes that are not in repeating units
                level = 0 if _topology_graph.nodes[group_node_index]["level"] == 0 else \
                _topology_graph.nodes[group_node_index]["level"] - 1
                adj_nodes = [x for x in find_neighbors(_atomistic_graph, group_node_index)]
                non_ru_adj = [x for x in adj_nodes
                              if (level == _atomistic_graph.nodes[x]["level"] != 0)
                              or (level == 0 and _atomistic_graph.nodes[x]["explicit_atom_ids"])]
                # Only change indices if all such nodes' edges are index 1
                if_change_ids = all(
                    ["1" in _atomistic_graph.edges[x, group_node_index]["bond_type"] for x in non_ru_adj]
                    ) if non_ru_adj else False
                if if_change_ids:
                    skip = True
                    break
                # Add to ending bonding descriptor list
                ending_bonding_descriptors.append(group_node_index)
            else:
                # Get adjacent bonding descriptors
                adj_bonding_descriptors = [x for x in find_neighbors(_topology_graph, group_node_index)
                                           if check_if_bonding_descriptor(_topology_graph.nodes[x]["symbol"])]
                for adj in adj_bonding_descriptors:
                    # Adjacent node to bonding descriptor with ids equal to group_id
                    group_node = [x for x in find_neighbors(_atomistic_graph, adj)
                                  if _atomistic_graph.nodes[x]["ids"] == group_id][0]
                    # If the bond between the end group and the bonding descriptor is not index 1, swap indices of
                    # all edges connected to the bonding descriptor
                    if "1" not in _atomistic_graph.edges[adj, group_node]["bond_type"]:
                        skip = True
                        break


        # Skip to next combination of starts and ends if necessary
        if skip:
            continue

        # Check if there is any group with only type 1 or only type 2 bonds (only check the ones that are not BD)
        atomistic_group_ids = {n: id for n, id in nx.get_node_attributes(_atomistic_graph, "ids").items()
                               if not check_if_bonding_descriptor(_atomistic_graph.nodes[n]["symbol"])}
        # Group atoms of the same group
        atoms_in_groups = [[n for n in atomistic_group_ids.keys() if atomistic_group_ids[n] == id]
                           for id in atomistic_group_ids.values()]
        # Remove duplicates
        atoms_in_groups = list(set(map(lambda i: tuple(sorted(i)), atoms_in_groups)))
        for group in atoms_in_groups:
            bond_type_list = []
            # For each atom
            for atom in group:
                _edges = _atomistic_graph[atom]
                # Check the bond type with the adjacent atoms
                for adj in _edges:
                    # Get the bond type
                    bond_type = _atomistic_graph[atom][adj]["bond_type"]
                    if "1" in bond_type:
                        bond_type = 1
                    elif "2" in bond_type:
                        bond_type = 2
                    else:
                        bond_type = 0
                    # Append to list
                    bond_type_list.append(bond_type)
            # There must be 1 type 2 and 1 or many type 1. If there are 0 or more than 1 type 2, we must skip this graph
            number_type_1 = len([x for x in bond_type_list if x == 1])
            number_type_2 = len([x for x in bond_type_list if x == 2])
            # If there is only one type 2, it is a start. So it can continue
            if number_type_1 == 0 and number_type_2 == 1:
                skip = False
            # IF there is only one type 2, it is an end. So it can continue
            elif number_type_1 == 1 and number_type_2 == 0:
                skip = False
            # If there are many type 1 and one type 2, it is a linear transition or a branch point. So it can continue
            elif number_type_1 >= 1 and number_type_2 == 1:
                skip = False
            # Any other case will have either o or many type 2, which is not right
            else:
                skip = True
                break


        # Skip to next combination of starts and ends if necessary
        if skip:
            continue

        final_states, _transitions = generate_tree_transitions(atomistic=_atomistic_graph, topology=_topology_graph,
                                                               ending_bonding_descriptors=ending_bonding_descriptors,
                                                               starting_bonding_descriptors=[])
        transitions_list.append([final_states, _transitions, _atomistic_graph])  # Added the atomistic graph to check

    return transitions_list


def find_neighbors(graph, node):
    # Created this function because for bonds with H, nx.neighbors return [] because the parameters are not filled in atlas

    _neighbors = []
    for edge in graph.edges():
        if node in edge:
            if edge[0] != node:
                _neighbors.append(edge[0])
            else:
                _neighbors.append(edge[1])

    # Remove duplicates
    _neighbors = list(set(_neighbors))

    return sorted(_neighbors)


def transition_table_to_automaton(transition_table, end_states):
    """
    This function converts a table of transition rules into the tree automaton object
    Args:
        transition_table: transition rules
        end_states: ending states

    Returns: tree automaton

    """
    transitions_list = []
    for row in transition_table.iterrows():
        transition = tree_automata.Transitions(input=row[1]["input"], alphabet=row[1]['alphabet'],
                                               smiles=re.sub(r"_\d", "", row[1]['smiles']),
                                               output=row[1]['output'])
        transitions_list.append(transition)

    # Create Tree Automata object
    tree = tree_automata.TreeAutomata(transitions=transitions_list, end_states=end_states)

    return tree


def generate_transition_tables(graphs):
    """
    This function generates the transition table for every polymer graph. A transition table is a pandas table with
    the transition rules.
    Args:
        graphs: list of graph

    Returns: list of transition tables

    """
    # Initialize the list of transition tables
    transitions = []

    # For every graph, generate the table with transition rules
    for index, g in enumerate(graphs):
        # Convert graphs into tree automatons
        _transitions = atomistic_to_transitions(bigsmarts_graphs=g)
        # Add to transitions list
        if _transitions:
            _transitions[0].append(g["atomistic"])
        transitions += _transitions

    return transitions


def choose_one_bigsmiles(bigsmiles_list):
    """
    This function takes the list of all BigSMILES generated and chooses one
    Args:
        bigsmiles_list: list of BigSMILES

    Returns: one preferred BigSMILES

    """
    # Remove duplicates
    bigsmiles_list = list(set(bigsmiles_list))
    # Sort the list of BigSMILES
    bigsmiles_list = sorted(bigsmiles_list, reverse=True, key=lambda x: [-len(x), x])
    # Choose the first
    preferred_bigsmiles = bigsmiles_list[0]

    return preferred_bigsmiles


def save_bigsmiles_file(original_bigsmiles, canonical_bigsmiles, bigsmiles_list, output_folder, filename="Canonical_BigSMILES.txt"):
    """
    This function saves a text file containing the input BigSMILES, the preferred canonical string and the other options
    Args:
        original_bigsmiles: original bigsmiles
        canonical_bigsmiles: preferred BigSMILES
        bigsmiles_list: list of canonical BigSMILES
        output_folder: output folder
        filename: file name

    Returns: None

    """
    # Create path
    _path = os.path.join(output_folder, filename)

    # Save the file
    with open(_path, "w") as f:
        # Write initial BigSMILES
        f.write(f"Initial: {original_bigsmiles} \n")
        # Write canonical BigSMILES
        f.write(f"Canonical: {canonical_bigsmiles}")
        # Write other options
        f.write("\nOther options:\n")
        for bg in bigsmiles_list:
            if bg != canonical_bigsmiles:
                f.write(f"{bg}\n")


def canonicalize_bigsmiles(bigsmiles, output_folder="Output", plot=False):
    """
    This function provides a canonical BigSMILES string
    Args:
        bigsmiles: BigSMILES string
        output_folder: Folder all files will be saved
        plot: if True, generates figures of the tree automatons

    Returns: canonical BigSMILES

    """
    # Generate all possible atomistic and topology graphs
    graphs = generate_all_possible_graphs(input_string=bigsmiles)

    # Generate tree transition tables for every graph
    transitions = generate_transition_tables(graphs=graphs)

    # Convert every transition table into tree automaton, minimize it and convert it into BigSMILES
    index = 0    # Index that will distinguish each tree
    # List of bigsmiles
    bigsmiles_list = []
    for end_states, transition_table, _, atomistic_graph in transitions:

        # Create a folder for each tree
        _output_folder = output_folder + f"\\Tree_{index}\\"

        # Save atomistic graph
        visualize_atomistic_graph(atomistic_graph, filename=f"Atomistic_Graph_{index}", output_folder=_output_folder,
                                  show_state_index=False)

        # Create Tree Automata object
        tree = transition_table_to_automaton(transition_table=transition_table, end_states=end_states)
        # Minimize state machine
        tree.generate_minimial_tree(plot=plot, tree_name=index, draw_alphabet_function=draw_alphabets,
                                    output_folder=_output_folder)
        # Convert tree automaton into BigSMILES
        _bigsmiles = string_conversion.to_bigsmiles(dfta=tree, tree_name=index, output_folder=_output_folder,
                                                    draw_alphabets=draw_alphabets)

        # Add to list of BigSMILES
        bigsmiles_list.append(_bigsmiles)

        # Update index
        index += 1

    # Choose one string from the set of BigSMILES strings
    canonical_bigsmiles = choose_one_bigsmiles(bigsmiles_list)

    # Save a text file with the initial BigSMILES, canonical BigSMILEs and other options
    save_bigsmiles_file(original_bigsmiles=bigsmiles,
                        canonical_bigsmiles=canonical_bigsmiles,
                        bigsmiles_list=bigsmiles_list,
                        output_folder=output_folder)

    return canonical_bigsmiles


def run_validation(file_names, output_folder):
    """
    This function automatically runs the validation
    Args:
        file_names: list of file names. Every element is a list: [file name, file extension]
        output_folder: output folder

    Returns: None

    """
    # For every file, run the code
    for file_name, ext in file_names:

        # Read the file
        dataset = pd.read_excel(file_name + ext)

        # If output directory does not exist, create it
        _output_folder = os.path.join(output_folder, file_name)

        try:
            os.makedirs(_output_folder)
        except:
            pass

        # Run the canonicalization function for every row of the file
        for index, row in dataset.iterrows():
            try:
            # Set the name of the folder with the results of each input
                subfolder_name = f"{index}"
                # Get the initial BigSMILES
                bigsmiles = row["Input"]
                # Canonicalize BigSMILES
                canonical = canonicalize_bigsmiles(bigsmiles=bigsmiles,
                                                   output_folder=os.path.join(_output_folder, subfolder_name),
                                                   plot=True)
                # Add canonical BigSMILES to file
                dataset.loc[index, "Canonical"] = canonical
            except:
                print(f"Error index {index}")

        # Save answers
        dataset.to_excel(os.path.join(_output_folder, file_name + ext))


# Main -----------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Set the names of the files and their extensions
    file_names = [["Validation I", ".xlsx"], ["Validation II", ".xlsx"]]

    # Run validation
    run_validation(file_names=file_names, output_folder="Validation")

