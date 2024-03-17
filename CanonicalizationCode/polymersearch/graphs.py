import re
import copy
import time

import rdkit
from rdkit import Chem
import networkx as nx

from polymersearch.adjacency import convert_tuple_string, feed_to_bonds_n

desc_regex = r"\[(\$|\<|\>|\$=|\<=|\>=)\d+\]"


def get_comp(d):
    if "<" in d:
        return d.replace("<", ">")
    elif ">" in d:
        return d.replace(">", "<")
    else:
        return d


def fragment_notation_to_bigsmiles(bigsmiles):
    if bigsmiles.find(".{") == -1:
        return bigsmiles

    base = bigsmiles[:bigsmiles.find(".{")]
    fragments = bigsmiles[bigsmiles.find(".{") + 1:]
    dot_delimited = fragments.split(".")
    for i in range(len(dot_delimited)):
        a = dot_delimited[i].find("#")
        b = dot_delimited[i].find("=")
        frag_id = "[" + dot_delimited[i][a:b] + "]"
        value = dot_delimited[i][b + 1: -1]
        base = base.replace(frag_id, value)
    return base


def get_objects(ends):
    i = 0
    o = []
    start = []
    while i < len(ends):
        if ends[i] == "{":
            start.append(i)
            object = "{"
            count = 1
            i += 1
            while count != 0:
                if ends[i] == "{":
                    count += 1
                if ends[i] == "}":
                    count -= 1
                object += ends[i]
                i += 1
            o.append(object)
        else:
            i += 1
    return [o, start]


def get_repeats(object):
    object = object[object.find("]") + 1:]  # between the terminal descriptors
    object = object[:object.rfind("[")]

    def between_brackets(object, i):
        j = i
        merge1 = False
        while j >= 0:
            if object[j] == "]":
                return False
            elif object[j] == "[":
                merge1 = True
                break
            j -= 1
        if not merge1:
            return False
        j = i
        while j < len(object):
            if object[j] == "[":
                return False
            elif object[j] == "]":
                return True
            j += 1
        return False

    i = 0
    o = []
    count = 0
    repeat = ""
    object += ","
    while i < len(object):
        if object[i] == "{":
            count += 1
        if object[i] == "}":
            count -= 1
        if object[i] == "," and not between_brackets(object, i) and count == 0:
            o.append(repeat)
            repeat = ""
        else:
            repeat += object[i]
        i += 1

    repeats = [[], []]
    index = 0
    for i in range(len(o)):
        if ";" in o[i] and not between_brackets(o[i], o[i].find(";")):
            o[i].split(";")
            repeats[index].append(o[i][:o[i].index(";")])
            index += 1
            repeats[index].append(o[i][o[i].index(";") + 1:])
        else:
            repeats[index].append(o[i])

    return repeats[0], repeats[1]


def sub_obj_with_Bk(bigsmiles):
    smiles = ""
    counter = 0
    objects = get_objects(bigsmiles)[0]
    indices = get_objects(bigsmiles)[1]
    for i in range(len(objects)):
        smiles += bigsmiles[counter:indices[i]]
        smiles += "[Bk]"
        counter += (indices[i] - counter) + len(objects[i])
    smiles += bigsmiles[counter:]
    return smiles, objects


def modify_string(smiles, object_list):
    Bk_locations = [(d.start(0), d.end(0)) for d in re.finditer(r"\[Bk\]", smiles)]
    insert_No = []
    for i in range(len(Bk_locations)):
        repeats, implicit_ends = get_repeats(object_list[i])
        left_terminal = object_list[i][1:object_list[i].find("]") + 1]
        right_terminal = object_list[i][object_list[i].rfind("["):-1]

        if i == 0 and Bk_locations[i][0] == 0 and left_terminal != "[]":
            insert_No.append(0)
        if i == len(Bk_locations) - 1 and Bk_locations[i][1] == len(smiles) and right_terminal != "[]":
            insert_No.append(Bk_locations[i][1])
        if Bk_locations[i][1] < len(smiles) and smiles[Bk_locations[i][1]] == ")" and right_terminal != "[]":
            insert_No.append(Bk_locations[i][1])

        repeat_list = ""
        for r in repeats:
            repeat_list += r + ","
        r_tot_desc = [(d.start(0), d.end(0)) for d in re.finditer(desc_regex, repeat_list)]

        if r_tot_desc == []:
            # there are no repeat units in the object
            if object_list[i] == "{[][]}":
                # replace wildcard objects with a cycle wildcard cycle
                object_list[i] = "{[<1][>1][Fm][Md][<1][>1]}"
            else:
                # if there are localization elements, keep them in the object
                object_list[i] = "{[<1][>1][Fm][Md][<1]," + object_list[i][3:-3] + "[>1]}"
        else:

            # there are repeat units in the object
            repeat_list = left_terminal + repeat_list + right_terminal
            r_tot_desc = [(d.start(0), d.end(0)) for d in re.finditer(desc_regex, repeat_list)]

            # I changed this piece of code so that now guarantees the left and right end bonding descriptor will be compatible
            if_no_left_endgroup = object_list[i].find("{[]") == 0
            if_no_right_endgroup = object_list[i][-3:] == "[]}"
            if if_no_left_endgroup:
                # find last descriptor and insert compatible at beginning
                a = r_tot_desc[-1][0]
                b = r_tot_desc[-1][1]
                object_list[i] = "{" + get_comp(repeat_list[a:b]) + object_list[i][3:]
            if if_no_right_endgroup:
                if if_no_left_endgroup:
                    # find the compatible descriptor that was inserted at the left end
                    left_bd = [(d.start(0), d.end(0)) for d in re.finditer(desc_regex, object_list[i])][0]
                    left_bd = object_list[i][left_bd[0]:left_bd[1]]
                    object_list[i] = object_list[i][0:-3] + get_comp(left_bd) + "}"
                else:
                    # find the first descriptor and insert compatible at end
                    a = r_tot_desc[0][0]
                    b = r_tot_desc[0][1]
                    object_list[i] = object_list[i][0:-3] + get_comp(repeat_list[a:b]) + "}"

    if len(insert_No) > 0:
        smiles_new = smiles[0: insert_No[0]]
        for i in range(len(insert_No) - 1):
            smiles_new += "[No]" + smiles[insert_No[i]:insert_No[i + 1]]
        smiles_new += "[No]" + smiles[insert_No[-1]:]
        smiles = smiles_new

    while True:
        smiles_prev = smiles.replace("[Bk][Bk]", "[Bk][Es][Bk]")
        if smiles_prev == smiles:
            break
        smiles = smiles_prev
    while True:
        smiles_prev = smiles.replace("[Bk])", "[Bk][Es])")
        if smiles_prev == smiles:
            break
        smiles = smiles_prev
    while True:
        smiles_prev = smiles.replace("[Cf][Bk]", "[Cf][Es][Bk]")
        if smiles_prev == smiles:
            break
        smiles = smiles_prev
    while True:
        smiles_prev = smiles.replace("[Bk][Cf]", "[Bk][Es][Cf]")
        if smiles_prev == smiles:
            break
        smiles = smiles_prev

    if smiles.find("[Bk]") == 0:
        smiles = "[Es]" + smiles
    if smiles.rfind("[Bk]") == len(smiles) - 4 and smiles.rfind("[Bk]") != -1:
        smiles = smiles + "[Es]"

    return smiles, object_list


def RDKit_to_networkx_graph(mol, is_bigsmarts, level):
    # https://github.com/maxhodak/keras-molecules/pull/32/files
    G = nx.Graph()
    for atom in mol.GetAtoms():

        # difference between query and target is the hydrogen specification
        if is_bigsmarts:
            num_hs = atom.GetNumExplicitHs()
        else:
            num_hs = atom.GetTotalNumHs()
        G.add_node(atom.GetIdx(),
                   symbol=atom.GetSymbol(),
                   formal_charge=atom.GetFormalCharge(),
                   is_aromatic=atom.GetIsAromatic(),
                   chirality=atom.GetChiralTag(),
                   num_hs=num_hs,
                   stoch_el=[[], []],
                   active=False,
                   level=level,
                   map_num=atom.GetAtomMapNum())

    atoms_added = set()
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=str(bond.GetBondType()),
                   bond_type_object=bond.GetBondType(),
                   bond_dir_object=bond.GetBondDir(),
                   bond_stereo_object=bond.GetStereo())
        atoms_added.add(bond.GetBeginAtomIdx())
        atoms_added.add(bond.GetEndAtomIdx())

    if len(atoms_added) != G.number_of_nodes():
        formal_charge = nx.get_node_attributes(G, "formal_charge")
        ionic_bond = rdkit.Chem.rdchem.BondType.IONIC
        ionic = []
        for key in formal_charge:
            if key not in atoms_added:
                ionic.append(key)
        for ionic_atom in ionic:
            for all_atom in formal_charge:
                if all_atom not in ionic and formal_charge[ionic_atom] * formal_charge[all_atom] < 0:
                    G.add_edge(ionic_atom, all_atom, bond_type=str(ionic_bond), bond_type_object=ionic_bond,
                               bond_dir_object=Chem.BondDir(0), bond_stereo_object=Chem.BondStereo.NONE)

    return G


def orientation(networkx_graph, index):
    neighbors = []
    symbols = nx.get_node_attributes(networkx_graph, "symbol")

    i = -1
    for key in symbols:
        if symbols[key] in ["[#98]", "[Cf]", "Cf"]:
            i += 1
        if i == index:
            source = key
            break
    for t in nx.bfs_edges(networkx_graph, source=source):
        if symbols[t[1]] in ["[#97]", "[Bk]", "Bk"]:
            if t[1] > t[0]:
                neighbors.append([t[1], "1"])
            else:
                neighbors.append([t[1], "2"])
    neighbors = sorted(neighbors, key=lambda x: x[0])
    return [n[1] for n in neighbors]


def insert_terminals(graph, terminals, neighbors, bond):
    index = int(bond) - 1
    left_terminal = terminals[index][0]
    right_terminal = terminals[index][1]
    l_index = graph.number_of_nodes()
    graph.add_node(l_index, symbol=left_terminal, active=True)
    if left_terminal == right_terminal:
        r_index = l_index
    else:
        r_index = graph.number_of_nodes()
        graph.add_node(r_index, symbol=right_terminal, active=True)
    if bond == "1":
        graph.add_edge(neighbors[0], l_index, bond_type="1")
        graph.add_edge(neighbors[1], r_index, bond_type="2")
        terminals_attachment = [["2"], ["1"]]
    else:
        graph.add_edge(neighbors[0], l_index, bond_type="2")
        graph.add_edge(neighbors[1], r_index, bond_type="1")
        terminals_attachment = [["1"], ["2"]]
    return graph, terminals_attachment, l_index, r_index


def single_path_chemistries(repeats):
    repeats = copy.deepcopy(repeats)
    forced = {"1": [], "2": []}
    for i in range(len(repeats)):
        descriptor_locations = [(d.start(0), d.end(0)) for d in re.finditer(desc_regex, repeats[i])]
        for d in descriptor_locations:
            desc1 = repeats[i][d[0]:d[1]]
            if "$" in desc1:
                continue
            no_compatible = True
            for j in range(len(repeats)):
                descriptor_locations = [(l.start(0), l.end(0)) for l in re.finditer(desc_regex, repeats[j])]
                for l in descriptor_locations:
                    desc2 = repeats[j][l[0]:l[1]]
                    if desc1 == get_comp(desc2):
                        no_compatible = False
            if no_compatible:
                forced["1"].append(desc1)

    descriptors = []
    for i in range(len(repeats)):
        for value in forced["1"]:
            repeats[i] = repeats[i].replace(value, "[]")
        descriptor_locations = [(d.start(0), d.end(0)) for d in re.finditer(desc_regex, repeats[i])]
        descriptors.append([])
        for d in descriptor_locations:
            desc1 = repeats[i][d[0]:d[1]]
            if "$" in desc1:
                return forced
            descriptors[-1].append(desc1)
    try:
        head = [descriptors[0][0], get_comp(descriptors[0][0])]
        for i in range(len(head)):
            h = head[i]
            t = get_comp(h)
            found_dendrimer = True
            for j in range(len(descriptors)):
                a = descriptors[j].count(h) + descriptors[j].count(t) == len(descriptors[j]) and descriptors[j].count(
                    h) == 1
                if not a:
                    found_dendrimer = False
                    break
            if found_dendrimer:
                forced["2"].append(h)
        return forced
    except:
        return forced


def single_atom_cycle(descriptors, smiles):
    if len(descriptors) == 2 and descriptors[0] == get_comp(descriptors[1]):
        rdkit_graph = Chem.MolFromSmiles(smiles)
        networkx_graph = RDKit_to_networkx_graph(mol=rdkit_graph, is_bigsmarts=False, level=0)
        symbols = nx.get_node_attributes(networkx_graph, "symbol")
        cf_nodes = []
        for key in symbols:
            if symbols[key] == "Cf":
                cf_nodes.append(key)
        path = nx.shortest_path(G=networkx_graph, source=cf_nodes[0], target=cf_nodes[1])
        if len(path) == 3:
            return "[Cf][Es]" + smiles[4:-4] + "[Cf]"
        else:
            return smiles
    else:
        return smiles


def build_atomistic(bigsmiles, is_bigsmarts):
    if "!{[][]}" in bigsmiles:
        no_other_objects = True
    else:
        no_other_objects = False
    # add integer ids if they are not specified in the string
    replace = {";!{[][]}": "", "!{[][]}": "", "[H]{": "{", "}[H]": "}", "[<]": "[<1]", "[>]": "[>1]", "[$]": "[$1]",
               "?*": "[Fm][Md]", "[<=]": "[<=1]", "[>=]": "[>=1]", "[$=]": "[$=1]"}
    for key in replace:
        bigsmiles = bigsmiles.replace(key, replace[key])

    # if fragment notation is specified, convert it into equivalent BigSMARTS or BigSMILES representation
    bigsmiles = fragment_notation_to_bigsmiles(bigsmiles)

    # substitute stochastic objects with Bk to convert into SMARTS or SMILES, save each stochastic object string
    smarts_smiles, object_list = sub_obj_with_Bk(bigsmiles)
    smarts_smiles, object_list = modify_string(smarts_smiles, object_list)
    # print(smarts_smiles)
    # print(object_list)

    # convert SMARTS or SMILES into RDKit molecular graph
    if is_bigsmarts:
        rdkit_graph = Chem.MolFromSmiles(smarts_smiles)
    else:
        rdkit_graph = Chem.MolFromSmiles(smarts_smiles)

    # convert rdkit molecular graph into networkx graph with atom and bond info from RDKit
    networkx_graph = RDKit_to_networkx_graph(mol=rdkit_graph, is_bigsmarts=is_bigsmarts, level=0)
    # return networkx_graph, no_other_objects

    # call recursive function to add repeat units to the graph
    return build_level(atomistic=networkx_graph,
                       is_bigsmarts=is_bigsmarts,
                       object_list=object_list,
                       object_orientation=["1"] * len(object_list),
                       was_implicit=[],
                       level=0), no_other_objects


def build_all_atomistic_graphs(bigsmiles):
    """
    This function generates all possible atomistic graphs from a bigsmiles string
    Args:
        bigsmiles: bigsmiles string

    Returns: list of atomistic graphs

    """
    if "!{[][]}" in bigsmiles:
        no_other_objects = True
    else:
        no_other_objects = False
    # add integer ids if they are not specified in the string
    replace = {";!{[][]}": "", "!{[][]}": "", "[H]{": "{", "}[H]": "}", "[<]": "[<1]", "[>]": "[>1]", "[$]": "[$1]",
               "?*": "[Fm][Md]", "[<=]": "[<=1]", "[>=]": "[>=1]", "[$=]": "[$=1]"}
    for key in replace:
        bigsmiles = bigsmiles.replace(key, replace[key])

    # if fragment notation is specified, convert it into equivalent BigSMARTS or BigSMILES representation
    bigsmiles = fragment_notation_to_bigsmiles(bigsmiles)

    # substitute stochastic objects with Bk to convert into SMARTS or SMILES, save each stochastic object string
    smarts_smiles, object_list = sub_obj_with_Bk(bigsmiles)
    smarts_smiles, object_list = modify_string(smarts_smiles, object_list)

    # convert SMARTS or SMILES into RDKit molecular graph
    rdkit_graph = Chem.MolFromSmiles(smarts_smiles)

    # convert rdkit molecular graph into networkx graph with atom and bond info from RDKit
    networkx_graph = RDKit_to_networkx_graph(mol=rdkit_graph, is_bigsmarts=False, level=0)

    # Everything that is not a stochastic object (i.e., a Bk) is explicit
    for n in networkx_graph.nodes:
        if networkx_graph.nodes[n]["symbol"] != 'Bk':
            networkx_graph.nodes[n]["explicit_atom_ids"] = True

    # Find ends, nonstochastic atoms and stochastic objects
    ends = []
    nonstochastic_atoms = []
    stochastic_objects = []
    first_endgroup = []
    check_first_endgroup = True
    for node in networkx_graph.nodes:
        if networkx_graph.degree(node) == 1:
            ends.append(node)
        if networkx_graph.nodes[node]["symbol"] == "Bk":
            check_first_endgroup = False
            stochastic_objects.append(node)
        if networkx_graph.nodes[node]["symbol"] != "Bk":
            nonstochastic_atoms.append(node)
        if check_first_endgroup:  # The first end group has all atoms until a stochastic object appears
            first_endgroup.append(node)

    # Get one atom from each end group to be the start. This means that the path between this atom and any other from
    # the list cannot have a Bk. This is done to remove graph replicates in the final results
    # Get list of atoms that will not be starts
    not_starts = []
    # Get one node
    for i, node_i in enumerate(nonstochastic_atoms):
        # If this node is in not_starts, skip it
        if node_i in not_starts:
            continue
        # Get another node
        for j, node_j in enumerate(nonstochastic_atoms):
            # If its index is less than i or if it is in not_start, skip it
            if j <= i or node_j in not_starts:
                continue
            # Get shortest path
            path = nx.shortest_path(networkx_graph, node_i, node_j)
            # If there is not a Bk along the path, node_j cannot be a start
            symbols = [networkx_graph.nodes[n]["symbol"] for n in path]
            if "Bk" not in symbols:
                not_starts.append(node_j)
    # Get list of possible starts
    possible_starts = [x for x in nonstochastic_atoms if x not in not_starts]

    # Generate all possible graphs by choosing one end to be the start
    list_of_graphs = []
    for i, node in enumerate(possible_starts):  # enumerate(ends):
        # Dictionary that maps nodes onto stochastic objects
        node_to_object = dict(zip(stochastic_objects, object_list))
        # Get all indices, making the chosen one the first
        list_of_indices = list(nx.dfs_tree(networkx_graph, source=node).nodes())
        # Get all nodes and normalize their indices
        list_of_nodes = [(index, networkx_graph.nodes[x]) for index, x in enumerate(list_of_indices)]
        # Get all edges and replace old node indices by new ones
        list_of_edges = networkx_graph.edges(data=True)
        list_of_edges = [(list_of_indices.index(x[0]), list_of_indices.index(x[1]), x[2]) for x in list_of_edges]
        list_of_edges = sorted(
            list_of_edges)  # For the function to work properly, the edges between lowest index nodes must appear first
        # Create new graph that will result in the atomistic graph
        G = nx.Graph()
        G.add_nodes_from(list_of_nodes)  # Add nodes
        G.add_edges_from(list_of_edges)  # Add edges

        # Loop over all end groups and check when it is necessary to swap end group bonding descriptors.
        _object_list = copy.deepcopy(object_list)
        # List that will tell whether or not the object end groups have already been swapped
        swapped_endgroups = [False for _ in _object_list]
        for j, end in enumerate(ends):
            # Take the associated stochastic object
            paths_to_objects = [nx.shortest_path(networkx_graph, source=end, target=x) for x in stochastic_objects]
            sto_obj_node = sorted(paths_to_objects, key=lambda x: len(x))[0][-1]  # Get the closest stochastic object
            object = node_to_object[sto_obj_node]  # _object_list[j]
            obj_index = _object_list.index(object)  # Get object index
            path_i_j = nx.shortest_path(networkx_graph, source=node, target=end)  # Get the path between i and j
            is_same_endgroup = all(["Bk" not in networkx_graph.nodes[x]['symbol'] for x in
                                    path_i_j])  # If there are no bonding descriptor between i and j, they belong to the same group
            is_starting_endgroup = is_same_endgroup
            is_first_endgroup = end in first_endgroup
            # If the end group is the starting end group but it is not the first (left-most) of the string, swap
            if (is_starting_endgroup) and (not is_first_endgroup) and (
            not swapped_endgroups[obj_index]):  # end != nonstochastic_atoms[0]:
                object = swap_end_bonding_descriptor(object)
                swapped_endgroups[obj_index] = True
            # If it is not the starting end group but it is the first (left-most) one, swap
            elif not (is_starting_endgroup) and (is_first_endgroup) and (
            not swapped_endgroups[obj_index]):  # end == nonstochastic_atoms[0]:
                object = swap_end_bonding_descriptor(object)
                swapped_endgroups[obj_index] = True
            # Update list of stochastic objects
            _object_list[obj_index] = object
            # Update node_to_object
            node_to_object[sto_obj_node] = object

        # Take the list of stochastic objects and make the one associated with the starting end group the first element
        paths_to_objects = [nx.shortest_path(networkx_graph, source=node, target=x) for x in stochastic_objects]
        sto_obj_node = sorted(paths_to_objects, key=lambda x: len(x))[0][-1]  # Get the closest stochastic object
        object = node_to_object[sto_obj_node]  # _object_list[j]
        obj_index = _object_list.index(object)
        object = _object_list[obj_index]  # _object_list[i]
        del _object_list[obj_index]  # _object_list[i]
        _object_list = [object] + _object_list

        # Generate graph
        graph = build_level(atomistic=G,
                            is_bigsmarts=False,
                            object_list=_object_list,
                            object_orientation=["1"] * len(_object_list),
                            was_implicit=[],
                            level=0)
        list_of_graphs.append(graph)

    return list_of_graphs

    # call recursive function to add repeat units to the graph
    # return build_level(atomistic=networkx_graph,
    #                    is_bigsmarts=False,
    #                    object_list=object_list,
    #                    object_orientation=["1"] * len(object_list),
    #                    was_implicit=[],
    #                    level=0), no_other_objects


def swap_end_bonding_descriptor(stochastic_object):
    """
    Given a stochastic object, this function swaps the bonding descriptors of the end groups
    """
    start_left, end_left, start_right, end_right = 0, 0, 0, 0
    # Find start of the left end group bonding descriptor
    for j, char in enumerate(stochastic_object):
        if char == "[":
            start_left = j
            break
    # Find end of the left end group bonding descriptor
    for j, char in enumerate(stochastic_object):
        if char == "]":
            end_left = j
            break
    for j in range(len(stochastic_object) - 1, 0, -1):
        char = stochastic_object[j]
        if char == "[":
            start_right = j
            break
    for j in range(len(stochastic_object) - 1, 0, -1):
        char = stochastic_object[j]
        if char == "]":
            end_right = j
            break
    # Get left and right end group BDs and swap them
    left_bd = stochastic_object[start_left:end_left + 1]
    right_bd = stochastic_object[start_right:end_right + 1]
    stochastic_object = stochastic_object[0:start_left] + right_bd + stochastic_object[end_left + 1:start_right] \
                        + left_bd + stochastic_object[end_right + 1::]

    return stochastic_object


def build_level(atomistic, is_bigsmarts, object_list, object_orientation, was_implicit, level):
    symbols = nx.get_node_attributes(atomistic, "symbol")
    index_Bk = []
    for key in symbols:
        if symbols[key] in ["[#97]", "[Bk]", "Bk"]:
            neighbors = list(atomistic[key])
            if len(neighbors) == 2:
                index_Bk.append(key)

    endgroup_atom_ids = []
    for key in symbols:
        if symbols[key] not in ["Bk", "Es"]:
            endgroup_atom_ids.append(key)
    terminals_indices = []

    # keep track of "1" single path endgroups for duplication
    one_single = []

    # iterate through each Bk in NetworkX graph, which contains the saved stochastic object strings
    nested_objects = []
    was_implicit_now = []
    nested_orientation = []
    for o in range(len(object_list)):
        # Parse repeat units, implicit endgroups, and terminal descriptors from each stochastic object string
        repeats, implicit_ends = get_repeats(object_list[o])
        if "[>1][Fm][Md][<1]" in repeats:
            wildcard_cluster = True
        else:
            wildcard_cluster = False

        stochastic_descriptor = []

        left_terminal = object_list[o][1:object_list[o].find("]") + 1]
        right_terminal = object_list[o][object_list[o].rfind("["):-1]
        single_path = single_path_chemistries(repeats)
        one_single.append(single_path["1"])

        # Connect terminal descriptors to Bk's neighbors.
        neighbors = list(atomistic[index_Bk[o]])
        for n in neighbors:
            atomistic.remove_edge(index_Bk[o], n)

        terminals = [[left_terminal, get_comp(right_terminal)], [get_comp(left_terminal), right_terminal]]
        if len(single_path["2"]) == 1:
            value = single_path["2"][0]
            if value == get_comp(left_terminal):
                atomistic, terminals_attachment, l_index, r_index = insert_terminals(atomistic, terminals, neighbors,
                                                                                     "1")
            elif value == get_comp(right_terminal):
                atomistic, terminals_attachment, l_index, r_index = insert_terminals(atomistic, terminals, neighbors,
                                                                                     "2")
        elif len(single_path["2"]) > 1:
            for value in single_path["2"]:
                if object_orientation[o] == "1":
                    if value == get_comp(left_terminal):
                        atomistic, terminals_attachment, l_index, r_index = insert_terminals(atomistic, terminals,
                                                                                             neighbors, "1")
                        single_path["2"] = [value]
                        break
                else:
                    if value == get_comp(right_terminal):
                        atomistic, terminals_attachment, l_index, r_index = insert_terminals(atomistic, terminals,
                                                                                             neighbors, "2")
                        single_path["2"] = [value]
                        break
        else:
            atomistic, terminals_attachment, l_index, r_index = insert_terminals(atomistic, terminals, neighbors,
                                                                                 object_orientation[o])
        terminals_indices.append([l_index, r_index])

        if len(implicit_ends) > 0:
            symbols = nx.get_node_attributes(atomistic, "symbol")
            for i in range(2):
                n = list(atomistic[neighbors[i]])
                expl_a = symbols[neighbors[i]] not in ["[#99]", "[Es]", "Es"]
                expl_b = symbols[neighbors[i]] in ["[#99]", "[Es]", "Es"] and len(list(atomistic[neighbors[i]])) >= 2
                if not (expl_a or expl_b):
                    terminals_attachment[i] = ["1", "2"]
            if left_terminal == get_comp(right_terminal):
                a = list(set(terminals_attachment[0]) & set(terminals_attachment[1]))
                terminals_attachment = [a, a]

        # Iterate through reach parsed repeat unit and implicit endgroup
        ru_local_el = set()
        for smiles in repeats:
            # Replace nested objects with Bk, descriptors with Cf, and save nested object string.
            smiles, nested_object_list = sub_obj_with_Bk(smiles)

            descriptor_locations = [(d.start(0), d.end(0)) for d in re.finditer(desc_regex, smiles)]
            descriptors = []
            for d in descriptor_locations:
                descriptors.append(smiles[d[0]:d[1]])
            for d in descriptors:
                smiles = smiles.replace(d, "[Cf]")
            smiles = single_atom_cycle(descriptors, smiles)

            duplication = len(descriptors)
            for n in range(len(descriptors)):
                if descriptors[n] in single_path["1"]:
                    duplication -= 1
                if get_comp(descriptors[n]) in single_path["2"]:
                    duplication -= 1

            if "[Bk]" in smiles:
                smiles, nested_object_list = modify_string(smiles, nested_object_list)
                for d in range(duplication):
                    for n in nested_object_list:
                        nested_objects.append(n)
                        was_implicit_now.append(False)

            ## store stochastic elements
            if duplication == 0:
                ru_local_el.add(smiles)
                continue

            # Convert SMARTS or SMILES into RDKit molecular graph and NetworkX graph
            if is_bigsmarts:
                rdkit_graph = Chem.MolFromSmiles(smiles)
            else:
                rdkit_graph = Chem.MolFromSmiles(smiles)

            if level > 0 and was_implicit[o]:
                networkx_graph = RDKit_to_networkx_graph(mol=rdkit_graph, is_bigsmarts=is_bigsmarts, level=level - 1)
            else:
                networkx_graph = RDKit_to_networkx_graph(mol=rdkit_graph, is_bigsmarts=is_bigsmarts, level=level)

            for d in range(duplication):
                atomistic = nx.disjoint_union(atomistic, networkx_graph)

            symbols = nx.get_node_attributes(atomistic, "symbol")
            neighbors = []
            for key in symbols:
                if symbols[key] in ["[#98]", "[Cf]", "Cf"]:
                    n = list(atomistic[key])
                    if len(n) == 1:
                        neighbors.append(n[0])
            for key in symbols:
                if symbols[key] in ["[#98]", "[Cf]", "Cf"]:
                    n = list(atomistic[key])
                    if len(n) == 1:
                        atomistic.remove_edge(key, n[0])

            n = 0
            for inside in range(len(descriptors)):
                if descriptors[inside] in single_path["1"] or get_comp(descriptors[inside]) in single_path["2"]:
                    continue
                for outside in range(len(descriptors)):
                    symbols = nx.get_node_attributes(atomistic, "symbol")
                    bonds = nx.get_edge_attributes(atomistic, "bonds")
                    active = nx.get_node_attributes(atomistic, "active")
                    if inside == outside:
                        input = descriptors[inside]
                        for key in symbols:
                            if symbols[key] == get_comp(input) and active[key]:
                                atomistic.add_edge(key, neighbors[n], bond_type="2")
                                nodes = orientation(networkx_graph, inside)
                                for j in nodes:
                                    nested_orientation.append(j)
                                break
                            elif key == len(symbols) - 1:
                                added = atomistic.number_of_nodes()
                                atomistic.add_node(added, symbol=get_comp(input), active=True)
                                stochastic_descriptor.append(added)
                                atomistic.add_edge(added, neighbors[n], bond_type="2")
                                nodes = orientation(networkx_graph, inside)
                                for j in nodes:
                                    nested_orientation.append(j)
                                break
                    else:
                        output = descriptors[outside]
                        for key in symbols:
                            if symbols[key] == output and active[key]:
                                atomistic.add_edge(key, neighbors[n], bond_type="1")
                                break
                            elif key == len(symbols) - 1:
                                added = atomistic.number_of_nodes()
                                atomistic.add_node(added, symbol=output, active=True)
                                stochastic_descriptor.append(added)
                                atomistic.add_edge(added, neighbors[n], bond_type="1")
                                break
                    n += 1

        if len(implicit_ends) > 0:
            symbols = nx.get_node_attributes(atomistic, "symbol")
            active = nx.get_node_attributes(atomistic, "active")
            junctions = []
            for key in symbols:
                if active[key]:
                    descriptor_locations = [(d.start(0), d.end(0)) for d in re.finditer(desc_regex, symbols[key])]
                    if len(descriptor_locations) == 1:
                        junctions.append(key)

        endgrp_local_el = set()
        for smiles in implicit_ends:
            smiles, nested_object_list = sub_obj_with_Bk(smiles)

            descriptor_locations = [(d.start(0), d.end(0)) for d in re.finditer(desc_regex, smiles)]

            ## store stochastic elements
            if len(descriptor_locations) == 0:
                endgrp_local_el.add(smiles)
                continue

            descriptor = smiles[descriptor_locations[0][0]:descriptor_locations[0][1]]
            smiles = smiles.replace(descriptor, "[Cf]")
            if "[Bk]" in smiles:
                smiles, nested_object_list = modify_string(smiles, nested_object_list)

            if is_bigsmarts:
                rdkit_graph = Chem.MolFromSmiles(smiles)
            else:
                rdkit_graph = Chem.MolFromSmiles(smiles)

            networkx_graph = RDKit_to_networkx_graph(mol=rdkit_graph, is_bigsmarts=is_bigsmarts, level=level)
            # If the level is zero, set explicit_atom_ids of all nodes but bonding descriptors (Cf) and
            # nested objects (Bk) to True. This is done so that end groups listed after ";" have the attribute
            # explicit_atom_ids set to True
            if level == 0:
                for n in networkx_graph.nodes:
                    if networkx_graph.nodes[n]["symbol"] not in ["Cf", "Bk"]:
                        networkx_graph.nodes[n]["explicit_atom_ids"] = True

            def allowed_to_add(graph_descriptor, implicit_descriptor, bond_type):
                if bond_type == "1" and graph_descriptor == implicit_descriptor:
                    return True
                if bond_type == "2" and get_comp(graph_descriptor) == implicit_descriptor:
                    return True
                return False

            def add_terminal(graph, smiles, graph_key, b_type):
                graph = nx.disjoint_union(graph, smiles)
                symbols = nx.get_node_attributes(graph, "symbol")
                for key in symbols:
                    if symbols[key] in ["[#98]", "[Cf]", "Cf"]:
                        n = list(graph[key])
                        if len(n) == 1:
                            neighbor = n[0]
                            graph.remove_edge(key, neighbor)
                            graph.add_edge(graph_key, neighbor, bond_type=b_type)
                return graph

            for key in junctions:
                if symbols[key] == left_terminal:
                    iteration = terminals_attachment[0]
                elif symbols[key] == right_terminal:
                    iteration = terminals_attachment[1]
                else:
                    iteration = ["1", "2"]
                for b_type in iteration:
                    if allowed_to_add(symbols[key], descriptor, b_type):
                        atomistic = add_terminal(atomistic, networkx_graph, key, b_type)
                        for n in nested_object_list:
                            nested_objects.append(n)
                            was_implicit_now.append(True)
                            nested_orientation.append("1")

        nx.set_node_attributes(atomistic, False, "active")

        local_el = nx.get_node_attributes(atomistic, "local_el")
        nodes = set(stochastic_descriptor + terminals_indices[-1])
        for n in nodes:
            local_el[n] = {"wildcard_cluster": wildcard_cluster, "ru_local_el": ru_local_el,
                           "endgrp_local_el": endgrp_local_el}
        nx.set_node_attributes(atomistic, local_el, "local_el")

        # Add attribute level to bonding descriptor nodes to indicate what level they are
        # level_dict = dict(zip(nodes, len(nodes) * [level]))
        # For each node, get the levels of the neighbors
        level_dict = dict({})
        for n in nodes:
            # Neighbor levels
            neighbor_levels = [atomistic.nodes[x]["level"] for x in find_neighbors(atomistic, n) if
                               "level" in atomistic.nodes[x]]
            # Take the maximum level
            max_level = max(neighbor_levels)
            # Assign it to the bonding descriptor
            level_dict[n] = max_level
        nx.set_node_attributes(atomistic, level_dict, "level")

    if len(nested_objects) > 0:
        atomistic = build_level(atomistic=atomistic,
                                is_bigsmarts=is_bigsmarts,
                                object_list=nested_objects,
                                object_orientation=nested_orientation,
                                was_implicit=was_implicit_now,
                                level=level + 1)

    if level > 0:
        return atomistic

    root = 0
    symbols = nx.get_node_attributes(atomistic, "symbol")
    for key in symbols:
        if key == root:
            continue
        if not nx.has_path(atomistic, root, key):
            atomistic.remove_node(key)
        else:
            neighbors = list(atomistic[key])
            for d in re.finditer(desc_regex, symbols[key]):
                if len(neighbors) == 2:
                    atomistic.remove_node(key)
                    if "=" not in symbols[key]:
                        atomistic.add_edge(*tuple(neighbors), bond_type=str(rdkit.Chem.rdchem.BondType.SINGLE))
                    else:
                        atomistic.add_edge(*tuple(neighbors), bond_type=str(rdkit.Chem.rdchem.BondType.DOUBLE))
                break
            if symbols[key] in ["[#99]", "[Es]", "Es"] and len(neighbors) == 1:
                atomistic.remove_node(key)
    if symbols[0] in ["[#99]", "[Es]", "Es"] and len(list(atomistic[0])) == 1:
        atomistic.remove_node(0)

    symbols = nx.get_node_attributes(atomistic, "symbol")
    for key in symbols:
        if symbols[key] == "No":
            symbols[key] = "H"
    nx.set_node_attributes(atomistic, symbols, "symbol")

    hydro = nx.get_node_attributes(atomistic, "num_hs")
    explicit_atom_ids = copy.deepcopy(hydro)
    for key in explicit_atom_ids:
        # If the atom is within an end group or already has explicit_atom_ids set to True, set explicit_atom_ids to True
        if key in endgroup_atom_ids or (
                "explicit_atom_ids" in atomistic.nodes[key] and atomistic.nodes[key]["explicit_atom_ids"]):
            explicit_atom_ids[key] = True
        else:
            explicit_atom_ids[key] = False
    nx.set_node_attributes(atomistic, explicit_atom_ids, "explicit_atom_ids")

    return atomistic


def build_topology(atomistic):
    def extract_atoms(graph, extracted, start_atom, descriptors):
        extracted.add(start_atom)
        neighbors = list(graph[start_atom])
        next_atoms = []
        for n in neighbors:
            if n not in descriptors and n not in extracted:
                extracted.add(n)
                next_atoms.append(n)
        for n in next_atoms:
            extracted = extract_atoms(graph, extracted, n, descriptors)
        return extracted

    elements = nx.get_node_attributes(atomistic, "symbol")
    bonds = nx.get_edge_attributes(atomistic, "bond_type")

    descriptors = []
    for key in elements:
        d = list(re.finditer(desc_regex, elements[key]))
        if len(d) != 0:
            descriptors.append(key)

    edge_labels = nx.get_edge_attributes(atomistic, "bond_type")
    start = []
    for i in edge_labels:
        if edge_labels[i] == "1":
            d = list(re.finditer(desc_regex, elements[i[0]]))
            if len(d) == 0:
                start.append(i[0])
            else:
                start.append(i[1])
    if len(start) == 0:
        start = [0]

    extracted = []
    for i in range(len(start)):
        extracted.append(extract_atoms(atomistic, set(), start[i], descriptors))
    ids = dict()
    counter = 1
    for i in extracted:
        for k in i:
            ids[k] = counter
        counter += 1
    nx.set_node_attributes(atomistic, ids, "ids")

    start = []
    for i in edge_labels:
        if edge_labels[i] == "2":
            d = list(re.finditer(desc_regex, elements[i[0]]))
            if len(d) == 0:
                if i[0] not in ids:
                    start.append(i[0])
            else:
                if i[1] not in ids:
                    start.append(i[1])

    extracted = []
    for i in range(len(start)):
        extracted.append(extract_atoms(atomistic, set(), start[i], descriptors))
    for i in extracted:
        for k in i:
            ids[k] = counter
        counter += 1
    for i in descriptors:
        ids[i] = counter
        counter += 1
    nx.set_node_attributes(atomistic, ids, "ids")

    topology = atomistic.copy()
    edge_labels = nx.get_edge_attributes(topology, "bond_type")
    for d in descriptors:
        neighbors = topology[d]
        same_frag = dict()
        for n in neighbors:
            same_frag[n] = ids[n]
        for i in same_frag:
            for j in same_frag:
                if i != j and same_frag[i] == same_frag[j]:
                    a = edge_labels[(min(d, i), max(d, i))]
                    b = edge_labels[(min(d, j), max(d, j))]
                    if [a, b] == ["1", "2"] or [b, a] == ["1", "2"]:
                        edge_labels[(min(d, i), max(d, i))] = "3"
                        edge_labels[(min(d, j), max(d, j))] = "3"
    nx.set_edge_attributes(topology, edge_labels, "bond_type")

    edge_labels = nx.get_edge_attributes(topology, "bond_type")
    vals = list(edge_labels.values())
    while vals.count("1") + vals.count("2") + vals.count("3") < len(vals):
        for i in edge_labels:
            if edge_labels[i] not in ["1", "2", "3"]:
                topology = nx.contracted_edge(topology, i, self_loops=False)
                break
        edge_labels = nx.get_edge_attributes(topology, "bond_type")
        vals = list(edge_labels.values())

    topology_undir = copy.deepcopy(topology)

    topology = nx.to_directed(topology)
    topology = nx.DiGraph(topology)

    edge_labels = nx.get_edge_attributes(topology, "bond_type")
    for i in edge_labels:
        a = edge_labels[i] == "1" and i[0] in descriptors
        b = edge_labels[i] == "2" and i[1] in descriptors
        if a or b:
            topology.remove_edge(*i)
    edge_labels = nx.get_edge_attributes(topology, "bond_type")
    for i in edge_labels:
        edge_labels[i] = ""
    nx.set_edge_attributes(topology, edge_labels, "bond_type")

    ############# topology adjustments
    # if [Md] has a degree of 2, then delete node
    # replace [Md] with *
    # replace [Fm] with ?*
    symbols = nx.get_node_attributes(atomistic, "symbol")
    for key in symbols:
        if symbols[key] == "Md" and atomistic.degree[key] == 2:

            # wildcard cycle, do not delete Md: get neighbor of Md, check whether Md and Fm connect to the same descriptor
            connections_Md = list(atomistic[key].keys())
            con1 = -1
            con2 = -2
            for node in connections_Md:
                if node in descriptors:
                    con1 = node
                if symbols[node] == "Fm":
                    connections_Fm = list(atomistic[node].keys())
                    for node in connections_Fm:
                        if node in descriptors:
                            con2 = node
            if con1 == con2:
                continue

            # get two neighbors of the Md
            connections = list(atomistic[key].keys())

            # get the bond type connecting Md to the atom that is not Fm
            for atom in atomistic[key]:
                if symbols[atom] != "Fm":
                    new_bond = atomistic[key][atom]['bond_type']

            # delete Md from the graph
            atomistic.remove_node(key)

            # reconnect the Fm to the other neighbor
            atomistic.add_edge(connections[0], connections[1], bond_type=new_bond)
    nx.set_node_attributes(atomistic, symbols, "symbol")

    symbols = nx.get_node_attributes(atomistic, "symbol")
    for key in symbols:
        if symbols[key] == "Md":
            symbols[key] = "*"
        elif symbols[key] == "Fm":
            symbols[key] = "?*"
        elif symbols[key] == "Es":
            symbols[key] = ""
    nx.set_node_attributes(atomistic, symbols, "symbol")
    #############

    ############ multidigraph depiction for networks
    symbols = nx.get_node_attributes(atomistic, "symbol")
    ids = nx.get_node_attributes(atomistic, "ids")
    bonds = nx.get_edge_attributes(atomistic, "bond_type")

    # iterate through each symbol, determine # of bonds between ids and descriptor
    bonds_ids_desc = {}
    for key in symbols:
        if key in descriptors:
            # get neighbors of descriptors
            neighbors = atomistic[key]
            for n in neighbors:
                if n not in descriptors:
                    # populate vector that maps number of connections to descriptor
                    v = (ids[key], ids[n], bonds[feed_to_bonds_n(n, key)])
                    if v not in bonds_ids_desc:
                        bonds_ids_desc[v] = 0
                    else:
                        bonds_ids_desc[v] += 1

    multidigraph = nx.MultiDiGraph(topology)
    multi_ids = nx.get_node_attributes(multidigraph, "ids")
    for conn in bonds_ids_desc:
        # conn stores (desc, node, bond_type)
        # establish key1 and key2 in the multigraph
        for key in multi_ids:
            if multi_ids[key] == conn[0]:
                desc = key
            elif multi_ids[key] == conn[1]:
                node = key
        if conn[2] == "2":
            multidigraph.add_edges_from([(desc, node)] * bonds_ids_desc[conn])
        else:
            multidigraph.add_edges_from([(node, desc)] * bonds_ids_desc[conn])
    ###########

    bonds = nx.get_edge_attributes(atomistic, "bond_type")
    symbols = nx.get_node_attributes(atomistic, "symbol")
    for key in bonds:
        if "1" in bonds[key] or "2" in bonds[key]:
            if "=" in symbols[key[0]] + symbols[key[1]]:
                bonds[key] = bonds[key] + "_DOUBLE"
            else:
                bonds[key] = bonds[key] + "_SINGLE"
    nx.set_edge_attributes(atomistic, bonds, "bond_type")

    return topology, topology_undir, multidigraph, descriptors, ids


def find_neighbors(graph, node):
    """
    This function finds all neighbors of a node in a graph
    Args:
        graph: graph
        node: node

    Returns: list of neighbors

    """

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