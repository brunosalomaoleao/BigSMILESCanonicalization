import pandas as pd
import time
import copy
import re

import rdkit
from rdkit import Chem
import networkx as nx

from polymersearch import topology
from polymersearch.graphs import build_atomistic, build_topology, get_objects, get_repeats, build_all_atomistic_graphs
from polymersearch.adjacency import feed_to_bonds_n

def generate_NetworkX_graphs(input_string, is_bigsmarts = False):
    atomistic_directed, no_other_objects = build_atomistic(input_string, is_bigsmarts)
    topology, topology_undir, multidigraph, descriptors, ids = build_topology(atomistic_directed)
    return {"string": input_string, 
            "atomistic": atomistic_directed,
            "topology": topology, 
            "top_undir": topology_undir, 
            "ids": ids,
            "descriptors": descriptors, 
            "multidigraph": multidigraph,
            "no_other_objects":no_other_objects
            }
def generate_all_possible_graphs(input_string):
    """
    This function generates all possible graphs (atomistic and topology), given a BigSMILES string
    Args:
        input_string: BigSMILES
    Returns: List of dictionaries, whose keys are "atomistic" and "topology"
    """
    # Generate all atomistic graphs
    atomistic_list = build_all_atomistic_graphs(input_string)

    # Generate all topology graphs
    topology_list = []
    for atomistic in atomistic_list:
        topology, _, _, _, _ = build_topology(atomistic)
        topology_list.append(topology)

    # Generate list of graphs
    graphs = []
    for atomistic, topology in zip(atomistic_list, topology_list):
        graphs.append({"atomistic": atomistic,
                       "topology": topology})

    return graphs
def identify_cycles(graphs):
    topology = graphs["topology"]

    ids = nx.get_node_attributes(topology, "ids")
    individual_cycles = []
    for i in nx.simple_cycles(topology):
        individual_cycles.append([ids[j] for j in i])
    
    explicit = nx.get_node_attributes(graphs["atomistic"], "explicit_atom_ids")
    ids = nx.get_node_attributes(graphs["atomistic"], "ids")
    treat_as_cycle = []
    for key in explicit:
        if explicit[key]:
            remove = -1
            for i in individual_cycles:
                if ids[key] in i:
                    remove = i
                    break
            if remove != -1:
                individual_cycles.remove(remove)
                treat_as_cycle.append(remove)
    
    # https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements
    def to_graph(l):
        G = nx.Graph()
        for part in l:
            G.add_nodes_from(part)
            G.add_edges_from(to_edges(part))
        return G

    def to_edges(l):
        it = iter(l)
        last = next(it)
        for current in it:
            yield last, current
            last = current   
    
    individual_clusters = []
    G = to_graph(individual_cycles)
    for cycles in nx.connected_components(G):
        individual_clusters.append(cycles)
        
    level = nx.get_node_attributes(graphs["top_undir"], "level")
    ids = nx.get_node_attributes(graphs["top_undir"], "ids")
    inv_map = {v: k for k, v in ids.items()}
    bonds = nx.get_edge_attributes(graphs["top_undir"],"bond_type")
    nested = []
    for i in range(len(individual_clusters)):
        count_0 = 0
        count_1 = 0
        for j in individual_clusters[i]:
            id = inv_map[j]
            if id in level and level[id] == 1:
                count_1 = 1
            if id in level and level[id] == 0:
                count_0 = 1
        if count_0 == 0 and count_1 == 1 and individual_clusters[i] not in nested:
            nested.append(individual_clusters[i]) 
    for i in range(len(nested)):
        if nested[i] not in individual_clusters:
            continue
        for j in range(len(individual_clusters)):
            adjacent = False
            for k in nested[i]:
                for l in individual_clusters[j]:
                    if feed_to_bonds_n(inv_map[k], inv_map[l]) in bonds:
                        adjacent = True
            if adjacent:
                individual_clusters[j].update(nested[i])
                individual_clusters.remove(nested[i])
                break

    group_cycles = []
    for i in range(len(individual_clusters)):
        group_cycles.append([])
        for j in range(len(individual_cycles)):
            if individual_cycles[j][0] in individual_clusters[i]:
                group_cycles[-1].append(individual_cycles[j]) 
    return group_cycles, individual_clusters, treat_as_cycle

def cycles_to_ring_SMILES(graphs, cycle):
    # https://github.com/maxhodak/keras-molecules/pull/32/files            

    symbols = nx.get_node_attributes(graphs["atomistic"], "symbol")
    formal_charge = nx.get_node_attributes(graphs["atomistic"], "formal_charge")
    is_aromatic = nx.get_node_attributes(graphs["atomistic"], "is_aromatic")
    bonds = nx.get_edge_attributes(graphs["atomistic"], "bond_type_object")
    ids = nx.get_node_attributes(graphs["atomistic"], "ids")
    
    rings = []
    for i in range(len(cycle)):
        for j in range(len(cycle[i])):
            mol = Chem.RWMol()
            node_to_idx = {}
            for node in graphs["atomistic"].nodes():
                if ids[node] in cycle[i][j] and node not in graphs["descriptors"] and symbols[node] != "":
                    a = Chem.Atom(symbols[node])
                    a.SetFormalCharge(formal_charge[node])
                    a.SetIsAromatic(is_aromatic[node])
                    idx = mol.AddAtom(a)
                    node_to_idx[node] = idx
            
            d_bonds = []
            already_added = set()
            for edge in graphs["atomistic"].edges():
                first, second = edge
                if ids[first] in cycle[i][j] and ids[second] in cycle[i][j]:
                    if first in graphs["descriptors"] or second in graphs["descriptors"]:
                        d_bonds.append([(first, second), rdkit.Chem.rdchem.BondType.SINGLE])
                    else:
                        ifirst = node_to_idx[first]
                        isecond = node_to_idx[second]
                        bond_type_object = bonds[first, second]
                        if tuple(sorted([ifirst, isecond])) not in already_added:
                            mol.AddBond(ifirst, isecond, bond_type_object)
                            already_added.add(tuple(sorted([ifirst, isecond])))

            for k in range(len(d_bonds)):
                for l in range(k + 1, len(d_bonds)):
                    descriptor = set(d_bonds[k][0]).intersection(d_bonds[l][0])
                    if len(descriptor) == 1:
                        try:
                            a = set(d_bonds[k][0]).difference(descriptor).pop()
                            b = set(d_bonds[l][0]).difference(descriptor).pop()
                            ifirst = node_to_idx[a]
                            isecond = node_to_idx[b]
                            if tuple(sorted([ifirst, isecond])) not in already_added:
                                mol.AddBond(ifirst, isecond, d_bonds[k][1])
                                already_added.add(tuple(sorted([ifirst, isecond])))
                        except:
                            continue
            
            Chem.SanitizeMol(mol)
            rings.append(Chem.MolToSmiles(mol))

    return rings 

def get_repeats_as_rings(graphs):
    cycles, clusters = identify_cycles(graphs["topology"])
    ring_smiles = cycles_to_ring_SMILES(graphs, cycles)
    return ring_smiles

def contains_substructure(bigsmarts_graphs, bigsmiles_graphs):
    q_cycles, q_clusters, q_macrocycle = identify_cycles(bigsmarts_graphs)   
    t_cycles, t_clusters, t_macrocycle = identify_cycles(bigsmiles_graphs) 
    search = topology.Topology_Graph_Matcher(bigsmarts_graphs, bigsmiles_graphs, 
                                            q_cycles, q_clusters, 
                                            t_cycles, t_clusters, 
                                            q_macrocycle, t_macrocycle)
    return search.search_repeats_endgroups() 
    
def logical_repeat_unit_search(bigsmarts):
    # Assumptions: 
    # only 1 object in the query has logical operators
    # only one type of logical operation per query: "!", "or", "xor" along with "and"

    # determine all objects in the string
    objects = get_objects(bigsmarts)

    # iterate through each object
    for object in objects[0]:

        # get all repeat units in the object
        repeats = get_repeats(object)

        # map the logical operator to the repeat units
        logical = {} 
        for repeat in repeats[0]:

            # group logical operator with repeat units or SMARTS
            if repeat.find("[or") == 0 and repeat[0:5] not in logical:
                logical[repeat[0:5]] = [repeat[5:]]
            elif repeat.find("[or") == 0 and repeat[0:5] in logical:
                logical[repeat[0:5]].append(repeat[5:])
            
            elif repeat.find("[xor") == 0 and repeat[0:6] not in logical:
                logical[repeat[0:6]] = [repeat[6:]]
            elif repeat.find("[xor") == 0 and repeat[0:6] in logical:
                logical[repeat[0:6]].append(repeat[6:])

            elif repeat.find("!") == 0 and repeat != "!*" and "!" not in logical:
                logical["!"] = [repeat[1:]]
            elif repeat.find("!") == 0 and repeat != "!*" and repeat[1:] in logical:
                logical["!"].append(repeat[1:])

            elif "and" not in logical:
                logical["and"] = [repeat]
            else:
                logical["and"].append(repeat)
        
        # is this the object with the logical operators?
        logic = list(logical.keys())
        logic = [i for i in logic if i != "and"]

        # if not, continue
        if len(logic) == 0:
            continue
        
        # list of object strings that convert logical strings into valid BigSMARTS
        objects_replaced = []
        logic_return = "and"
        for logic in logical:

            if "or" in logic or "xor" in logic:
                logic_return = logic
                for repeat in logical[logic]:
                    # delete logical operator and repeat unit
                    if logic + repeat + "," in object:
                        # this is for every other repeat unit in the stochastic object
                        replaced = object.replace(logic + repeat + ",", "")
                    else:
                        # this is for the last repeat unit in the stochastic object only
                        replaced = object.replace("," + logic + repeat, "")
                    replaced = replaced.replace(logic, "")
                    objects_replaced.append(replaced)
            
            elif "!" in logic:
                logic_return = logic
                for repeat in logical[logic]:
                    # delete logical operator
                    replaced = object.replace("!", "")
                    objects_replaced.append(replaced)
                # delete logical operator and repeat unit
                if logic + repeat + "," in object:
                    # this is for every other repeat unit in the stochastic object
                    replaced = object.replace(logic + repeat + ",", "")
                else:
                    # this is for the last repeat unit in the stochastic object only
                    replaced = object.replace("," + logic + repeat, "")
                replaced = replaced.replace(logic, "")
                objects_replaced.append(replaced)

        bigsmarts_replaced = []
        for o in objects_replaced:
            bigsmarts_replaced.append(bigsmarts.replace(object, o))
    
        return bigsmarts_replaced, logic_return
    
    return [bigsmarts], "and"
                
def search_matches(bigsmarts, bigsmiles):
    bigsmarts_list, logic = logical_repeat_unit_search(bigsmarts) 
    matches = []
    for bigsmarts in bigsmarts_list:
        bigsmarts_graphs = generate_NetworkX_graphs(input_string = bigsmarts, is_bigsmarts = True) 
        bigsmiles_graphs = generate_NetworkX_graphs(input_string = bigsmiles, is_bigsmarts = False) 
        # visualize_NetworkX_graphs(graphs = bigsmarts_graphs, id = "bigsmarts")
        # visualize_NetworkX_graphs(graphs = bigsmiles_graphs, id = "bigsmiles")
        m = contains_substructure(bigsmarts_graphs = bigsmarts_graphs, bigsmiles_graphs = bigsmiles_graphs)
        if "or" in logic and "xor" not in logic and m:
            return True
        elif "and" in logic:
            return m
        matches.append(m)
    if "!" in logic:
        if matches[0] == False and matches[1] == True:
            return True
        return False 
    if "xor" in logic:
        if matches[0] == False and matches[1] == True or matches[0] == True and matches[1] == False:
            return True
        return False
    if "or" in logic:
        return matches[1]
    return False
    