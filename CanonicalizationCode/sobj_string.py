import networkx as nx
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
import copy
import re

from polymersearch.graphs import desc_regex, get_comp
from parser import BigSMILES_BigSmilesObj

def to_bigsmiles(tree):
    # works for single objects

    polymer = tree.DFTA_to_networkx()
    alphabet = nx.get_edge_attributes(polymer, "smiles")
    node_alphabet = nx.get_node_attributes(polymer, "smiles")
    isStart  = nx.get_node_attributes(polymer, "is_empty") 
    isBranch = nx.get_node_attributes(polymer, "is_branch")

    for branch in isBranch:
        if isBranch[branch]:
            for neighbor in polymer[branch]:
                alphabet[(branch, neighbor)] = node_alphabet[branch]

    def traversal(current):
        weight = 0
        next_state = None
        for neighbor in polymer[current]:
            if (current, neighbor) in traversal_order:
                continue
            molar_mass = 0
            for key in alphabet:
                if key[0] == current and key[1] == neighbor:
                    molar_mass += Descriptors.ExactMolWt(Chem.MolFromSmiles(alphabet[key]))
            if molar_mass > weight:
                weight = molar_mass
                next_state = neighbor
        if next_state is None:
            if current == root and len(node_prev) == 0:
                return 
            max = 0 
            curr_next = 0
            for key in node_prev:
                if node_prev[key] > max and key[1] == current:
                    curr_next = key
                    max = node_prev[key]
            try:
                del node_prev[curr_next]
                traversal(curr_next[0])
            except:
                return
        else:
            traversal_order.append((current, next_state))
            node_prev[(current, next_state)] = len(node_prev)
            traversal(next_state) 
    
    def transitions_to_object(traversal_order):
        def smiles_to_RU(smiles, ids_to_replace):
            s = copy.deepcopy(smiles)
            s = s.replace("[*:1]", "[<" + str(ids_to_replace[0]) + "]")
            for i in range(1, len(ids_to_replace)):
                s = s.replace("[*:" + str(i + 1) + "]", "[>" + str(ids_to_replace[i]) + "]")
            return s 

        object = []
        edges = polymer.edges
        branches_added = []
        for i in range(len(traversal_order)):
            alphabet_between_nodes = {}
            for bonds in alphabet:
                if traversal_order[i][0] == bonds[0] and traversal_order[i][1] == bonds[1]:
                    smiles = alphabet[bonds]
                    heavy = copy.deepcopy(smiles)
                    heavy = re.sub(r"\[\*:\d+\]", "[Bk]", heavy) 
                    alphabet_between_nodes[smiles] = [Descriptors.ExactMolWt(Chem.MolFromSmiles(heavy)), bonds]
            sorted_r = sorted(alphabet_between_nodes.items(), key=lambda e: e[1][0])
            for key in sorted_r:
                if isBranch[traversal_order[i][0]] or isBranch[traversal_order[i][1]]:
                    if isBranch[traversal_order[i][0]]:
                        b = traversal_order[i][0]
                    else:
                        b = traversal_order[i][1]
                    if b in branches_added:
                        continue
                    branches_added.append(b)
                    neighbors = []
                    for e in edges:
                        if e[1] == b:
                            neighbors.append(e[0])
                    for e in edges:
                        if e[0] == b:
                            neighbors.append(e[1])
                else:
                    smiles = key[0]
                    neighbors = key[1][1]
                o = smiles_to_RU(smiles, neighbors)
                if "Es" not in o:
                    object.append(o)

        return object  

    def merge_deterministic(object):
        isEnd  = nx.get_node_attributes(polymer, "is_end")
        deterministic = set()
        for bonds in polymer.edges:
            for i in range(2):
                incoming = len(polymer.in_edges(bonds[i]))
                outgoing = len(polymer.out_edges(bonds[i]))
                if incoming == 1 and outgoing == 1 and not isBranch[bonds[i]]:
                    add = True
                    for b in polymer.edges:
                        if bonds[i] == b[0]:
                            if isBranch[b[1]]:
                                add = False 
                                break
                        elif bonds[i] == b[1]:
                            if isBranch[b[0]]:
                                add = False
                                break
                    if add and not isStart[bonds[i]] and not isEnd[bonds[i]]:
                        deterministic.add(bonds[i]) 
        
        def merge(smiles1, smiles2, desc):
            smiles1 = smiles1.replace("[<","[Bk:")
            smiles1 = smiles1.replace("[>","[Cf:")
            smiles2 = smiles2.replace("[<","[Bk:")
            smiles2 = smiles2.replace("[>","[Cf:")
            s1 = Chem.MolFromSmiles(smiles1)
            s2 = Chem.MolFromSmiles(smiles2)
            combo = Chem.CombineMols(s1,s2)
            edcombo = Chem.RWMol(combo)
            edcombo = Chem.MolFromSmiles((Chem.MolToSmiles(combo)))
            rwmol = Chem.RWMol(edcombo)

            add = []
            remove = []
            for at in edcombo.GetAtoms():
                if desc == at.GetAtomMapNum():
                    remove.append(at.GetIdx())
                    neighbors = at.GetNeighbors()
                    for n in neighbors:
                        add.append(n.GetIdx())
            rwmol.AddBond(add[0], add[1], order=Chem.rdchem.BondType.SINGLE)
            for index in sorted(remove, reverse=True):
                rwmol.RemoveAtom(index)
            smiles = Chem.MolToSmiles(rwmol)
            smiles = smiles.replace("Bk:", "<")     
            smiles = smiles.replace("Cf:", ">")  
            return smiles

        for d in deterministic: 
            try:
                index = None
                merging = []
                for j in range(len(object)):
                    if object[j].count("[>") + object[j].count("[<") > 2:
                        continue
                    a = "[<" + str(d) + "]" 
                    b = "[>" + str(d) + "]"
                    if a in object[j] or b in object[j]:
                        merging = [object[j], object[j + 1]]
                        index = j
                        break
                if index is not None:
                    smiles = merge(merging[0], merging[1], d)
                    object.insert(index, smiles)
                    object.remove(merging[0])
                    object.remove(merging[1])
            except:
                continue
        return object

    def descriptors_to_ends(object):
        for i in range(len(object)):
            try:
                d = []
                repeat = "{[]" + object[i] + "[]}"
                desc = [(d.start(0), d.end(0)) for d in re.finditer(desc_regex, repeat)] 
                for j in desc:
                    d.append(repeat[j[0]:j[1]])
                mapping = dict()
                mapping_reverse = dict()
                count = 1
                for r in d:
                    new = r[0:2] + str(count) + "]"
                    mapping[r] = new
                    mapping_reverse[new] = r
                    repeat = repeat.replace(r, mapping[r])
                    count += 1
                p = BigSMILES_BigSmilesObj.BigSMILES(repeat)
                p.writeStandard()
                object[i] = str(p[0][0])
                object[i] = object[i].replace("]-", "]")
                object[i] = object[i].replace("-[", "[")
                for r in mapping_reverse:
                    object[i] = object[i].replace(r, mapping_reverse[r])
            except:
                continue
        return object

    def relabel_bonding_descriptors(repeat_units):    
        d = []
        for i in range(len(repeat_units)):
            desc = [(d.start(0), d.end(0)) for d in re.finditer(desc_regex, repeat_units[i])] 
            for j in desc:
                d.append(repeat_units[i][j[0]:j[1]])
        d2 = copy.deepcopy(d)

        # relabel the bonding id number
        counter = 1
        already_treated = []
        for i in range(len(d)):
            if i not in already_treated:
                already_treated.append(i)
                for j in range(len(d)):
                    if d[i] == d[j] or d[i] == get_comp(d[j]):
                        d2[j] = "[" + d[j][1] + str(counter) + "]"
                        already_treated.append(j)
                counter += 1

        counter = 0
        # replace repeat unit descriptors with modified descriptors in d2
        for i in range(len(repeat_units)):
            repeat_units[i] = re.sub(desc_regex, "[Bk]", repeat_units[i])
            while "Bk" in repeat_units[i]:
                repeat_units[i] = repeat_units[i].replace("[Bk]", d2[counter], 1)
                counter += 1
        
        return repeat_units

    def add_dollar(repeats):
        if len(repeats) > 2:
            return repeats
        import re
        replace = []
        delete = []
        for i in range(len(repeats)):
            for j in range(len(repeats)):
                if i == j:
                    continue

                pair = [repeats[i], repeats[j]]
                desc = []
                for k in range(len(pair)):
                    r_tot_desc = [(d.start(0), d.end(0)) for d in re.finditer(desc_regex, pair[k])]
                    desc.append([])
                    for r in r_tot_desc:
                        desc[-1].append(pair[k][r[0]:r[1]])
                        pair[k] = pair[k].replace(pair[k][r[0]:r[1]], "[Bk]")
                    pair[k] = Chem.MolToSmiles(Chem.MolFromSmiles(pair[k]))
                if pair[0] == pair[1] and len(desc[0]) == len(desc[1]) == 2:
                    if get_comp(desc[0][0]) == desc[0][1] and get_comp(desc[1][0]) == desc[1][1] and desc[0][0] in desc[1]: 
                        if j not in replace:
                            replace.append(i)
                            delete.append(j)
        repeats_final = []
        for i in range(len(repeats)):
            if i in delete:
                continue
            elif i in replace:
                repeats[i] = repeats[i].replace("<", "$")
                repeats[i] = repeats[i].replace(">", "$")   
            repeats_final.append(repeats[i])
        return repeats_final

    def traversal_to_string(traversal_order):
        object = transitions_to_object(traversal_order)
        object = merge_deterministic(object)
        object = descriptors_to_ends(object) 
        object = relabel_bonding_descriptors(object)
        object = add_dollar(object)

        bigsmiles = "{[]"
        for o in object:
            bigsmiles += o + ","
        bigsmiles = bigsmiles[:-1] + "[]}"

        return bigsmiles

    starts = []
    for root in isStart:
        if isStart[root]:
            starts.append(root)
    
    bigsmiles = []
    for i in range(len(starts)):
        traversal_order = []
        node_prev = dict()
        traversal(starts[i]) 
        for j in range(len(starts)):
            if i == j:
                continue
            traversal(starts[j]) 
        bigsmiles.append(traversal_to_string(traversal_order))

    bigsmiles = sorted(bigsmiles)[0]
    return bigsmiles