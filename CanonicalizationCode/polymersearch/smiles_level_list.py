import re
import numpy as np

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Descriptors import ExactMolWt

import run

def find_branch_smiles_list(bigsmiles):
    line = bigsmiles

    matchObj = re.match( r'(.*?){(.*?){(.*?)}(.*)}(.*)', line, re.M|re.I) 

    branch_smiles_list = []

    while matchObj:
    
        raw_branch_smiles=matchObj.group(3)
        #print(raw_branch_smiles)
        if '$' in raw_branch_smiles:
            branch_smiles = raw_branch_smiles.replace('$','')[4:-4]
        
            mol = Chem.MolFromSmiles(branch_smiles)
            branch_smiles = Chem.MolToSmiles(mol)
            branch_smiles_list.append(branch_smiles)

        elif '<' in raw_branch_smiles or '>' in raw_branch_smiles:
            branch_smiles = raw_branch_smiles.replace('<','').replace('>','')[4:-4]
            #if ('Si' in branch_smiles)==False:
            #    branch_smiles = branch_smiles[0]+'1'+branch_smiles[1:]+'1'

            mol = Chem.MolFromSmiles(branch_smiles)
            branch_smiles = Chem.MolToSmiles(mol)
            branch_smiles_list.append(branch_smiles)
        
        else:
           print("Error")

        raw_branch_smiles_x = '{' + raw_branch_smiles + '}'
        line = line.replace(raw_branch_smiles_x,'',1)
        #print(line)
        matchObj = re.match( r'(.*?){(.*?){(.*?)}(.*)}(.*)', line, re.M|re.I) 


    return branch_smiles_list


def check_weight(smilesA,smiles_list):
    for i in range(0, len(smiles_list)):
        molA = Chem.MolFromSmiles(smilesA)
        mol = Chem.MolFromSmiles(smiles_list[i])
        if abs(ExactMolWt(molA) - ExactMolWt(mol))<=2.015651 and abs(ExactMolWt(molA) - ExactMolWt(mol)) >= 2.015650: # two H atom weights
           return True
        if abs(ExactMolWt(molA) - ExactMolWt(mol))<=1.007826 and abs(ExactMolWt(molA) - ExactMolWt(mol)) >= 1.007825: # one H atom weights
           return True

    return False


def get_smiles_level_list(bigsmiles):
    branch_smiles_list_x = find_branch_smiles_list(bigsmiles)
    smiles_list_x = run.get_repeats_as_rings(bigsmiles)
    smiles_list_length = len(smiles_list_x)
 
    smiles_level_list = []
    if len(branch_smiles_list_x) < 1:
        for i in range(0, smiles_list_length):
            smiles_level_list.append(1)
    else:
        for i in range(0, smiles_list_length):
            if smiles_list_x[i] in branch_smiles_list_x:
                smiles_level_list.append(2)
                
            elif  check_weight(smiles_list_x[i],branch_smiles_list_x):
                smiles_level_list.append(2)
                
            else:
                smiles_level_list.append(1)    


    return smiles_level_list        

if __name__ == "__main__":
    test_query = "{[][<]NCCC{[>][<][Si](C)(C)O[>][<]}[Si](C)(C)CCCN[<],[>]C(=O)NC(CC1)CCC1CC(CC2)CCC2NC(=O)[>][]}"
    branch_list = find_branch_smiles_list(test_query)
    print(branch_list)
    smiles_level_list = get_smiles_level_list(test_query)
    print(smiles_level_list)
