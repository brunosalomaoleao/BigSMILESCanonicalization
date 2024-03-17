"""
© Copyright 2022
JIALE SHI
"""

import numpy as np

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

# conda install -c conda-forge pyomo
# conda install -c conda-forge coincbc
from pyomo.environ import *

from polymersearch import search_tools


def Similarity_Score_EMD(query_smiles_list = None, 
                         query_smiles_level_list = None, 
                         target_smiles_list = None, 
                         target_smiles_level_list = None,
                         level_weight = False,
                         level_ratio = 3,
                         embedding_function = 'RDKFingerprint', # MorganFingerprint, MACCSkeys
                         similarity_score_function = 'Tanimoto', # Dice, Cosine
                         restrain_emd = False):
  
    #obtain the length of query smiles list and target smiles list
    if query_smiles_list != None:
        query_smiles_list_length = len(query_smiles_list)
    else:
        print ("Missing query smiles list")
        return    

    if target_smiles_list != None:
        target_smiles_list_length = len(target_smiles_list)
    else:
        print ("Missing target smiles list")
        return

    if embedding_function == 'RDKFingerprint':    
        query_mol_list = [Chem.MolFromSmiles(x) for x in query_smiles_list]
        query_fingerprint_list = [Chem.RDKFingerprint(x) for x in query_mol_list]
        target_mol_list = [Chem.MolFromSmiles(x) for x in target_smiles_list]
        target_fingerprint_list = [Chem.RDKFingerprint(x) for x in target_mol_list]
	
    elif embedding_function == 'MorganFingerprint':    
        query_mol_list = [Chem.MolFromSmiles(x) for x in query_smiles_list]
        query_fingerprint_list = [AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=1024) for x in query_mol_list]
        target_mol_list = [Chem.MolFromSmiles(x) for x in target_smiles_list]
        target_fingerprint_list = [AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=1024) for x in target_mol_list]

    elif embedding_function == 'MACCSkeys':    
        query_mol_list = [Chem.MolFromSmiles(x) for x in query_smiles_list]
        query_fingerprint_list = [MACCSkeys.GenMACCSKeys(x) for x in query_mol_list]
        target_mol_list = [Chem.MolFromSmiles(x) for x in target_smiles_list]
        target_fingerprint_list = [MACCSkeys.GenMACCSKeys(x) for x in target_mol_list]


    else:
        print(embedding_function + " is not included in the current vision, please choose an available embedding function.");
        return  


    Demand = {}
    Supply = {}
    T = {}

    if query_smiles_level_list == None or len(set(query_smiles_level_list)) ==1 or level_weight == False:
        for i in range(0, query_smiles_list_length):
            Demand["P" + str(i+1)] = 1/query_smiles_list_length
    else:
        print("Query smiles list has different levels")
        #query_weight_sum = sum(query_smiles_level_list)
        #level 1, weight = 1; level 2, weight = 3^1;  level n, weight = 3^n-1
        query_weight_sum = 0.0
        for i in range(0, query_smiles_list_length):
            query_weight_sum = query_weight_sum + pow(level_ratio,query_smiles_level_list[i]-1)

        for i in range(0, query_smiles_list_length):
            Demand["P" + str(i+1)] = pow(level_ratio,query_smiles_level_list[i]-1)/query_weight_sum
            #print("P" + str(i+1), Demand["P" + str(i+1)])


    if target_smiles_level_list == None or len(set(target_smiles_level_list)) ==1 or level_weight == False:       
        for j in range(0,target_smiles_list_length):
            Supply["Q" + str(j+1)] = 1/target_smiles_list_length
    else:
        print("Target smiles list has different levels")
        
        target_weight_sum = 0.0
        for j in range(0, target_smiles_list_length):
            target_weight_sum = target_weight_sum + pow(level_ratio,target_smiles_level_list[j]-1)

        for j in range(0, target_smiles_list_length):
            Supply["Q" + str(j+1)] = pow(level_ratio,target_smiles_level_list[j]-1)/target_weight_sum
            #print("Q" + str(j+1), Supply["Q" + str(j+1)])


    # embedding function and similarity 
    if similarity_score_function == 'Tanimoto':
        for i in range(0,query_smiles_list_length):
            for j in range(0,target_smiles_list_length):
                # calculate the fingerprint similarityscore between query[i],target[j] and input the distance = 1- similarityscore
                T[("P" + str(i+1), "Q" + str(j+1))] = 1 - DataStructs.FingerprintSimilarity(query_fingerprint_list[i],target_fingerprint_list[j])
                #print("P" + str(i+1), "->Q" + str(j+1), T[("P" + str(i+1), "Q" + str(j+1))] )
		
    elif similarity_score_function == 'Dice':
        for i in range(0,query_smiles_list_length):
            for j in range(0,target_smiles_list_length):
                # calculate the fingerprint similarityscore between query[i],target[j] and input the distance = 1- similarityscore
                T[("P" + str(i+1), "Q" + str(j+1))] = 1 - DataStructs.FingerprintSimilarity(query_fingerprint_list[i],target_fingerprint_list[j], metric=DataStructs.DiceSimilarity)
                #print("P" + str(i+1), "->Q" + str(j+1), T[("P" + str(i+1), "Q" + str(j+1))] )

    elif similarity_score_function == 'Cosine':
        for i in range(0,query_smiles_list_length):
            for j in range(0,target_smiles_list_length):
                # calculate the fingerprint similarityscore between query[i],target[j] and input the distance = 1- similarityscore
                T[("P" + str(i+1), "Q" + str(j+1))] = 1 - DataStructs.FingerprintSimilarity(query_fingerprint_list[i],target_fingerprint_list[j], metric=DataStructs.CosineSimilarity)
                #print("P" + str(i+1), "->Q" + str(j+1), T[("P" + str(i+1), "Q" + str(j+1))] )
       
    else:
        print(similarity_score_function + " is not included in the current vision, please choose an available similarity function.");
        return

    
    ## the pyomo optimization code is adapted from Kantor's github notebook
    ## https://github.com/jckantor/ND-Pyomo-Cookbook
    ## https://jckantor.github.io/ND-Pyomo-Cookbook/03.01-Transportation-Networks.html
    #print(len(Demand), len(Supply), len(T))
    # Step 0: Create an instance of the model
    model = ConcreteModel()
    model.dual = Suffix(direction=Suffix.IMPORT)

    # Step 1: Define index sets
    CUS = list(Demand.keys())
    SRC = list(Supply.keys())

    # Step 2: Define the decision 
    model.x = Var(CUS, SRC, domain = NonNegativeReals)

    # Step 3: Define Objective
    model.Cost = Objective(
    expr = sum([T[c,s]*model.x[c,s] for c in CUS for s in SRC]),
    sense = minimize)

    # Step 4: Constraints
    model.src = ConstraintList()
    for s in SRC:
        model.src.add(sum([model.x[c,s] for c in CUS]) == Supply[s])
        
    model.dmd = ConstraintList()
    for c in CUS:
        model.dmd.add(sum([model.x[c,s] for s in SRC]) == Demand[c])

    # add restrain to the EMD
    if restrain_emd == True: 
        model.restrain = ConstraintList()
        for i in range(0,query_smiles_list_length):
            model.restrain.add(model.x[CUS[i],SRC[i]] == Supply["Q" + str(i+1)])
    
    results = SolverFactory('cbc').solve(model)


    if 'ok' == str(results.Solver.status):
        #print("EMD(P,Q) = ",model.Cost())
        #print ("\n")
        #print("S(P,Q) = ", 1- model.Cost())
        SimilarityScore = 1- model.Cost()
        return SimilarityScore
        
    else:
        print("No Valid Solution Found")
        return False



# Graph Edit Distance
def Similarity_Score_Graph_Edit_Distance(Graph1 = None, 
                         Graph2 = None, 
                         alpha = 1):
    if Graph1 == None:
        print("Missing Graph1")
        return
    if Graph2 == None:
        print("Missing Graph2")
        return
    
    # Since Graph1 is the subgraph of Graph2, the calculation of graph edit distance can be simpilified.
    Graph1_number = Graph1.number_of_nodes() + Graph1.number_of_edges()
    Graph2_number = Graph2.number_of_nodes() + Graph2.number_of_edges()

    graph_edit_distance = abs(Graph2_number - Graph1_number)

    # utilize the exponential decay function to turn the graph edit distance to similarity score
    similarity_score = np.exp(-graph_edit_distance/(alpha * Graph1_number))

    return similarity_score 


def Combined_Similarity_Score(Repeat_Unit_Similarity_Score = None,
                              Repeat_Unit_Weight = 0.5,
                              Graph_Similarity_Score = None,
                              Graph_Weight = 0.5,
                              End_Group_Similarity_Score = None,
                              End_Group_Weight = 0.0,
                              Mean_Function = 'arithmetic'):
  
    # Verify whether the weight sum is normalized.
    if  abs(Repeat_Unit_Weight + Graph_Weight +  End_Group_Weight -1 ) >=0.000000000001:
        print("Weight Sum is not normalized.")
        return False

    # Not consider the end group
    if End_Group_Similarity_Score == None:

        if Mean_Function == 'arithmetic':
            combined_similarity_score = (Repeat_Unit_Weight * Repeat_Unit_Similarity_Score + Graph_Weight * Graph_Similarity_Score)

        elif Mean_Function == 'geometric':
            combined_similarity_score = pow(Repeat_Unit_Similarity_Score,Repeat_Unit_Weight)*pow(Graph_Similarity_Score,Graph_Weight)

        else:
            print("Your input mean function ", Mean_Function, " is not implemented, please choose those implemented mean function, like arithmetic, geometric")
    
    # consider the end group
    else:
      
        if Mean_Function == 'arithmetic':
            combined_similarity_score = Repeat_Unit_Weight * Repeat_Unit_Similarity_Score + Graph_Weight * Graph_Similarity_Score + End_Group_Weight * End_Group_Similarity_Score

        elif Mean_Function == 'geometric':
            combined_similarity_score = pow(Repeat_Unit_Similarity_Score,Repeat_Unit_Weight)*pow(Graph_Similarity_Score,Graph_Weight)*pow(End_Group_Similarity_Score, End_Group_Weight)

        else:
            print("Your input mean function ", Mean_Function, " is not implemented, please choose those implemented mean function, like arithmetic, geometric")
            
    return combined_similarity_score


def Similarity_Score_Two_Polymer(query = None,
                                 target = None,
                                 level_weight = True,
                                 level_ratio = 3,
                                 embedding_function = 'RDKFingerprint', #Embedding function
                                 similarity_score_function = 'Tanimoto', # Similarity function for two vectors
                                 restrain_emd = False, # Whether to restrain the emd
                                 alpha=1, #reduced parameter for the exponential decay function
                                 Repeat_Unit_Weight=0.5,
                                 Graph_Weight=0.5,
                                 End_Group_Weight = 0.0,
                                 Mean_Function = 'geometric',
                                 details_print = False):

    if query == None or target == None:
        print ("Either query polymer or target polymer is missing! Please check your input.")
        return False
    S_repeat_unit = Similarity_Score_EMD(query_smiles_list = query.repeat_unit_smiles_list, 
                     query_smiles_level_list = query.repeat_unit_smiles_level_list, 
                     target_smiles_list = target.repeat_unit_smiles_list, 
                     target_smiles_level_list = target.repeat_unit_smiles_level_list,
                     level_weight = level_weight,
                     level_ratio = level_ratio,		 		 
                     embedding_function = embedding_function,
                     similarity_score_function = similarity_score_function,
                     restrain_emd = restrain_emd)
    
    S_graph = Similarity_Score_Graph_Edit_Distance(Graph1=query.graph_representation, 
                                                   Graph2=target.graph_representation, 
                                                   alpha=alpha)
    
    if End_Group_Weight == 0.0: 
        S_combined = Combined_Similarity_Score(Repeat_Unit_Similarity_Score=S_repeat_unit ,
                                       Repeat_Unit_Weight=Repeat_Unit_Weight,
                                       Graph_Similarity_Score=S_graph,
                                       Graph_Weight=Graph_Weight,
                                       End_Group_Similarity_Score = None,
                                       End_Group_Weight = End_Group_Weight,
                                       Mean_Function = Mean_Function)
        if details_print == True:
            print("Details of the Similarity Score:\n")
            print("Similarity score on Repeating Unit = ", S_repeat_unit, ", Weight for Repeating Unit = ", Repeat_Unit_Weight)
            print("Similarity score on Graph = ", S_graph, ", Weight for Graph = ", Graph_Weight)
            print("Similarity score on End Group = ", "None", ", Weight for End Group = ", End_Group_Weight)
            print("Similarity score Combined in " + Mean_Function + " mean = ", S_combined)
            print("\n")

        return S_combined 

    else: 
        S_end_group = Similarity_Score_EMD(query_smiles_list = query.end_group_smiles_list, 
                     query_smiles_level_list = query.end_group_smiles_level_list, 
                     target_smiles_list = target.end_group_smiles_list, 
                     target_smiles_level_list = target.end_group_smiles_level_list,
                     level_weight = level_weight,
                     level_ratio = level_ratio,	
                     embedding_function = embedding_function,
                     similarity_score_function = similarity_score_function,
                     restrain_emd = restrain_emd)
            
        S_combined = Combined_Similarity_Score(Repeat_Unit_Similarity_Score=S_repeat_unit ,
                                       Repeat_Unit_Weight=Repeat_Unit_Weight,
                                       Graph_Similarity_Score=S_graph,
                                       Graph_Weight=Graph_Weight,
                                       End_Group_Similarity_Score = S_end_group,
                                       End_Group_Weight = End_Group_Weight,
                                       Mean_Function = Mean_Function)
        
        if details_print == True:
            print("Details of the Similarity Score:\n")
            print("Similarity score on Repeating Unit = ", S_repeat_unit, ", Weight for Repeating Unit = ", Repeat_Unit_Weight)
            print("Similarity score on Graph = ", S_graph, ", Weight for Graph = ", Graph_Weight)
            print("Similarity score on End Group = ", S_end_group, ", Weight for End Group = ", End_Group_Weight )
            print("Similarity score Combined in " + Mean_Function + " mean = ", S_combined)
            print("\n")

        return S_combined     


class Polymer:
  def __init__(self, 
               repeat_unit_smiles_list=None, 
               repeat_unit_smiles_level_list=None, 
               end_group_smiles_list=None, 
               end_group_smiles_level_list=None,
               graph_representation=None):
    
      self.repeat_unit_smiles_list = repeat_unit_smiles_list
      self.repeat_unit_smiles_level_list = repeat_unit_smiles_level_list
      self.end_group_smiles_list = end_group_smiles_list
      self.end_group_smiles_level_list = end_group_smiles_level_list
      self.graph_representation = graph_representation


if __name__ == "__main__":

	query = "{[][<]CCO[>][]}"
        
	Polymer0010 = Polymer(repeat_unit_smiles_list = search_tools.get_repeats_as_rings(query), 
                      graph_representation = search_tools.generate_NetworkX_graphs(query)["topology"])
	
	target = "{[][<]CCCO[>][]}"
	Polymer0011 = Polymer(repeat_unit_smiles_list= search_tools.get_repeats_as_rings(target),
                      graph_representation = search_tools.generate_NetworkX_graphs(target)["topology"])


	score = Similarity_Score_Two_Polymer(query = Polymer0010,
                             target = Polymer0011,
                             Repeat_Unit_Weight=0.5,
                             Graph_Weight=0.5)
	print(score)
	
	#one more example , both the query and target have level 2 smiles
    query = "{[][<]NCCC{[>][<][Si](C)(C)O[>][<]}[Si](C)(C)CCCN[<],[>]C(=O)NC(CC1)CCC1CC(CC2)CCC2NC(=O)[>][]}"
    print("Query Polymer:")
    print("BigSMILES: ",query)
    print("Repeat Unit List:",search_tools.get_repeats_as_rings(query))
    query_smiles_level_list = smiles_level_list.get_smiles_level_list(query)
    print("Repeat Unit Level List:",query_smiles_level_list)

    Polymer_query = Polymer(repeat_unit_smiles_list = search_tools.get_repeats_as_rings(query),
                    repeat_unit_smiles_level_list = smiles_level_list.get_smiles_level_list(query),
                    graph_representation = search_tools.generate_NetworkX_graphs(query)["topology"])

    # target = '{[][<]C(=O)Nc(cc1)ccc1Cc(cc2)ccc2NC(=O)[<],[>]O{[<][>]CC(C)O[<][>]}[>],[>]NCCN[>][]}'
    # print("Target Polymer:")
    # print("BigSMILES: ",query)
    # print("Repeat Unit List:",search_tools.get_repeats_as_rings(target))
    # target_smiles_level_list = smiles_level_list.get_smiles_level_list(target)
    # print("Repeat Unit Level List:",target_smiles_level_list)
    # Polymer_target = Polymer(repeat_unit_smiles_list= search_tools.get_repeats_as_rings(target),
    #                 repeat_unit_smiles_level_list = smiles_level_list.get_smiles_level_list(target),
    #                 graph_representation = search_tools.generate_NetworkX_graphs(target)["topology"])


    # score = Similarity_Score_Two_Polymer(query = Polymer_query,
    #                         target = Polymer_target,
    #                         level_weight = True,
    #                         level_ratio = 3,
    #                         embedding_function = 'RDKFingerprint', #Embedding function
    #                         Repeat_Unit_Weight=0.5,
    #                         Graph_Weight=0.5)
    # print(score)


