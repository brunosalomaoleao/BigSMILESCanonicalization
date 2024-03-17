"""
Author: Bruno Salomão Leão
Description: This file contains all functions and objects related to tree automaton and its operations
"""


# External imports -----------------------------------------------------------------------------------------------------
import itertools
import copy
import networkx as nx
import pydot
import os
from itertools import product
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt


# Global ---------------------------------------------------------------------------------------------------------------
FORMS_OF_EMPTY_ALPHABET = ["[*:1][Es][*:2]", "[*:1][Es]", "[Es][*:2]", "[Es]",
                           "[*:2][Es][*:1]", "[Es][*:1]", "[*:2][Es]",
                           "[Es]([*:2])[*:1]", "[Es]([*:1])[*:2]",
                           "[*:2]", "[*:1]", "[*:2][*:1]", "[*:1][*:2]"]

FORMS_OF_STARTING_EMPTY_ALPHABET = ["[Es][*:1]", "[*:1][Es]", "[*:1]"]

EPSILON_CHARACTER = "<&#949;>"


# Functions ------------------------------------------------------------------------------------------------------------
def generate_all_combinations(items: list):
    """
    Generate all combinations with the number of elements ranging from 1 to the length of the list
    Args:
        items: sample of items to be combined
    Returns: all combinations
    """
    list_combinations = list()

    items = set(items)
    for n in range(len(items) + 1):
        list_combinations += list(itertools.combinations(items, n))

    return [x for x in list_combinations if x]


def check_if_epsilon(transition, include_start=True, canonicalize=True):
    """
    Checks if a transition is an empty transition
    Args:
        transition: transition
        include_start: if True, checks the empty starting transitions. If False, checks all transitions but starting
        canonicalize: if True, it canonicalizes the SMILES when comparing

    Returns: True if it is an empty transition

    """

    if include_start:
        forms_of_Es = [Chem.MolToSmiles(Chem.MolFromSmiles("[Es]")),
                       Chem.MolToSmiles(Chem.MolFromSmiles("[Es][*:1]")),
                       Chem.MolToSmiles(Chem.MolFromSmiles("[*:2][Es][*:1]")),
                       Chem.MolToSmiles(Chem.MolFromSmiles("[Es][*:2]")),
                       ""]
    else:
        forms_of_Es = [Chem.MolToSmiles(Chem.MolFromSmiles("[Es]")),
                       Chem.MolToSmiles(Chem.MolFromSmiles("[*:2][Es][*:1]")),
                       Chem.MolToSmiles(Chem.MolFromSmiles("[Es][*:2]")),
                       ""]

    # If it is an Es or "", it is an empty transition
    if canonicalize:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(transition.smiles))
    else:
        smiles = transition.smiles
    if smiles in forms_of_Es:
        return True
    else:
        return False


# Classes --------------------------------------------------------------------------------------------------------------
class Transitions:
    """
    This class represents the transitions of a Bottom-Up Tree Automaton
    """

    def __init__(self, input: list, alphabet: str, smiles: str, output: int):
        """
        Create a transition
        :param input: states that are the input for the transition
        :param alphabet: symbol that represents the transition
        :param smiles: atomistic meaning of the alphabet
        :param output: state that the transition leads to
        """
        # self.input = sorted(input)    # Sorts the list because it makes it easier to compare later
        self.input = input
        self.alphabet = alphabet
        self.smiles = smiles
        self.output = output


class TreeAutomata:
    """
    This class represents the Bottom-Up Tree Automata and its operations.
    This type of automata contains transitions that may have 0, 1 or many input states but only one output.
    """

    def __init__(self, transitions: list = [], states: list = [], end_states: list = []):
        """
        Represents a bottom-up Tree Automata
        :param transitions: list of Transitions
        :param states: list of states
        :param end_states: list of end states
        """
        self.transitions = transitions
        self.end_states = end_states
        if states:
            self.states = states
        else:
            self.states = self.get_states()
        self.transition_map = self.generate_transition_map()

    def normalize_state_names(self):
        """
        Converts all the state names into indices and sorts when they are in the same list.
        Example: "$" and "S2" to 0 and 1
        :return: None
        """
        # List all states
        states = set({})
        for transition in self.transitions:
            # Add inputs
            for state in transition.input:
                states.add(state)
            # Add outputs
            states.add(transition.output)

        # Create a dictionary to map names onto indices
        statename_to_index = {name: index for index, name in enumerate(states)}

        # Replace state names by indices
        for transition in self.transitions:
            # Replace inputs
            _inputs = sorted([statename_to_index[x] for x in transition.input])
            transition.input = _inputs
            # Replace outputs
            transition.output = statename_to_index[transition.output]

        # Store in the states variable
        self.states = list(statename_to_index.values())

        # Normalize end states
        self.end_states = [statename_to_index[x] for x in self.end_states]

    def remove_duplicates(self):
        """
        Remove transitions that are the exact same
        :return: None
        """
        _transitions = []
        for i, trans1 in enumerate(self.transitions):
            duplicated = False
            # Loop over the remaining of the list to check if there is another equal transition
            for j, trans2 in enumerate(self.transitions[i + 1:]):
                # Checks if inputs are the same
                check_inputs = trans1.input == trans2.input
                # Checks outputs
                check_outputs = trans1.output == trans2.output
                # Checks alphabet
                check_alphabet = trans1.alphabet == trans2.alphabet
                # If inputs, outputs and alphabets are the same, it is duplicated
                if check_inputs and check_outputs and check_alphabet:
                    duplicated = True
            # If it is not duplicated, add to the transitions list
            if not duplicated:
                _transitions.append(trans1)
        # Update transitions list
        self.transitions = _transitions

    def add_transitions_with_merger(self, old_states, new_state, transitions):
        """
        Add transitions that read the new state that has just been created. This state is a merger of others
        Args:
            old_states: list of states that were merged
            new_state: new state
            transitions: list of transitions

        Returns: None

        """
        new_transitions = []
        for tr in transitions:
            # Check if the merged states are inputs to this transition
            old_in_input = [x for x in tr.input if x in old_states]
            # Generate all combinations with old_in_input
            combinations = generate_all_combinations(old_in_input)
            for combination in combinations:
                # Generate a list of all possible combinations with the new state as input
                input = copy.deepcopy(tr.input)
                possible_inputs = []
                for replaced_state in combination:
                    for i in range(len(input)):
                        if input[i] == replaced_state:
                            input[i] = new_state
                            possible_inputs.append(sorted(input))
                # For every possibility, create a transition
                for input in possible_inputs:
                    # Check if there is already a transition that has the same input, alphabet and output
                    new_transition = True
                    for tr2 in [x for x in transitions if x != tr]:
                        if_same_input = tr2.input == input
                        if_same_alphabet = tr2.alphabet == tr.alphabet
                        if_same_output = tr2.output == tr.output
                        if if_same_input and if_same_alphabet and if_same_output:
                            new_transition = False
                            break
                    # Add new transition if there is no other transition
                    if new_transition:
                        new_transitions.append(Transitions(input, tr.alphabet, tr.smiles, tr.output))

        return transitions + new_transitions

    # def determinization(self):
    #     """
    #     This function converts any non-deterministic bottom-up tree automata into deterministic tree automata
    #     Returns: None
    #     """
    #     # Dictionary of merged states
    #     merged_states = {}
    #     # List that keeps track of all states that were once ending states
    #     _end_states = copy.deepcopy(self.end_states)
    #     # Set initial variables
    #     max_state = max(self.states) + 1
    #     number_of_mergers = 1
    #     # Deterministic transitions
    #     det_transitions = copy.deepcopy(self.transitions)
    #
    #     # Variable that checks whether self.transitions has changed
    #     difference = True
    #     while number_of_mergers != 0:
    #         number_of_mergers = 0
    #         for i in range(len(self.transitions)-1):
    #             for j in range(i+1, len(self.transitions)):
    #                 # If the transitions have the same input and same alphabet, the output states must be merged
    #                 if_same_input = self.transitions[i].input == self.transitions[j].input
    #                 if_same_alphabet = self.transitions[i].alphabet == self.transitions[j].alphabet
    #                 if_different_output = self.transitions[i].output != self.transitions[j].output
    #                 if_determinize = if_same_input and if_same_alphabet and if_different_output
    #                 if if_determinize:
    #                     # Create new state
    #                     max_state += 1
    #                     # Merge old states
    #                     old_states = sorted([self.transitions[i].output, self.transitions[j].output])
    #                     # If the states have already been merged, use the state previously created
    #                     create_new_state = True
    #                     new_state = max_state
    #                     for new, merger in merged_states.items():
    #                         if old_states == merger:
    #                             new_state = new
    #                             create_new_state = False
    #                             break
    #                     # If at least one of the 2 states is an ending state, the resulting state will be an ending state
    #                     if (self.transitions[i].output in _end_states) or (self.transitions[j].output in _end_states):
    #                         _end_states.append(new_state)
    #                     # Add transition to new_state
    #                     det_transitions.append(Transitions(self.transitions[i].input, self.transitions[i].alphabet,
    #                                         self.transitions[i].smiles, new_state))
    #                     # If it is a new state, check what happens when any alphabet takes it as input
    #                     if create_new_state:
    #                         # Update dict of merged states
    #                         merged_states[new_state] = old_states
    #                         # Add transitions that read the new_state
    #                         det_transitions = self.add_transitions_with_merger(old_states, new_state, det_transitions)
    #                     # Remove transitions from list
    #                     _list = []
    #                     for t in det_transitions:
    #                         for t2 in (self.transitions[i], self.transitions[j]):
    #                             if_same_input = t.input == t2.input
    #                             if_same_alphabet = t.alphabet == t2.alphabet
    #                             if_same_output = t.output == t2.output
    #                             same_transition = if_same_input and if_same_alphabet and if_same_output
    #                             if same_transition:
    #                                 break
    #                         if not same_transition:
    #                             _list.append(t)
    #                     det_transitions = copy.deepcopy(_list)
    #
    #                     # Count merger
    #                     number_of_mergers += 1
    #                     # Break
    #                     break
    #             # If it merged states once, start over
    #             if number_of_mergers != 0:
    #                 break
    #
    #         # Update tree transitions with deterministic transitions
    #         self.transitions = copy.deepcopy(det_transitions)
    #         # Remove duplicates
    #         self.remove_duplicates()
    #
    #     # Update states from list of transitions
    #     self.states = self.get_states()
    #     # Define end states
    #     self.end_states = [q for q in self.states if q in _end_states]
    #     # Remove duplicates
    #     self.remove_duplicates()
    #     # Reduce (remove inaccessible states)
    #     self.reduction()
    #     # Normalize state names
    #     self.normalize_state_names()

    def choose_deterministic_input(self, transitions, det_states, i, preserve):
        # input = []
        # if preserve:
        #     return transitions[i][0]
        # else:
        #     for x in transitions[i][0]:
        #         if type(x) == list:
        #             _in = x
        #         else:
        #             _in = [x]
        #         new_in = []
        #         for y in det_states:
        #             if any([_x in y for _x in _in]):
        #                 # _in = y
        #                 # break
        #                 new_in += y
        #         new_in = set(new_in)
        #         new_in = sorted(list(new_in))
        #         input.append(new_in)

        input = []
        if preserve:
            return transitions[i][0]
        else:
            for x in transitions[i][0]:
                if type(x) == list:
                    _in = x
                else:
                    _in = [x]
                possible_input = []
                for y in det_states:
                    if any([_x in y for _x in _in]):
                        possible_input.append(y)
                input.append(possible_input)

        all_possible_inputs = list(product(*input))

        return all_possible_inputs

    def collapse_states(self, transitions, det_transitions, det_states, det_state_list, new_states,
                        remove_duplicates=False):
        """
        This function executes one step of determinization. It loops over all nondeterministic transitions and, if the
        deterministic automaton already contains its inputs, it adds both the transition and the output states to
        the deterministic automaton.

        Also, it looks for nondeterministic transitions, i.e., transitions with same input and alphabet but different
        output. Then, it merges the outputs to create a new state and adds transitions from the new states based on
        the states that were grouped.

        Args:
            transitions: list of nondeterministic transitions
            det_transitions: list of deterministic transitions
            det_states: list of merged states
            det_state_list: list of initial states
            new_states: list of merged states
            remove_duplicates: if True, removes duplicates

        Returns: a flag that says if changes have been made to the deterministic state machine,
        list of deterministic transitions, list of merged states, list of initial states and list of merged states
        """
        loop = False

        this_loop = True
        while this_loop:
            # Set this variable to False so loop stops in this iteration
            this_loop = False

            # Loop over each transition from the nondeterministic automaton
            for i in range(len(transitions)):

                # Get transition i input
                input_i = transitions[i][0]
                # Get transition i alphabet
                alphabet_i = transitions[i][1]
                # Get transition i output
                if type(transitions[i][2]) == list:
                    output_i = copy.deepcopy(transitions[i][2])
                else:
                    output_i = [copy.deepcopy(transitions[i][2])]

                # Check if inputs are in det_states
                input_in_det = True
                for _in in input_i:
                    if type(_in) == list:
                        states_in_det_state_list = all([x in det_state_list for x in _in])
                    else:
                        states_in_det_state_list = _in in det_state_list
                    if not states_in_det_state_list:
                        input_in_det = False

                # Only add this transition if the inputs of transition i are in the deterministic automaton
                if input_in_det:
                    # Get states from deterministic automaton that contain the states from the input of transition i. Get the combinations
                    possible_inputs = self.choose_deterministic_input(transitions, det_states, i,
                                                                      remove_duplicates)
                    # For each possible input, see what the output is. It will be the result of when the alphabets read
                    # each element of the possible input
                    for input in possible_inputs:

                        output = copy.deepcopy(output_i)
                        for j in range(len(transitions)):

                            # Get transition i input
                            input_j = transitions[j][0]
                            # Get transition i alphabet
                            alphabet_j = transitions[j][1]
                            # Get transition i output
                            if type(transitions[j][2]) == list:
                                output_j = copy.deepcopy(transitions[j][2])
                            else:
                                output_j = [copy.deepcopy(transitions[j][2])]

                            # Check if transition i and j have the same alphabet
                            if_same_alphabet = alphabet_i == alphabet_j
                            # Check if input and input_j have the same length and all elements of input_j are in input
                            if_same_len = len(input) == len(input_j)
                            if_same_input = False
                            if if_same_len:
                                if_same_input = True
                                for l in range(len(input)):
                                    element_input = input[l]
                                    element_input_j = input_j[l]
                                    # If at least one element from input_j is not within input, they do not have the same input
                                    if set(element_input).isdisjoint(element_input_j):
                                        if_same_input = False
                                        break

                            # If transitions i and j have the same alphabet and input but are different transitions, merge
                            if if_same_input and if_same_len and if_same_alphabet and i != j:
                                # Merge outputs and remove duplicates
                                output += copy.deepcopy(output_j)
                                output = set(output)
                                output = sorted(list(output))

                        # Check if the transition has already been added to the deterministic state machine
                        add = True
                        for det_trans in det_transitions:
                            if_same_input = det_trans[0] == input
                            if_same_alphabet = det_trans[1] == alphabet_i
                            if_same_output = det_trans[2] == output
                            if if_same_input and if_same_alphabet and if_same_output:
                                add = False
                                break
                        if add:
                            # Signal that the outer loop needs to continue
                            loop = True
                            this_loop = True
                            # Add new transition
                            new_transition = [input, alphabet_i, output]
                            det_transitions.append(new_transition)
                            # Add new state
                            if output not in det_states:
                                det_states.append(output)
                            if output not in new_states:
                                new_states.append(output)
                            # Add merged states to list
                            for s in output:
                                det_state_list.append(s)

        return loop, det_transitions, det_states, det_state_list, new_states

    def determinize(self, transitions, remove_duplicates=False):
        """
        This function executes at least one step of determinization. It executes the function collapse_states() until
        no changes are made to the state machine
        Args:
            transitions: list of transitions. Each element must have 3 elements: list of input states, alphabet and
            output states
            remove_duplicates: if True, removes duplicates

        Returns: list of deterministic transitions, list of merged states, list of initial states and list
        of merged states

        """

        # Initialize
        det_transitions = []
        det_states = []
        det_state_list = []
        new_states = []

        loop = True
        while loop:
            loop, det_transitions, det_states, det_state_list, new_states = self.collapse_states(transitions,
                                                                                                 det_transitions,
                                                                                                 det_states,
                                                                                                 det_state_list,
                                                                                                 new_states,
                                                                                                 remove_duplicates)
        return det_transitions, det_states, det_state_list, new_states

    def determinization(self):
        """
        This function converts any non-deterministic bottom-up tree automata into deterministic tree automata
        Returns: None
        """
        # List that keeps track of all states that were once ending states
        _end_states = copy.deepcopy(self.end_states)

        # Initialize
        transitions = []
        for tr in self.transitions:
            transitions.append([[[x] for x in tr.input], tr.alphabet, [tr.output]])

        # Final states
        end_states = [x for x in self.end_states]

        # First determinization
        det_transitions, det_states, det_state_list, new_states = self.determinize(transitions=transitions)

        # Deterministic end states
        det_end_states = [x for x in det_states if not set(x).isdisjoint(
            end_states)]  # Select all states that contain at least one final state of the original automata

        # Replace state groups by indices
        new_det_transitions = []
        new_det_end_states = []

        # Replace transitions
        for tr in det_transitions:
            # Input
            input = []
            for s in tr[0]:
                try:
                    index = det_states.index(s)
                    input.append(index)
                except:
                    pass
            # Output
            output = []
            try:
                index = det_states.index(tr[2])
                output = index
            except:
                pass
            # Add new transition
            new_det_transitions.append([input, tr[1], output])

        # Replace end states
        for s in det_end_states:
            try:
                index = det_states.index(s)
            except:
                continue
            new_det_end_states.append(index)

        # Update transitions
        _transitions = []
        for tr in new_det_transitions:
            # Find the smiles
            smiles = ""
            for tr2 in self.transitions:
                if tr2.alphabet == tr[1]:
                    smiles = tr2.smiles
                    break
            _transitions.append(Transitions(input=tr[0], alphabet=tr[1], output=tr[2], smiles=smiles))
        self.transitions = _transitions
        # Update end states
        self.end_states = new_det_end_states
        # Update states
        self.states = [i for i, el in enumerate(det_states)]

    def get_output(self, alphabet, input):
        """
        Gets the output state of a deterministic tree automaton, given the alphabet and input. Returns the output
        state and, if there is not one, None.
        Args:
            alphabet: alphabet
            input: list of inputs

        Returns: the output state or None

        """
        # Loop over all transitions
        for transition in self.transitions:
            # If it has the same alphabet and same input, return the output
            if_same_alphabet = alphabet == transition.alphabet
            if_same_input = sorted(transition.input) == sorted(input)
            if if_same_alphabet and if_same_input:
                return transition.output

        # If no transition is found, return None
        return None

    def check_equivalence(self, stateA, stateB, equivalence_map, not_determined=set([])):
        """
        Checks if stateA and stateB are equivalent. They are equivalent if, upon replacing stateA by stateB
        (and vice-versa) in all transition rules, the output is equivalent to the original output.
        Implemented according to the algorithm from https://doi.org/10.1007/978-3-540-76336-9_13
        Args:
            stateA: one state
            stateB: another state
            equivalence_map: dictionary that maps each pair of state into True, False or None
            not_determined: set of pairs whose equivalence could not be determined yet

        Returns: True, if the states are equivalent in all transitions
        """

        # If the equivalence_map for both states is not None, i.e. it is either True or False, return it
        if equivalence_map[stateA][stateB] is not None:
            return equivalence_map[stateA][stateB]

        # # If both states are the same, return True
        # if stateA == stateB:
        #     return True
        #
        # # If one of the states is an accepting state and the other is not, they are not equivalent
        # elif {(stateA in self.end_states), (stateB in self.end_states)} == {True, False}:
        #     return False

        # List of alphabets that read stateA
        alphabets_A = sorted(list(set(tr.alphabet for tr in self.transitions if stateA in tr.input)))
        # List of alphabets that read stateB
        alphabets_B = sorted(list(set(tr.alphabet for tr in self.transitions if stateB in tr.input)))
        # If they are different, stateA and stateB are not equivalent
        if alphabets_A != alphabets_B:
            # Populate state map
            equivalence_map[stateA][stateB] = False
            equivalence_map[stateB][stateA] = False
            return False

        # Set equivalence to False
        is_equivalent = False

        # Loop over all transitions that to check if stateA and stateB are interchangeable
        for transition in self.transitions:

            # Get transition alphabet, input and output
            alphabet = transition.alphabet
            input = transition.input
            output = transition.output

            # If stateA is an input, replace it by stateB in the input list
            if stateA in input:
                # Replace stateA by stateB
                replaced_input = sorted([x if x != stateA else stateB for x in input])
                # Output of the replaced input
                replaced_output = self.get_output(alphabet, replaced_input)
                # If both outputs are the same or if they loop back, the states might be equivalent
                if (replaced_output == output) or (output in [stateA, stateB] and replaced_output in [stateA, stateB]):
                    is_equivalent = True
                # If there is no output for the replaced input, they are not equivalent
                elif replaced_output == None:
                    is_equivalent = False
                # If the outputs are different, check their equivalence
                else:
                    # If output and replaced output could not be determined, do not check them (prevents infinite loop)
                    if tuple({output, replaced_output}) in not_determined:
                        continue
                    else:
                        not_determined.add(tuple({output, replaced_output}))
                        not_determined.add(tuple({stateA, stateB}))
                        is_equivalent = self.check_equivalence(output, replaced_output, equivalence_map, not_determined)
                        # Their equivalence could be determined, so they should be removed from the set
                        # If it fails to remove, it is because the states were not in the set anymore
                        try:
                            not_determined.remove(tuple({output, replaced_output}))
                        except:
                            pass
                        try:
                            not_determined.remove(tuple({stateA, stateB}))
                        except:
                            pass

                # If it is not equivalent in at least one transition, it is not equivalent at all
                if not is_equivalent:
                    # Populate state map
                    equivalence_map[stateA][stateB] = is_equivalent
                    equivalence_map[stateB][stateA] = is_equivalent
                    return is_equivalent

            # If stateB is an input, replace it by stateA in the input list
            if stateB in input:
                # Replace stateB by stateA
                replaced_input = sorted([x if x != stateB else stateA for x in input])
                # Output of the replaced input
                replaced_output = self.get_output(alphabet, replaced_input)
                # If both outputs are the same or if they loop back, the states might be equivalent
                if (replaced_output == output) or (output in [stateA, stateB] and replaced_output in [stateA, stateB]):
                    is_equivalent = True
                # If there is no output for the replaced input, they are not equivalent
                elif replaced_output == None:
                    is_equivalent = False
                # If the outputs are different, check their equivalence
                else:
                    # If output and replaced output could not be determined, do not check them (prevents infinite loop)
                    if tuple({output, replaced_output}) in not_determined:
                        continue
                    else:
                        not_determined.add(tuple({output, replaced_output}))
                        not_determined.add(tuple({stateA, stateB}))
                        is_equivalent = self.check_equivalence(output, replaced_output, equivalence_map, not_determined)
                        # Their equivalence could be determined, so they should be removed from the set
                        # If it fails to remove, it is because the states were not in the set anymore
                        try:
                            not_determined.remove(tuple({output, replaced_output}))
                        except:
                            pass
                        try:
                            not_determined.remove(tuple({stateA, stateB}))
                        except:
                            pass

                # If it is not equivalent in at least one transition, it is nt equivalent at all
                if not is_equivalent:
                    # Populate state map
                    equivalence_map[stateA][stateB] = is_equivalent
                    equivalence_map[stateB][stateA] = is_equivalent
                    return is_equivalent

        # Populate state map
        equivalence_map[stateA][stateB] = is_equivalent
        equivalence_map[stateB][stateA] = is_equivalent

        return is_equivalent

    def get_states(self):
        """
        Gets all states from a tree based on its transitions
        Returns: list of states
        """
        # Set as empty
        states = []

        # Get all states from transitions (input and output)
        for tr in self.transitions:
            states.append(tr.output)
            states = states + tr.input

        # Remove duplicates
        states = list(dict.fromkeys(states))

        return sorted(states)

    def reduction(self):
        """
        This function reduces a tree, i. e., removes inaccessible states
        Returns:None
        """
        # Get all states
        states = copy.deepcopy(self.states)
        # Get all transitions
        transitions = copy.deepcopy(self.transitions)
        # Number of accessible states
        n_access_states_before = 0
        n_access_states_after = len(states)

        # Set accessible states list to empty
        access_states = []
        # Find accessible states
        while (n_access_states_before - n_access_states_after) != 0:

            # Find accessible states and add to list
            states_to_check = [x for x in states if x not in access_states]  # Select states that we do not know yet
            for q in states_to_check:
                for tr in transitions:
                    # If the transition points at q and all inputs are in access_states, it is accessible
                    if tr.output == q and all([x in access_states for x in tr.input]):
                        access_states.append(q)
                        break

            # Update count of accessible states
            n_access_states_before = n_access_states_after
            n_access_states_after = len(access_states)

        # Remove transitions that read inaccessible states
        new_transitions = []
        for tr in transitions:
            # If all input states are accessible, add transition. Otherwise, do not add
            add = True
            for q in tr.input:
                if q not in access_states:
                    add = False
                    break
            if add:
                new_transitions.append(tr)
        transitions = new_transitions

        # Update tree
        self.transitions = transitions
        self.states = sorted(access_states)
        self.end_states = [q for q in self.end_states if q in access_states]

    def minimization_step(self):
        """
        This function executes one minimization step over a bottom-up tree automata
        Returns: None
        """
        # Normalize state names
        self.normalize_state_names()


        # Initialize the dictionary that groups equivalent states
        equivalent_states = [self.end_states, [x for x in self.states if not (x in self.end_states)]]

        # This variable stores the equivalence of states to increase the efficiency of the check_equivalence function.
        # It is a nested dictionary where, if the value is True the states are equivalent; if it is False, they are not;
        # if it is None it is not determined
        equivalence_map = {s1: {s2: True if s1 == s2 else None for s2 in self.states} for s1 in self.states}
        for s1 in equivalent_states[0]:
            for s2 in equivalent_states[1]:
                equivalence_map[s1][s2] = False
                equivalence_map[s2][s1] = False


        # Variable that checks whether equivalent_states changed
        difference = True
        while difference:

            # Keep old value of equivalent_states
            old_equivalent_states = copy.deepcopy(equivalent_states)

            # Loop over each group of states
            for group_index in range(len(equivalent_states)):
                # Set variables that help group the states
                refinements = []
                # Compare the states within a group
                for i in range(len(equivalent_states[group_index])):
                    # new_class += 1
                    for j in range(len(equivalent_states[group_index])):
                        # Get state names from indices
                        i_state = equivalent_states[group_index][i]
                        j_state = equivalent_states[group_index][j]
                        if self.check_equivalence(i_state, j_state, equivalence_map):
                            # They are assigned the same class
                            refinements.append([i_state, j_state])
                        else:
                            refinements.append([i_state])
                            refinements.append([j_state])
                            # changes += 1
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
                        equivalent_states[group_index] = refinements[0]
                    if len(refinements) > 1:
                        for group in refinements[1:]:
                            equivalent_states.append(group)
            # Check if equivalent_states changed
            difference = sorted(equivalent_states) != sorted(old_equivalent_states)

        # Merge states within the same group
        new_state = max(self.states)
        for group in equivalent_states:
            new_state += 1
            for tr in self.transitions:
                # Replace states by merger in transitions
                tr.input = [new_state if x in group else x for x in tr.input]
                tr.output = new_state if tr.output in group else tr.output
                self.end_states = list(
                    set([new_state if x in group else x for x in self.end_states]))  # removing duplicates

        # Get states
        self.states = self.get_states()
        # Remove inaccessible
        self.reduction()
        # Normalize state names
        self.normalize_state_names()
        # Remove duplicates
        self.remove_duplicates()

    def minimization(self):
        """
        This function minimizes a deterministic, bottom-up tree automata
        Returns: None
        """

        # Se the numbers of transitions in the previous step
        number_of_transitions_old = len(self.transitions)

        # Minimize until no changes are made
        difference = True
        while difference:
            # Minimize
            self.minimization_step()
            # Check if there are any changes
            number_of_transitions = len(self.transitions)
            difference = number_of_transitions != number_of_transitions_old
            # Update number_of_transitions_old
            number_of_transitions_old = len(self.transitions)

    def generate_minimial_tree(self, plot, tree_name, draw_alphabet_function, output_folder):
        """
        This function removes empty transitions, converts a non-deterministic tree automata into deterministic,
        and minimizes it.

        Args:
            plot: if True, plots the state machine
            tree_name: name of the tree that will appear in the file name
            draw_alphabet_function: function that draws alphabets
            output_folder: output folder

        Returns: None

        """
        # Plot initial state machine
        if plot:
            self.plot(tree_name=f"NFTA_{tree_name}",
                      draw_alphabet_function=draw_alphabet_function,
                      output_folder=output_folder)

        # Remove empty transitions and plot
        self.eliminate_epsilon_transitions()
        if plot:
            self.plot(tree_name=f"NFTA_no_epsilon_{tree_name}", output_folder=output_folder)

        # Determinize and plot
        self.determinization()
        if plot:
            self.plot(tree_name=f"DFTA_{tree_name}", output_folder=output_folder)

        # Minimize and plot
        self.minimization()
        if plot:
            self.plot(tree_name=f"MFTA_{tree_name}", output_folder=output_folder)

    def plot(self, tree_name="tree", output_folder="Output", draw_alphabet_function=None, *args, **kwargs):
        """
        This function plots the tree automaton
        Args:
            tree_name: file name of the output graph
            draw_alphabet_function: function that outputs the alphabets in a specific file format
        Returns: None

        """

        # Create graph
        graph = pydot.Dot(graph_type='graph', rankdir="LR")

        # Index that represent the branch points that will be created. This will also be used for transitions without input
        new_index = max(self.states) + 1

        # Translate each transition
        for tr in self.transitions:

            # If alphabet is empty, replace it by epsilon
            if tr.smiles in FORMS_OF_EMPTY_ALPHABET:
                alphabet = EPSILON_CHARACTER
            else:
                alphabet = tr.alphabet

            # If there is no input
            if tr.input == []:
                # Create and add empty node
                empty_node = pydot.Node(new_index, style="invis")
                graph.add_node(empty_node)
                # Create and add output node
                node = pydot.Node(tr.output, label=tr.output, shape="circle")
                graph.add_node(node)
                # Create and add edge
                edge = pydot.Edge(empty_node, node, label=alphabet, dir="forward", penwidth=4, color="darkgreen")
                graph.add_edge(edge)
                # Update new_index
                new_index += 1
            # If it is not a branch point
            elif len(tr.input) == 1:
                # Create input and output nodes
                node1 = pydot.Node(tr.input[0], label=tr.input[0], shape="circle")
                node2 = pydot.Node(tr.output, label=tr.output, shape="circle")
                # Add nodes
                graph.add_node(node1)
                graph.add_node(node2)
                # Create edge
                edge = pydot.Edge(node1, node2, label=alphabet, dir="forward")
                # Add edge
                graph.add_edge(edge)
            # If it is a branch point
            else:
                # Create a node that represents a branch point
                branch = pydot.Node(new_index, label=alphabet, shape="circle",
                                    fixedsize="true", width=0.2, color="white")
                graph.add_node(branch)
                # Create each input node and connect it to the branch point
                for n in tr.input:
                    node = pydot.Node(n, label=n, shape="circle")
                    edge = pydot.Edge(node, branch, label="", dir="none")
                    graph.add_node(node)
                    graph.add_edge(edge)
                # Create the output node and connect it to the branch point
                node = pydot.Node(tr.output, label=tr.output, shape="circle")
                edge = pydot.Edge(branch, node, label="", dir="forward")
                graph.add_node(node)
                graph.add_edge(edge)
                # Update new_index
                new_index += 1

        # Format end states
        for n in self.end_states:
            # Create and add node
            node = pydot.Node(n, label=n, shape="doublecircle", color="red", style="filled")
            graph.add_node(node)

        # Ouput folder
        output_folder = os.path.join(os.getcwd(), output_folder)
        # If it does not exist, create it
        try:
            os.makedirs(output_folder)
        except:
            pass

        # If the filename does not end in ".svg", add it
        if tree_name[-4:] != ".svg":
            tree_name += ".svg"
        # Create the names of the files that will have the alphabet definition and the tree graph
        alphabet_file_name = os.path.join(output_folder, "alphabets_" + tree_name)
        tree_filename = os.path.join(output_folder, tree_name)

        # Write the graph in an .svg file
        graph.write_svg(tree_filename)

        # Write the alphabets in an .svg file, only if the function is specified
        if draw_alphabet_function is not None:
            alphabet_dictionary = {tr.alphabet: tr.smiles for tr in self.transitions}
            draw_alphabet_function(alphabet_dictionary=alphabet_dictionary, filename=alphabet_file_name, *args,
                                   **kwargs)

    def get_starting_transitions(self):
        """
        This function find all starting transitions
        Returns: list of starting transitions

        """
        # List of starting transitions
        starts = [tr for tr in self.transitions if tr.input == []]

        return starts

    def get_nonstarting_transitions(self):
        """
        This function finds all transitions that are not starting transitions
        Returns: list of non starting transitions

        """

        # List of starting transitions
        starts = self.get_starting_transitions()

        # Look for non starting transitions
        nonstarts = [tr for tr in self.transitions if tr not in starts]

        return nonstarts

    def get_ending_transitions(self):
        """
        This functions finds all transitions that result in an ending state
        Returns: list of ending transitions
        """

        ending_transitions = [tr for tr in self.transitions if tr.output in self.end_states]

        return ending_transitions

    def generate_transition_map(self):
        """
        This function generates a dictionary whose keys are the states. The values are dictionaries with keys "in"
        and "out" that represent the transitions going into and out of the states.
        Returns: dictionary that is the transition map

        """

        transition_map = {state: {"in": [], "out": []} for state in self.states}

        for tr in self.transitions:
            # For each input, set the transition as an output of the state
            for s in tr.input:
                if tr not in transition_map[s]["out"]:  # To guarantee there will not be duplicates
                    transition_map[s]["out"].append(tr)
            # The transition output will be the input of the output state
            if tr not in transition_map[tr.output]["in"]:  # To guarantee there will not be duplicates
                transition_map[tr.output]["in"].append(tr)

        # Update transition map
        self.transition_map = transition_map

        return transition_map

    def sort(self):
        """
        This function relabels the states and sorts the transition rules.

        It executes a graph traversal starting from each starting transition. Then, it takes the one with
        the highest score (mass*position). Given a node, the graph traversal will prioritize output transitions
        with the lowest molar mass.

        It also updates the transition map.

        Returns: None

        """
        # Update transition map
        self.generate_transition_map()

        # Rank of states
        state_rank = []
        # Rank of transitions
        transition_rank = []
        # Mass of every transition
        transition_mass = []

        # Starting transitions
        start_transitions = self.get_starting_transitions()

        # Initialize the list that contains the ranks
        ranks = []
        # score = 0

        # For each start, compute the path
        for start_transition in start_transitions:

            # Starting state
            start_state = start_transition.output
            # Mass of the first transition
            forms_of_Es = [Chem.MolToSmiles(Chem.MolFromSmiles("[Es]")),
                           Chem.MolToSmiles(Chem.MolFromSmiles("[Es][*:1]")),
                           Chem.MolToSmiles(Chem.MolFromSmiles("[*:2][Es][*:1]")),
                           Chem.MolToSmiles(Chem.MolFromSmiles("[Es][*:2]"))]
            if Chem.MolToSmiles(Chem.MolFromSmiles(start_transition.smiles)) in forms_of_Es:
                molar_wt = 0
            else:
                molar_wt = ExactMolWt(Chem.MolFromSmiles(start_transition.smiles))

            # Initiate the lists
            _state_rank = [start_state]
            _transition_rank = [start_transition]
            _transition_mass = [molar_wt]

            # Traverse the state machine
            self.traverse(root=start_state, state_rank=_state_rank, transition_rank=_transition_rank,
                          transition_mass=_transition_mass)

            # Calculate score
            _score = sum([m * (i + 1) for i, m in enumerate(_transition_mass)])

            ranks.append([_score, _state_rank, _transition_rank])
            # # Compare score to update lists
            # if _score > score:
            #     # Update score
            #     score = _score
            #     # Update the lists
            #     state_rank = _state_rank
            #     transition_rank = _transition_rank
            #     transition_mass = _transition_mass

        # Choose the longest, heaviest rank
        ranks = sorted(ranks, key=lambda x: x[0], reverse=True)
        chosen_rank = ranks[0]
        state_rank = chosen_rank[1]
        transition_rank = chosen_rank[2]

        # Complete the chosen rank with the other ranks
        for other_rank in ranks[1:]:
            other_state_rank = other_rank[1]
            other_transition_rank = other_rank[2]
            # Add to chosen rank the states from other rank that are not in chosen rank
            state_rank += [s for s in other_state_rank if s not in state_rank]
            # Add to chosen rank the transitions from other rank that are not in chosen rank
            transition_rank += [s for s in other_transition_rank if s not in transition_rank]


        # Update the list of transitions with the list of transitions that has been generated
        self.transitions = transition_rank

        # Replace state labels by the indices from the state_rank list
        old_new_state_map = {s: i for i, s in enumerate(state_rank)}  # Dict that maps old states onto new states
        # List of states
        self.states = [old_new_state_map[s] for s in self.states]
        # List of ending states
        self.end_states = [old_new_state_map[s] for s in self.end_states]
        # List of transitions
        for tr in self.transitions:
            tr.input = [old_new_state_map[s] for s in tr.input]
            tr.output = old_new_state_map[tr.output]

        # Update transition map
        self.generate_transition_map()

    def traverse(self, root, state_rank, transition_rank, transition_mass):
        """
        Traverse the graph and generates a list of sorted states, transitions and transition mass.
        Given a node, the graph traversal will prioritize output transitions with the lowest molar mass. To untie,
        it compares the canonical smiles of the alphabets.
        Args:
            root: state from which the traversal will start
            state_rank: rank (order) of states
            transition_rank: order of transitions
            transition_mass: mass of transitions

        Returns: None

        """

        # Create list of transitions. They can be both the inputs and outputs. Es are removed from alphabets so
        # they do not have mass. It prioritizes the output transitions
        forms_of_Es = [Chem.MolFromSmiles("[Es]"), Chem.MolFromSmiles("[Es][*:1]"),
                       Chem.MolFromSmiles("[*:2][Es][*:1]"),
                       Chem.MolFromSmiles("[Es][*:2]")]
        possible_transitions = []
        # List output transitions
        for tr in self.transition_map[root]["out"]:
            _mol = Chem.MolFromSmiles(tr.smiles)
            _smiles = Chem.MolToSmiles(_mol)
            # If it is an empty transition, set mass to 0 and smiles to an empty string
            if _smiles in forms_of_Es:
                _mass = 0
                _smiles = ""
            else:
                _mass = ExactMolWt(_mol)
            # Add to list of transitions
            possible_transitions.append([tr, "out", _mass, _smiles])

        # List output transitions
        for tr in self.transition_map[root]["in"]:
            _mol = Chem.MolFromSmiles(tr.smiles)
            _smiles = Chem.MolToSmiles(_mol)
            # If it is an empty transition, set mass to 0 and smiles to an empty string
            if _smiles in forms_of_Es:
                _mass = 0
                _smiles = ""
            else:
                _mass = ExactMolWt(_mol)
            # Add to list of transitions
            possible_transitions.append([tr, "in", _mass, _smiles])

        # possible_transitions = [[tr, "out",
        #                          ExactMolWt(Chem.MolFromSmiles(tr.smiles.replace("[Es]", ""))),
        #                          Chem.MolToSmiles(Chem.MolFromSmiles(tr.smiles.replace("[Es]", "")))]
        #                         for tr in self.transition_map[root]["out"]
        #                         ]
        # possible_transitions += [[tr, "in",
        #                           ExactMolWt(Chem.MolFromSmiles(tr.smiles.replace("[Es]", ""))),
        #                           Chem.MolToSmiles(Chem.MolFromSmiles(tr.smiles.replace("[Es]", "")))]
        #                          for tr in self.transition_map[root]["in"]
        #                          ]
        # Sort the transitions in terms of molar mass, prioritizing the output transitions
        possible_transitions = sorted(possible_transitions, key=lambda x: [x[1] != "out", x[2], x[3]])

        # Choose the lightest transition
        next_transition = None
        next_transition_mass = None
        next_state = None
        for tr, direction, mass, _ in possible_transitions:
            # If the transition has already been chosen, skip it
            if tr in transition_rank:
                continue

            # If it is an outward transition, take the output state as the next state
            if direction == "out":
                # Choose the transition
                next_transition = tr
                next_transition_mass = mass
                next_state = tr.output
                # Update lists
                if next_state not in state_rank:  # Only add a state that has not been added
                    state_rank.append(next_state)
                if next_transition not in transition_rank:
                    transition_rank.append(next_transition)
                    transition_mass.append(next_transition_mass)
                # Traverse
                self.traverse(root=next_state, state_rank=state_rank, transition_rank=transition_rank,
                              transition_mass=transition_mass)

            # If it is inward and a starting transition, add to transitions and mass lists
            else:
                # Choose the transition
                next_transition = tr
                next_transition_mass = mass
                # If the transition has no input (starting transition), just add the transition and mass to lists
                if not tr.input:
                    transition_rank.append(next_transition)
                    transition_mass.append(next_transition_mass)

        # If not transition has been chosen, terminate the traversal
        if next_transition is None:
            return

    def DFTA_to_networkx(self):
        """
        This function converts a DFTA into a Networkx directed graph
        Returns: networkx directed graph

        """

        # Create directed graph
        G = nx.MultiDiGraph()

        # Add nodes
        G.add_nodes_from(self.states, is_state=True, is_branch=False, is_empty=False, is_end=False, smiles="",
                         alphabet="")

        # Get maximum node label
        max_node = max(self.states) + 1

        # Create edges based on the transition rules. New nodes will be created to designate transitions with
        # multiple inputs or no inputs
        for tr in self.transitions:
            # If there is no input, create a state to be the empty input
            if not tr.input:
                G.add_node(max_node, is_state=False, is_branch=False, is_empty=True, is_end=False, smiles="",
                           alphabet="")
                # Add edge
                G.add_edge(max_node, tr.output, smiles=tr.smiles, alphabet=tr.alphabet)
                # Update max_node
                max_node += 1
            # If there are many inputs, create a node that will be the branch point
            elif len(tr.input) > 1:
                G.add_node(max_node, is_state=False, is_branch=True, is_empty=False, is_end=False, smiles=tr.smiles,
                           alphabet=tr.alphabet)
                # Add edges from inputs to branch
                for s in tr.input:
                    G.add_edge(s, max_node, smiles=tr.smiles, alphabet=tr.alphabet)
                # Add edge from branch to output state
                G.add_edge(max_node, tr.output, smiles="", alphabet="")
                # Update max_node
                max_node += 1
            # If there is one input and one output, just connect them
            else:
                G.add_edge(tr.input[0], tr.output, smiles=tr.smiles, alphabet=tr.alphabet)

        # Set final states
        for s in self.end_states:
            G.nodes[s]["is_end"] = True

        return G

    def e_closure(self, root, list_of_states, canonicalize=True):
        """
        Give a root state, this function lists all states that are achieved by taking epsilon transitions (including
        the root state itself)
        Args:
            root: root state
            list_of_states: list of states connected by empty transitions. Initially, it is an empty list
            canonicalize: if True, canonicalizes the alphabet when comparing to empty alphabets

        Returns: None
        """
        # Initialize list of states with the root if it is empty
        if not list_of_states:
            list_of_states.append(root)

        # Create list of empty transitions
        possible_transitions = []
        # List output transitions
        for tr in self.transition_map[root]["out"]:
            if check_if_epsilon(tr, include_start=False, canonicalize=canonicalize):
                # Add to list of transitions
                possible_transitions.append(tr)

        # Take each empty transition to traverse
        next_transition = None
        for tr in possible_transitions:

            # If the transition has already been traversed, skip it
            next_state = tr.output
            if next_state in list_of_states:
                continue

            # Choose the next transition
            next_transition = tr

            # Append state to list of states
            list_of_states.append(next_state)

            # Traverse next state
            self.e_closure(root=next_state, list_of_states=list_of_states, canonicalize=canonicalize)

        # If not transition has been chosen, terminate the traversal
        if next_transition is None:
            return

    def eliminate_epsilon_transitions(self, canonicalize=True):
        """
        This function eliminates epsilon (empty) transitions as described in TATA. It takes groups of states connected
        by e-transitions and redirects the transitions that point to the root state to each of one them.
        Args:
            canonicalize: if True, canonicalizes the SMILES when comparing to empty transitions
        Returns: None

        """
        # Initialize list with new transitions
        new_transitions = []

        # Check each state of the automaton
        for state in self.states:
            # Take the list of states connected by empty transitions
            e_closure = []
            self.e_closure(root=state, list_of_states=e_closure, canonicalize=canonicalize)
            # Redirect the transitions to state to each of the states in e_closure
            transitions_into_state = [x for x in self.transition_map[state]["in"] if not check_if_epsilon(x, include_start=False, canonicalize=canonicalize)]    # Take all transitions but empty ones
            for q2 in e_closure:
                # Copy transitions into state
                list_of_transitions = copy.deepcopy(transitions_into_state)
                # Redirect them to q2
                for tr in list_of_transitions:
                    tr.output = q2
                # Append to list of new transitions
                new_transitions += list_of_transitions

        # Replace the transitions of the automaton by the new transitions
        self.transitions = new_transitions

        # Update the transition map
        self.generate_transition_map()

    def merge_states(self, stateA, stateB):
        """
        This function merges 2 states into one. It chooses one state to keep and another to delete, and redirects all
        transitions from/to the state that will be deleted from/to the state to keep.

        Args:
            stateA: one state to merge
            stateB: another state to merge

        Returns: None

        """

        if stateA > stateB:
            to_keep = stateA
            to_replace = stateB
        else:
            to_keep = stateB
            to_replace = stateA

        # Remove from list of states
        self.states.remove(to_replace)
        # If the state to be removed is a final state, remove it from list of final states and add to_keep
        if to_replace in self.end_states:
            self.end_states.remove(to_replace)
            if to_keep not in self.end_states:
                self.end_states.append(to_keep)

        # Change all transitions
        for tr in self.transition_map[to_replace]["in"]:
            tr.output = to_keep
            self.transition_map[to_keep]["in"].append(tr)
        for tr in self.transition_map[to_replace]["out"]:
            tr.input = [to_keep if s == to_replace else s for s in tr.input]
            self.transition_map[to_keep]["out"].append(tr)
        # Delete information about to_replace from transition map
        del self.transition_map[to_replace]

    def remove_transition(self, transition):
        """
        This function removes a transition from the transition list and the transition map.
        Args:
            transition: transition to be removed

        Returns: None

        """
        try:
            self.transitions.remove(transition)
        except:
            pass

        # Remove from transition map
        # Inputs
        for s in transition.input:
            try:
                self.transition_map[s]["out"].remove(transition)
            except:
                pass
        # Output
        try:
            self.transition_map[transition.output]["in"].remove(transition)
        except:
            pass

    def add_transition(self, transition):
        """
        This function adds a transition to the transition list and to the transition map.
        Args:
            transition: transition to be added

        Returns: None

        """
        # Add to transition list
        self.transitions.append(transition)

        # Add to transition map
        self.transition_map[transition.output]["in"].append(transition)
        for _in in transition.input:
            if transition not in self.transition_map[_in]["out"]:
                self.transition_map[_in]["out"].append(transition)

    def remove_state(self, state):
        """
        This functions deletes a state from the list of states and the transition map
        Args:
            state: state to be deleted

        Returns: None

        """
        self.states.remove(state)
        del self.transition_map[state]

# Main -----------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # tree = TreeAutomata([
    #                     Transitions(input=[], output=2, alphabet="e1", smiles="a"),
    #                     Transitions(input=[], output=0, alphabet="e2", smiles="a"),
    #                     Transitions(input=[2, 2], output=2, alphabet="C", smiles="a"),
    #                     Transitions(input=[0, 0], output=0, alphabet="C", smiles="a"),
    #                     Transitions(input=[0], output=2, alphabet="A", smiles="a"),
    #                     Transitions(input=[2], output=0, alphabet="A", smiles="a")],
    #                     end_states=[2,0]
    #                     )
    # tree = TreeAutomata([
    #                     Transitions(input=[], output=1, alphabet="B", smiles="d"),
    #                     Transitions(input=[], output=2, alphabet="B", smiles="d"),
    #                     Transitions(input=[1], output=1, alphabet="A", smiles="e"),
    #                     Transitions(input=[2], output=2, alphabet="A", smiles="k"),
    #                     Transitions(input=[1, 2], output=3, alphabet="D", smiles="a")],
    #                     end_states=[3]
    #                     )
    # tree = TreeAutomata([
    #                     Transitions(input=[], output=0, alphabet="A", smiles="d"),
    #                     Transitions(input=[], output=2, alphabet="A1", smiles="d"),
    #                     Transitions(input=[0], output=1, alphabet="B", smiles="d"),
    #                     Transitions(input=[2], output=1, alphabet="B", smiles="d")],
    #                     end_states=[1]
    #                     )
    # tree = TreeAutomata([
    #                     Transitions(input=[], output=1, alphabet="e", smiles="d"),
    #                     Transitions(input=[1], output=2, alphabet="C", smiles="d"),
    #                     Transitions(input=[2], output=3, alphabet="C", smiles="d"),
    #                     Transitions(input=[3], output=4, alphabet="O", smiles="d"),
    #                     Transitions(input=[4], output=5, alphabet="C", smiles="d"),
    #                     Transitions(input=[5], output=6, alphabet="C", smiles="d"),
    #                     Transitions(input=[6], output=4, alphabet="O", smiles="d")],
    #                     end_states=[4]
    #                     )
    # tree = TreeAutomata([
    #                     Transitions(input=[], output=1, alphabet="e", smiles="d"),
    #                     Transitions(input=[1], output=2, alphabet="A", smiles="d"),
    #                     Transitions(input=[1], output=3, alphabet="A", smiles="d"),
    #                     Transitions(input=[2,3], output=1, alphabet="B", smiles="d")],
    #                     end_states=[1]
    #                     )
    # tree = TreeAutomata([
    #                     Transitions(input=[], output=1, alphabet="C", smiles="d"),
    #                     Transitions(input=[], output=2, alphabet="C", smiles="d"),
    #                     Transitions(input=[1], output=2, alphabet="A", smiles="d"),
    #                     Transitions(input=[2], output=1, alphabet="A", smiles="d"),
    #                     Transitions(input=[1, 1], output=1, alphabet="B", smiles="d"),
    #                     Transitions(input=[2, 2], output=2, alphabet="B", smiles="d")],
    #                     end_states=[1, 2]
    #                     )
    # tree = TreeAutomata([
    #                     Transitions(input=[], output=10, alphabet="E1", smiles="d"),
    #                     Transitions(input=[], output=20, alphabet="E2", smiles="d"),
    #                     Transitions(input=[10], output=1, alphabet="C", smiles="d"),
    #                     Transitions(input=[1], output=10, alphabet="C", smiles="d"),
    #                     Transitions(input=[30], output=10, alphabet="E", smiles="d"),
    #                     Transitions(input=[30], output=30, alphabet="S", smiles="d"),
    #                     Transitions(input=[20], output=30, alphabet="E", smiles="d"),
    #                     Transitions(input=[20], output=40, alphabet="E", smiles="d"),
    #                     Transitions(input=[40], output=40, alphabet="S", smiles="d"),
    #                     Transitions(input=[], output=20, alphabet="E2", smiles="d"),
    #                     Transitions(input=[40], output=10, alphabet="E", smiles="d"),
    #                     Transitions(input=[10, 20], output=2, alphabet="C", smiles="d"),
    #                     Transitions(input=[10, 2], output=20, alphabet="C", smiles="d"),
    #                     Transitions(input=[10, 20], output=4, alphabet="C", smiles="d"),
    #                     Transitions(input=[10, 4], output=20, alphabet="C", smiles="d")
    #                     ],
    #                     end_states=[10, 20]
    #                     )
    # tree = TreeAutomata([
    #                     Transitions(input=[], output=1, alphabet="E1", smiles="d"),
    #                     Transitions(input=[], output=2, alphabet="E2", smiles="d"),
    #                     Transitions(input=[1], output=1, alphabet="C", smiles="d"),
    #                     Transitions(input=[2], output=2, alphabet="C", smiles="d"),
    #                     Transitions(input=[1, 2], output=3, alphabet="D", smiles="d"),],
    #                     end_states=[3]
    #                     )
    # tree = TreeAutomata([
    #                     Transitions(input=[], output=1, alphabet="E1", smiles="d"),
    #                     Transitions(input=[], output=2, alphabet="E2", smiles="d"),
    #                     Transitions(input=[1], output=1, alphabet="C", smiles="d"),
    #                     Transitions(input=[2], output=2, alphabet="C", smiles="d"),
    #                     Transitions(input=[2], output=3, alphabet="D", smiles="d"),
    #                     Transitions(input=[1], output=3, alphabet="D", smiles="d")],
    #                     end_states=[3]
    #                     )
    # tree = TreeAutomata([
    #                     Transitions(input=[], output=1, alphabet="E1", smiles="d"),
    #                     Transitions(input=[], output=2, alphabet="E2", smiles="d"),
    #                     Transitions(input=[1], output=1, alphabet="C", smiles="d"),
    #                     Transitions(input=[2], output=2, alphabet="C", smiles="d"),
    #                     Transitions(input=[1, 2], output=3, alphabet="D", smiles="d"),],
    #                     end_states=[3]
    #                     )
    # tree = TreeAutomata([
    #                     Transitions(input=[], output=0, alphabet="E", smiles="d"),
    #                     Transitions(input=[0], output=4, alphabet="C", smiles="d"),
    #                     Transitions(input=[4], output=1, alphabet="C", smiles="d"),
    #                     Transitions(input=[1], output=7, alphabet="O", smiles="d"),
    #                     Transitions(input=[7], output=6, alphabet="C", smiles="d"),
    #                     Transitions(input=[6], output=5, alphabet="C", smiles="d"),
    #                     Transitions(input=[5], output=7, alphabet="O", smiles="d"),
    #                     Transitions(input=[7], output=2, alphabet="C", smiles="d"),
    #                     Transitions(input=[2], output=3, alphabet="C", smiles="d"),
    #                     Transitions(input=[3], output=0, alphabet="O", smiles="d")],
    #                     end_states=[0]
    #                     )
    # tree = TreeAutomata([
    #                     Transitions(input=[], output=1, alphabet="E", smiles="d"),
    #                     Transitions(input=[1], output=2, alphabet="A", smiles="d"),
    #                     Transitions(input=[2], output=3, alphabet="B", smiles="d"),
    #                     Transitions(input=[3], output=5, alphabet="A", smiles="d"),
    #                     Transitions(input=[5], output=3, alphabet="B", smiles="d"),
    #                     Transitions(input=[3], output=4, alphabet="A", smiles="d"),
    #                     Transitions(input=[4], output=1, alphabet="B", smiles="d")],
    #                     end_states=[1]
    #                     )
    # tree = TreeAutomata([
    #                     Transitions(input=[], output=1, alphabet="e", smiles="C[*:2]"),
    #                     Transitions(input=[], output=2, alphabet="e", smiles="C[*:2]"),
    #                     Transitions(input=[1, 2], output=3, alphabet="A", smiles="[*:1]O[*:2]"),
    #                     Transitions(input=[3], output=2, alphabet="B", smiles="[*:1]N[*:2]")],
    #                     end_states=[3]
    #                     )
    # tree = TreeAutomata([
    #                     Transitions(input=[], output=0, alphabet="e", smiles="C[*:2]"),
    #                     Transitions(input=[0], output=1, alphabet="A", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[1], output=2, alphabet="B", smiles="[*:1]O[*:2]"),
    #                     Transitions(input=[2], output=0, alphabet="B", smiles="[*:1]O[*:2]"),
    #                     Transitions(input=[0], output=3, alphabet="A", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[3], output=4, alphabet="B", smiles="[*:1]O[*:2]"),
    #                     Transitions(input=[4], output=5, alphabet="B", smiles="[*:1]O[*:2]"),
    #                     Transitions(input=[5], output=6, alphabet="A", smiles="[*:2]N[*:2]")],
    #                     end_states=[6]
    #                     )
    # tree = TreeAutomata([
    #                     Transitions(input=[], output=3, alphabet="C", smiles="Es[*:2]"),
    #                     Transitions(input=[], output=12, alphabet="E", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[3], output=40, alphabet="D", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[40], output=12, alphabet="D", smiles="[*:1]O[*:2]"),
    #                     Transitions(input=[12], output=27, alphabet="A", smiles="[*:1]O[*:2]"),
    #                     Transitions(input=[27], output=47, alphabet="B", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[47], output=46, alphabet="B", smiles="[*:1]O[*:2]"),
    #                     Transitions(input=[46], output=27, alphabet="A", smiles="[*:1]O[*:2]"),
    #                     Transitions(input=[27], output=43, alphabet="B", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[43], output=42, alphabet="B", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[42], output=3, alphabet="A", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[12], output=44, alphabet="A", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[44], output=45, alphabet="B", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[45], output=33, alphabet="B", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[33], output=48, alphabet="A", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[48], output=49, alphabet="B", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[49], output=33, alphabet="B", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[33], output=3, alphabet="A", smiles="[*:2]N[*:2]")],
    #                     end_states=[3, 12]
    #                     )
    # tree = TreeAutomata([
    #                     Transitions(input=[], output=3, alphabet="C", smiles="Es[*:2]"),
    #                     Transitions(input=[], output=12, alphabet="E", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[3], output=12, alphabet="D", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[12], output=27, alphabet="A", smiles="[*:1]O[*:2]"),
    #                     Transitions(input=[27], output=46, alphabet="B", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[46], output=27, alphabet="A", smiles="[*:1]O[*:2]"),
    #                     Transitions(input=[27], output=42, alphabet="B", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[42], output=3, alphabet="A", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[12], output=44, alphabet="A", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[44], output=33, alphabet="B", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[33], output=48, alphabet="A", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[48], output=33, alphabet="B", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[33], output=3, alphabet="A", smiles="[*:2]N[*:2]")],
    #                     end_states=[3, 12]
    #                     )
    # tree = TreeAutomata([
    #                     Transitions(input=[], output=7, alphabet="C", smiles="Es[*:2]"),
    #                     Transitions(input=[7], output=22, alphabet="B", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[22], output=23, alphabet="A", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[23], output=7, alphabet="A", smiles="[*:1]O[*:2]"),
    #                     Transitions(input=[7], output=20, alphabet="B", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[20], output=21, alphabet="A", smiles="[*:1]O[*:2]"),
    #                     Transitions(input=[21], output=13, alphabet="A", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[13], output=24, alphabet="B", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[24], output=25, alphabet="D", smiles="[*:2]N[*:2]"),
    #                     Transitions(input=[25], output=13, alphabet="A", smiles="[*:2]N[*:2]")],
    #                     end_states=[13]
    #                     )
    # tree = TreeAutomata([
    #     Transitions(input=[], output=0, alphabet="A", smiles="[Br][*:1]"),
    #     Transitions(input=[0, 0], output=0, alphabet="B", smiles="[*:3]n1cc([*:1])c([*:2])c1")
    # ])
    tree = TreeAutomata([
        Transitions(input=[], output=1, alphabet="E", smiles="Es[*:1]"),
        Transitions(input=[1], output=2, alphabet="B", smiles="[*:2]C[*:1]"),
        Transitions(input=[2, 3], output=1, alphabet="C", smiles="[*:1]C([*:2])[*:3]"),
        Transitions(input=[4], output=3, alphabet="D", smiles="[*:1]C(=O)[*:2]"),
        Transitions(input=[5], output=4, alphabet="F", smiles="[*:1]O[*:2]"),
        Transitions(input=[], output=5, alphabet="H", smiles="[*:1]Es"),
        Transitions(input=[5], output=7, alphabet="F", smiles="[*:1]O[*:2]"),
        Transitions(input=[7], output=6, alphabet="B", smiles="[*:1]C[*:2]"),
        Transitions(input=[6], output=5, alphabet="B", smiles="[*:1]C[*:2]")],
        end_states=[1]
    )
    tree = TreeAutomata([
        Transitions(input=[], output=2, alphabet="D", smiles="Es[*:1]"),
        Transitions(input=[], output=3, alphabet="E", smiles="Es[*:1]"),
        Transitions(input=[], output=1, alphabet="F", smiles="[*:2]C[*:1]"),
        Transitions(input=[2], output=2, alphabet="A", smiles="[*:2]C[*:1]"),
        Transitions(input=[3], output=3, alphabet="A", smiles="[*:2]C[*:1]"),
        Transitions(input=[1, 2, 3], output=4, alphabet="C", smiles="[*:1]C([*:2])[*:3]"),
        Transitions(input=[2, 2, 1], output=4, alphabet="C", smiles="[*:1]C([*:2])[*:3]"),
        Transitions(input=[3, 3, 1], output=4, alphabet="C", smiles="[*:1]C([*:2])[*:3]"), ],
        end_states=[4]
    )
    tree = TreeAutomata([
        Transitions(input=[], output=0, alphabet="E", smiles="Es[*:1]"),
        Transitions(input=[0], output=1, alphabet="B", smiles="Es[*:1]"),
        Transitions(input=[1], output=2, alphabet="B", smiles="[*:2]C[*:1]"),
        Transitions(input=[2], output=3, alphabet="A", smiles="[*:2]C[*:1]"),
        Transitions(input=[3], output=4, alphabet="B", smiles="[*:2]C[*:1]"),
        Transitions(input=[4], output=2, alphabet="B", smiles="[*:1]C([*:2])[*:3]"),
        Transitions(input=[4], output=5, alphabet="C", smiles="[*:1]C([*:2])[*:3]"),
        Transitions(input=[5], output=6, alphabet="B", smiles="[*:1]C([*:2])[*:3]"),
        Transitions(input=[6], output=5, alphabet="C", smiles="[*:1]C([*:2])[*:3]"),
        Transitions(input=[5], output=7, alphabet="D", smiles="[*:1]C([*:2])[*:3]"),
        Transitions(input=[3], output=7, alphabet="D", smiles="[*:1]C([*:2])[*:3]"),
    ],
        end_states=[7]
    )
    tree = TreeAutomata([
        Transitions(input=[], output=0, alphabet="D", smiles="Es[*:1]"),
        Transitions(input=[], output=1, alphabet="B", smiles="Es[*:1]"),
        Transitions(input=[0, 1], output=2, alphabet="A", smiles="[*:2]C[*:1]"),
        Transitions(input=[2], output=5, alphabet="C", smiles="[*:2]C[*:1]"),
        Transitions(input=[5], output=6, alphabet="D", smiles="[*:2]C[*:1]"),
        Transitions(input=[6, 3], output=4, alphabet="A", smiles="[*:2]C[*:1]"),
        Transitions(input=[4], output=5, alphabet="C", smiles="[*:1]C([*:2])[*:3]"),
        Transitions(input=[], output=3, alphabet="B", smiles="[*:1]C([*:2])[*:3]"),
    ],
        end_states=[5]
    )
    tree = TreeAutomata([
        Transitions(input=[], output=0, alphabet="D", smiles="Es[*:1]"),
        Transitions(input=[], output=1, alphabet="B", smiles="Es[*:1]"),
        Transitions(input=[0, 1], output=2, alphabet="A", smiles="[*:2]C[*:1]"),
        Transitions(input=[2], output=3, alphabet="C", smiles="[*:2]C[*:1]"),
        Transitions(input=[3], output=0, alphabet="D", smiles="[*:2]C[*:1]"),
        Transitions(input=[0, 4], output=5, alphabet="A", smiles="[*:2]C[*:1]"),
        Transitions(input=[5], output=6, alphabet="C", smiles="[*:1]C([*:2])[*:3]"),
        Transitions(input=[], output=4, alphabet="B", smiles="[*:1]C([*:2])[*:3]"),
    ],
        end_states=[6]
    )
    tree = TreeAutomata([
        Transitions(input=[], output=0, alphabet="A", smiles="Es[*:1]"),
        Transitions(input=[0], output=0, alphabet="B", smiles="Es[*:1]"),
        Transitions(input=[0], output=1, alphabet="C", smiles="[*:2]C[*:1]"),
        Transitions(input=[], output=2, alphabet="D", smiles="[*:2]C[*:1]"),
        Transitions(input=[1, 2], output=3, alphabet="E", smiles="[*:2]C[*:1]"),
        Transitions(input=[3], output=4, alphabet="F", smiles="[*:2]C[*:1]"),
    ],
        end_states=[4]
    )
    tree = TreeAutomata([
        Transitions(input=[], output=0, alphabet="A", smiles="Es[*:1]"),
        Transitions(input=[0], output=0, alphabet="B", smiles="Es[*:1]"),
        Transitions(input=[0], output=1, alphabet="C", smiles="[*:2]C[*:1]"),
        Transitions(input=[3], output=4, alphabet="D", smiles="[*:2]C[*:1]"),
        Transitions(input=[1, 2], output=3, alphabet="E", smiles="[*:2]C[*:1]"),
        Transitions(input=[], output=2, alphabet="F", smiles="[*:2]C[*:1]"),
    ],
        end_states=[4]
    )

    tree = TreeAutomata([
        Transitions(input=[], output=0, alphabet="D", smiles="Es[*:1]"),
        Transitions(input=[0], output=3, alphabet="F", smiles="[*:2]C[*:1]"),
        Transitions(input=[3], output=0, alphabet="F", smiles="[*:2]C[*:1]"),
        Transitions(input=[3], output=4, alphabet="H", smiles="[*:2]C[*:1]"),
        Transitions(input=[4], output=3, alphabet="H", smiles="[*:2]C[*:1]"),
        Transitions(input=[], output=1, alphabet="B", smiles="Es[*:1]"),
        Transitions(input=[1], output=1, alphabet="G", smiles="[*:2]C[*:1]"),
        Transitions(input=[0, 1], output=2, alphabet="A", smiles="[*:2]C([*:1])[*:3]"),
        Transitions(input=[2], output=2, alphabet="C", smiles="[*:2]C[*:1]"),
    ],
        end_states=[2]
    )
    tree = TreeAutomata([
        # Transitions(input=[], output=0, alphabet="e", smiles="Es[*:1]"),
        Transitions(input=[], output=1, alphabet="C", smiles="Es[*:1]"),
        Transitions(input=[1], output=2, alphabet="C", smiles="[*:2]C[*:1]"),
        Transitions(input=[2], output=3, alphabet="O", smiles="[*:2]C[*:1]"),
        Transitions(input=[3], output=4, alphabet="C", smiles="[*:2]C[*:1]"),
        Transitions(input=[4], output=5, alphabet="C", smiles="[*:2]C[*:1]"),
        Transitions(input=[5], output=3, alphabet="O", smiles="Es[*:1]"),
        Transitions(input=[3], output=7, alphabet="C", smiles="[*:2]C[*:1]"),
        Transitions(input=[7], output=6, alphabet="CB", smiles="[*:2]C([*:1])[*:3]"),
        Transitions(input=[6], output=7, alphabet="C", smiles="[*:2]C[*:1]"),
    ],
        end_states=[3,6]
    )
    tree = TreeAutomata([
        Transitions(input=[], output=1, alphabet="E", smiles="Es[*:1]"),
        Transitions(input=[1], output=2, alphabet="C", smiles="[*:2]C[*:1]"),
        Transitions(input=[2], output=3, alphabet="C", smiles="[*:2]C[*:1]"),
        Transitions(input=[3], output=1, alphabet="O", smiles="[*:2]C[*:1]"),
        Transitions(input=[1], output=4, alphabet="C", smiles="[*:2]C[*:1]"),
        Transitions(input=[4], output=5, alphabet="C", smiles="Es[*:1]"),
        Transitions(input=[5], output=6, alphabet="O", smiles="[*:2]C[*:1]"),
        Transitions(input=[6], output=7, alphabet="C", smiles="[*:2]C([*:1])[*:3]"),
        Transitions(input=[7], output=6, alphabet="CB", smiles="[*:2]C[*:1]"),
    ],
        end_states=[6]
    )
    tree = TreeAutomata([
        Transitions(input=[], output=0, alphabet="E", smiles="Es[*:1]"),
        Transitions(input=[0], output=0, alphabet="A", smiles="[*:2]C[*:1]"),
        Transitions(input=[0], output=1, alphabet="A", smiles="[*:2]C[*:1]"),
        Transitions(input=[1], output=1, alphabet="A", smiles="[*:2]C[*:1]"),
    ],
        end_states=[0, 1]
    )
    tree = TreeAutomata([
        Transitions(input=[], output=0, alphabet="E", smiles="Es[*:1]"),
        Transitions(input=[], output=1, alphabet="E2", smiles="[*:2]C[*:1]"),
        Transitions(input=[], output=2, alphabet="E3", smiles="[*:2]C[*:1]"),
        Transitions(input=[0], output=0, alphabet="A", smiles="[*:2]C[*:1]"),
        Transitions(input=[0], output=1, alphabet="A", smiles="[*:2]C[*:1]"),
        Transitions(input=[1], output=1, alphabet="B", smiles="[*:2]C[*:1]"),
        Transitions(input=[1], output=2, alphabet="B", smiles="[*:2]C[*:1]"),
        Transitions(input=[2], output=2, alphabet="C", smiles="[*:2]C[*:1]"),
        Transitions(input=[0], output=2, alphabet="A", smiles="[*:2]C[*:1]"),
    ],
        end_states=[2]
    )
    tree = TreeAutomata([
        Transitions(input=[], output=0, alphabet="E)", smiles="[Es][*:1]"),
        Transitions(input=[0], output=0, alphabet="A", smiles="[*:2]C[*:1]"),
        Transitions(input=[0], output=1, alphabet="E", smiles="[*:2][Es][*:1]"),
        Transitions(input=[1], output=1, alphabet="B", smiles="[*:2]C[*:1]"),
        Transitions(input=[1], output=2, alphabet="E", smiles="[*:2][Es][*:1]"),
        Transitions(input=[2], output=2, alphabet="C", smiles="[*:2]C[*:1]"),
    ],
        end_states=[2]
    )
    tree = TreeAutomata([
        Transitions(input=[], output=0, alphabet="E0", smiles="[Es][*:1]"),
        Transitions(input=[0], output=0, alphabet="A", smiles="[*:2]C[*:1]"),
        Transitions(input=[0], output=1, alphabet="E", smiles="[*:2][Es][*:1]"),
        Transitions(input=[1], output=1, alphabet="A", smiles="[*:2]C[*:1]"),
    ],
        end_states=[0, 1]
    )
    tree = TreeAutomata([
        Transitions(input=[], output=6, alphabet="E0", smiles="[Es][*:1]"),
        Transitions(input=[6], output=5, alphabet="B", smiles="[*:2]C[*:1]"),
        Transitions(input=[5], output=3, alphabet="B", smiles="[*:2]C[*:1]"),
        Transitions(input=[3], output=0, alphabet="A", smiles="[*:2]O[*:1]"),
        Transitions(input=[0], output=4, alphabet="B", smiles="[*:2]C[*:1]"),
        Transitions(input=[4], output=3, alphabet="B", smiles="[*:2]C[*:1]"),
        Transitions(input=[4], output=2, alphabet="C", smiles="[*:2]C(c1ccccc1)[*:1]"),
        Transitions(input=[2], output=1, alphabet="B", smiles="[*:2]C[*:1]"),
        Transitions(input=[1], output=2, alphabet="C", smiles="[*:2]C(c1ccccc1)[*:1]"),
        Transitions(input=[0], output=7, alphabet="E", smiles="[*:2][Es][*:1]"),
        Transitions(input=[2], output=7, alphabet="E", smiles="[*:2][Es][*:1]"),
    ],
        end_states=[7]
    )
    tree = TreeAutomata([
        Transitions(input=[], output=0, alphabet="E0", smiles="[Es][*:1]"),
        Transitions(input=[0], output=0, alphabet="D", smiles="[*:2]C[*:1]"),
        Transitions(input=[0], output=1, alphabet="C", smiles="[*:2]C[*:1]"),
        Transitions(input=[1], output=1, alphabet="C", smiles="[*:2]O[*:1]"),
        Transitions(input=[1], output=2, alphabet="B", smiles="[*:2]C[*:1]"),
        Transitions(input=[2], output=2, alphabet="B", smiles="[*:2]C[*:1]"),
        Transitions(input=[0], output=2, alphabet="B", smiles="[*:2]C(c1ccccc1)[*:1]"),
        Transitions(input=[0], output=3, alphabet="E", smiles="[*:2][Es][*:1]"),
        Transitions(input=[1], output=3, alphabet="E", smiles="[*:2][Es][*:1]"),
        Transitions(input=[2], output=3, alphabet="E", smiles="[*:2][Es][*:1]"),
    ],
        end_states=[3]
    )
    tree = TreeAutomata([
        Transitions(input=[], output=0, alphabet="E0", smiles="[Es][*:1]"),
        Transitions(input=[0], output=0, alphabet="D", smiles="[*:2]C[*:1]"),
        Transitions(input=[0], output=1, alphabet="E", smiles="[*:2][Es][*:1]"),
        Transitions(input=[1], output=1, alphabet="C", smiles="[*:2]O[*:1]"),
        Transitions(input=[1], output=2, alphabet="E", smiles="[*:2][Es][*:1]"),
        Transitions(input=[2], output=2, alphabet="B", smiles="[*:2]C[*:1]"),
        Transitions(input=[2], output=3, alphabet="E", smiles="[*:2][Es][*:1]"),
    ],
        end_states=[3]
    )
    tree = TreeAutomata([
        Transitions(input=[], output=0, alphabet="H", smiles="[Es][*:1]"),
        Transitions(input=[], output=2, alphabet="F", smiles="[Es][*:1]"),
        Transitions(input=[], output=3, alphabet="G", smiles="[Es][*:1]"),
        Transitions(input=[0, 3], output=1, alphabet="A", smiles="[*:2]C(C[*:1])[*:3]"),
        Transitions(input=[0, 1], output=3, alphabet="A", smiles="[*:2]C(C[*:1])[*:3]"),
        Transitions(input=[1, 3], output=2, alphabet="A", smiles="[*:2]C(C[*:1])[*:3]"),
        Transitions(input=[1, 2], output=3, alphabet="A", smiles="[*:2]C(C[*:1])[*:3]"),
        Transitions(input=[2, 3], output=1, alphabet="A", smiles="[*:2]C(C[*:1])[*:3]"),
        Transitions(input=[2], output=2, alphabet="EE", smiles="[*:2]SS[*:1]"),
        Transitions(input=[3], output=3, alphabet="BCB", smiles="[*:2]CC=CC[*:1]"),
    ],
        end_states=[0, 2, 3]
    )
    tree = TreeAutomata([
        Transitions(input=[], output=0, alphabet="A", smiles="Br[*:1]"),
        Transitions(input=[], output=1, alphabet="B", smiles="CC[*:1]"),
        Transitions(input=[0, 1], output=2, alphabet="C", smiles="[*:2]C([*:1])[*:3]"),
        Transitions(input=[0, 2], output=2, alphabet="C", smiles="[*:2]C([*:1])[*:3]"),
        Transitions(input=[2], output=3, alphabet="D", smiles="[*:2]C[*:1]"),
    ],
        end_states=[3]
    )
    tree = TreeAutomata([
        Transitions(input=[], output=0, alphabet="A", smiles="Br[*:1]"),
        Transitions(input=[], output=1, alphabet="B", smiles="CC[*:1]"),
        Transitions(input=[1], output=2, alphabet="E", smiles="[*:2][Es][*:1]"),
        Transitions(input=[0, 2], output=2, alphabet="C", smiles="[*:2]C([*:1])[*:3]"),
        Transitions(input=[2], output=3, alphabet="D", smiles="[*:2]C[*:1]"),
    ],
        end_states=[3]
    )
    tree = TreeAutomata([
        Transitions(input=[], output=0, alphabet="A", smiles="Br[*:1]"),
        Transitions(input=[0], output=0, alphabet="B", smiles="CC[*:1]"),
        Transitions(input=[0], output=1, alphabet="E", smiles="[*:2][Es][*:1]"),
        Transitions(input=[1], output=1, alphabet="C", smiles="CC[*:1]"),
        Transitions(input=[1], output=2, alphabet="E", smiles="[*:2][Es][*:1]"),
    ],
        end_states=[2]
    )
    folder = "Cycle_Detection_Tests\\Test12"
    tree.plot(tree_name="initial", output_folder=folder)
    tree.eliminate_epsilon_transitions()
    tree.plot(tree_name="no_empty_transitions", output_folder=folder)
    tree.determinization()
    tree.plot(tree_name="determinized", output_folder=folder)
    tree.minimization()
    tree.plot(tree_name="minimized", output_folder=folder)
    pass