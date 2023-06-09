"""
Automaton module

This module contains the Automaton class and its methods.
The Automaton class is used to represent a finite automaton.
The automaton can be constructed from a regular expression using Thompson's construction.
The automaton can be converted to a DFA using the subset construction.
"""

from graphviz import Digraph


EPS = 'ε'  # Epsilon symbol (empty string)
CONCATENATION = '·'  # Concatenation symbol
special_regex_chars = {'|', '+', '*', '?', '(', ')', CONCATENATION}

LETTER_SPECIAL_CHAR = chr(251)
DIGIT_SPECIAL_CHAR = chr(252)
ANYTHING_SPECIAL_CHAR = chr(253)

global_automata_counter = 0
global_state_register = dict()  # {automata_id: {state_name: State}}
global_state_counter = 0


class Automaton:
    """
    Automaton class
    """

    def __init__(self, regex=None, depth=0, id=None):
        """
        Initialize the automaton, either from a regular expression or from an id.
        """
        global global_automata_counter
        self.depth = depth
        self.alphabet = set()
        if id is None:
            self.id = global_automata_counter
            global_automata_counter += 1
            global_state_register[self.id] = dict()
        else:
            self.id = id

        if regex:
            preprocessed_regex = self._preprocess_regex(regex)
            self.from_regex(preprocessed_regex)
        else:
            self.states = dict()  # {state_name: State}
            self.is_deterministic = False
            self.initial_state = None
            self.final_states = set()
            self.transitions = dict()

    def _preprocess_regex(self, regex):
        """
        Preprocess the regular expression to insert concatenation symbols.
        """
        if CONCATENATION in regex:
            raise Exception(
                f"Symbols {CONCATENATION} are not allowed in the regex")

        processed_regex = []  # {literal: char, escaped: bool}

        unary_operators = {'+', '*', '?'}
        parentheses = {'(', ')'}
        binary_operators = {'|'}

        def is_other(literal):
            result = literal["char"] not in (unary_operators | binary_operators | parentheses) or (
                literal["char"] in (unary_operators | binary_operators | parentheses) and literal["escaped"])
            if result:
                self.alphabet.add(literal["char"])
            return result

        # Process escape sequences
        i = 0
        while i < len(regex):
            char = regex[i]

            if char == '\\':
                processed_regex.append(
                    {'char': regex[i+1], 'escaped': True})
                i += 2
            else:
                processed_regex.append({'char': char, 'escaped': False})
                i += 1

        # Insert concatenation symbols
        i = 0
        processed_concatenation_regex = []
        while i < len(processed_regex):
            literal = processed_regex[i]
            char = literal["char"]

            if i < len(processed_regex) - 1:
                next_literal = processed_regex[i+1]
                next_char = next_literal["char"]
            else:
                next_literal = None
                next_char = None

            processed_concatenation_regex.append(literal)

            if next_literal:
                is_concatenation_needed = (
                    char in ({")"} | unary_operators) or is_other(literal)) and (next_char in {'('} or is_other(next_literal))

                if is_concatenation_needed:
                    processed_concatenation_regex.append(
                        {'char': CONCATENATION, 'escaped': False})

            i += 1

        return processed_concatenation_regex

    def regex_to_postfix(self, regex):
        """
        Convert a regular expression from infix to postfix notation using the shunting-yard algorithm.
        """
        # Define the precedence of operators
        precedence = {'|': 0, CONCATENATION: 1, '+': 2, '?': 3, '*': 3}

        # Initialize the operator stack and output queue
        operator_stack = []
        output_queue = []

        # Define special characters
        special_chars = special_regex_chars

        # Process each character in the regular expression
        for literal in regex:
            char = literal["char"]
            escaped = literal["escaped"]
            if char not in special_chars or escaped:
                # Literal character, add to output queue
                output_queue.append(literal)
            elif char == '(':
                # Left parenthesis, push onto operator stack
                operator_stack.append(literal)
            elif char == ')':
                # Right parenthesis, pop operators from stack to output queue until left parenthesis is found
                while operator_stack and operator_stack[-1]["char"] != '(':
                    output_queue.append(operator_stack.pop())
                operator_stack.pop()  # Pop the left parenthesis
            else:
                # Operator, pop operators with higher or equal precedence from stack to output queue
                while (operator_stack and operator_stack[-1]["char"] != '(' and
                       precedence[char] <= precedence[operator_stack[-1]["char"]]):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(literal)

        # Pop remaining operators from stack to output queue
        while operator_stack:
            output_queue.append(operator_stack.pop())

        return output_queue

    def from_regex(self, regex):
        """
        Convert a regular expression to an NFA using Thompson's construction.
        """

        postfix_regex = self.regex_to_postfix(regex)

        self.is_deterministic = False
        # Initialize the stack
        stack = []

        # Process each character in the postfix regular expression
        for literal in postfix_regex:
            char = literal["char"]
            escaped = literal["escaped"]

            if escaped:
                # Treat the character as a normal symbol
                nfa = self._symbol(char)
                stack.append(nfa)

            else:
                match char:
                    case '|':
                        nfa2 = stack.pop()
                        nfa1 = stack.pop()
                        nfa = nfa1._union(nfa2)
                        stack.append(nfa)

                    case '+':
                        nfa = stack.pop()
                        nfa = nfa._plus(nfa)
                        stack.append(nfa)

                    case '?':
                        nfa = stack.pop()
                        nfa = nfa._question(nfa)
                        stack.append(nfa)

                    case '*':
                        nfa = stack.pop()
                        nfa = nfa._star(nfa)
                        stack.append(nfa)

                    case _ if char == CONCATENATION:
                        nfa2 = stack.pop()
                        nfa1 = stack.pop()
                        nfa = nfa1._concatenate(nfa2)
                        stack.append(nfa)

                    case _:
                        nfa = self._symbol(char)
                        stack.append(nfa)

        # The final NFA is the only NFA left on the stack
        nfa = stack.pop()

        # Set the initial and final states
        self.initial_state = nfa.initial_state
        self.final_states = nfa.final_states

        # Set the states and transitions
        self.states = nfa.states
        self.transitions = nfa.transitions

    def add_state(self, is_initial=False, is_final=False):
        """
        Add a new state to the NFA.
        """
        global global_state_counter
        state_name = f"{str(self.depth)}.{str(global_state_counter)}"
        global_state_counter += 1
        state = self.State(
            state_name, is_initial=is_initial, is_final=is_final)
        self.states[state_name] = state
        if is_initial:
            self.initial_state = state_name
        if is_final:
            self.final_states.add(state_name)

        state.parent_nfa = self

        global_state_register[self.id][state_name] = state
        return state_name

    def add_transition(self, source, symbol, target):
        """
        Add a new transition to the NFA.
        """
        if source not in self.transitions:
            self.transitions[source] = dict()
        if symbol not in self.transitions[source]:
            self.transitions[source][symbol] = set()

        self.transitions[source][symbol].add(target)
        self.states[source].add_transition(symbol, target)

    def visualize(self):
        """
        Visualize the NFA using Graphviz.
        """
        transitions = self.get_all_transitions()
        dot = Digraph(comment='NFA')

        # Add states as nodes
        for state_name, state in self.states.items():
            attributes = {}
            if state.is_initial:
                attributes['shape'] = 'pentagon'
                if state.is_final:
                    attributes['shape'] = 'doubleoctagon'
            elif state.is_final:
                attributes['shape'] = 'doublecircle'
            else:
                attributes['shape'] = 'circle'
            dot.node(state_name, state.name, **attributes)

        # Create nodes for the rest of the states
        for src, label, dest in transitions:
            if src not in self.states:
                dot.node(src, src, shape='circle')

        # Add transitions as edges
        for src, label, dest in transitions:
            if label == LETTER_SPECIAL_CHAR:
                label = 'letter'
            elif label == DIGIT_SPECIAL_CHAR:
                label = 'digit'
            elif label == ANYTHING_SPECIAL_CHAR:
                label = 'anything'
            dot.edge(src, dest, label=label)

        # Render and display the graph
        dot.view()

    def __repr__(self):
        return f"Automaton(states={self.states}, initial_state={self.initial_state}, final_states={self.final_states}, transitions={self.transitions})"

    def get_all_transitions(self):
        """
        Return all transitions in the NFA.
        """
        # Return all transitions using BFS to traverse the states
        visited = set()
        queue = [self.initial_state]

        transitions = set()

        while queue:
            state = queue.pop(0)
            state = global_state_register[self.id][state]
            if state in visited:
                continue
            visited.add(state)
            if state in self.final_states:
                continue
            for symbol, targets in state.transitions.items():
                for target in targets:
                    transitions.add((state.name, symbol, target))
                    queue.append(target)

        return transitions

    def _plus(self, nfa):
        """
        Return the NFA for the plus operator.
        """
        new_nfa = Automaton(depth=self.depth + 1, id=self.id)
        initial = new_nfa.add_state(is_initial=True)
        final = new_nfa.add_state(is_final=True)

        # Add epsilon transition from the new initial state to the initial state of nfa
        new_nfa.add_transition(initial, EPS, nfa.initial_state)
        global_state_register[self.id][nfa.initial_state].is_initial = False

        # Add epsilon transitions from the final states of nfa to its initial state and the new final state
        for final_state in nfa.final_states:
            nfa.add_transition(final_state, EPS, nfa.initial_state)
            nfa.add_transition(final_state, EPS, final)
            global_state_register[self.id][final_state].is_final = False

        nfa.final_states = {}
        nfa.initial_state = None

        return new_nfa

    def _question(self, nfa):
        """
        Return the NFA for the question operator.
        """
        new_nfa = Automaton(depth=self.depth + 1, id=self.id)
        initial = new_nfa.add_state(is_initial=True)
        final = new_nfa.add_state(is_final=True)

        # Add epsilon transition from the new initial state to the initial state of nfa
        new_nfa.add_transition(initial, EPS, nfa.initial_state)
        global_state_register[self.id][nfa.initial_state].is_initial = False

        # Add epsilon transition from the new initial state to the new final state (zero occurrence case)
        new_nfa.add_transition(initial, EPS, final)

        # Add epsilon transitions from the final states of nfa to the new final state
        for final_state in nfa.final_states:
            nfa.add_transition(final_state, EPS, final)
            global_state_register[self.id][final_state].is_final = False

        nfa.final_states = {}
        nfa.initial_state = None

        return new_nfa

    def _concatenate(self, nfa):
        """
        Return the NFA for the concatenation operator.
        """
        for final_state in self.final_states:
            self.add_transition(final_state, EPS, nfa.initial_state)
            global_state_register[self.id][final_state].is_final = False

        # Set the final states to the other NFA's final states
        self.final_states = nfa.final_states

        # Add the other NFA's states and transitions
        self.states.update(nfa.states)
        self.transitions.update(nfa.transitions)

        # Remove the other NFA's initial state
        del self.states[nfa.initial_state]
        nfa.initial_state = None

        return self

    def _star(self, nfa):
        """
        Return the NFA for the star operator.
        """
        new_nfa = Automaton(depth=self.depth + 1, id=self.id)
        initial = new_nfa.add_state(is_initial=True)
        final = new_nfa.add_state(is_final=True)

        new_nfa.add_transition(initial, EPS, final)
        new_nfa.add_transition(initial, EPS, nfa.initial_state)
        global_state_register[self.id][nfa.initial_state].is_initial = False

        for final_state in nfa.final_states:
            nfa.add_transition(final_state, EPS, final)
            nfa.add_transition(final_state, EPS, nfa.initial_state)
            global_state_register[self.id][final_state].is_final = False

        nfa.final_states = {}
        nfa.initial_state = None

        return new_nfa

    def _union(self, nfa):
        """
        Return the NFA for the union operator.
        """
        new_nfa = Automaton(depth=self.depth + 1, id=self.id)
        initial = new_nfa.add_state(is_initial=True)
        final = new_nfa.add_state(is_final=True)

        # Add epsilon transitions from the new initial state to the initial states of nfa1 and nfa2
        new_nfa.add_transition(initial, EPS, self.initial_state)
        new_nfa.add_transition(initial, EPS, nfa.initial_state)
        global_state_register[self.id][self.initial_state].is_initial = False
        global_state_register[self.id][nfa.initial_state].is_initial = False

        # Add epsilon transitions from the final states of nfa1 and nfa2 to the new final state
        for final_state in self.final_states:
            self.add_transition(final_state, EPS, final)
            global_state_register[self.id][final_state].is_final = False

        for final_state in nfa.final_states:
            nfa.add_transition(final_state, EPS, final)
            global_state_register[self.id][final_state].is_final = False

        self.final_states = {}
        self.initial_state = None
        nfa.final_states = {}
        nfa.initial_state = None

        return new_nfa

    def _symbol(self, symbol):
        """
        Return the NFA for the symbol operator.
        """
        nfa = Automaton(depth=self.depth + 1, id=self.id)
        initial = nfa.add_state(is_initial=True)
        final = nfa.add_state(is_final=True)

        nfa.add_transition(initial, symbol, final)

        return nfa

    def make_deterministic(self):
        """
        Make the automaton deterministic using the subset construction algorithm.
        """
        dfa_transitions, final_states = self.nfa_to_dfa(
            self.get_all_transitions(),
            self.initial_state
        )

        # Create a new Automaton (DFA).
        dfa = Automaton()
        dfa = dfa.from_transitions(dfa_transitions, '0', final_states)

        dfa.is_deterministic = True

        # Make self a DFA.
        self.is_deterministic = True
        self.states = dfa.states
        self.initial_state = '0'
        self.final_states = dfa.final_states
        self.transitions = dfa.transitions
        self.id = dfa.id

        # Build a transition table.
        transition_table = {}

        transitions = self.get_all_transitions()

        for (source, symbol, target) in transitions:
            if source not in transition_table:
                transition_table[source] = {}
            transition_table[source][symbol] = target

        self.transition_table = transition_table

    def minimize(self):
        # TODO: Minimize the DFA using the Hopcroft algorithm.
        pass

    class State:
        """
        A state in an automaton.
        """

        def __init__(self, name, is_initial=False, is_final=False, parent_nfa=None):
            """
            Create a new State.
            """
            self.name = name
            self.is_initial = is_initial
            self.is_final = is_final
            self.transitions = dict()
            self.parent_nfa = None

        def add_transition(self, symbol, target):
            """
            Add a transition from this state to another state.
            """
            if symbol not in self.transitions:
                self.transitions[symbol] = set()
            self.transitions[symbol].add(target)

        def __repr__(self):
            return f"State {self.name} (initial={self.is_initial}, final={self.is_final})"

    def from_transitions(self, transitions, initial_state, final_states):
        """
        Create a new Automaton from a list of transitions. A transition is a tuple (source, symbol, target).
        """

        # Create a new Automaton.
        automaton = Automaton()

        # Add the initial state.
        automaton.initial_state = initial_state
        automaton.states[initial_state] = Automaton.State(
            initial_state, parent_nfa=automaton, is_initial=True, is_final=initial_state in final_states)

        # Add the final states.
        for state in final_states:
            automaton.final_states.add(state)

        # Add the transitions.
        for (source, symbol, target) in transitions:
            if source not in automaton.states:
                automaton.states[source] = Automaton.State(
                    source, parent_nfa=automaton, is_initial=source == initial_state, is_final=source in final_states)
            if target not in automaton.states:
                automaton.states[target] = Automaton.State(
                    target, parent_nfa=automaton, is_final=target in final_states, is_initial=False)

            global_state_register[automaton.id][source] = automaton.states[source]
            global_state_register[automaton.id][target] = automaton.states[target]
            automaton.states[source].add_transition(symbol, target)

        return automaton

    def nfa_to_dfa(self, nfa_transitions, initial_state):
        """
        Convert an NFA to a DFA using the subset construction algorithm.
        """

        def epsilon_closure(state, transitions):
            closure = set([state])
            for (s1, symbol, s2) in transitions:
                if s1 == state and symbol == 'ε':
                    closure.update(epsilon_closure(s2, transitions))
            return closure

        def epsilon_closure_set(states, transitions):
            closure = set()
            for state in states:
                closure.update(epsilon_closure(state, transitions))
            return closure

        def move(states, symbol, transitions):
            next_states = set()
            for (s1, sym, s2) in transitions:
                if s1 in states and sym == symbol:
                    next_states.add(s2)
            return next_states

        alphabet = {symbol for (_, symbol, _)
                    in nfa_transitions if symbol != 'ε'}

        initial_closure = frozenset(
            epsilon_closure(initial_state, nfa_transitions))

        dfa_state_counter = 0
        A = [initial_closure]

        nfa_dfa_state_map = {
            initial_closure: str(dfa_state_counter)
        }

        dfa_transitions = []

        for i in A:
            for symbol in alphabet:

                move_i = move(i, symbol, nfa_transitions)
                alphabet_closure = frozenset(
                    epsilon_closure_set(move_i, nfa_transitions))

                if len(alphabet_closure) == 0:
                    continue

                if alphabet_closure not in nfa_dfa_state_map:
                    dfa_state_counter += 1
                    A.append(alphabet_closure)
                    nfa_dfa_state_map[alphabet_closure] = str(
                        dfa_state_counter)

                dfa_transitions.append(
                    (nfa_dfa_state_map[i], symbol, nfa_dfa_state_map[alphabet_closure]))

        # Add final states
        final_states = []
        for state in A:
            for final_state in self.final_states:
                if final_state in state:
                    final_states.append(nfa_dfa_state_map[state])

        return dfa_transitions, final_states

    def match(self, string):
        """
        Match a string against the automaton.
        """
        if not self.is_deterministic:
            raise Exception("Automaton is not deterministic.")

        if CONCATENATION in string or LETTER_SPECIAL_CHAR in string or DIGIT_SPECIAL_CHAR in string or ANYTHING_SPECIAL_CHAR in string:
            raise Exception(
                f"String contains either special characters {CONCATENATION}, {LETTER_SPECIAL_CHAR}, {DIGIT_SPECIAL_CHAR}, {ANYTHING_SPECIAL_CHAR} or is not a valid string.")

        # Match the string.
        current_state = self.initial_state
        for symbol in string:
            is_letter = symbol in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            is_digit = symbol in "0123456789"
            special_symbol = None

            if is_letter:
                special_symbol = LETTER_SPECIAL_CHAR
            elif is_digit:
                special_symbol = DIGIT_SPECIAL_CHAR

            # Match special characters if there was no match for the current symbol.
            if symbol not in self.transition_table[current_state]:
                if special_symbol in self.transition_table[current_state]:
                    symbol = special_symbol
                elif ANYTHING_SPECIAL_CHAR in self.transition_table[current_state]:
                    symbol = ANYTHING_SPECIAL_CHAR
                else:
                    return False

            current_state = self.transition_table[current_state][symbol]

        return current_state in self.final_states

    def transition(self, state, symbol):
        """
        Transition from a state to another state given a symbol.
        """

        # If state is an object, get its name.
        if isinstance(state, Automaton.State):
            state = state.name

        is_letter = symbol in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        is_digit = symbol in "0123456789"

        special_symbol = None

        if is_letter:
            special_symbol = LETTER_SPECIAL_CHAR
        elif is_digit:
            special_symbol = DIGIT_SPECIAL_CHAR

        if state not in self.transition_table:
            return None

        if symbol not in self.transition_table[state]:
            # Match special characters if there was no match for the current symbol.
            if special_symbol in self.transition_table[state]:
                symbol = special_symbol
            elif ANYTHING_SPECIAL_CHAR in self.transition_table[state]:
                symbol = ANYTHING_SPECIAL_CHAR
            else:
                return None

        state_name = self.transition_table[state][symbol]
        return global_state_register[self.id][state_name]

    def get_initial_state(self):
        """
        Return the object of the initial state.
        """
        return self.states[self.initial_state]


if __name__ == "__main__":

    automaton = Automaton(regex='a(b|c)*d')
    automaton.make_deterministic()
    automaton.visualize()
    print(automaton.get_all_transitions())
    print(automaton.match('abbbcd'))
