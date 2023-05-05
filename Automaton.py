from graphviz import Digraph


EPS = 'ε'  # Epsilon symbol (empty string)
CONCATENATION = '·'  # Concatenation symbol
global_state_register = dict()  # {state_name: State}
global_state_counter = 0
special_regex_chars = {'|', '+', '*', '?', '(', ')', CONCATENATION}


class Automaton:
    def __init__(self, regex=None, depth=0):
        self.depth = depth
        if regex:
            if CONCATENATION in regex:
                raise Exception(
                    f"Symbol '{CONCATENATION}' is not allowed in the regex")

            preprocessed_regex = self._preprocess_regex(regex)
            self.from_regex(preprocessed_regex)
        else:
            self.states = dict()  # {state_name: State}
            self.is_deterministic = False
            self.initial_state = None
            self.final_states = set()
            self.transitions = dict()

    def _preprocess_regex(self, regex):
        processed_regex = []  # {literal: char, escaped: bool}

        unary_operators = {'+', '*', '?'}
        parentheses = {'(', ')'}
        binary_operators = {'|'}

        def is_other(literal):
            return literal["char"] not in (unary_operators | binary_operators | parentheses) or (literal["char"] in (unary_operators | binary_operators | parentheses) and literal["escaped"])

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
        # Convert a regular expression to an NFA using Thompson's construction.

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

        global_state_register[state_name] = state
        return state_name

    def add_transition(self, source, symbol, target):
        if source not in self.transitions:
            self.transitions[source] = dict()
        if symbol not in self.transitions[source]:
            self.transitions[source][symbol] = set()

        self.transitions[source][symbol].add(target)
        self.states[source].add_transition(symbol, target)

    def visualize(self):
        transitions = self.get_all_transitions()
        dot = Digraph(comment='NFA')

        # Add states as nodes
        for state_name, state in self.states.items():
            attributes = {}
            if state.is_initial:
                attributes['shape'] = 'pentagon'
            if state.is_final:
                attributes['shape'] = 'doublecircle'
            dot.node(state_name, state.name, **attributes)

        # Add transitions as edges
        for src, label, dest in transitions:
            dot.edge(src, dest, label=label)

        # Render and display the graph
        dot.view()

    def __repr__(self):
        return f"Automaton(states={self.states}, initial_state={self.initial_state}, final_states={self.final_states}, transitions={self.transitions})"

    def get_all_transitions(self):
        # Return all transitions using BFS to traverse the states
        visited = set()
        queue = [self.initial_state]

        transitions = set()

        while queue:
            state = queue.pop(0)
            state = global_state_register[state]
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
        pass

    def _question(self, nfa):
        pass

    def _concatenate(self, nfa):
        for final_state in self.final_states:
            self.add_transition(final_state, EPS, nfa.initial_state)
            global_state_register[final_state].is_final = False

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
        new_nfa = Automaton(depth=self.depth + 1)
        initial = new_nfa.add_state(is_initial=True)
        final = new_nfa.add_state(is_final=True)

        new_nfa.add_transition(initial, EPS, final)
        new_nfa.add_transition(initial, EPS, nfa.initial_state)
        global_state_register[nfa.initial_state].is_initial = False

        for final_state in nfa.final_states:
            nfa.add_transition(final_state, EPS, final)
            nfa.add_transition(final_state, EPS, nfa.initial_state)
            global_state_register[final_state].is_final = False

        nfa.final_states = {}
        nfa.initial_state = None

        return new_nfa

    def _union(self, nfa):
        new_nfa = Automaton(depth=self.depth + 1)
        initial = new_nfa.add_state(is_initial=True)
        final = new_nfa.add_state(is_final=True)

        # Add epsilon transitions from the new initial state to the initial states of nfa1 and nfa2
        new_nfa.add_transition(initial, EPS, self.initial_state)
        new_nfa.add_transition(initial, EPS, nfa.initial_state)
        global_state_register[self.initial_state].is_initial = False
        global_state_register[nfa.initial_state].is_initial = False

        # Add epsilon transitions from the final states of nfa1 and nfa2 to the new final state
        for final_state in self.final_states:
            self.add_transition(final_state, EPS, final)
            global_state_register[final_state].is_final = False

        for final_state in nfa.final_states:
            nfa.add_transition(final_state, EPS, final)
            global_state_register[final_state].is_final = False

        self.final_states = {}
        self.initial_state = None
        nfa.final_states = {}
        nfa.initial_state = None

        return new_nfa

    def _symbol(self, symbol):
        nfa = Automaton(depth=self.depth + 1)
        initial = nfa.add_state(is_initial=True)
        final = nfa.add_state(is_final=True)

        nfa.add_transition(initial, symbol, final)

        return nfa

    def make_deterministic(self):
        # Convert the NFA to a DFA using the subset construction.
        pass

    def union(self, nfa1, nfa2):
        # Union of two NFAs
        pass

    class State:
        def __init__(self, name, is_initial=False, is_final=False, parent_nfa=None):
            self.name = name
            self.is_initial = is_initial
            self.is_final = is_final
            self.transitions = dict()
            self.parent_nfa = None

        def add_transition(self, symbol, target):
            if symbol not in self.transitions:
                self.transitions[symbol] = set()
            self.transitions[symbol].add(target)

        def __repr__(self):
            return f"State {self.name} (initial={self.is_initial}, final={self.is_final})"


if __name__ == "__main__":

    # Example 1: a simple automaton that recognizes (a|b)*abb
    # automaton = Automaton()
    # automaton.add_state(automaton.State(0, is_initial=True))
    # automaton.add_state(automaton.State(1))
    # automaton.add_state(automaton.State(2))
    # automaton.add_state(automaton.State(3, is_final=True))

    # automaton.add_transition(0, 'a', 0)
    # automaton.add_transition(0, 'b', 0)
    # automaton.add_transition(0, 'a', 1)
    # automaton.add_transition(1, 'b', 2)
    # automaton.add_transition(2, 'b', 3)

    # automaton.visualize()

    # Example 2: a simple automaton that recognizes (a|b)*abb directly from a regular expression
    automaton = Automaton(regex='a(b|c)*d')
    # print(automaton)
    # print(automaton.get_all_transitions())
    automaton.visualize()

    # # Example 3: a simple automaton that recognizes (a|b)*abb directly from a regular expression and then converts it to a DFA
    # automaton = Automaton()
    # automaton.from_regex('(a|b)*abb')
    # automaton.make_deterministic()
    # automaton.visualize()
