from graphviz import Digraph


EPS = 'ε'  # Epsilon symbol (empty string)
CONCATENATION = '·'  # Concatenation symbol
global_state_register = dict()  # {state_name: State}
global_state_counter = 0
special_regex_chars = {'|', '+', '*', '?', '(', ')', CONCATENATION}


class Automaton:
    def __init__(self, regex=None, depth=0):
        self.depth = depth
        self.alphabet = set()
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

        def epsilon_closure(state):
            # Return the epsilon closure of the given state.
            closure = set([state])
            stack = [state]

            while stack:
                current_state = stack.pop()
                current_state_obj = global_state_register[current_state]

                if EPS in current_state_obj.transitions:
                    for next_state in current_state_obj.transitions[EPS]:
                        if next_state not in closure:
                            closure.add(next_state)
                            stack.append(next_state)

            return closure

        def move(states, symbol):
            # Return the set of states reachable from the given set of states
            # using the given symbol.
            reachable = set()

            for state in states:
                state_obj = global_state_register[state]
                if symbol in state_obj.transitions:
                    for next_state in state_obj.transitions[symbol]:
                        reachable.add(next_state)

            return reachable

        print(self.get_all_transitions())

        # Create a new Automaton (DFA).
        dfa = Automaton()
        dfa.is_deterministic = True

    def minimize(self):
        # Minimize the DFA using the Hopcroft algorithm.
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
    automaton = Automaton(regex='(a|b)*abb')
    print(automaton.get_all_transitions())
    automaton.make_deterministic()
    automaton.visualize()

    def nfa_to_dfa(nfa_transitions, initial_state):

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
            initial_closure: dfa_state_counter
        }

        dfa_transitions = []

        for i in A:
            found_transitions = False
            for symbol in alphabet:

                move_i = move(i, symbol, nfa_transitions)
                alphabet_closure = frozenset(
                    epsilon_closure_set(move_i, nfa_transitions))

                if len(alphabet_closure) == 0:
                    continue

                found_transitions = True

                if alphabet_closure not in nfa_dfa_state_map:
                    dfa_state_counter += 1
                    A.append(alphabet_closure)
                    nfa_dfa_state_map[alphabet_closure] = dfa_state_counter

                print("i: " + str(nfa_dfa_state_map[i]))
                print(str(i))
                print("symbol: " + symbol)
                print("alphabet_closure: " +
                      str(nfa_dfa_state_map[alphabet_closure]))
                print(str(alphabet_closure))

                dfa_transitions.append(
                    (nfa_dfa_state_map[i], symbol, nfa_dfa_state_map[alphabet_closure]))

            if not found_transitions:
                break

        print(nfa_dfa_state_map)
        print("IIIII")
        print(dfa_transitions)

        # Start of subset construction
        dfa_transitions = set()

        return dfa_transitions

    dfa_transitions = nfa_to_dfa(
        automaton.get_all_transitions(), automaton.initial_state)
    print(dfa_transitions)
