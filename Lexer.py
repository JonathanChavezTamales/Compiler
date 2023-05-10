from Automaton import Automaton, DIGIT_SPECIAL_CHAR, LETTER_SPECIAL_CHAR
import os


class Lexer:
    def __init__(self, keywords, special_symbols, identifier_regex, string_regex, number_regex) -> None:
        self.keywords = keywords
        self.special_symbols = special_symbols
        self.identifier_regex = identifier_regex
        self.string_regex = string_regex
        self.number_regex = number_regex
        self.automaton = None

        self.build_automaton()

    def tokenize(self, input_string):
        """Return the sequence of tokens corresponding to the input string."""

        tokens = []

        automaton = self.build_automaton()

    def build_automaton(self):
        """Build the automaton that will tokenize the input string."""

        keywords_automaton = Automaton()
        special_symbols_automaton = Automaton()
        identifier_automaton = Automaton()
        string_automaton = Automaton()
        number_automaton = Automaton()


class LexerAutomaton:
    """
    This class is used to build an automaton that will tokenize the input string.
    It's a stateful class, so it will keep track of the current state of the automaton during the tokenization.
    """

    def __init__(self, regex, name) -> None:
        self.regex = regex
        self.name = name
        self.automaton = Automaton(regex)
        self.automaton.make_deterministic()
        self.initial_state = self.automaton.get_initial_state()

        self.current_state = self.initial_state
        self.stack = []

    def transition(self, symbol):
        """Return the next state of the automaton."""
        if self.current_state is None:
            return None

        next_state = self.automaton.transition(self.current_state, symbol)

        self.current_state = next_state
        return next_state

    def is_accepting(self):
        if self.current_state is None:
            return False
        return self.current_state.is_final

    def reset(self):
        self.current_state = self.initial_state

    def find_longest_match(self, input_string):
        """Return the longest match in the input string."""
        self.reset()

        substring = ""
        last_accepting_substring = ""

        for symbol in input_string:
            next_state = self.transition(symbol)
            if next_state is None:
                break
            substring += symbol
            if self.is_accepting():
                last_accepting_substring = substring

        self.reset()

        return last_accepting_substring


# [a-z]
aToz_regex = f"({'|'.join([chr(i) for i in range(ord('a'), ord('z') + 1)])})"
# [A-Z]
AToZ_regex = f"({'|'.join([chr(i) for i in range(ord('A'), ord('Z') + 1)])})"
# [0-9]
zeroToNine_regex = f"({'|'.join([chr(i) for i in range(ord('0'), ord('9') + 1)])})"

if __name__ == "__main__":

    keywords = ["int", "float", "string", "for", "if",
                "else", "while", "return", "read", "write", "void"]
    escaped_symbols = ["\+", "-", "\*", "/", "<", "<=", ">", ">=", "==", "!=",
                       "=", ";", ",", "\"", ".", "\(", "\)", "[", "]", "{", "}", "/\*", "\*/"]
    symbols = ["+", "-", "*", "/", "<", "<=", ">", ">=", "==", "!=",
               "=", ";", ",", "\"", ".", "(", ")", "[", "]", "{", "}", "/*", "*/"]
    whitespaces = (" ", "\n", "\t")

    # Convert the regex to a regex that this program can understand (this regex engine can't understand ranges)'

    whitespaces_regex = f'({"|".join(whitespaces)})+'
    keywords_regex = f'({"|".join(keywords)})'
    symbols_regex = f'({"|".join(escaped_symbols)})'

    # Same as "[a-zA-Z]([a-zA-Z]|[0-9])*""
    identifier_regex = f"{LETTER_SPECIAL_CHAR}({LETTER_SPECIAL_CHAR}|{DIGIT_SPECIAL_CHAR})*"
    # Same as "\".*\""
    string_regex = f"\"({LETTER_SPECIAL_CHAR}|({'|'.join(whitespaces)})|{DIGIT_SPECIAL_CHAR})*\""
    # Same as "[0-9]+(\.[0-9]+)?"
    number_regex = f"{DIGIT_SPECIAL_CHAR}+(.{DIGIT_SPECIAL_CHAR}+)?"
    # Same as "/\*.*\*/"
    comment_regex = f"/\*({LETTER_SPECIAL_CHAR}|{DIGIT_SPECIAL_CHAR})*\*/"

    # Build the automata

    keywords_automaton = LexerAutomaton(keywords_regex, "keywords")
    identifier_automaton = LexerAutomaton(identifier_regex, "identifiers")
    string_automaton = LexerAutomaton(string_regex, "strings")
    number_automaton = LexerAutomaton(number_regex, "numbers")
    comment_automaton = LexerAutomaton(comment_regex, "comments")

    print(string_automaton.automaton.match("\"df df\""))

    symbols_automaton = LexerAutomaton(symbols_regex, "symbols")
    whitespace_automaton = LexerAutomaton(whitespaces_regex, "whitespaces")

    # Fix:
    # 1. Comments are not working
    # 2. 0. and .8 ERROR
    # 3. 3abc4 ERROR
    # 4. 3.4.5 ERROR
    # 5. !abe = 8, -jojo = 3 ERROR
    # 5. @ in the middle of the string ERROR
    # 6. /* in the middle of the string ERROR
    # 7. 6 += i ERROR

    # Read programs from ./example_programs folder
    program_names = ["program_1.c", "program_2_error.c", "program_3_error.c",
                     "program_4_error.c", "program_5_error.c", "program_6_error.c"]
    programs = []
    for program_name in program_names:
        with open(f"./example_programs/{program_name}", "r") as program_file:
            programs.append(program_file.read())

    # Generate the token ids
    token_ids = {}
    id_counter = 0

    for keyword in keywords:
        token_ids[keyword] = id_counter
        id_counter += 1
    for symbol in symbols:
        token_ids[symbol] = id_counter
        id_counter += 1
    token_ids["identifiers"] = id_counter
    id_counter += 1
    token_ids["strings"] = id_counter
    id_counter += 1
    token_ids["numbers"] = id_counter
    id_counter += 1
    token_ids["comments"] = id_counter
    id_counter += 1

    def tokenize(program):
        symbol_table = {}  # {token_type: {token_value: position_in_symbol_table}}}
        scanner_output = []  # [(token_id, position_in_symbol_table)]

        automata = [keywords_automaton, identifier_automaton, string_automaton,
                    number_automaton, comment_automaton, symbols_automaton, whitespace_automaton]

        i = 0

        keywords_automaton.find_longest_match("\"Hello world\"")

        while i < len(program):
            longest_match = None
            longest_match_token_type = None

            for automaton in automata:
                match = automaton.find_longest_match(program[i:])
                print(program[i:], automaton.name, match)
                if longest_match is None or len(match) > len(longest_match):
                    longest_match = match
                    longest_match_token_type = automaton.name

            if longest_match is None:
                raise Exception("No automaton found a match")

            if longest_match_token_type == "whitespaces":
                i += len(longest_match)
                continue

            remaining = program[i+len(longest_match):]
            longest_remaining_match = None
            longest_remaining_match_token_type = None

            if remaining:
                for automaton in automata:
                    match = automaton.find_longest_match(remaining)
                    if longest_remaining_match is None or len(match) > len(longest_remaining_match):
                        longest_remaining_match = match
                        longest_remaining_match_token_type = automaton.name

                # Tokens should be separated by whitespaces or symbols, so if the longest match is not followed by a whitespace or a symbol, then it is not a valid token
                if longest_match_token_type in ["identifiers", "strings", "numbers"] and longest_remaining_match_token_type not in ["whitespaces", "symbols"]:
                    raise Exception("Invalid token", longest_match,
                                    longest_remaining_match)

            # Add the token to the symbol table
            if longest_match_token_type == "identifiers" or longest_match_token_type == "strings" or longest_match_token_type == "numbers":
                if longest_match_token_type not in symbol_table:
                    symbol_table[longest_match_token_type] = {}

                if longest_match not in symbol_table[longest_match_token_type]:
                    symbol_table[longest_match_token_type][len(
                        symbol_table[longest_match_token_type])] = longest_match

            def get_symbol_table_position(token_type, token_value):
                if token_type not in symbol_table:
                    return None
                else:
                    # {position_in_symbol_table: token_value}
                    symbol_table_positions = symbol_table[token_type]
                    for position, value in symbol_table_positions.items():
                        if value == token_value:
                            return position

            def get_token_id(token_type, token_value):
                if token_type == "identifiers" or token_type == "strings" or token_type == "numbers" or token_type == "comments":
                    return token_ids[token_type]
                elif token_type == "symbols" or token_type == "keywords":
                    return token_ids[token_value]
                else:
                    raise Exception("Invalid token type")

            scanner_output.append((get_token_id(longest_match_token_type, longest_match), get_symbol_table_position(
                longest_match_token_type, longest_match)))

            i += len(longest_match)

            print(longest_match)
            print(longest_match_token_type)

        return symbol_table, scanner_output

    symbol_table, scanner_output = tokenize(programs[0])

    print(programs[0])

    print("Token ids:")
    print(token_ids)
    print("\nSymbol table:")
    print(symbol_table)
    print("\nScanner output:")
    print(scanner_output)

    def recognize_token(token_id, symbol_table_position=None):
        for token_name, token_id_ in token_ids.items():
            if token_id_ == token_id:
                if token_name == "identifiers" or token_name == "strings" or token_name == "numbers" or token_name == "comments":
                    return symbol_table[token_name][symbol_table_position]
                else:
                    return token_name
        return None

    recognized_tokens = []
    for token_id, symbol_table_position in scanner_output:
        recognized_tokens.append(recognize_token(
            token_id, symbol_table_position))

    print("\nRecognized tokens:")
    print(recognized_tokens)
