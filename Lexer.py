from Automaton import Automaton, DIGIT_SPECIAL_CHAR, LETTER_SPECIAL_CHAR, ANYTHING_SPECIAL_CHAR
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


if __name__ == "__main__":

    keywords = ["int", "float", "string", "for", "if",
                "else", "while", "return", "read", "write", "void"]
    escaped_symbols = ["\+", "-", "\*", "/", "<", "<=", ">", ">=", "==", "!=",
                       "=", ";", ",", "\"", "\(", "\)", "[", "]", "{", "}", "/\*", "\*/"]  # . is not included because it is not a symbol in the grammar
    symbols = ["+", "-", "*", "/", "<", "<=", ">", ">=", "==", "!=",
               "=", ";", ",",  "(", ")", "[", "]", "{", "}", "/*", "*/"]  # " and . are not included because they are not symbols in the grammar
    whitespaces = (" ", "\n", "\t")

    # Convert the regex to a regex that this program can understand (this regex engine can't understand ranges)'

    whitespaces_regex = f'({"|".join(whitespaces)})+'
    keywords_regex = f'({"|".join(keywords)})'
    symbols_regex = f'({"|".join(escaped_symbols)})'

    # Same as "[a-zA-Z]([a-zA-Z]|[0-9])*""
    identifier_regex = f"{LETTER_SPECIAL_CHAR}({LETTER_SPECIAL_CHAR}|{DIGIT_SPECIAL_CHAR})*"
    # Same as "\".*\""
    string_regex = f"\"({ANYTHING_SPECIAL_CHAR})*\""
    # Same as "[0-9]+(\.[0-9]+)?"
    number_regex = f"{DIGIT_SPECIAL_CHAR}+(.{DIGIT_SPECIAL_CHAR}+)?"
    # Same as "/\*.*\*/"
    comment_regex = f"/\*({ANYTHING_SPECIAL_CHAR})*\*/"

    # Build the automata

    keywords_automaton = LexerAutomaton(keywords_regex, "keywords")
    identifier_automaton = LexerAutomaton(identifier_regex, "identifiers")
    string_automaton = LexerAutomaton(string_regex, "strings")
    number_automaton = LexerAutomaton(number_regex, "numbers")
    comment_automaton = LexerAutomaton(comment_regex, "comments")

    symbols_automaton = LexerAutomaton(symbols_regex, "symbols")
    symbols_automaton.automaton.visualize()
    whitespace_automaton = LexerAutomaton(whitespaces_regex, "whitespaces")

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

        while i < len(program):
            longest_match = None
            longest_match_token_type = None

            for automaton in automata:
                match = automaton.find_longest_match(program[i:])
                if longest_match is None or len(match) > len(longest_match):
                    longest_match = match
                    longest_match_token_type = automaton.name

            if longest_match is None or len(longest_match) == 0:
                raise Exception("Syntax error: invalid token at",
                                i, "artifact: ", program[i:min(len(program), i+10)])

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
                    raise Exception(
                        "Syntax error: delimiter expected after", longest_match, "but", program[i+len(longest_match)], "found at", i+len(longest_match))

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

        return symbol_table, scanner_output

    # Fix:
    # 8. /* in the middle of the string should throw an error
    # 7. How to handle +=, two operators together

    # Read programs from ./example_programs folder
    program_names = ["program_1.c", "program_2_error.c", "program_3_error.c",
                     "program_4_error.c", "program_5_error.c", "program_6_error.c", "program_7_error.c", "program_8_error.c", "program_9_error.c"]
    programs = []
    for program_name in program_names:
        with open(f"./example_programs/{program_name}", "r") as program_file:
            programs.append(program_file.read())

    symbol_table, scanner_output = tokenize(programs[8])

    print("Token ids:")
    for token_name, token_id in token_ids.items():
        print(token_name, token_id)

    print("\nSymbol table:")
    print(symbol_table)
    for token_type, token_values in symbol_table.items():
        print(token_type)
        for position, value in token_values.items():
            print(position, value)

    print("\nScanner output:")
    for i in scanner_output:
        print(i)

    def recognize_token(token_id, symbol_table_position=None):
        for token_name, token_id_ in token_ids.items():
            if token_id_ == token_id:
                if token_name == "identifiers" or token_name == "strings" or token_name == "numbers":
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
