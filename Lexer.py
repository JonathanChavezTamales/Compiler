from Automaton import Automaton, DIGIT_SPECIAL_CHAR, LETTER_SPECIAL_CHAR, ANYTHING_SPECIAL_CHAR
import os


class Lexer:
    def __init__(self, keywords, symbols, escaped_symbols, whitespaces, identifier_regex, string_regex, number_regex, comments_regex) -> None:
        self.keywords = keywords
        self.symbols = symbols
        self.whitespaces = whitespaces

        self.whitespaces_regex = f'({"|".join(whitespaces)})+'
        self.keywords_regex = f'({"|".join(keywords)})'
        self.symbols_regex = f'({"|".join(escaped_symbols)})'

        self.identifier_regex = identifier_regex
        self.string_regex = string_regex
        self.number_regex = number_regex
        self.comments_regex = comments_regex

        self.build_token_id_table()

        self.build_automaton()

    def tokenize(self, program):
        symbol_table = {}  # {token_type: {token_value: position_in_symbol_table}}}
        scanner_output = []  # [(token_id, position_in_symbol_table)]

        automata = [self.keywords_automaton, self.identifier_automaton, self.string_automaton,
                    self.number_automaton, self.comment_automaton, self.symbols_automaton, self.whitespace_automaton]

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
                    return self.token_ids[token_type]
                elif token_type == "symbols" or token_type == "keywords":
                    return self.token_ids[token_value]
                else:
                    raise Exception("Invalid token type")

            scanner_output.append((get_token_id(longest_match_token_type, longest_match), get_symbol_table_position(
                longest_match_token_type, longest_match)))

            i += len(longest_match)

        return symbol_table, scanner_output

    def build_token_id_table(self):
        # Generate the token ids
        self.token_ids = {}
        id_counter = 0

        for keyword in self.keywords:
            self.token_ids[keyword] = id_counter
            id_counter += 1
        for symbol in self.symbols:
            self.token_ids[symbol] = id_counter
            id_counter += 1
        self.token_ids["identifiers"] = id_counter
        id_counter += 1
        self.token_ids["strings"] = id_counter
        id_counter += 1
        self.token_ids["numbers"] = id_counter
        id_counter += 1
        self.token_ids["comments"] = id_counter
        id_counter += 1

    def recognize_token(self, token_id, symbol_table_position=None):
        for token_name, token_id_ in self.token_ids.items():
            if token_id_ == token_id:
                if token_name == "identifiers" or token_name == "strings" or token_name == "numbers":
                    return symbol_table[token_name][symbol_table_position]
                else:
                    return token_name
        return None

    def build_automaton(self):
        """Build the automaton that will tokenize the input string."""

        self.keywords_automaton = LexerAutomaton(
            self.keywords_regex, "keywords")
        print(self.symbols_regex)
        self.symbols_automaton = LexerAutomaton(self.symbols_regex, "symbols")
        self.identifier_automaton = LexerAutomaton(
            self.identifier_regex, "identifiers")
        self.string_automaton = LexerAutomaton(self.string_regex, "strings")
        self.number_automaton = LexerAutomaton(self.number_regex, "numbers")
        self.comment_automaton = LexerAutomaton(
            self.comments_regex, "comments")
        self.whitespace_automaton = LexerAutomaton(
            self.whitespaces_regex, "whitespaces")


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

    # Same as "[a-zA-Z]([a-zA-Z]|[0-9])*""
    identifier_regex = f"{LETTER_SPECIAL_CHAR}({LETTER_SPECIAL_CHAR}|{DIGIT_SPECIAL_CHAR})*"
    # Same as "\".*\""
    string_regex = f"\"({ANYTHING_SPECIAL_CHAR})*\""
    # Same as "[0-9]+(\.[0-9]+)?"
    number_regex = f"{DIGIT_SPECIAL_CHAR}+(.{DIGIT_SPECIAL_CHAR}+)?"
    # Same as "/\*.*\*/"
    comment_regex = f"/\*({ANYTHING_SPECIAL_CHAR})*\*/"

    lexer = Lexer(keywords, symbols, escaped_symbols, whitespaces, identifier_regex,
                  string_regex, number_regex, comment_regex)

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

    symbol_table, scanner_output = lexer.tokenize(programs[7])
    token_id_table = lexer.token_ids

    print("Token ids:")
    for token_name, token_id in token_id_table.items():
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

    recognized_tokens = []
    for token_id, symbol_table_position in scanner_output:
        recognized_tokens.append(lexer.recognize_token(
            token_id, symbol_table_position))

    print("\nRecognized tokens:")
    print(recognized_tokens)
