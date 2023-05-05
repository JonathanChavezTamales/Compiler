class Lexer:
    def __init__(self, keywords, special_symbols, identifier_regex, string_regex, number_regex) -> None:
        self.keywords = keywords
        self.special_symbols = special_symbols
        self.identifier_regex = identifier_regex
        self.string_regex = string_regex
        self.number_regex = number_regex
        self.automaton = None

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


# [a-z]
aToz_regex = f"({'|'.join([chr(i) for i in range(ord('a'), ord('z') + 1)])})"
# [A-Z]
AToZ_regex = f"({'|'.join([chr(i) for i in range(ord('A'), ord('Z') + 1)])})"
# [0-9]
zeroToNine_regex = f"({'|'.join([chr(i) for i in range(ord('0'), ord('9') + 1)])})"

if __name__ == "__main__":

    keywords = ["int", "float", "string", "for", "if",
                "else", "while", "return", "read", "write", "void"]
    symbols = ["+", "-", "*", "/", "<", "<=", ">", ">=", "==", "!=",
               "=", ";", ",", "\"", ".", "(", ")", "[", "]", "{", "}", "/*", "*/"]

    # Convert the regex to a regex that this program can understand (this regex engine can't understand ranges)

    # Same as "[a-zA-Z]([a-zA-Z]|[0-9])*""
    identifier_regex = f"({aToz_regex}|{AToZ_regex})({aToz_regex}|{AToZ_regex}|{zeroToNine_regex})*"
    # Same as "\".*\""
    string_regex = f"\"({aToz_regex}|{AToZ_regex}|{zeroToNine_regex})*\""
    # Same as "[0-9]+(\.[0-9]+)?"
    number_regex = f"{zeroToNine_regex}+(.{zeroToNine_regex}+)?"

    print(identifier_regex)

    # lexer = Lexer(keywords, symbols, identifier_regex,
    #               string_regex, number_regex)
