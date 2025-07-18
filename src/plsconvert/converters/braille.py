from pathlib import Path
from plsconvert.converters.abstract import Converter


class brailleConverter(Converter):
    def __init__(self):
        super().__init__()
        self.mapping = {
            ' ': '⠀',
            '!': '⠮',
            '"': '⠐',
            '#': '⠼',
            '$': '⠫',
            '%': '⠩',
            '&': '⠯',
            "'": '⠄',
            '(': '⠷',
            ')': '⠾',
            '*': '⠡',
            '+': '⠬',
            ',': '⠠',
            '-': '⠤',
            '.': '⠨',
            '/': '⠌',
            '0': '⠴',
            '1': '⠂',
            '2': '⠆',
            '3': '⠒',
            '4': '⠲',
            '5': '⠢',
            '6': '⠖',
            '7': '⠶',
            '8': '⠦',
            '9': '⠔',
            ':': '⠱',
            ';': '⠰',
            '<': '⠣',
            '=': '⠿',
            '>': '⠜',
            '?': '⠹',
            '@': '⠈',
            'a': '⠁',
            'b': '⠃',
            'c': '⠉',
            'd': '⠙',
            'e': '⠑',
            'f': '⠋',
            'g': '⠛',
            'h': '⠓',
            'i': '⠊',
            'j': '⠚',
            'k': '⠅',
            'l': '⠇',
            'm': '⠍',
            'n': '⠝',
            'o': '⠕',
            'p': '⠏',
            'q': '⠟',
            'r': '⠗',
            's': '⠎',
            't': '⠞',
            'u': '⠥',
            'v': '⠧',
            'w': '⠺',
            'x': '⠭',
            'y': '⠽',
            'z': '⠵',
            '[': '⠪',
            '\\': '⠳',
            ']': '⠻',
            '^': '⠘',
            '_': '⠸'
        }
        # Create reverse mapping for Braille to text conversion
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}

    def convert_text(self, input_str: str, to_braille: bool = True) -> str:
        """
        Convert text to/from Braille
        
        Args:
            input_str: Input string to convert
            to_braille: If True, convert text to Braille; if False, convert Braille to text
        
        Returns:
            Converted string
        """
        out = ""
        if to_braille:
            # Convert text to Braille
            for input_char in input_str:
                if input_char.lower() in self.mapping:
                    out += self.mapping[input_char.lower()]
                elif input_char in self.mapping:
                    out += self.mapping[input_char]
                else:
                    # If character not in mapping, keep original
                    out += input_char
        else:
            # Convert Braille to text
            for braille_char in input_str:
                if braille_char in self.reverse_mapping:
                    out += self.reverse_mapping[braille_char]
                else:
                    # If Braille character not in mapping, keep original
                    out += braille_char
        return out

    def adjConverter(self) -> dict[str, list[list[str]]]:
        return {
            "braille": ["txt"],
            "txt": ["braille"],
        }

    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        with open(input, 'r', encoding='utf-8') as file:
            content = file.read()
        
        if input_extension == "txt" and output_extension == "braille":
            # Convert text to Braille
            converted_content = self.convert_text(content, to_braille=True)
        elif input_extension == "braille" and output_extension == "txt":
            # Convert Braille to text
            converted_content = self.convert_text(content, to_braille=False)
        else:
            raise ValueError(f"Unsupported conversion: {input_extension} to {output_extension}")
        
        with open(output, 'w', encoding='utf-8') as file:
            file.write(converted_content)

    def metDependencies(self) -> bool:
        return True
