from pathlib import Path
from plsconvert.converters.abstract import Converter
from plsconvert.converters.registry import register_converter
from plsconvert.utils.graph import ConversionAdj
from plsconvert.utils.graph import conversionFromToAdj
from plsconvert.utils.dependency import Dependencies


@register_converter
class brailleConverter(Converter):
    """
    Braille converter using Unicode Braille patterns.
    """

    @property
    def name(self) -> str:
        return "Braille Converter"

    @property
    def dependencies(self) -> Dependencies:
        return Dependencies.empty()

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

    def convert_to_svg(self, braille_str: str) -> str:
        """
        Convert Braille text to SVG visual representation
        
        Args:
            braille_str: Braille string to convert
            
        Returns:
            SVG string representation
        """
        # Braille Unicode range starts at U+2800
        braille_base = 0x2800
        dot_size = 8
        cell_width = 30
        cell_height = 40
        margin = 10
        
        # Calculate SVG dimensions
        chars_per_line = 30
        lines = []
        current_line = ""
        
        for char in braille_str:
            if char == '\n':
                if current_line:
                    lines.append(current_line)
                    current_line = ""
            else:
                current_line += char
                if len(current_line) >= chars_per_line:
                    lines.append(current_line)
                    current_line = ""
        
        if current_line:
            lines.append(current_line)
        
        svg_width = chars_per_line * cell_width + 2 * margin
        svg_height = len(lines) * cell_height + 2 * margin
        
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">
<style>
.dot-filled {{ fill: #808080; }}
.dot-empty {{ fill: none; stroke: #ccc; stroke-width: 1; }}
.braille-char {{ font-family: monospace; font-size: 12px; fill: #666; }}
</style>
'''

        for line_idx, line in enumerate(lines):
            for char_idx, braille_char in enumerate(line):
                if braille_char == ' ':
                    continue
                    
                # Get the Braille pattern
                char_code = ord(braille_char) - braille_base
                
                # Calculate position
                x = margin + char_idx * cell_width
                y = margin + line_idx * cell_height
                
                # Draw dots for this character
                # Braille dot positions (1-indexed):
                # 1 4
                # 2 5  
                # 3 6
                dot_positions = [
                    (x + 8, y + 8),   # dot 1
                    (x + 8, y + 18),  # dot 2
                    (x + 8, y + 28),  # dot 3
                    (x + 18, y + 8),  # dot 4
                    (x + 18, y + 18), # dot 5
                    (x + 18, y + 28), # dot 6
                ]
                
                # Check which dots are raised
                for dot_num in range(6):
                    dot_x, dot_y = dot_positions[dot_num]
                    is_raised = (char_code & (1 << dot_num)) != 0
                    
                    if is_raised:
                        svg_content += f'<circle cx="{dot_x}" cy="{dot_y}" r="{dot_size//2}" class="dot-filled"/>\n'
        
        svg_content += '</svg>'
        return svg_content

    def adjConverter(self) -> ConversionAdj:
        return conversionFromToAdj(["braille"], ["txt", "svg"]) + conversionFromToAdj(["txt"], ["braille"])

    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        if input_extension == "txt" and output_extension == "braille":
            # Read text file and convert to Braille
            with open(input, 'r', encoding='utf-8') as file:
                content = file.read()
            converted_content = self.convert_text(content, to_braille=True)
            # Write as binary braille file
            with open(output, 'wb') as file:
                file.write(converted_content.encode('utf-8'))
                
        elif input_extension == "braille" and output_extension == "txt":
            # Read binary braille file and convert to text
            try:
                # First try to read as UTF-8 text (for text-based braille files)
                with open(input, 'r', encoding='utf-8') as file:
                    content = file.read()
            except UnicodeDecodeError:
                # If that fails, read as binary and try to decode
                with open(input, 'rb') as file:
                    binary_content = file.read()
                    try:
                        content = binary_content.decode('utf-8')
                    except UnicodeDecodeError:
                        # If still can't decode, treat as raw binary data
                        # Convert binary data to braille characters
                        content = ""
                        for byte in binary_content:
                            if byte < 0x80:  # ASCII range
                                content += chr(0x2800 + byte)  # Map to braille range
                            else:
                                content += chr(0x2800 + (byte % 64))  # Map to braille range
            converted_content = self.convert_text(content, to_braille=False)
            with open(output, 'w', encoding='utf-8') as file:
                file.write(converted_content)
                
        elif input_extension == "braille" and output_extension == "svg":
            # Read binary braille file and convert to SVG
            try:
                # First try to read as UTF-8 text
                with open(input, 'r', encoding='utf-8') as file:
                    content = file.read()
            except UnicodeDecodeError:
                # If that fails, read as binary and try to decode
                with open(input, 'rb') as file:
                    binary_content = file.read()
                    try:
                        content = binary_content.decode('utf-8')
                    except UnicodeDecodeError:
                        # If still can't decode, treat as raw binary data
                        content = ""
                        for byte in binary_content:
                            if byte < 0x80:  # ASCII range
                                content += chr(0x2800 + byte)  # Map to braille range
                            else:
                                content += chr(0x2800 + (byte % 64))  # Map to braille range
            converted_content = self.convert_to_svg(content)
            with open(output, 'w', encoding='utf-8') as file:
                file.write(converted_content)
        else:
            raise ValueError(f"Unsupported conversion: {input_extension} to {output_extension}")
