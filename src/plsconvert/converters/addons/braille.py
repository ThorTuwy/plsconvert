from pathlib import Path
from plsconvert.converters.abstract import Converter
from plsconvert.converters.registry import addMethodData, registerConverter
from plsconvert.utils.dependency import Dependencies
from plsconvert.utils.graph import PairList

@registerConverter
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

    def _read_braille_file(self, input_path: Path) -> str:
        """
        Read braille file and return content as string
        
        Args:
            input_path: Path to the braille file
            
        Returns:
            Content as string
        """
        try:
            # First try to read as UTF-8 text (for text-based braille files)
            with open(input_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # If that fails, read as binary and try to decode
            with open(input_path, 'rb') as file:
                binary_content = file.read()
                try:
                    return binary_content.decode('utf-8')
                except UnicodeDecodeError:
                    # If still can't decode, treat as raw binary data
                    content = ""
                    for byte in binary_content:
                        if byte < 0x80:  # ASCII range
                            content += chr(0x2800 + byte)  # Map to braille range
                        else:
                            content += chr(0x2800 + (byte % 64))  # Map to braille range
                    return content

    @addMethodData(PairList.all2all(["txt", "braille"], ["braille", "txt"]), True)
    def convert_text(self, input: Path, output: Path, input_extension: str, output_extension: str) -> None:
        """
        Convert text to/from Braille
        
        Args:
            input: Input file path
            output: Output file path
            input_extension: Input file extension
            output_extension: Output file extension
        """
        # Read input file
        if input_extension == "txt":
            with open(input, 'r', encoding='utf-8') as file:
                content = file.read()
            # Convert text to Braille
            converted_content = self._convert_text_string(content, to_braille=True)
            # Write as binary braille file
            with open(output, 'wb') as file:
                file.write(converted_content.encode('utf-8'))
        else:  # braille to txt
            content = self._read_braille_file(input)
            converted_content = self._convert_text_string(content, to_braille=False)
            with open(output, 'w', encoding='utf-8') as file:
                file.write(converted_content)

    def _convert_text_string(self, input_str: str, to_braille: bool = True) -> str:
        """
        Convert text string to/from Braille
        
        Args:
            input_str: Input string to convert
            to_braille: If True, convert text to Braille; if False, convert Braille to text
        
        Returns:
            Converted string
        """
        out = ""
        pbar = self.pbInit(len(input_str))

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
                pbar.update(1)
        else:
            # Convert Braille to text
            for braille_char in input_str:
                if braille_char in self.reverse_mapping:
                    out += self.reverse_mapping[braille_char]
                else:
                    # If Braille character not in mapping, keep original
                    out += braille_char
                pbar.update(1)
        return out

    @addMethodData(PairList(("braille", "svg")), True)
    def convert_to_svg(self, input: Path, output: Path, input_extension: str, output_extension: str) -> None:
        """
        Convert Braille text to SVG visual representation
        
        Args:
            input: Input braille file path
            output: Output SVG file path
            input_extension: Input file extension
            output_extension: Output file extension
        """
        content = self._read_braille_file(input)
        svg_content = self._convert_to_svg_string(content)
        with open(output, 'w', encoding='utf-8') as file:
            file.write(svg_content)

    def _convert_to_svg_string(self, braille_str: str) -> str:
        """
        Convert Braille text string to SVG visual representation
        
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

        pbar = self.pbInit(len(braille_str) + len(braille_str) * 6)
        
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
            pbar.update(1)
        
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

                    pbar.update(1)

        pbar.close()
        svg_content += '</svg>'
        return svg_content
