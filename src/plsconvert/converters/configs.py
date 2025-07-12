from plsconvert.converters.abstract import Converter
from pathlib import Path

from plsconvert.utils.graph import conversionFromToAdj
from plsconvert.utils.dependency import checkLibsDependencies


class configParser(Converter):
    def adjConverter(self) -> dict[str, list[list[str]]]:
        return conversionFromToAdj(
            ["json", "toml", "yaml", "ini"],
            ["json", "toml", "yaml", "ini"],
        )

    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        import json
        import tomlkit
        import yaml
        import configparser

        if input_extension == "json":
            with open(input, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif input_extension == "toml":
            with open(input, "rb") as f:  # tomllib requires binary mode
                data = tomlkit.load(f)
        elif input_extension == "yaml":
            with open(input, "r", encoding="utf-8") as f:
                data = yaml.load(f, Loader=yaml.Loader)
        elif input_extension == "ini":
            config = configparser.ConfigParser()
            config.read(input)
            data = {section: dict(config[section]) for section in config.sections()}

        if output_extension == "json":
            with open(output, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

        elif output_extension == "toml":
            with open(output, "w", encoding="utf-8") as f:
                tomlkit.dump(data, f)
        elif output_extension == "yaml":
            with open(output, "w", encoding="utf-8") as f:
                yaml.dump(data, f, indent=2, sort_keys=False)
        elif output_extension == "ini":
            config = configparser.ConfigParser()
            for section, values in data.items():
                if not isinstance(values, dict):
                    raise ValueError(
                        f"Cannot convert non-dictionary data for section '{section}' to INI format."
                    )
                config[section] = values
            with open(output, "w", encoding="utf-8") as f:
                config.write(f)

    def metDependencies(self) -> bool:
        return checkLibsDependencies(["tomlkit", "yaml"])
