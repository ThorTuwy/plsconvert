from pathlib import Path
from plsconvert.converters.abstract import Converter
from plsconvert.utils.dependency import checkLibsDependencies


class spectrogramMaker(Converter):
    def adjConverter(self) -> dict[str, list[list[str]]]:
        return {
            "wav": ["png"],
        }

    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        import matplotlib.pyplot as plt
        from scipy.io import wavfile

        FS, data = wavfile.read(input)
        if data.ndim > 1 and data.shape[1] == 2:
            data = data.mean(axis=1)
        plt.specgram(data, Fs=FS, NFFT=128, noverlap=0)
        plt.savefig(output, format="png")
        plt.close()

    def metDependencies(self) -> bool:
        return checkLibsDependencies(["matplotlib", "scipy"])

class textToSpeech(Converter):
    def adjConverter(self) -> dict[str, list[list[str]]]:
        return {
            "txt": ["mp3"],
        }

    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        import pyttsx3

        with open(input, 'r', encoding='utf-8') as file:
            text = file.read()

        engine = pyttsx3.init()
        engine.save_to_file(text, output.as_posix())
        engine.runAndWait()

    def metDependencies(self) -> bool:
        return checkLibsDependencies(["pyttsx3"])

class audioFromMidi(Converter):
    def adjConverter(self) -> dict[str, list[list[str]]]:
        return {
            "mid": ["wav"],
        }

    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        from midi2audio import FluidSynth
        FluidSynth().midi_to_audio(str(input), str(output))


    def metDependencies(self) -> bool:
        return checkLibsDependencies(["midi2audio"])
