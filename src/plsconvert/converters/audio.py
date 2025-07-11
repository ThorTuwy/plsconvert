import matplotlib.pyplot as plt
from scipy.io import wavfile

from pathlib import Path
from plsconvert.converters.abstract import Converter


class spectrogramMaker(Converter):
    def adjConverter(self) -> dict[str, list[list[str]]]:
        return {
            "wav": ["png"],
        }

    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        FS, data = wavfile.read(input)
        if data.ndim > 1 and data.shape[1] == 2:
            data = data.mean(axis=1)
        plt.specgram(data, Fs=FS, NFFT=128, noverlap=0)
        plt.savefig(output, format="png")
        plt.close()
