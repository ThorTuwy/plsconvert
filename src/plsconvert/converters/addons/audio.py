from pathlib import Path
from plsconvert.converters.abstract import Converter
from plsconvert.converters.registry import addMethodData, registerConverter
from plsconvert.utils.graph import PairList
from plsconvert.utils.dependency import Dependencies, LibDependency as Lib


@registerConverter
class spectrogramMaker(Converter):
    """
    Spectrogram maker using matplotlib.
    """

    @property
    def name(self) -> str:
        return "Spectrogram Maker"

    @property
    def dependencies(self) -> Dependencies:
        return Dependencies([Lib("matplotlib"), Lib("scipy")])

    @addMethodData(PairList(("wav", "png")), False)
    def wav_to_spectrogram(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        import matplotlib.pyplot as plt
        from scipy.io import wavfile

        FS, data = wavfile.read(input)
        if data.ndim > 1 and data.shape[1] == 2: # type: ignore
            data = data.mean(axis=1)
        plt.specgram(data, Fs=FS, NFFT=128, noverlap=0)
        plt.savefig(output, format="png")
        plt.close()

@registerConverter
class textToSpeech(Converter):
    """
    Text to speech converter using pyttsx3.
    """

    @property
    def name(self) -> str:
        return "Text to Speech Converter"

    @property
    def dependencies(self) -> Dependencies:
        return Dependencies([Lib("pyttsx3")])

    @addMethodData(PairList(("txt", "mp3")), False)
    def txt_to_audio(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        import pyttsx3

        with open(input, 'r', encoding='utf-8') as file:
            text = file.read()

        engine = pyttsx3.init()
        engine.save_to_file(text, output.as_posix())
        engine.runAndWait()

@registerConverter
class audioFromMidi(Converter):
    """
    Audio from midi converter using midi2audio.
    """

    @property
    def name(self) -> str:
        return "Midi2Audio Converter"

    @property
    def dependencies(self) -> Dependencies:
        return Dependencies([Lib("midi2audio")])

    @addMethodData(PairList(("mid", "wav")), False)
    def midi_to_audio(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        from midi2audio import FluidSynth
        FluidSynth().midi_to_audio(str(input), str(output))
