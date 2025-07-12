# plsconvert
Convert any file to any other.
* * *
## About The Project
**plsconvert** is a CLI tool built in Python. It wraps a multitude of converters using a graph-based approach, enabling complex or unusual conversions that would typically require searching the internet for multiple commands. 
The project also heavily focuses on being easily expandable, welcoming contributions for new converters.

## Getting Started

You can simple installed from Pypi as follow:
>[!TIP]
>If you dont want all converters, you can install only the ones you want, for that I recommend cheking the [pyproject.toml](pyproject.toml)

```sh
pip install "plsconvert[all]"
```

Next, you can check if you have all the converters you want using:
```sh
plsconvert --dependencies 
```

And with that, you can now start converting your owns files to 😎:
```sh
plsconvert pipe.mp3 funny.pdf
```
<img width="493" height="517" alt="image" src="https://github.com/user-attachments/assets/d25a7d55-2a69-4571-b1f2-b36280bb4f1f" />

## Actual converters used

* Compression
  * **7z**
  * **Tar+gz+bz2+xz**
* Docs
  * **pandoc**
  * **pdf2docx**
* Media
  * **ffmpeg**
  * **imagemagick**
* Audio
  * **matplotlib+scipy** (For the spectograms)
* Config
  * Native Python Libs + **pyyaml+tomlkit**
