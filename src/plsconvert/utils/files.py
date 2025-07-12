import subprocess


def runCommand(command: list[str]) -> None:
    subprocess.run(
        command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def fileType(extension: str) -> str:
    return {
        "jpg": "image",
        "jpeg": "image",
        "png": "image",
        "gif": "image",
        "bmp": "image",
        "svg": "image",
        "webp": "image",
        "tiff": "image",
        "mp4": "video",
        "mov": "video",
        "avi": "video",
        "mkv": "video",
        "flv": "video",
        "wmv": "video",
        "webm": "video",
        "mpeg": "video",
        "mp3": "audio",
        "wav": "audio",
        "ogg": "audio",
        "aac": "audio",
        "flac": "audio",
        "m4a": "audio",
        "wma": "audio",
        "alac": "audio",
    }.get(extension, "other")
