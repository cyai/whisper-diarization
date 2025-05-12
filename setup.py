from setuptools import setup, find_packages

# Read requirements
with open("requirements.txt") as f:
    requirements = [l.strip() for l in f if l.strip() and not l.startswith("#")]

setup(
    name="whisper_diarization",
    version="0.1.0",
    description="Speaker-aware transcription using OpenAI Whisper + NeMo diarization",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    license="Apache-2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=requirements,
    include_package_data=True,      # needed to include YAML files
    entry_points={
        "transformers_pipelines": [
            # This registers your custom Pipeline class once we implement it
            "whisper-diarization = whisper_diarization.pipeline:WhisperDiarizationPipeline"
        ]
    },
    python_requires=">=3.8",
)
