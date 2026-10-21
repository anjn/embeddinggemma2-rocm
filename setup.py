from pathlib import Path
import shutil
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        source = Path(__file__).parent / "profiles"
        if source.exists():
            shutil.copytree(
                source,
                Path(self.build_lib) / "embeddinggemma2_rocm/profiles",
                dirs_exist_ok=True,
            )


setup(cmdclass={"build_py": BuildPy})
