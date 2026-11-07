from setuptools import Command, setup
from setuptools.command.build_py import build_py
from setuptools.command.bdist_wheel import bdist_wheel
from pathlib import Path
import subprocess
import sys


class BuildFlite(Command):
    description = "build the Flite text-to-speech helper"
    user_options = [
        ("source=", None, "existing Flite source checkout (otherwise downloaded)"),
        ("build-dir=", None, "directory for intermediate native build files"),
    ]

    def initialize_options(self):
        self.source = None
        self.build_dir = None

    def finalize_options(self):
        pass

    def run(self):
        command = [sys.executable, str(Path(__file__).parent / "pywizard" / "build_flite.py")]
        for option, value in (("--source", self.source), ("--build-dir", self.build_dir)):
            if value is not None:
                command.extend([option, str(value)])
        subprocess.run(command, check=True)


class BuildPy(build_py):
    def run(self):
        self.run_command("build_flite")
        super().run()


class FliteWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        tag = super().get_tag()
        return ("py3", "none", tag[2])


if len(sys.argv) == 1:
    sys.argv.append("build_flite")

setup(
    cmdclass={"build_flite": BuildFlite, "build_py": BuildPy, "bdist_wheel": FliteWheel},
    name = "pywizard",
    version = '1.4.3',
    description = 'A tool to convert Wave-Files to LPC bytestreams used by TMS5220 chips and emulators like the Arduino Talkie library',
    url = 'http://github.com/ptwz/python_wizard',
    author = 'Peter Turczak (python port), Patrick J. Collins (original code and most of the work), Special thanks to: Richard Wiggins Jonathan Gevaryahu Gene Frantz Frank Palazzolo',
    license = 'MIT',
    packages = ["pywizard", "lpcplayer"],
    install_requires=[
        'numpy>=2.1',
        'scipy>=1.14.1',
    ],
    python_requires = '>=3.13',
    scripts = ['python_wizard', 'python_wizard_gui']
)
