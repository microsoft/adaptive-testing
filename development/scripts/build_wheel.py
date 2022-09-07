import os
import logging
import platform
import subprocess

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def build_client():
    # Find our initial directory
    _logger.info("Starting build_client")
    _logger.info("Running npm install")
    # Spawning npm appears to work differently in PowerShell
    spawn_shell = platform.system() == "Windows"
    subprocess.run(
        ["npm", "install", "--loglevel", "verbose"],
        shell=spawn_shell,
        cwd="client",
        check=True,
    )
    _logger.info("Running npx webpack")
    subprocess.run(["npx", "webpack"], shell=spawn_shell, cwd="client", check=True)
    _logger.info("Ending build_client")


def build_wheel():
    _logger.info("Starting build_wheel")
    subprocess.run(["python", "setup.py", "sdist", "bdist_wheel"], check=True)
    _logger.info("Ending build_wheel")


def main():
    assert "setup.py" in os.listdir(), "Must be run from repo root"
    # Build the client
    build_client()

    # Build the wheel
    build_wheel()

    _logger.info("Completed")


if __name__ == "__main__":
    main()
