"""Build the limited homework text and single slide using LuaLaTeX/BibTeX."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def step(command, directory):
    process = subprocess.run(command, cwd=directory, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if process.returncode:
        print(process.stdout)
        raise RuntimeError(command)
    print("Built:", directory.name, " ".join(command))


def main():
    flags = ["-interaction=nonstopmode","-halt-on-error","-file-line-error","-no-shell-escape","main.tex"]
    step(["lualatex",*flags],ROOT/"paper")
    step(["bibtex","main"],ROOT/"paper")
    for _ in range(2):
        step(["lualatex",*flags],ROOT/"paper")
        step(["lualatex",*flags],ROOT/"slides")


if __name__ == "__main__":
    main()
