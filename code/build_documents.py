"""Build current documents; preserve old sections and the archived one-slide PDF."""
from pathlib import Path
import subprocess
import pypandoc
import re
import argparse

ROOT = Path(__file__).resolve().parents[1]


def step(command, directory):
    process = subprocess.run(command, cwd=directory, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if process.returncode:
        print(process.stdout)
        raise RuntimeError(command)
    print("Built:", directory.name, " ".join(command))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['all', 'article', 'slides'], default='all')
    parser.add_argument('--refresh-markdown', action='store_true',
                        help='Refresh added views only; original abstract/introduction remain untouched')
    args = parser.parse_args()
    flags = ['-interaction=nonstopmode', '-halt-on-error', '-file-line-error', '-no-shell-escape']
    directories = []
    if args.target in ['all', 'article']:
        paper = ROOT / 'paper'
        article = ['lualatex', *flags, 'main.tex']
        step(article, paper)
        step(['bibtex', 'main'], paper)
        step(article, paper)
        step(article, paper)
        from render_additions import main as render
        render()
        directories.append(paper)
    if args.target in ['all', 'slides']:
        for _ in range(2):
            step(['lualatex', *flags, 'main.tex'], ROOT / 'slides')
        directories.append(ROOT / 'slides')
    for directory in directories:
        log = (directory / 'main.log').read_text()
        issues = re.findall(r'[^\n]*(?:Overfull|undefined|multiply defined|Missing character|Rerun to get)[^\n]*', log)
        if issues:
            raise RuntimeError('\n'.join(issues))
    print('Built current documents; slides/one_slide.pdf and old prose preserved.')


if __name__ == "__main__":
    main()
