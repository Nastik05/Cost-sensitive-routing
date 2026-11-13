"""Render added sections while retaining every byte of the old experiment text."""
from pathlib import Path
import csv
import re
import subprocess
import pypandoc

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
MARKER = '\n\n<!-- expanded-experiments -->\n'


def expand(text):
    def replace(match):
        path = PAPER / match.group(1)
        if not path.suffix:
            path = path.with_suffix('.tex')
        return expand(path.read_text())
    return re.sub(r'\\input\{([^}]+)\}', replace, text)


def main():
    pandoc = pypandoc.get_pandoc_path()
    rows = list(csv.DictReader((PAPER / 'additions/generated/claims.csv').open()))
    macros = {}
    number = lambda value, digits=2: f'{float(value):.{digits}f}'.replace('.', '{,}')
    for row in rows:
        macros[row['key'] + 'Effect'] = number(row['difference_pp'])
        macros[row['key'] + 'CI'] = f"[{number(row['low'])}; {number(row['high'])}]"
        macros[row['key'] + 'P'] = number(row['adjusted_p'], 4)

    def convert(text):
        text = expand(text)
        for name, value in macros.items():
            text = re.sub(r'\\' + name + r'(?:\{\})?(?![A-Za-z])', lambda m, v=value: v, text)
        command = [pandoc, '--from=latex', '--to=gfm-raw_html+tex_math_dollars', '--citeproc', '--wrap=none',
                   '--bibliography=references.bib', '--bibliography=additions/references.bib',
                   '--metadata=lang:ru', '--metadata=reference-section-title:Литература']
        result = subprocess.run(command, cwd=PAPER, input=text, text=True, capture_output=True, check=True)
        output = result.stdout
        output = re.sub(r'(?<!/)additions/figures/([^\s)]+)\.pdf', r'../additions/figures/\1.png', output)
        output = re.sub(r'\$`([^`]+)`\$', r'$\1$', output)
        def math_block(match):
            body = re.sub(r'\\label\{[^}]+\}', '', match.group(1))
            body = re.sub(r'\\(?:begin|end)\{equation\*?\}', '', body).strip()
            if not body.startswith(r'\begin{aligned}'):
                body = '\\begin{aligned}\n' + body + '\n\\end{aligned}'
            return '\n\n$$\n' + body + '\n$$\n\n'
        output = re.sub(r'```\s*math\s*\n(.*?)\n```', math_block, output, flags=re.S)
        output = output.replace(r'$\to$', '→')
        # GFM drops LaTeX float/equation anchors; keep readable reference text.
        output = re.sub(r'\[([^\]]+)\]\(#(?:tab|fig|eq|sec):[^)]+\)', r'\1', output)
        output = re.sub(r'(?<!\$)\$(?!\$)([^$]+)\$(?!\$)',
                        lambda m: '$' + re.sub(r'\s*\n\s*', ' ', m.group(1)) + '$', output)
        output = re.sub(r'\n{3,}', '\n\n', output)
        return output

    sections = PAPER / 'paper_sections'
    method = (PAPER / 'additions/method.tex').read_text() + '\n' + (PAPER / 'additions/theory_addition.tex').read_text()
    (sections / 'method.md').write_text(convert(method))
    (sections / 'conclusion.md').write_text(convert((PAPER / 'additions/conclusion.tex').read_text()))
    target = sections / 'вычислительные эксперименты.md'
    original = target.read_text().split(MARKER, 1)[0]
    text = '\n'.join((PAPER / 'additions' / f'{name}.tex').read_text()
                     for name in ['protocol', 'results', 'multiclass', 'discussion'])
    target.write_text(original + MARKER + '\n# Расширенные эксперименты\n\n' + convert(text))
    print('Rendered method, expanded experiments and conclusion; original experiment text retained.')


if __name__ == '__main__':
    main()
