"""Build the article draft, readable section files and research slide."""
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
    parser.add_argument("--refresh-markdown",action="store_true",
                        help="Explicitly regenerate existing Markdown views from the TeX sources")
    args = parser.parse_args()
    paper = ROOT / "paper"
    section_dir = paper / "paper_sections"
    flags = ["-interaction=nonstopmode","-halt-on-error","-file-line-error","-no-shell-escape"]
    article = ["lualatex",*flags,"main.tex"]
    step(article,paper)
    step(["bibtex","main"],paper)
    for _ in range(2):
        step(article,paper)
        step(["lualatex",*flags,"main.tex"],ROOT/"slides")
    sections = [("abstract.tex","аннотация.md"),("introduction.tex","введение.md"),
                ("related_works.tex","related_works.md"),
                ("problem_statement.tex","постановка задачи.md"),("experiments.tex","вычислительные эксперименты.md")]
    pandoc = pypandoc.get_pandoc_path()

    def markdown(tex,heading_shift=0):
        converted = subprocess.run([pandoc,"--from=latex","--to=gfm-raw_html+tex_math_dollars","--citeproc",
                                    "--bibliography=references.bib","--metadata=lang:ru",
                                    "--metadata=reference-section-title:Литература",
                                    "--shift-heading-level-by="+str(heading_shift),"--wrap=none"],
                                   cwd=paper,input=tex,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
        text = converted.stdout.replace("figures/error_curve.pdf","../figures/error_curve.png").replace(
            "figures/decision_profiles.pdf","../figures/decision_profiles.png")
        text = re.sub(r"\$`([^`]+)`\$",lambda m:"$"+m.group(1)+"$",text)

        def display_math(match):
            body = re.sub(r"\\label\{[^}]+\}","",match.group(1))
            body = re.sub(r"\\(?:begin|end)\{equation\*?\}","",body).strip()
            if not body.startswith(r"\begin{aligned}"):
                body = "\\begin{aligned}\n"+body+"\n\\end{aligned}"
            return "\n\n$$\n"+body+"\n$$\n\n"

        text = re.sub(r"```\s*math\s*\n(.*?)\n```",display_math,text,flags=re.S)
        text = re.sub(r"\[([^\]]+)\]\(#(?:tab|fig|eq):[^)]+\)",r"\1",text)
        text = re.sub(r"\n{3,}","\n\n",text)
        return text

    for source,target in sections:
        target_path = section_dir / target
        # Existing Markdown may contain the author's manual edits.
        if target_path.exists() and not args.refresh_markdown:
            continue
        tex = (section_dir / source).read_text()
        if source == "abstract.tex":
            tex = "\\section*{Аннотация}\n" + tex
        target_path.write_text(markdown(tex))
    print("Built paper/main.pdf and section views in paper/paper_sections/")


if __name__ == "__main__":
    main()
