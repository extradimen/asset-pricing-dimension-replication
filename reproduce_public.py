#!/usr/bin/env python3
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent/'reproduction';paper=root/'main_paper_a_irfa'
for script in ['audit/rebuild_temporal_exhibit_data.py','audit/verify_exhibit_sources.py','visuals/build_publication_figures.py','visuals/build_publication_tables.py','visuals/build_supplement_tables.py']:
    subprocess.run([sys.executable,str(paper/script)],cwd=root,check=True)
for name in ['main_full','main_anonymous','title_page']:
    subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',name+'.tex'],cwd=paper/'manuscript',check=True)
subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','supplement.tex'],cwd=paper/'manuscript/supplement',check=True)
print('Public exhibit replay and manuscript builds completed.')
