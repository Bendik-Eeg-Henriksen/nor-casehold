# NOR-CASEHOLD GitHub Setup
# Run these commands from your local machine

# 1. Install Git LFS (once, if not already installed)
git lfs install

# 2. Create the repo structure
mkdir nor-casehold && cd nor-casehold
git init
git lfs track "data/*.jsonl"

# 3. Add files
mkdir data scripts
# Copy in your files:
#   README.md, LICENSE, .gitattributes, requirements.txt, benchmark.py
#   data/train.jsonl, data/val.jsonl, data/test.jsonl
#   scripts/cleanup_nor_casehold.py, scripts/cleanup_report.json

git add .gitattributes
git add .
git commit -m "Initial release: NOR-CASEHOLD v1.1"

# 4. Create repo on GitHub (go to github.com/new)
#    Name: nor-casehold
#    Visibility: Public
#    Do NOT initialize with README (you have one already)

# 5. Push
git remote add origin https://github.com/Bendik-Eeg-Henriksen/nor-casehold.git
git branch -M main
git push -u origin main
