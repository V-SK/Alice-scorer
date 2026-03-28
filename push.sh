#!/bin/bash
set -e
cd /tmp/alice-scorer
git init
git add .
git commit -m 'Initial release: Alice Protocol Scoring Worker'
git branch -M main
git remote add origin git@github.com:V-SK/alice-scorer.git
git push -u origin main
echo 'Done! https://github.com/V-SK/alice-scorer'
