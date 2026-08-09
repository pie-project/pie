#!/bin/bash
# Full occurrence sweep for a symbol: bare kernel name + qualified name
cd /root/Workspace/pie-new-driver
sym="$1"
fam="${sym%%::*}"
name="${sym##*::}"
echo "############ $sym ############"
echo "--- occurrences of full path '$sym' ---"
git grep -n -F -- "$sym" -- $(git ls-files | sed 's/.*//' >/dev/null; echo .) 2>/dev/null | head -80
echo "--- occurrences of bare name '$name' (word) ---"
git grep -n -w -- "$name" 2>/dev/null | head -120
