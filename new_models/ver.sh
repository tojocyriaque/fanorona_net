# Voir les 10 plus gros fichiers dans l'historique Git
git rev-list --objects --all
git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)'
awk '/^blob/ {print substr($0,6)}'
sort --numeric-sort --key=2
cut --complement --characters=13-40
numfmt --field=2 --to=iec
