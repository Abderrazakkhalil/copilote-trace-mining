"""Contrôle du format et des empreintes des données, sans entraînement."""
import csv
import hashlib
import json
from collections import Counter

from src.config import DATA_DIR, RAW_DIR


def main():
    manifest = json.loads((DATA_DIR / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        path = RAW_DIR / name
        if not path.exists():
            raise SystemExit(f'Fichier manquant : {path}. Voir data/README.md.')
        text = path.read_text(encoding='utf-8')
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if digest != expected['sha256_normalized_utf8']:
            raise SystemExit(f'Empreinte différente : {name}. Vérifier la version des données.')
        rows = text.splitlines()
        if len(rows) != expected['lines']:
            raise SystemExit(f'Nombre de lignes différent : {name}')
        print(f'{name}: {len(rows)} lignes, SHA-256 conforme')
    train = (RAW_DIR / 'train.csv').read_text(encoding='utf-8').splitlines()
    counts = Counter(row.split(',', 1)[0] for row in train)
    if len(counts) != 247 or (min(counts.values()), max(counts.values())) != (4, 75):
        raise SystemExit('Cible inattendue : attendre 247 utilisateurs, de 4 à 75 sessions chacun.')
    with (RAW_DIR / 'sample_submission.csv').open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if reader.fieldnames != ['RowId', 'prediction'] or [int(r['RowId']) for r in rows] != list(range(1, 325)):
            raise SystemExit('Format de sample_submission.csv incorrect.')
    print('3279 sessions train, 324 sessions test, 247 utilisateurs : données prêtes.')


if __name__ == '__main__':
    main()
