"""Read and hash the paired examples; no model inference or training."""
from collections import Counter
from pathlib import Path
from typing import Any
import ast
import hashlib
import json
import re
import numpy as np
from PIL import Image


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'src/mobilenet_enterprise/data/wheat_seed_hsi.py'
    names = {'_parse_hdr', '_dtype_from_envi', '_read_hsi_cube'}
    nodes = [n for n in ast.parse(source.read_text(encoding='utf-8')).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    namespace = {'np': np, 'Path': Path, 'Any': Any, 're': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    manifest = json.loads((root / 'datasets/examples/manifest.json').read_text(encoding='utf-8'))
    counts = Counter(s['cultivar'] for s in manifest['samples'])
    assert len({s['sample_id'] for s in manifest['samples']}) == len(manifest['samples'])
    if 'samples_per_cultivar' in manifest:
        assert set(counts) == {'CX186', 'XK129', 'ZM22', 'ZM379'}
        assert all(n == manifest['samples_per_cultivar'] for n in counts.values())
        assert len(manifest['samples']) == manifest['total_pairs']
    result = []
    for sample in manifest['samples']:
        assert len(sample['files']) == 3
        assert {x['role'] for x in sample['files']} == {'rgb', 'hsi_header', 'hsi_binary'}
        for item in sample['files']:
            path = root / item['file']
            assert path.stat().st_size == item['bytes']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        hdr = next(root / x['file'] for x in sample['files'] if x['role'] == 'hsi_header')
        cube = namespace['_read_hsi_cube'](hdr)
        assert cube.ndim == 3 and cube.shape[-1] == 454 and np.isfinite(cube).all()
        rgb = next(root / x['file'] for x in sample['files'] if x['role'] == 'rgb')
        with Image.open(rgb) as im:
            assert [im.height, im.width, len(im.getbands())] == sample['rgb_shape']
            im.verify()
        result.append({'sample_id': sample['sample_id'], 'hsi_shape': list(cube.shape),
                       'rgb_shape': sample['rgb_shape']})
    for cultivar, count in counts.items():
        folder = root / 'datasets/examples' / cultivar
        expected = {str((root / f['file']).resolve())
                    for s in manifest['samples'] if s['cultivar'] == cultivar for f in s['files']}
        actual = {str(p.resolve()) for p in folder.iterdir() if p.is_file()}
        assert actual == expected and len(actual) == count * 3
    print(json.dumps({'passed': True, 'pairs_per_cultivar': dict(counts),
                      'total_pairs': len(result), 'total_sample_files': len(result) * 3,
                      'model_inference': False,
                      'historical_accuracy_reproduced': False}, indent=2))


if __name__ == '__main__':
    main()
