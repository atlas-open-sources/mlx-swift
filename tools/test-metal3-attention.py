#!/usr/bin/env python3
"""Compile-only Metal 3 regression matrix; never loads a model or executes a GPU.

Run from any directory: python3 tools/test-metal3-attention.py --output /tmp/evidence
Requires the pinned MLX source submodule and an installed Apple Metal toolchain.
The full attention TU uses native bfloat, requiring Metal >=3.1. The shared
constant-loop control is additionally compiled in Metal 3.0. All positive
compiles use -Werror, with no warning suppression. Negative controls must fail
because the body was instantiated at the expected compile-time index.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'Source/Cmlx/mlx'
GENERATED = ROOT / 'Source/Cmlx/mlx-generated'
ATTENTION = 'mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal'


def preamble(path):
    matches = re.findall(r'R"preamble\((.*?)\)preamble"', path.read_text(), re.S)
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one generated preamble: {path}')
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # This matches the production JIT composition (utils + steel_attention +
    # template definitions), using all actual offline dtype/mask/shape definitions.
    definitions = re.sub(r'^#include .*$', '', (SOURCE / ATTENTION).read_text(), flags=re.M)
    jit = output / 'jit-attention.metal'
    # utils includes shared reduction constants unused by attention. Reference
    # them only in harness static assertions so -Werror checks the raw JIT
    # source without suppressing warnings or changing production constants.
    constants = ('MAX_REDUCE_SPECIALIZED_DIMS', 'REDUCE_N_READS',
                 'REDUCE_N_WRITES', 'SOFTMAX_N_READS', 'RMS_N_READS',
                 'RMS_LOOPED_LIMIT')
    assertions = '\n'.join(f'static_assert({name} > 0, \"shared positive constant\");'
                           for name in constants)
    jit.write_text(preamble(GENERATED / 'utils.cpp') + '\n' +
                   preamble(GENERATED / 'steel_attention.cpp') + '\n' +
                   definitions + '\n' + assertions)
    results = []
    for sdk, target in [('macosx', 'air64-apple-macosx15.2'),
                        ('iphoneos', 'air64-apple-ios18.2')]:
        for language in ('3.0', '3.1', '3.2'):
            units = [('controls', ROOT / 'tools/tests/metal3-constexpr-controls.metal')]
            if language != '3.0':
                units += [('source', SOURCE / ATTENTION),
                          ('generated', GENERATED / 'metal/steel/attn/kernels/steel_attention.metal'),
                          ('jit', jit)]
            for name, unit in units:
                label = f'{sdk}-metal{language}-{name}'
                command = ['xcrun', '-sdk', sdk, 'metal', f'-std=metal{language}',
                           '-target', target, '-Werror', '-I', str(SOURCE),
                           '-c', str(unit), '-o', str(output / f'{label}.air')]
                result = subprocess.run(command, capture_output=True, text=True)
                (output / f'{label}.log').write_text(result.stdout + result.stderr)
                results.append({'name': label, 'command': command, 'exit_code': result.returncode,
                                'diagnostics_empty': not (result.stdout or result.stderr),
                                'source_sha256': hashlib.sha256(unit.read_bytes()).hexdigest()})
                if result.returncode or result.stdout or result.stderr:
                    raise RuntimeError(f'{label} failed or emitted diagnostics; see output log')
            label = f'{sdk}-metal{language}-negative'
            command = ['xcrun', '-sdk', sdk, 'metal', f'-std=metal{language}',
                       '-target', target, '-Werror', '-I', str(SOURCE),
                       '-DEXPECT_INSTANTIATION_FAILURE', '-c',
                       str(ROOT / 'tools/tests/metal3-constexpr-controls.metal'),
                       '-o', str(output / f'{label}.air')]
            result = subprocess.run(command, capture_output=True, text=True)
            (output / f'{label}.log').write_text(result.stdout + result.stderr)
            if result.returncode == 0 or 'negative control must instantiate index two' not in result.stderr:
                raise RuntimeError(f'{label} did not reject the expected instantiation')
            results.append({'name': label, 'command': command, 'exit_code': result.returncode,
                            'expected_rejection': True})
    (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f'{len(results)} compile checks passed; evidence: {output}')


if __name__ == '__main__':
    main()
