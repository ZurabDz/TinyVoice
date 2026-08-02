#!/usr/bin/env python3
"""Rewrite an ACUITY inputmeta file for plain tensor inputs.

Runs *inside* the SDK container (Python 3.8, no f-string niceties assumed).

``pegasus generate inputmeta`` assumes every input is an image: it emits
``category: image``, a TEXT database of JPEG paths and a mean/scale
preprocessing block.  This model takes a log-mel spectrogram and a frame mask,
so each port is switched to ``category: undefined`` backed by an NPY database
of pre-computed tensors, with preprocessing reduced to identity.

Ports are matched by the ONNX tensor name they came from -- ACUITY suffixes the
layer id (``mel`` becomes ``mel_1234``), so a prefix match is what's reliable.

    python3 npu_inputmeta.py <name>_inputmeta.yml mel=dataset_mel.txt mask=dataset_mask.txt
"""

import sys

from ruamel.yaml import YAML


def main():
    if len(sys.argv) < 3:
        sys.stderr.write(__doc__)
        return 1

    path = sys.argv[1]
    datasets = dict(pair.split("=", 1) for pair in sys.argv[2:])

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(path) as handle:
        meta = yaml.load(handle)

    ports = []
    for database in meta["input_meta"]["databases"]:
        ports.extend(database["ports"])

    databases = []
    for name, dataset in datasets.items():
        matched = [p for p in ports if str(p["lid"]).startswith(name)]
        if not matched:
            sys.stderr.write(
                "no input port matching '%s'; ports are %s\n"
                % (name, [str(p["lid"]) for p in ports])
            )
            return 1

        for port in matched:
            port["category"] = "undefined"
            port["dtype"] = "float32"
            # Identity preprocessing: the .npy files are already model-ready.
            port["preprocess"] = {
                "reverse_channel": False,
                "mean": [0],
                "scale": [1.0],
                "preproc_node_params": {"add_preproc_node": False},
            }
        databases.append({"path": dataset, "type": "NPY", "ports": matched})

    meta["input_meta"]["databases"] = databases
    with open(path, "w") as handle:
        yaml.dump(meta, handle)

    for database in databases:
        print(
            "  %-16s <- %s  shape=%s"
            % (
                database["ports"][0]["lid"],
                database["path"],
                list(database["ports"][0]["shape"]),
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
