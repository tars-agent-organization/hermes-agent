#!/bin/sh
set -eu

launcher=$0
while [ -L "$launcher" ]; do
    link=$(readlink "$launcher")
    case "$link" in
        /*) launcher=$link ;;
        *) launcher=$(dirname "$launcher")/$link ;;
    esac
done

script_dir=$(CDPATH= cd -- "$(dirname -- "$launcher")" && pwd)
python_bin=${HERMES_LOCAL_STT_PYTHON:-"$HOME/.hermes/local-stt-venv/bin/python"}
exec "$python_bin" "$script_dir/tars_local_stt.py" "$@"
