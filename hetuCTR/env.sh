#!/bin/bash
path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$path/build/lib:$PYTHONPATH
