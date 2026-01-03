#!/bin/bash
scriptDir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")

cd $scriptDir

cd ../../

docker buildx build -t eugeniusz-gienek/justai-video-cli .
