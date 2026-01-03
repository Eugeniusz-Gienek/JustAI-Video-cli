#!/bin/bash
scriptDir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")

cd $scriptDir

cd ../../

docker buildx build -t justai-video-cli:latest .
