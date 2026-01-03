#!/bin/bash

scriptDir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")

cd "${scriptDir}/docker"
./docker_installer.sh
