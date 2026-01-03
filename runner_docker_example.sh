#!/bin/bash

# Many thanks to Liam and his manual: https://blog.smallsec.ca/dockerizing-cli-tools/

# Check whether our first positional argument is a directory
if [[ -d "$1" ]]; then
    # It is a directory. Store the argument in a variable.
    dir_name="$1"
    # Remove the first argument from our list of arguments
    shift
else
    # It's not a directory, just use the current working directory
    dir_name="$(pwd)"
fi

# Run the container:
#     - in interactive mode
#     - remove it when finished
#     - mount a shared volume
# and then pass the rest of the arguments ("$@") to our container

# Please use paths the "/share/...."-style ones - e.g. script inside docker will consider the paths as related to root directory "/share" (which is inside docker container)

DOCKER_CONTAINER_NAME="JustAI-Video-cli"

docker run --rm -i -v "$dir_name":/share "${DOCKER_CONTAINER_NAME}" "$@"

# Usage of this script will look like this_script.sh FOLDER_IN_YOUR_SYSTEM_WHICH_WILL_BE_MOUNTED_INSIDE_DOCKER arguments1 argument2 argument3 ...
