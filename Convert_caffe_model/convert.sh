#!/usr/bin/env bash

join_path() {
    echo "${1:+$1/}$2" | sed 's#//#/#g'
}

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  TARGET="$(readlink "$SOURCE")"
  if [[ $TARGET == /* ]]; then
    echo "SOURCE '$SOURCE' is an absolute symlink to '$TARGET'"
    SOURCE="$TARGET"
  else
    DIR="$( dirname "$SOURCE" )"
    echo "SOURCE '$SOURCE' is a relative symlink to '$TARGET' (relative to '$DIR')"
    SOURCE="$DIR/$TARGET" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
  fi
done

echo "SOURCE is '$SOURCE'"
RDIR="$( dirname "$SOURCE" )"
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
if [ "$DIR" != "$RDIR" ]; then
  echo "DIR '$RDIR' resolves to '$DIR'"
fi
echo "DIR is '$DIR'"


# access to folder models
MODELS_PATH="models"
MODELS_TAR_NAME="models_VGGNet_VOC0712_SSD_300x300.tar.gz"
DIR_MODELS="$( join_path $DIR $MODELS_PATH )"
FILDE_MODELS_PATH="$( join_path $DIR_MODELS $MODELS_TAR_NAME )"
echo "MODEL NAME is '$FILDE_MODELS_PATH'"

WEIGHTS_NAME="weights"
DIR_WEIGHTS="$( join_path $DIR $WEIGHTS_NAME )"

if [ -d "$DIR_WEIGHTS" ] ; then
    echo "Folder: '$WEIGHTS_NAME' is existing !! we need to remove"
    rm -rf "$DIR_WEIGHTS" && mkdir "$DIR_WEIGHTS"
else
    echo "Folder: '$WEIGHTS_NAME' isn't existing !! we need to create"
    mkdir "$DIR_WEIGHTS"
fi


# extraction
tar -zxf $FILDE_MODELS_PATH --directory $DIR_WEIGHTS





