#!/usr/bin/env bash


# location path
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


# create dataset folder IF IT EXISTS
DATASET_FOLDER_NAME='dataset_OD'
DATASET_FOLDER_PATH="$( join_path $DIR $DATASET_FOLDER_NAME )"

if [ -d "$DATASET_FOLDER_PATH" ] ; then
    echo "Folder: '$DATASET_FOLDER_NAME' is existing !! we need to remove"
    rm -rf "$DATASET_FOLDER_PATH" && mkdir "$DATASET_FOLDER_PATH"
else
    echo "Folder: '$DATASET_FOLDER_NAME' isn't existing !! we need to create"
    mkdir "$DATASET_FOLDER_PATH"
fi
echo "DATASET FOLDER IS '$DATASET_FOLDER_PATH'"

# downloads dataset
DOWNLOAD_PATH='http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar'
NAME_FILE_TAR='VOCtrainval_06-Nov-2007.tar'
TAR_NAME_PATH="$( join_path $DATASET_FOLDER_PATH $NAME_FILE_TAR )"
echo "TAR NAME is '$TAR_NAME_PATH'"

wget $DOWNLOAD_PATH -P $DATASET_FOLDER_PATH
tar -xvf $TAR_NAME_PATH -C $DATASET_FOLDER_PATH
rm -rf $TAR_NAME_PATH

