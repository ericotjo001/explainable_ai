#!/bin/bash

search_dir=./
for entry in "$search_dir"/*
do
    file=${entry##*/} # to extract only te name (no path)
    if [ "$file" = "dataset" ] || [ "$file" = "xresearchlog" ] || [ "$file" = "xresearchlog.resnet50.nscc.1" ] || [ "$file" = "xresearchlog.resnet50.nscc.2" ] || [ "$file" = "checkpoint" ] || [ "$file" = "xunformattedlog" ]; then
        echo "EXCEPTING THIS ONE $entry $file"
    else
        echo "$entry $file"
        scp -r $file ericotjo@ntu.nscc.sg:scratch/wsolevaluation-master/$file
fi 
done

