#!/bin/sh
# bash applyVHACD input_directory output_directory vhacd_directory number_of_models
# /home/jonathan/Desktop/Projects/v-hacd/build/linux/test/testVHACD
# must be run in vhacdify directory

if [ ! -f "$PWD/convert.py" ]; then
    echo "Error: vrml2 to obj conversion file doesn't exist!"
    exit 1
fi

if [ "$#" !=  4 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

echo "Applying V-HACD to files..."
count=1
echo $4
for file1_parentdir in $1/*
do
    for file1 in $file1_parentdir/*
    do
        if (( $count > $4 )); then
            echo "asdfasdfasdfa"
            echo $count
            echo $4
            break
        fi
        echo $count
        echo "parentdir" $file1_parentdir
        echo "file1" $file1
        classIDdir=$2/"$(basename "$file1_parentdir")"
        mkdir $classIDdir > out.log
        newdir=$classIDdir/"$(basename "$file1")"
        echo "model_normalized.obj path:" $file1/models/model_normalized.obj
        echo "newdir" $newdir
        mkdir $newdir > out.log
        # touch $newdir/model.wrl
        $3 --input $file1/models/model_normalized.obj --output $newdir/model.wrl > out.log 
        count=$((count+1))
    done
done

count=1
echo ""
echo "Converting vrml files to obj files..."
for file2_parentdir in $2/*
do
    for file2 in $file2_parentdir/*
    do
        echo $count
        echo $file2
        classIDdir=$2/"$(basename "$file2_parentdir")"
        newdir=$classIDdir/"$(basename "$file2")"
        # touch $file2/model.obj
        blender --background --python $PWD/convert.py -- $newdir/model.wrl $newdir/model.obj > out.log 
        count=$((count+1))
    done
done

echo""
echo "Finished Successfully"
