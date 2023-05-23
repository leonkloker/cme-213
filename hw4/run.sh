#!/bin/bash

if [ -f "result.txt" ]; then
    rm result.txt
fi

for order in 2 4 8
do
    sed -i "4s/.*/${order}/" params.in
    for n in 256 512 1024 2048 4096
    do
        sed -i "1s/.*/${n}\ ${n}/" params.in
        ./main >> result.txt
        echo >> result.txt
    done
done
