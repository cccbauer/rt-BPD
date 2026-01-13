#!/bin/bash

if [ "$1" == "--help" ]; then
    echo "$0 [TR(ms) [port [hostname]]]"
    exit 0
fi

tr=1250
if [ "$1" ]; then
    tr=$1
fi

port=15000
if [ "$2" ]; then
   port=$2
fi

host=${HOST}
if [ "$3" ]; then
   host=$3
fi

sleep=0
if [ "$4" ]; then
   sleep=$4
fi

preHeader=1
if [ x"$5" != x ]; then
   preHeader=$5
fi

sleep $sleep
servepath=`which servenii`

if [ x"$servepath" == x ]; then
    echo "servenii not found, please add it to your path"
    exit 1
fi

servenii img/img 1 240 0 75 $tr $port $host $preHeader
