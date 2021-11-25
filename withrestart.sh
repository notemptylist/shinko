#!/bin/bash

runit() {
    bash ./fitsome.sh
}

while runit; do
    echo "Sleeping...";
    sleep 120;
done
