#!/bin/bash

runit() {
    bash ./fitsome.sh
}

while runit; do
    echo "Sleeping...";
    sleep 20;
done
