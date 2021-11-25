#!/bin/bash

runit() {
    bash ./fitsome.sh
}

until runit; do
    echo "Stopped wtih exit code $?. Restarting..";
    sleep 30;
done
