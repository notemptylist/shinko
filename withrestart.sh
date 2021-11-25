#!/bin/bash

runit() {
    bash ./fitsome.sh
}

while runit; do
    sleep 30;
done
