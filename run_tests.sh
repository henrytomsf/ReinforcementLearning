#!/bin/bash

echo "Starting tests...."

python3 DDPG/tests.py
test_status=$(echo $?)

exit $test_status