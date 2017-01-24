#!/usr/bin/env bash
# 2017-01-23
#
# Test the goods

cd $(dirname $0)/..
nodemon -e '*.*' -x "nosetests && beep || espeak 'failed ya dumb bitch'"
