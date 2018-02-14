#!/usr/bin/env bash

for file in *.csv; do python generate_vector.py ${file} & done