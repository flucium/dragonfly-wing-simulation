#!/bin/bash
set -e
ls -lh ./notebooks/*.ipynb && \
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace ./notebooks/*.ipynb && \
ls -lh ./notebooks/*.ipynb