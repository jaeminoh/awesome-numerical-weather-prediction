#!/bin/bash

mkdocs gh-deploy --force --no-history
rm -rf site/
cp docs/index.md README.md