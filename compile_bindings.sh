#!/bin/bash

PROMPT=$(cat compile_bindings.txt)
echo "$PROMPT"

codex --full-auto -q "$PROMPT"
