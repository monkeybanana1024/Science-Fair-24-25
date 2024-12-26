#!/bin/bash

# Enable verbose mode
set -x

# Check if a commit message was provided
if [ -z "$1" ]; then
    echo "Error: No commit message provided."
    echo "Usage: ./your_script.sh 'Your commit message'"
    exit 1
fi

# Add all changes to the staging area
git add .
git restore --staged "Remote\ Sensing/Venv/"

# Commit the changes with the provided message
git commit -m "$1"

# Push the changes to the remote repository
git push

# Disable verbose mode (optional)
set +x

echo "Done!"
