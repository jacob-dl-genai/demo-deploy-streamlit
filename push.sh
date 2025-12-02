#!/bin/bash

# 1. Initialize Git if not already done
if [ ! -d ".git" ]; then
    echo "Initializing new Git repository..."
    git init
    git branch -M main  # Ensure branch is named main
else
    echo "Git repository already initialized."
fi

# 2. Initialize Git LFS
echo "🐘 Initializing Git LFS..."
git lfs install

# 3. Track large files in the specific directory
# We track .bin (PyTorch default) and .safetensors (Newer HuggingFace default)
git lfs track "tiny_bert_sentiment/*.bin"
git lfs track "tiny_bert_sentiment/*.safetensors"
git lfs track "tiny_bert_sentiment/*.pth"

# 4. Stage all files (LFS config, model weights, app code, requirements)
git add .

# 5. Commit changes
git commit -m "Initialize Git LFS, add model weights and application code"

# 6. Push to remote (Interactive check)

# Check if 'origin' remote exists
if ! git remote | grep -q "origin"; then
    echo "No remote 'origin' found."
    echo "To push, we need a destination URL."
    read -p "Enter your GitHub repository URL (e.g., https://github.com/user/repo.git): " REPO_URL
    
    if [ -n "$REPO_URL" ]; then
        git remote add origin "$REPO_URL"
        echo "Remote 'origin' added."
    else
        echo "No URL provided. Skipping push. Run 'git remote add origin <url>' and 'git push' later."
        exit 1
    fi
fi

# Push to main
echo "Pushing to origin main..."
git push -u origin main

echo "Done! Code and LFS files pushed successfully."