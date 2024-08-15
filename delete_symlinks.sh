dest_directory="/home/filip/Desktop/mammo_internship/train_img"

for symlink in "$dest_directory"/*; do
    if [ -L "$symlink" ]; then
        rm "$symlink"
        echo "Deleted symlink: $symlink"
    fi
done