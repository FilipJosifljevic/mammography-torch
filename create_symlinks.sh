src_directory="/home/filip/rsna_img/train_images"
dest_directory="/home/filip/train_img"

mkdir -p "$dest_directory"

for patient_dir in "$src_directory"/*; do
    if [ -d "$patient_dir" ]; then
        patient_id=$(basename "$patient_dir")

        for file in "$patient_dir"/*.dcm; do
            if [ -f "$file" ]; then
                filename=$(basename "$file")
                symlink_path="$dest_directory/$filename"

                # Create symlink
                ln -s "$file" "$symlink_path"
                echo "Created symlink: $symlink_path -> $file"
            fi
        done
    fi
done
