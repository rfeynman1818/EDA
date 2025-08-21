import shutil, os

def copy_files_flat(file_paths, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for file_path in file_paths:
        if os.path.isfile(file_path):
            try:
                shutil.copy(file_path, os.path.join(target_dir, os.path.basename(file_path)))
                print(f"Copied: {file_path}")
            except Exception as e:
                print(f"Failed to copy {file_path}: {e}")
        else:
            print(f"Not found: {file_path}")

# Example usage
files = [
    "/path/to/file1.txt",
    "/path/to/file2.pdf",
    "/path/to/nonexistent.docx"
]
destination = "/path/to/target_directory"
copy_files_flat(files, destination)
