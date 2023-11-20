wget --quiet --show-progress "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
mkdir TinyStories
tar -xf TinyStories_all_data.tar.gz -C TinyStories
rm TinyStories_all_data.tar.gz
