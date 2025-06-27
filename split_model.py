from progress_bar import progress_bar
import os

def split_bytes(path, max_size, output_dir):
    with open(path, 'rb') as f:
        byte_str = f.read()
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    start = 0
    end = max_size
    i = 0
    while start < len(byte_str):
        with open(os.path.join(output_dir, f'{path}.{i}'), "wb") as f:
            f.write(byte_str[start:min(end, len(byte_str))])
            start = end
            end += max_size
            progress_bar(end/len(byte_str), 20)
            i += 1

if __name__ == "__main__":
    split_bytes("ascii_reader.keras", 52428800, "split_model")