import os
import requests
from tqdm import tqdm
import zipfile

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def main():
    # Create directories
    os.makedirs('data/coco/train2017', exist_ok=True)
    os.makedirs('data/coco/val2017', exist_ok=True)
    os.makedirs('data/coco/annotations', exist_ok=True)

    # URLs for COCO dataset
    train_url = 'http://images.cocodataset.org/zips/train2017.zip'
    val_url = 'http://images.cocodataset.org/zips/val2017.zip'
    annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

    # Download files
    print("Downloading train2017.zip...")
    download_file(train_url, 'data/coco/train2017.zip')
    
    print("Downloading val2017.zip...")
    download_file(val_url, 'data/coco/val2017.zip')
    
    print("Downloading annotations_trainval2017.zip...")
    download_file(annotations_url, 'data/coco/annotations_trainval2017.zip')

    # Extract files
    print("Extracting train2017.zip...")
    with zipfile.ZipFile('data/coco/train2017.zip', 'r') as zip_ref:
        zip_ref.extractall('data/coco')
    
    print("Extracting val2017.zip...")
    with zipfile.ZipFile('data/coco/val2017.zip', 'r') as zip_ref:
        zip_ref.extractall('data/coco')
    
    print("Extracting annotations_trainval2017.zip...")
    with zipfile.ZipFile('data/coco/annotations_trainval2017.zip', 'r') as zip_ref:
        zip_ref.extractall('data/coco')

    # Clean up zip files
    os.remove('data/coco/train2017.zip')
    os.remove('data/coco/val2017.zip')
    os.remove('data/coco/annotations_trainval2017.zip')

    print("Download and extraction complete!")

if __name__ == '__main__':
    main()
