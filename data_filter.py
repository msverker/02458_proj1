import os

def filter_data(subset_path, utkcropped_path):
    if not os.path.exists(subset_path):
        os.makedirs(subset_path)

    for filename in os.listdir(utkcropped_path):
        if filename.endswith('.jpg'):
            parts = filename.split('_')
            if parts[0] in ['60', '61', '62', '63', '64', '65']:
                if parts[1] in ['0']:
                    if parts[2] in ['0']:
                        src = os.path.join(utkcropped_path, filename)
                        dst = os.path.join(subset_path, filename)
                        os.rename(src, dst)
                        print(f"Moved: {filename}")
    

if __name__ == "__main__":
    subset_path = 'subset'
    utkcropped_path = 'utkcropped/utkcropped' #path to utkcropped folder
    # filter_data(subset_path, utkcropped_path)
    print(len(os.listdir(subset_path)))