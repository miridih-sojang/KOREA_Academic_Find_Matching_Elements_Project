import os
import pandas as pd


class ImageDataset:
    def __init__(self):
        self.lowest_file_list = self.find_lowest_files()
        self.df = pd.DataFrame(self.lowest_file_list, columns=['raw_image_path'])
        self.transform()
        self.filter()

    def find_lowest_files(self, verbose_len=1000000):
        lowest_file_list = []
        for parent, _, files in os.walk(f'/opt/raw_data_storage/images'):
            for f in files:
                lowest_file_list.append(os.path.join(parent, f))
                if len(lowest_file_list) % verbose_len == 0:
                    print(f'Preprocessing : {len(lowest_file_list)}')
        return lowest_file_list

    def transform(self):
        self.df['extension'] = self.df.raw_image_path.str.split('.').str[-1]
        self.df['raw_image_path'] = self.df['raw_image_path'].str.replace(f'/opt/raw_data_storage/images/', '')
        self.df['image_file_name'] = self.df['raw_image_path'].str.split('/').str[-1]
        self.df['_folder'] = self.df['raw_image_path'].str.split('/').str[0]

    def filter(self):
        self.df = self.df[self.df.extension == 'png']
        # self.df = self.df[~self.df.raw_image_path.str.endswith('_removed.png')]
        # self.df = self.df[self.df._folder.isin(self.config['use_image_folder'])]