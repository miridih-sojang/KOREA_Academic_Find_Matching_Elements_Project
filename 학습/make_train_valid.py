import os
import glob
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random


def filter_corrupt_images(examples):
    """remove problematic images"""
    valid_images = []
    for elements_image_file, removed_image_path  in zip(examples["elements_image_path"], examples['removed_image_path']):
        if (os.path.exists(elements_image_file)) & (os.path.exists(removed_image_path)):
            valid_images.append(True)
        else:
            valid_images.append(False)

    return valid_images


def make_datasetdict(image_dir, val_percentage=0.1, discard_text_ratio=False, num_proc=32):

    random.seed(42)

    # 제거해야하는 이미지 리스트
    error_img_path_list = [file for file in os.listdir(image_dir) if file.startswith('null_elements')]
    error_img_list=[]

    for error_path in error_img_path_list:
        error_img_list.extend(pd.read_csv(os.path.join(image_dir,error_path)).image_file_name.values.tolist())


    # train zip 푼 파일들
    dir_list = []
    for dir in os.listdir(image_dir):
        if (dir.startswith('train_image'))|(dir.startswith('data_')):
            dir_list.append(dir)

    dir_list = sorted(dir_list)

    filtered_png_files = []

    # 모든 디렉터리를 순회하며 .png 파일을 검색
    for i in tqdm(dir_list):
        # 모든 .png 파일을 검색 (재귀적으로)
        all_png_files = glob.glob(os.path.join(image_dir, f'{i}/**/elements/*.png'), recursive=True)

        # 'removed'가 없는 파일만 필터링
        filtered_png_files.extend([file for file in all_png_files if 'removed' not in os.path.basename(file)])

    # 중복 제거를 위한 리스트의 집합 변환 후 다시 리스트로 변환
    filtered_png_list = list(set(filtered_png_files))
    filtered_png_list = [i for i in filtered_png_list if i not in error_img_list]

    # 요소 제외된 이미지 리스트 생성
    filtered_removed_png_list = [image_path.replace('.png','_removed.png') for image_path in filtered_png_list]

    # DataFrame으로 변환
    df = pd.DataFrame({'elements_image_path': filtered_png_list,
                        'removed_image_path' : filtered_removed_png_list})

    df['image_file_name'] = df['elements_image_path'].apply(lambda x: os.path.basename(x))

    # 메타데이터 로드
    meta_img_path_list = [file for file in os.listdir(image_dir) if file.startswith('metadata_elements')]

    meta_df = pd.DataFrame()
    for i in meta_img_path_list:
        meta_df = pd.concat([meta_df,pd.read_csv(os.path.join(image_dir,i))]).reset_index(drop=True)
        
    # 병합 및 필터링
    concated_df = pd.merge(df, meta_df, how='left')
    concated_df = concated_df[~concated_df['tag'].isin(['SHAPESVG','LineShapeItem','Barcode','QRcode'])].reset_index(drop=True)

    error_case_index = concated_df[(concated_df['img_height_resized']<2)|(concated_df['img_width_resized']<2)]['elements_image_path'].index.tolist()
    concated_df = concated_df.drop(error_case_index).reset_index(drop=True)

    # 텍스트 일정 비율 제거
    if discard_text_ratio:
        drop_text_index = concated_df[concated_df['tag']=='TEXT'].sample(frac=discard_text_ratio,random_state=42).index.tolist()
        concated_df = concated_df.drop(drop_text_index).reset_index(drop=True)

    df = concated_df[['elements_image_path','removed_image_path']]

    # train과 validation set으로 분리
    train_df, val_df = train_test_split(df, test_size=val_percentage, random_state=42)

    # HuggingFace datasets 객체로 변환
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)


    # Validity Check
    train_dataset = train_dataset.filter(
        filter_corrupt_images, batched=True, num_proc=num_proc, desc= 'Filtering Train Set..'
    )

    val_dataset = val_dataset.filter(
        filter_corrupt_images, batched=True, num_proc=num_proc, desc= 'Filtering Validation Set..'
    )

    # DatasetDict에 train과 validation dataset 추가
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    
    dataset_dict = dataset_dict.remove_columns('__index_level_0__')

    return dataset_dict


if __name__ == "__main__":
    train_image_dir = '/mnt/raid6/dltmddbs100/miricanbus/train/'

    dataset = make_datasetdict(train_image_dir)
    dataset.save_to_disk(os.path.join(train_image_dir,'train_v01'))