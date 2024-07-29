CSV_DIR="eval_keyword_first_set"
OUTPUT_DIR="output_5_first"

model_name="ckpt/eff4_v01_discard:1_epoch:5" # 평가에 사용할 모델 경로

mkdir -p $OUTPUT_DIR

for csv_file in $CSV_DIR/*.csv; do
    echo "Processing $csv_file..."
    CUDA_VISIBLE_DEVICES=0 python evaluation.py --csv_file $csv_file --output_dir $OUTPUT_DIR --num_samples 5 --image_column image_file_name --removed_column removed_elem_background_path --model_name_or_path $model_name
done

python 2_merge_eval.py --model_name $model_name --output_dir $OUTPUT_DIR

echo "All CSV files processed."
