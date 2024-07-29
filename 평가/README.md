# 검증 코드

- 검증 코드는 이전 중간발표 결과를 전달하기 위해, 테스트 셋으로 7월 26일 새로 전달해주신 키워드가 아닌 이전에 전달해주신 키워드 셋을 사용합니다.

## 이미지 데이터 및 모델 준비

### valid 이미지 준비
- 현재 폴더 내에 valid 폴더를 위치해 주세요.
- 예시로, 아래와 같습니다.
```
valid/
├── 00009785/
│   └── 000/
│       ├── elements/
│       └── images/
├── 00009839/
│   └── 000/
│       ├── elements/
│       └── images/
├── 00010081/
│   └── 000/
│       ├── elements/
│       └── images/
├── 00010196/
│   └── 000/
│       ├── elements/
│       └── images/
├── 00010463/
│   └── 000/
│       ├── elements/
│       └── images/
├── 00010480/
│   └── 000/
│       ├── elements/
│       └── images/
└── 00010498/
│   └── 000/
│       ├── elements/
│       └── images/
│
│ ... (생략)
```

## 모델 준비
- 현재 폴더 내에 ckpt 폴더를 만들고, 해당 폴더에 학습된 모델을 위치해 주세요. 단, 사전학습 모델인 `google/efficientnet-b4`은 위치하지 않아도 됩니다.

```
ckpt/
├── eff4_v01_discard:1_epoch:5
└── eff4_v01_epoch:5
```

## 각 코드 및 파일 설명
- `eval_keyword_first_set` : first 키워드 요소에 해당하는 background 이미지를 매핑한 csv 파일을 모아둔 폴더입니다. 해당 csv 파일을 사용하여 평가를 진행합니다.
- `keyword_first_second_images` : first와 second 키워드 요소 이미지를 모아둔 것입니다. 배경이미지가 존재하지 않거나, 이미지 프로세서에서 에러가 생기는 이미지는 제거 했습니다.
- `1_evaluation.py` : 모델을 통해 평가를 각 keyword 별 평가를 진행하는 evaluation 코드입니다. 입력은 csv 파일이며, 출력은 `output_5_first` 폴더 내 csv 파일을 참조하세요.
- `2_merge_eval.py` : `1_evaluation.py`를 통해 나온 결과를 모델별로 결합하는 코드입니다.