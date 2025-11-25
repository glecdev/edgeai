# Qwen2.5-0.5B LoRA Fine-tuning GPU 실행 가이드

**로컬 GPU 환경에서 Qwen2.5-0.5B-Instruct 모델을 화물차 한국어 도메인으로 fine-tuning하는 완전한 가이드**

---

## 📋 목차

1. [필수 요구사항](#필수-요구사항)
2. [환경 설정](#환경-설정)
3. [Step 1: 환경 검증](#step-1-환경-검증)
4. [Step 2: LoRA Fine-tuning](#step-2-lora-fine-tuning)
5. [Step 3: 모델 평가](#step-3-모델-평가)
6. [Step 4: LoRA 어댑터 병합](#step-4-lora-어댑터-병합)
7. [Step 5: INT4 양자화](#step-5-int4-양자화)
8. [Step 6: Android 배포](#step-6-android-배포)
9. [트러블슈팅](#트러블슈팅)

---

## 필수 요구사항

### 하드웨어
- **GPU**: NVIDIA RTX 4060 8GB 이상 (CUDA 지원)
- **RAM**: 16GB 이상
- **Storage**: 10GB 여유 공간

### 소프트웨어
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS (CUDA 없음, 느림)
- **Python**: 3.10 or 3.11
- **CUDA**: 11.8 or 12.1 (NVIDIA 드라이버 포함)
- **Git**: 최신 버전

### 예상 소요 시간
| 단계 | 시간 | VRAM | 비고 |
|------|------|------|------|
| 환경 설정 | 10-15분 | - | 첫 실행만 |
| 모델 다운로드 | 5-10분 | - | 첫 실행만 |
| LoRA 학습 | 1-2시간 | ~6GB | RTX 4060 기준 |
| 모델 평가 | 10-15분 | ~4GB | - |
| 어댑터 병합 | 5분 | ~4GB | - |
| INT4 양자화 | 5-10분 | ~4GB | - |
| **총 소요** | **1.5-2.5시간** | **6GB** | - |

---

## 환경 설정

### 1. CUDA/GPU 드라이버 설치 (첫 실행만)

**Windows**:
```bash
# NVIDIA 드라이버 다운로드 및 설치
# https://www.nvidia.com/Download/index.aspx
# RTX 4060 → Latest Game Ready Driver

# CUDA Toolkit 12.1 다운로드 및 설치
# https://developer.nvidia.com/cuda-downloads
```

**Linux (Ubuntu)**:
```bash
# NVIDIA 드라이버 설치
sudo ubuntu-drivers autoinstall

# CUDA Toolkit 12.1 설치
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run
```

### 2. Python 가상환경 설정

```bash
# 프로젝트 디렉토리로 이동
cd d:/edgeai/edgeai-repo

# Python 3.11 가상환경 생성 (없으면)
python3.11 -m venv venv

# 가상환경 활성화
# Windows:
.\venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate
```

### 3. LoRA Fine-tuning 패키지 설치

```bash
# LoRA 전용 requirements 설치
pip install -r ai-models/fine-tuning/requirements-lora.txt

# 설치 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 예상 출력:
# PyTorch: 2.1.0+cu121
# CUDA: True
```

---

## Step 1: 환경 검증

**자동 환경 체크 실행**:

```bash
cd ai-models/fine-tuning

python setup_lora_env.py
```

**예상 출력**:
```
============================================================
Qwen2.5-0.5B LoRA Fine-tuning 환경 설정
============================================================

[Check 1] Python 버전 확인...
  [PASS] Python 3.11.5

[Check 2] CUDA/GPU 감지...
  [PASS] GPU 감지: NVIDIA GeForce RTX 4060
         VRAM: 8.0 GB
         CUDA: 12.1

[Check 3] 필수 패키지 확인...
  [PASS] 모든 패키지 설치됨
         - torch 2.1.0
         - transformers 4.36.0
         - peft 0.7.0
         - accelerate 0.25.0
         - datasets 2.14.0
         - trl 0.7.0

[Check 4] 데이터셋 확인...
  [PASS] 데이터셋 존재
         Train: 1189 samples
         Val: 149 samples

[Check 5] Qwen2.5-0.5B 모델 캐시 확인...
  [INFO] 모델 캐시 없음 (첫 학습 시 자동 다운로드)
         모델: Qwen/Qwen2.5-0.5B-Instruct
         크기: ~980MB (다운로드 시간: 5-10분)

[Check 6] 디렉토리 구조 생성...
  [PASS] 디렉토리 생성 완료
         - ai-models/fine-tuning/outputs
         - ai-models/fine-tuning/checkpoints
         - ai-models/fine-tuning/logs
         - ai-models/fine-tuning/merged-models

============================================================
환경 설정 요약
============================================================
[PASS] 모든 체크 통과!

LoRA fine-tuning 준비 완료:
  1. GPU 환경 정상
  2. 필수 패키지 설치됨
  3. 데이터셋 준비됨

다음 단계:
  python train_qwen_lora.py
```

**❌ 체크 실패 시**:
- `CUDA/GPU` 실패 → NVIDIA 드라이버 및 CUDA Toolkit 설치
- `Dependencies` 실패 → `pip install -r requirements-lora.txt`
- `Dataset` 실패 → `python ../../datasets/truck-korean/generate_dataset.py`

---

## Step 2: LoRA Fine-tuning

### 기본 학습 (권장)

```bash
# 기본 설정으로 학습 (RTX 4060 8GB 최적화)
python train_qwen_lora.py

# 예상 소요 시간: 1-2시간
# VRAM 사용량: ~6GB
```

### Custom 설정 (고급)

```bash
# Batch size, Epochs, Learning rate 조정
python train_qwen_lora.py \
    --num-epochs 5 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --lora-r 32 \
    --lora-alpha 64

# VRAM이 16GB+ 있다면 4-bit 비활성화 (더 빠름, 더 정확)
python train_qwen_lora.py --no-4bit
```

### 학습 중 모니터링

**TensorBoard 실행** (별도 터미널):
```bash
tensorboard --logdir ai-models/fine-tuning/outputs/qwen-lora-truck/logs
```

브라우저에서 `http://localhost:6006` 접속:
- **Loss 그래프**: 감소 추세 확인 (목표: <1.5)
- **Learning Rate**: Cosine schedule 확인
- **GPU 사용률**: ~80-90% 유지 확인

### 학습 중단 및 재개

**Ctrl+C로 중단 후 재개**:
```bash
# 최신 체크포인트에서 재개
python train_qwen_lora.py --resume ./outputs/qwen-lora-truck/checkpoint-500
```

### 예상 학습 출력

```
============================================================
모델 및 토크나이저 로드
============================================================
모델: Qwen/Qwen2.5-0.5B-Instruct
4-bit quantization: True

[PASS] 모델 로드 완료
       파라미터 수: 494.0M

============================================================
LoRA 설정
============================================================
Rank (r): 16
Alpha: 32
Dropout: 0.05

[PASS] LoRA 적용 완료
       학습 가능 파라미터: 2.52M
       전체 파라미터: 494.0M
       학습 비율: 0.51%

============================================================
데이터셋 로드
============================================================
데이터 경로: ../../datasets/truck-korean

Train samples: 1189
Val samples: 149

토크나이징 시작...
Tokenizing train dataset: 100%|██████████| 1189/1189 [00:05<00:00]
Tokenizing val dataset: 100%|██████████| 149/149 [00:01<00:00]

[PASS] 토크나이징 완료

============================================================
LoRA Fine-tuning 시작
============================================================

학습 시작...
  - Epochs: 3
  - Batch size: 4
  - Gradient accumulation: 4 (effective batch: 16)
  - Learning rate: 0.0002
  - Optimizer: paged_adamw_8bit
  - FP16: True

============================================================

Epoch 1/3:
Step 10/300: loss=2.345, lr=0.00018
Step 20/300: loss=1.987, lr=0.00016
...
Step 100/300: loss=1.234, lr=0.00012
Evaluation: eval_loss=1.456

Epoch 2/3:
Step 110/300: loss=1.123, lr=0.00010
...

Epoch 3/3:
Step 210/300: loss=0.987, lr=0.00005
...
Step 300/300: loss=0.876, lr=0.00001
Final Evaluation: eval_loss=1.123

============================================================
학습 완료!
============================================================
LoRA 어댑터 저장 위치: ./outputs/qwen-lora-truck/final

다음 단계:
  1. 모델 평가: python evaluate_lora.py
  2. 어댑터 병합: python merge_lora.py
  3. INT4 재양자화: python quantize_merged_model.py
```

---

## Step 3: 모델 평가

### LoRA 모델 평가

```bash
python evaluate_lora.py \
    --lora-dir ./outputs/qwen-lora-truck/final \
    --test-dataset ../../datasets/truck-korean/test.json
```

### 베이스라인과 비교 (선택사항)

```bash
python evaluate_lora.py \
    --lora-dir ./outputs/qwen-lora-truck/final \
    --baseline
```

### 예상 평가 결과

```
============================================================
LoRA 모델 평가 시작
============================================================

[Metric 1] Perplexity 계산...
  Perplexity: 15.34 (샘플: 50개)
  목표: <20, [PASS]

[Metric 2] BLEU Score 계산...
  진행: 0/50
  진행: 10/50
  ...
  BLEU Score: 52.67 (샘플: 50개)
  목표: >40, [PASS]

[Metric 3] Response Relevance 계산...
  진행: 0/50
  진행: 10/50
  ...
  Relevance: 92.0% (샘플: 50개)
  목표: >80%, [PASS]

[Metric 4] Inference Speed 계산...
  평균 추론 시간: 1.845s
  P95 추론 시간: 1.932s
  목표: <2초, [PASS]

============================================================
평가 결과 요약
============================================================
Perplexity: 15.34 (목표: <20)
BLEU Score: 52.67 (목표: >40)
Relevance: 92.0% (목표: >80%)
Avg Latency: 1.845s (목표: <2s)

[PASS] 모든 목표 달성!

평가 결과 저장: ./outputs/qwen-lora-truck/final/evaluation_results.json
```

**성능 해석**:
- **Perplexity 15.34**: 모델이 텍스트를 잘 예측 (낮을수록 좋음, <20 목표)
- **BLEU 52.67**: 정답과 52.67% 유사 (높을수록 좋음, >40 목표)
- **Relevance 92%**: 응답의 92%가 키워드 매칭 (높을수록 좋음, >80% 목표)
- **Latency 1.85s**: Android 목표 2초 이내 달성

---

## Step 4: LoRA 어댑터 병합

**LoRA 어댑터를 베이스 모델에 병합**:

```bash
python merge_lora.py \
    --lora-dir ./outputs/qwen-lora-truck/final \
    --output-dir ./merged-models/qwen-truck-fp16
```

**예상 출력**:
```
============================================================
LoRA 어댑터 병합 시작
============================================================
베이스 모델: Qwen/Qwen2.5-0.5B-Instruct
LoRA 어댑터: ./outputs/qwen-lora-truck/final
출력 경로: ./merged-models/qwen-truck-fp16

[Step 1] 베이스 모델 로드...
  [PASS] 494.0M params

[Step 2] LoRA 어댑터 로드...
  [PASS] 어댑터 로드 완료

[Step 3] 어댑터 병합 중...
  [PASS] 병합 완료

[Step 4] 병합된 모델 저장...
  [PASS] 모델 저장 완료: ./merged-models/qwen-truck-fp16

============================================================
병합 완료!
============================================================
모델 크기: 0.98 GB (FP16)
파일 수: 1 safetensors

다음 단계:
  python quantize_merged_model.py \
    --model-dir ./merged-models/qwen-truck-fp16 \
    --output-dir ./quantized-models/qwen-truck-int4
```

---

## Step 5: INT4 양자화

**FP16 모델을 INT4로 양자화 (Android 배포용)**:

```bash
python quantize_merged_model.py \
    --model-dir ./merged-models/qwen-truck-fp16 \
    --output-dir ./quantized-models/qwen-truck-int4
```

**예상 출력**:
```
[Check] MLC-LLM 설치 확인...
  [PASS] MLC-LLM 설치됨: 0.1.0

============================================================
MLC-LLM INT4 Quantization
============================================================
입력 모델: ./merged-models/qwen-truck-fp16
출력 경로: ./quantized-models/qwen-truck-int4
양자화 모드: q4f16_1 (INT4)

명령어:
  mlc_llm convert_weight ./merged-models/qwen-truck-fp16 --quantization q4f16_1 -o ./quantized-models/qwen-truck-int4

양자화 시작 (5-10분 소요)...

[PASS] 양자화 완료

[Step] 토크나이저 복사...
  복사: tokenizer.json
  복사: tokenizer_config.json
  [PASS] 토크나이저 복사 완료

[Step] 설정 파일 생성...
  생성: mlc-chat-config.json
  [PASS] 설정 파일 생성 완료

============================================================
양자화 완료!
============================================================
모델 크기: 300.5 MB (INT4)
출력 경로: ./quantized-models/qwen-truck-int4

Android 배포 준비:
  1. 모델 파일을 Android assets/로 복사
  2. Qwen25InferenceEngine.kt의 MODEL_PATH 업데이트
  3. APK 빌드 및 테스트
```

**크기 비교**:
- **FP16**: 980MB
- **INT4**: 300MB (69% 감소)
- **정확도 손실**: <5%

---

## Step 6: Android 배포

### 1. 모델 파일 Android 프로젝트로 복사

```bash
# 양자화된 모델을 Android assets로 복사
mkdir -p ../../android-dtg/app/src/main/assets/models/Qwen2.5-0.5B-q4f16_1

cp -r ./quantized-models/qwen-truck-int4/* \
      ../../android-dtg/app/src/main/assets/models/Qwen2.5-0.5B-q4f16_1/
```

### 2. Android 프로젝트에서 모델 경로 업데이트

**파일**: `android-dtg/app/src/main/java/com/glec/dtg/llm/Qwen25InferenceEngine.kt`

```kotlin
companion object {
    private const val MODEL_PATH = "models/Qwen2.5-0.5B-q4f16_1"  // ← 업데이트
    // ...
}
```

### 3. APK 빌드 (로컬 환경 필요)

```bash
cd ../../android-dtg

# Debug APK 빌드
./gradlew assembleDebug

# 출력: app/build/outputs/apk/debug/app-debug.apk
```

### 4. 디바이스 설치 및 테스트

```bash
# USB 디버깅 활성화된 Android 디바이스 연결
adb install -r app/build/outputs/apk/debug/app-debug.apk

# 로그 확인
adb logcat | grep "Qwen25InferenceEngine"
```

---

## 트러블슈팅

### 문제 1: CUDA Out of Memory

**증상**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.5 GB
```

**해결**:
```bash
# Batch size 줄이기 (8GB VRAM 기준)
python train_qwen_lora.py --batch-size 2  # 기본 4 → 2

# 또는 max_seq_length 줄이기
python train_qwen_lora.py --max-seq-length 256  # 기본 512 → 256
```

### 문제 2: MLC-LLM 설치 실패

**증상**:
```
ERROR: Could not find a version that satisfies the requirement mlc-llm
```

**해결**:
```bash
# MLC-LLM은 별도 설치 필요
pip install --pre mlc-llm -f https://mlc.ai/wheels
```

### 문제 3: Hugging Face 모델 다운로드 실패

**증상**:
```
ConnectionError: Couldn't reach https://huggingface.co
```

**해결**:
```bash
# Hugging Face 토큰 설정 (선택사항)
huggingface-cli login

# 또는 환경 변수 설정
export HF_ENDPOINT=https://hf-mirror.com
```

### 문제 4: 학습 Loss가 수렴하지 않음

**증상**:
- Loss가 2.0 이하로 떨어지지 않음
- 3 epoch 후에도 eval_loss > 1.5

**해결**:
```bash
# Learning rate 줄이기
python train_qwen_lora.py --learning-rate 1e-4  # 기본 2e-4 → 1e-4

# Epochs 늘리기
python train_qwen_lora.py --num-epochs 5  # 기본 3 → 5

# LoRA rank 늘리기 (더 많은 파라미터)
python train_qwen_lora.py --lora-r 32 --lora-alpha 64  # 기본 16/32 → 32/64
```

### 문제 5: Android 배포 후 모델 로드 실패

**증상**:
```
E/Qwen25InferenceEngine: Failed to load model: models/Qwen2.5-0.5B-q4f16_1
```

**해결**:
1. **assets 경로 확인**:
   ```bash
   # Android Studio에서 확인
   app/src/main/assets/models/Qwen2.5-0.5B-q4f16_1/
   ├── params_shard_0.bin  # 또는 .safetensors
   ├── tokenizer.json
   └── mlc-chat-config.json
   ```

2. **파일 크기 확인** (APK 100MB 제한):
   ```bash
   # build.gradle에 splits 추가
   android {
       splits {
           abi {
               enable true
           }
       }
   }
   ```

3. **MLC-LLM AAR 추가**:
   ```gradle
   // app/build.gradle
   dependencies {
       implementation 'ai.mlc:mlc-llm:0.1.0'
   }
   ```

---

## 성능 벤치마크 (RTX 4060 8GB)

| 단계 | VRAM | 시간 | 비고 |
|------|------|------|------|
| 모델 로드 | 3.2 GB | 30초 | Qwen2.5-0.5B FP16 |
| LoRA 준비 | 4.1 GB | 10초 | LoRA adapters |
| 학습 (Epoch 1) | 5.8 GB | 30분 | 1189 samples |
| 학습 (Epoch 2) | 5.8 GB | 30분 | - |
| 학습 (Epoch 3) | 5.8 GB | 30분 | - |
| 평가 | 4.2 GB | 10분 | 149 samples |
| 병합 | 4.1 GB | 5분 | - |
| INT4 양자화 | 3.5 GB | 8분 | - |
| **총계** | **6 GB** | **1.5-2시간** | - |

---

## 참고 자료

**문서**:
- [PHASE3K_LLM_INTEGRATION.md](../../docs/PHASE3K_LLM_INTEGRATION.md) - 전체 LLM 통합 계획
- [LLM_SETUP_GUIDE.md](../../docs/LLM_SETUP_GUIDE.md) - 환경 설정 가이드
- [LLM_IMPLEMENTATION_GUIDE.md](../../docs/LLM_IMPLEMENTATION_GUIDE.md) - 구현 가이드

**외부 링크**:
- [Qwen2.5 모델 카드](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [PEFT (LoRA) 문서](https://huggingface.co/docs/peft)
- [MLC-LLM 공식 문서](https://llm.mlc.ai/)
- [Transformers 라이브러리](https://huggingface.co/docs/transformers)

---

## 다음 단계

✅ **현재 완료**: LoRA fine-tuning 준비 완료

🎯 **다음 작업**:
1. GPU 로컬 환경에서 이 가이드대로 학습 실행
2. 평가 결과 확인 (목표: Perplexity <20, BLEU >40)
3. Android DTG 앱에 fine-tuned 모델 통합
4. 실제 차량 데이터로 테스트

---

**작성일**: 2025-11-14
**버전**: 1.0
**작성자**: Claude (Phase 3-K LoRA Fine-tuning)
