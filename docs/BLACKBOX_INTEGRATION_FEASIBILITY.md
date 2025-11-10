# GLEC DTG - 블랙박스 영상 통합 기술 검증 보고서

**분석 일자**: 2025-01-10
**검증 대상**: 블랙박스 영상 데이터 엣지 분석 통합 가능성
**방법론**: 오픈소스 컴퓨터 비전 모델 + 현 DTG 하드웨어 스펙 분석

---

## 📋 Executive Summary

**결론**: ⚠️ **조건부 가능** - 경량 모델 및 샘플링 최적화 필수

### 핵심 요약
- **통합 가능 여부**: ✅ 가능 (단, 최적화 필수)
- **권장 접근법**: 이벤트 기반 분석 (실시간 전체 프레임 분석 아님)
- **모델 크기 증가**: +8~12MB (경량 모델 사용 시)
- **전력 소비 증가**: +0.5~1.0W (간헐적 분석 시)
- **추론 지연**: 100~200ms/이벤트 (허용 가능)

### 권장 아키텍처
```
블랙박스 → [이벤트 감지] → DTG 엣지 분석 → Fleet Platform
         (가속도/GPS)   (경량 CV 모델)   (MQTT 전송)
```

---

## 🎯 요구사항 분석

### 1. 블랙박스 연동 방법

#### A. 물리적 포트 연동 ✅ 권장
**인터페이스 옵션**:
1. **USB (권장)**: USB OTG로 블랙박스 연결
   - 전송 속도: USB 2.0 (480 Mbps)
   - 전력: USB 전원 공급 가능
   - 호환성: 대부분의 블랙박스 지원 (USB 저장 모드)

2. **Wi-Fi Direct**: 블랙박스 Wi-Fi 기능 활용
   - 전송 속도: 802.11n (150~300 Mbps)
   - 전력: 중간 소비 (+0.3~0.5W)
   - 호환성: 최신 블랙박스 (2020년 이후)

3. **SD Card Slot**: 블랙박스 SD 카드를 DTG에 삽입
   - 전송 속도: Class 10 (10 MB/s)
   - 전력: 최소 소비
   - 호환성: 모든 블랙박스

#### B. 블루투스 연동 ⚠️ 비권장
**제약사항**:
- 전송 속도: BLE 5.0 (2 Mbps 최대)
- **문제**: 영상 데이터 실시간 전송 불가능
- **용도**: 메타데이터 전송만 적합 (이벤트 알림, 파일 목록)

### 2. 한국 블랙박스 시장 현황

**주요 제조사**:
- 아이나비 (Inavy): Wi-Fi/USB 지원
- 파인뷰 (FineVu): Wi-Fi/USB 지원
- 팅크웨어 (Thinkware): Wi-Fi/LTE 지원
- 블랙뷰 (BlackVue): Wi-Fi/Cloud 지원

**공통 특징**:
- 해상도: Full HD (1920x1080) ~ 4K (3840x2160)
- 프레임레이트: 30 fps (일부 60 fps)
- 압축: H.264 / H.265 (HEVC)
- 저장: MP4 포맷, 1~3분 단위 파일

---

## 💻 DTG 현재 하드웨어 스펙

### Qualcomm Snapdragon 기반

**프로세서**:
- CPU: Octa-core (예: Snapdragon 660/710 급)
- GPU: Adreno 512/616
- **NPU/HTP**: Hexagon DSP (AI 가속)
- RAM: 4GB (일반적)

**현재 AI 워크로드**:
```
모델          크기      지연시간    메모리
─────────────────────────────────────────
LightGBM      5.7 MB    5-15 ms    ~10 MB
TCN (목표)    2-4 MB    15-25 ms   ~8 MB
LSTM-AE (목표) 2-3 MB   25-35 ms   ~7 MB
─────────────────────────────────────────
합계          ~12 MB    ~30 ms     ~25 MB
전력 소비: ~2W (추론 시)
```

**남은 리소스**:
- 모델 크기: ~2MB 여유 (목표 14MB 대비)
- 메모리: ~25MB 여유 (목표 50MB 대비)
- 전력: 제한적 (목표 2W, 현재 ~2W 사용)

---

## 🎬 블랙박스 영상 분석 요구사항

### 1. 실시간 전체 프레임 분석 (❌ 불가능)

**요구사항**:
- 해상도: 1920x1080
- 프레임레이트: 30 fps
- 처리: 30 frames/sec = **33ms/frame**

**문제점**:
1. **모델 크기**: 실시간 객체 감지 모델 20~50MB
2. **추론 지연**: 100~200ms/frame (NPU 사용 시)
3. **메모리**: 영상 버퍼 ~50MB + 모델 ~30MB = 80MB 초과
4. **전력**: GPU/NPU 상시 가동 시 +2~3W (총 4~5W)

**결론**: **DTG 스펙 초과** ❌

### 2. 이벤트 기반 샘플링 분석 (✅ 가능)

**접근법**: 특정 이벤트 발생 시에만 분석

**트리거 이벤트** (CAN 데이터 기반):
1. 급가속 (>3 m/s²)
2. 급감속 (<-4 m/s²)
3. 급회전 (각속도 > 임계값)
4. 충격 감지 (G-센서 >2G)
5. 과속 (>100 km/h)
6. 급정거 (완전 정지)

**분석 프로세스**:
```
1. 이벤트 감지 (CAN/IMU) → 0.1초
2. 블랙박스에서 전후 5초 영상 요청 → 1~2초
3. 키 프레임 추출 (3~5 프레임) → 0.2초
4. CV 모델 추론 (프레임당 150ms) → 0.5~0.75초
5. 결과 저장 및 전송 → 0.3초
─────────────────────────────────────────
총 소요 시간: 2~3초/이벤트
```

**빈도**: 평균 5~10 이벤트/시간 (정상 주행 시)

**리소스 사용**:
- 평균 전력: +0.1~0.2W (간헐적 분석)
- 피크 전력: +0.8~1.0W (분석 중)
- 평균 메모리: +10~15MB (버퍼)
- 피크 메모리: +40~50MB (분석 중)

**결론**: **DTG 스펙 허용 범위** ✅

---

## 🤖 오픈소스 컴퓨터 비전 모델 분석

### 1. 경량 객체 감지 모델

#### A. YOLOv5 Nano (✅ 권장)
**스펙**:
- 모델 크기: **3.8 MB** (INT8 양자화)
- 추론 지연: **50~80ms** (Snapdragon NPU)
- 정확도: mAP 28.0% (COCO dataset)
- 클래스: 80개 (차량, 사람, 신호등 등)

**장점**:
- 초경량, DTG에 통합 가능
- ONNX 변환 지원
- Qualcomm SNPE 최적화 가능

**단점**:
- 정확도 상대적으로 낮음 (nano 버전)

#### B. MobileNet SSD v2 (✅ 대안)
**스펙**:
- 모델 크기: **6.9 MB** (INT8 양자화)
- 추론 지연: **70~100ms** (Snapdragon NPU)
- 정확도: mAP 22.1% (COCO dataset)
- 클래스: 90개

**장점**:
- Google이 모바일용으로 최적화
- TFLite 직접 지원
- 낮은 전력 소비

**단점**:
- YOLOv5보다 느림

#### C. YOLOv8 Nano (✅ 최신 대안)
**스펙**:
- 모델 크기: **6.2 MB** (INT8 양자화)
- 추론 지연: **60~90ms** (Snapdragon NPU)
- 정확도: mAP 37.3% (COCO dataset)
- 클래스: 80개

**장점**:
- YOLOv5 대비 정확도 향상
- 최신 아키텍처 (2023년)
- 모바일 최적화

**단점**:
- YOLOv5보다 약간 큼

### 2. 차선 감지 모델

#### Ultra-Fast-Lane-Detection (✅ 권장)
**스펙**:
- 모델 크기: **2.3 MB** (CULane 데이터셋)
- 추론 지연: **15~25ms** (Snapdragon GPU)
- 정확도: 68.4% (CULane dataset)

**장점**:
- 극도로 빠름 (실시간 가능)
- 경량 모델
- 한국 도로 환경에서도 작동

**단점**:
- 복잡한 도로 (비, 눈)에서 정확도 하락

### 3. 운전자 행동 분석

#### Drowsiness Detection (Driver Monitoring)
**스펙**:
- 모델 크기: **4.5 MB** (Mediapipe Face Mesh 기반)
- 추론 지연: **40~60ms** (Snapdragon NPU)
- 감지: 졸음, 눈 깜빡임, 시선 방향

**단점**:
- 내부 카메라 필요 (블랙박스 전면 카메라와 별도)

---

## 📊 통합 시나리오 분석

### 시나리오 1: 최소 통합 (객체 감지만) ✅ 권장

**모델 구성**:
```
기존 AI 모델:
- LightGBM: 5.7 MB
- TCN: 3 MB (예상)
- LSTM-AE: 2.5 MB (예상)

추가 CV 모델:
- YOLOv5 Nano: 3.8 MB

─────────────────────────
총 모델 크기: 15.0 MB
```

**스펙 평가**:
| 항목 | 목표 | 현재 | 상태 |
|------|------|------|------|
| 모델 크기 | <14 MB | 15.0 MB | ⚠️ 7% 초과 |
| 추론 지연 (CAN) | <50 ms | ~30 ms | ✅ 여유 |
| 추론 지연 (CV) | <200 ms | ~80 ms | ✅ 양호 |
| 메모리 (평균) | <50 MB | ~40 MB | ✅ 여유 |
| 전력 (평균) | <2W | ~2.2W | ⚠️ 10% 초과 |

**완화 방법**:
1. **모델 크기**: TCN/LSTM-AE 추가 양자화 (목표 달성 가능)
2. **전력**: 이벤트 기반 분석 (평균 전력 유지)

**결론**: **통합 가능** ✅

### 시나리오 2: 완전 통합 (객체 + 차선 + 운전자) ⚠️ 제한적

**모델 구성**:
```
기존 AI 모델: 11.2 MB

추가 CV 모델:
- YOLOv5 Nano: 3.8 MB
- Ultra-Fast-Lane: 2.3 MB
- Driver Monitor: 4.5 MB

─────────────────────────
총 모델 크기: 21.8 MB
```

**스펙 평가**:
| 항목 | 목표 | 현재 | 상태 |
|------|------|------|------|
| 모델 크기 | <14 MB | 21.8 MB | ❌ 56% 초과 |
| 메모리 (피크) | <50 MB | ~70 MB | ❌ 40% 초과 |
| 전력 (피크) | <2W | ~3W | ❌ 50% 초과 |

**결론**: **DTG 스펙 초과** ❌

### 시나리오 3: 점진적 통합 (객체 → 차선) ✅ 최적

**Phase 1**: 객체 감지만 (YOLOv5 Nano)
- 모델 크기: 15.0 MB (⚠️ 약간 초과, 최적화 가능)
- 이벤트 기반 분석
- 전력 영향 최소

**Phase 2**: 차선 감지 추가 (선택적)
- 모델 크기: 17.3 MB (재평가 필요)
- 고속도로 주행 시에만 활성화
- 전력 영향 제어 가능

**결론**: **점진적 통합 권장** ✅

---

## 🔧 기술 구현 방안

### 1. 블랙박스 연동 아키텍처

#### A. USB OTG 연동 (권장)
```java
// Android DTG - USB Host 모드
class BlackboxManager {
    private UsbManager usbManager
    private UsbDevice blackboxDevice

    fun connectBlackbox() {
        // 블랙박스를 USB 저장 장치로 마운트
        val storageVolume = usbManager.getStorageVolumes()

        // MP4 파일 액세스
        val videoFiles = listVideoFiles(storageVolume)
    }

    fun fetchEventVideo(eventTimestamp: Long): File {
        // 이벤트 시간 기준 전후 5초 영상 추출
        val startTime = eventTimestamp - 5000
        val endTime = eventTimestamp + 5000

        // 해당 시간대의 블랙박스 파일 찾기
        val videoFile = findVideoByTimestamp(startTime, endTime)

        // 로컬로 복사 (분석용)
        return copyToLocal(videoFile)
    }
}
```

#### B. Wi-Fi Direct 연동 (대안)
```java
class BlackboxWiFiManager {
    fun connectViaWiFiDirect() {
        // 블랙박스 Wi-Fi AP 연결
        // RTSP 스트림 또는 HTTP 파일 다운로드
    }

    fun streamVideo(eventTimestamp: Long): InputStream {
        // RTSP 프로토콜로 실시간 스트림
        // 또는 HTTP로 파일 다운로드
    }
}
```

### 2. 이벤트 기반 분석 로직

```kotlin
// DTGForegroundService.kt
class DTGForegroundService : Service() {

    private val blackboxManager = BlackboxManager()
    private val cvAnalyzer = ComputerVisionAnalyzer()

    fun onCANDataReceived(canData: CANData) {
        // 1. 기존 AI 분석 (실시간)
        val behaviorResult = multiModelEngine.infer(canData)

        // 2. 이벤트 감지
        val event = detectEvent(canData)

        if (event != null) {
            // 3. 블랙박스 영상 분석 트리거 (비동기)
            GlobalScope.launch(Dispatchers.IO) {
                analyzeEventVideo(event)
            }
        }
    }

    private fun detectEvent(canData: CANData): Event? {
        return when {
            // 급가속
            canData.accelerationX > 3.0 -> Event.HARSH_ACCELERATION

            // 급감속
            canData.accelerationX < -4.0 -> Event.HARSH_BRAKING

            // 급회전
            abs(canData.gyroZ) > 30.0 -> Event.SHARP_TURN

            // 충격
            canData.accelerationZ > 11.81 -> Event.COLLISION

            // 과속
            canData.vehicleSpeed > 100.0 -> Event.SPEEDING

            else -> null
        }
    }

    private suspend fun analyzeEventVideo(event: Event) {
        // 1. 블랙박스에서 영상 가져오기 (전후 5초)
        val videoFile = blackboxManager.fetchEventVideo(event.timestamp)

        // 2. 키 프레임 추출 (3~5 프레임)
        val frames = extractKeyFrames(videoFile, count = 5)

        // 3. CV 모델 추론
        val results = frames.map { frame ->
            cvAnalyzer.detectObjects(frame)
        }

        // 4. 분석 결과 저장
        saveAnalysisResult(event, results)

        // 5. Fleet Platform으로 전송
        mqttClient.publish("dtg/events/${deviceId}", eventPayload)
    }
}
```

### 3. CV 모델 통합

```kotlin
class ComputerVisionAnalyzer {
    private lateinit var yoloSession: OrtSession

    fun initialize() {
        // ONNX Runtime 세션 생성
        val env = OrtEnvironment.getEnvironment()
        val sessionOptions = OrtSession.SessionOptions()

        // Qualcomm SNPE 백엔드 사용 (하드웨어 가속)
        sessionOptions.addNnapi()  // Android Neural Networks API

        // YOLOv5 Nano 모델 로드
        yoloSession = env.createSession(
            context.assets.open("yolov5n.onnx").readBytes(),
            sessionOptions
        )
    }

    fun detectObjects(frame: Bitmap): List<Detection> {
        // 1. 전처리: 640x640으로 리사이즈 + 정규화
        val input = preprocessFrame(frame)

        // 2. 추론
        val output = yoloSession.run(mapOf("images" to input))

        // 3. 후처리: NMS (Non-Maximum Suppression)
        val detections = postprocessOutput(output)

        return detections
    }

    private fun preprocessFrame(frame: Bitmap): OnnxTensor {
        // Resize to 640x640
        val resized = Bitmap.createScaledBitmap(frame, 640, 640, true)

        // Convert to float array [1, 3, 640, 640]
        val floatArray = FloatArray(1 * 3 * 640 * 640)

        // Normalize to [0, 1]
        // ... (생략)

        return OnnxTensor.createTensor(env, floatArray, shape)
    }

    data class Detection(
        val className: String,    // "car", "person", "traffic_light"
        val confidence: Float,     // 0.0 ~ 1.0
        val bbox: RectF           // Bounding box
    )
}
```

### 4. 키 프레임 추출

```kotlin
fun extractKeyFrames(videoFile: File, count: Int): List<Bitmap> {
    val retriever = MediaMetadataRetriever()
    retriever.setDataSource(videoFile.absolutePath)

    val duration = retriever.extractMetadata(
        MediaMetadataRetriever.METADATA_KEY_DURATION
    ).toLong() * 1000  // microseconds

    val interval = duration / (count + 1)

    return (1..count).map { i ->
        val timeUs = interval * i
        retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST)
            ?: throw Exception("Failed to extract frame at $timeUs")
    }
}
```

---

## 📈 성능 예측 및 최적화

### 1. 리소스 사용 예측

#### 시나리오: 1시간 주행 (평균 8개 이벤트)

**기존 DTG (CAN 데이터만)**:
```
평균 전력: 2.0W
피크 전력: 2.0W (AI 추론)
평균 메모리: 25MB
피크 메모리: 40MB
데이터 전송: ~50MB (MQTT)
```

**블랙박스 통합 후 (이벤트 기반)**:
```
평균 전력: 2.1W (+5%)
피크 전력: 2.8W (CV 추론)
평균 메모리: 30MB (+20%)
피크 메모리: 60MB (+50%)
데이터 전송: ~150MB (+200%)
  - CAN 데이터: 50MB
  - 이벤트 영상: 100MB (8 events × 10초 × 1.25MB/초)
```

**결론**: **허용 가능한 증가** ✅

### 2. 최적화 전략

#### A. 모델 크기 최적화
```python
# YOLOv5 Nano INT8 양자화
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic

# 동적 양자화 (3.8MB)
quantize_dynamic(
    model_input='yolov5n.onnx',
    model_output='yolov5n_int8.onnx',
    weight_type=QuantType.QInt8
)

# 추가 최적화: SNPE DLC 변환
# Qualcomm 도구로 변환 시 추가 20~30% 압축 가능
```

#### B. 프레임 샘플링 최적화
```kotlin
// 키 프레임만 분석 (전체 프레임 대신)
fun extractKeyFrames(videoFile: File): List<Bitmap> {
    // 5초 영상 (150 프레임 @ 30fps) → 5 프레임만 추출
    // 첫 프레임, 중간 3개, 마지막 프레임
    return listOf(
        getFrameAt(0.0),      // 이벤트 시작 (t-5초)
        getFrameAt(0.25),     // t-3.75초
        getFrameAt(0.5),      // t (이벤트 발생)
        getFrameAt(0.75),     // t+1.25초
        getFrameAt(1.0)       // 이벤트 종료 (t+5초)
    )
}
```

#### C. 전력 관리
```kotlin
class PowerAwareAnalyzer {
    fun shouldAnalyze(batteryLevel: Int, isCharging: Boolean): Boolean {
        return when {
            // 충전 중이면 항상 분석
            isCharging -> true

            // 배터리 30% 이상이면 분석
            batteryLevel > 30 -> true

            // 배터리 부족 시 중요 이벤트만 분석
            batteryLevel > 15 -> event.isCritical

            // 배터리 15% 이하면 분석 안 함
            else -> false
        }
    }
}
```

#### D. 네트워크 최적화
```kotlin
// 이벤트 영상 압축 전송
fun uploadEventVideo(videoFile: File) {
    // 1. H.265 재압축 (50% 크기 감소)
    val compressed = compressVideo(videoFile, codec = "hevc")

    // 2. Wi-Fi 연결 시에만 전송
    if (isWiFiConnected()) {
        mqttClient.publish("dtg/events/video", compressed)
    } else {
        // 3. 모바일 데이터 시 썸네일만 전송
        val thumbnail = extractThumbnail(videoFile)
        mqttClient.publish("dtg/events/thumbnail", thumbnail)
    }
}
```

---

## 🎯 권장 구현 로드맵

### Phase 1: 최소 기능 검증 (2주)

**목표**: 블랙박스 USB 연동 + 객체 감지 POC

**작업**:
1. USB OTG 연동 구현 (3일)
   - Android USB Host API 사용
   - 블랙박스 MP4 파일 읽기
   - 키 프레임 추출

2. YOLOv5 Nano 통합 (5일)
   - ONNX 모델 다운로드 및 양자화
   - Android ONNX Runtime 통합
   - SNPE/NNAPI 하드웨어 가속 테스트

3. 이벤트 기반 트리거 구현 (3일)
   - CAN 데이터로 이벤트 감지
   - 비동기 분석 파이프라인
   - 결과 저장 및 MQTT 전송

4. 성능 테스트 (3일)
   - 추론 지연 측정
   - 메모리 사용량 프로파일링
   - 전력 소비 측정

**산출물**:
- USB 블랙박스 연동 POC
- 객체 감지 분석 결과 (8개 클래스)
- 성능 벤치마크 리포트

### Phase 2: 최적화 및 안정화 (2주)

**목표**: 실제 차량 환경 테스트

**작업**:
1. 모델 최적화 (3일)
   - INT8 양자화 적용
   - SNPE DLC 변환
   - 추론 속도 개선

2. 전력 관리 구현 (2일)
   - 배터리 레벨 기반 분석 제어
   - 충전 상태 감지
   - Doze 모드 대응

3. 실차 테스트 (5일)
   - 다양한 블랙박스 모델 테스트
   - 실제 주행 환경 데이터 수집
   - 오감지 케이스 분석

4. 문서화 (4일)
   - 사용자 가이드 (블랙박스 연결)
   - API 문서
   - 성능 튜닝 가이드

**산출물**:
- 최적화된 CV 모델
- 실차 테스트 리포트
- 사용자 문서

### Phase 3: 고급 기능 추가 (선택, 2주)

**목표**: 차선 감지 또는 운전자 모니터링

**작업**:
1. Ultra-Fast-Lane 모델 통합
2. 차선 이탈 경고 기능
3. 고속도로/일반도로 자동 전환

**우선순위**: 낮음 (Phase 1/2 완료 후 재평가)

---

## 💰 비용 분석

### 1. 하드웨어 비용 (변동 없음)

**기존 DTG 구성**:
- Qualcomm Snapdragon 모듈: $50~80
- STM32 MCU: $10~15
- 센서/GPS/통신 모듈: $30~50
- **합계**: ~$100~150

**블랙박스 통합 후**:
- 추가 하드웨어 없음 (USB OTG 기본 지원)
- **합계**: ~$100~150 (동일)

### 2. 소프트웨어 비용 (오픈소스)

**오픈소스 모델**:
- YOLOv5 Nano: MIT License (무료)
- Ultra-Fast-Lane: Apache 2.0 (무료)
- ONNX Runtime: MIT License (무료)

**개발 비용**:
- Phase 1 (2주): 1명 × 2주 = ~$5,000
- Phase 2 (2주): 1명 × 2주 = ~$5,000
- **합계**: ~$10,000 (일회성)

### 3. 운영 비용 (증가)

**데이터 전송 비용**:
- 기존: 50MB/시간 (CAN 데이터)
- 추가: +100MB/시간 (이벤트 영상)
- **증가**: +200% (하지만 절대값은 여전히 낮음)

**월간 비용** (월 200시간 주행 기준):
- 기존: 10GB/월 × $0.10/GB = $1.00
- 추가: +20GB/월 × $0.10/GB = +$2.00
- **합계**: $3.00/월 (차량당)

---

## ⚠️ 리스크 및 제약사항

### 1. 기술적 리스크

#### A. 모델 크기 초과 (중간 리스크)
**문제**: 15.0MB > 14MB 목표
**완화**:
- TCN/LSTM-AE 추가 압축
- YOLOv5 Nano 대신 더 경량 모델 (MobileNet SSD)
- 온디맨드 모델 로딩 (사용 시에만 메모리 적재)

#### B. 전력 소비 증가 (낮은 리스크)
**문제**: 이벤트 분석 시 피크 2.8W
**완화**:
- 이벤트 기반 분석 (평균 영향 최소)
- 배터리 레벨 기반 제어
- 충전 중 우선 분석

#### C. 블랙박스 호환성 (중간 리스크)
**문제**: 다양한 블랙박스 모델 존재
**완화**:
- 주요 제조사 모델 테스트 (아이나비, 파인뷰, 팅크웨어)
- 표준 인터페이스 사용 (USB 저장 장치, MP4 포맷)
- 호환성 테스트 가이드 제공

### 2. 법적/규제 리스크

#### A. 개인정보 보호 (높은 리스크)
**문제**: 블랙박스 영상은 개인정보 포함 (차량 번호판, 사람 얼굴)
**완화**:
- 영상 암호화 저장 및 전송
- 사용자 동의 획득 (앱 설정)
- 번호판/얼굴 자동 블러 처리 (추가 CV 모델)
- 로컬 분석 후 메타데이터만 전송 (영상 삭제)

#### B. 블랙박스 제조사 규약
**문제**: 일부 블랙박스는 USB 접근 제한
**완화**:
- 제조사와 협력 (API 제공 요청)
- Wi-Fi Direct 대안 제공
- SD 카드 슬롯 활용

### 3. 운영 리스크

#### A. 데이터 전송 비용
**문제**: 이벤트 영상 전송 시 데이터 비용 증가
**완화**:
- Wi-Fi 전용 모드 제공
- 압축률 향상 (H.265)
- 썸네일 우선 전송 (전체 영상은 선택)

#### B. 저장 공간 부족
**문제**: 이벤트 영상 로컬 저장 시 공간 부족
**완화**:
- 자동 삭제 정책 (분석 후 즉시 삭제)
- 클라우드 동기화 후 로컬 삭제
- 압축 저장

---

## 📋 체크리스트 및 검증 항목

### 하드웨어 검증
- [ ] Qualcomm Snapdragon NPU 활용 가능 여부 확인
- [ ] USB OTG Host 모드 지원 확인
- [ ] 피크 전력 3W 허용 가능 여부 (전원 설계)
- [ ] 메모리 60MB 피크 허용 (4GB RAM 기준)

### 소프트웨어 검증
- [ ] ONNX Runtime Android 통합 테스트
- [ ] SNPE/NNAPI 하드웨어 가속 성능 측정
- [ ] YOLOv5 Nano 추론 속도 (<100ms) 달성
- [ ] 키 프레임 추출 성능 (<500ms) 달성

### 블랙박스 호환성 검증
- [ ] 아이나비 주요 모델 테스트 (3개)
- [ ] 파인뷰 주요 모델 테스트 (3개)
- [ ] 팅크웨어 주요 모델 테스트 (3개)
- [ ] USB 저장 장치 모드 확인
- [ ] MP4 파일 포맷 호환성 확인

### 성능 검증
- [ ] 이벤트 감지 → 분석 완료: <3초
- [ ] 평균 전력 증가: <10%
- [ ] 피크 메모리 사용: <60MB
- [ ] 데이터 전송 비용: 허용 범위 확인

### 법적 검증
- [ ] 개인정보 보호 정책 업데이트
- [ ] 사용자 동의 획득 프로세스 구현
- [ ] 영상 암호화 적용
- [ ] 데이터 보관 기간 정책 수립

---

## 🎯 최종 권고사항

### ✅ 권장: 조건부 통합

**권장 구현 범위**:
1. **Phase 1 필수**: 이벤트 기반 객체 감지 (YOLOv5 Nano)
2. **Phase 2 선택**: 차선 감지 (고속도로 전용)
3. **Phase 3 보류**: 운전자 모니터링 (별도 카메라 필요)

**기대 효과**:
- 급가속/급감속 이벤트 시 차량/사람 객체 감지
- 충격 이벤트 시 충돌 대상 파악
- 보험사 사고 분석 자료 제공
- 운전자 안전 개선 (위험 상황 식별)

**리소스 영향**:
- 모델 크기: +3.8MB (총 15.0MB, 약간 초과)
- 평균 전력: +0.1W (총 2.1W, 5% 증가)
- 피크 전력: +0.8W (총 2.8W, 분석 중에만)
- 메모리: 평균 +10MB, 피크 +20MB

**완화 조치**:
1. 기존 모델 추가 양자화 (목표 14MB 달성)
2. 이벤트 기반 분석 (평균 전력 영향 최소)
3. 배터리 레벨 기반 분석 제어

### ⚠️ 주의사항

1. **개인정보 보호 최우선**: 번호판/얼굴 블러 처리 필수
2. **블랙박스 호환성 테스트**: 주요 모델 사전 검증 필요
3. **점진적 롤아웃**: 베타 테스트 → 제한된 차량 → 전체 배포
4. **성능 모니터링**: 전력/메모리 실시간 모니터링

### 📈 비즈니스 가치

**시장 차별화**:
- 한국 최초 DTG + 블랙박스 통합 솔루션
- AI 기반 영상 분석 (객체 감지)
- 보험사/물류사 경쟁력 강화

**예상 ROI**:
- 개발 비용: $10,000 (일회성)
- 운영 비용 증가: $2/차량/월
- 예상 프리미엄: $5~10/차량/월 (추가 기능)
- **순익**: $3~8/차량/월

---

## 📝 결론

**블랙박스 영상 통합은 DTG 하드웨어 스펙 내에서 조건부 가능합니다.**

**핵심 요구사항**:
1. ✅ 이벤트 기반 분석 (실시간 전체 프레임 분석 아님)
2. ✅ 경량 모델 사용 (YOLOv5 Nano, 3.8MB)
3. ✅ 키 프레임 샘플링 (전체 영상 대신 5 프레임)
4. ✅ 배터리 레벨 기반 제어 (전력 관리)

**권장 로드맵**:
- Phase 1 (2주): USB 연동 + 객체 감지 POC
- Phase 2 (2주): 최적화 + 실차 테스트
- Phase 3 (선택): 차선 감지 추가

**최종 판단**: **통합 권장** ✅ (단, 최적화 필수)

---

**보고서 종료**
