# Logistics LLM Integration Guide

## Model Information

- **Model**: Qwen2.5-0.5B (Fine-tuned for Korean Truck Logistics)
- **Location**: `app/src/main/assets/models/qwen-truck-korean/`
- **Size**: Check actual size in assets folder
- **Format**: HuggingFace Transformers (FP16)

## Integration Steps

### 1. Add Dependencies

In `app/build.gradle`:

```gradle
dependencies {
    // PyTorch Android
    implementation 'org.pytorch:pytorch_android:1.13.1'
    implementation 'org.pytorch:pytorch_android_torchvision:1.13.1'

    // Or use ONNX Runtime Mobile
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:latest.release'

    // Or use Transformers.js (JavaScript)
    // implementation 'com.facebook.react:react-native:+'
}
```

### 2. Load Model in Kotlin

```kotlin
// app/src/main/java/com/glec/dtg/ai/LogisticsLLMEngine.kt

class LogisticsLLMEngine(context: Context) {
    private lateinit var tokenizer: Tokenizer
    private lateinit var model: Module

    fun initialize() {
        val modelDir = File(context.filesDir, "models/qwen-truck-korean")

        // Copy from assets if not exists
        if (!modelDir.exists()) {
            copyModelFromAssets(context, modelDir)
        }

        // Load tokenizer
        tokenizer = loadTokenizer(modelDir)

        // Load model (placeholder - actual implementation depends on runtime)
        // For PyTorch: model = Module.load(modelPath)
        // For ONNX: session = OrtEnvironment.getEnvironment().createSession(modelPath)
    }

    fun generateResponse(query: String, vehicleData: VehicleData): String {
        // Build prompt
        val context = buildContext(vehicleData)
        val prompt = "<|im_start|>user\n차량 상태: $context\n질문: $query<|im_end|>\n<|im_start|>assistant\n"

        // Tokenize
        val inputIds = tokenizer.encode(prompt)

        // Generate (placeholder)
        // val outputIds = model.generate(inputIds, maxLength = 256)

        // Decode
        // val response = tokenizer.decode(outputIds)

        // For now, return placeholder
        return "모델 응답 (구현 필요)"
    }

    private fun buildContext(data: VehicleData): String {
        return "속도: ${data.speed} km/h, RPM: ${data.rpm}, " +
               "연료: ${data.fuelLevel}%, 적재: ${data.loadWeight} kg"
    }
}
```

### 3. Integrate with Voice Assistant

```kotlin
// app/src/main/java/com/glec/dtg/voice/VoiceAssistant.kt

class VoiceAssistant(context: Context) {
    private val llmEngine = LogisticsLLMEngine(context)
    private val tts = TextToSpeech(context, null)

    init {
        llmEngine.initialize()
    }

    suspend fun handleVoiceQuery(query: String, vehicleData: VehicleData) {
        // Generate response
        val response = withContext(Dispatchers.Default) {
            llmEngine.generateResponse(query, vehicleData)
        }

        // Speak response
        tts.speak(response, TextToSpeech.QUEUE_FLUSH, null, null)
    }
}
```

### 4. Test Integration

```kotlin
// Test in MainActivity or service
lifecycleScope.launch {
    val vehicleData = VehicleData(
        speed = 80,
        rpm = 2000,
        fuelLevel = 65,
        loadWeight = 8000
    )

    val query = "연비를 개선하려면 어떻게 해야 하나요?"

    voiceAssistant.handleVoiceQuery(query, vehicleData)
}
```

## Performance Expectations

- **Response Time**: 5-10 seconds (first inference may be slower)
- **Memory Usage**: ~500 MB RAM
- **Model Size**: ~1 GB (FP16)
- **Accuracy**: 38.79% overall (물류 관련성 48.4%)

## Optimization Tips

1. **Reduce Model Size**:
   - INT8 quantization → 50% size reduction
   - Model pruning → 30-40% size reduction

2. **Improve Response Time**:
   - Use ONNX Runtime Mobile (faster inference)
   - Cache common queries
   - Limit max_tokens to 100-150

3. **Handle OOM**:
   - Monitor available memory before inference
   - Implement fallback to rule-based responses
   - Clear model from memory when not in use

## Troubleshooting

**Issue 1: Model fails to load**
- Check assets folder contains all model files
- Verify file permissions
- Check available storage (need 2-3x model size)

**Issue 2: Slow inference**
- Enable GPU acceleration if available
- Reduce max_tokens
- Consider model distillation

**Issue 3: Out of memory**
- Close other apps
- Reduce model precision (INT8)
- Implement lazy loading

## Next Steps

1. Implement actual model loading (PyTorch/ONNX/MLC-LLM)
2. Add error handling and fallbacks
3. Integrate with existing voice UI
4. Run field tests with real drivers
5. Collect feedback and iterate

## Resources

- Model evaluation report: `d:\edgeai\PHASE3K_LOGISTICS_AI_EVALUATION_REPORT.md`
- Training logs: `edgeai-repo/ai-models/fine-tuning/training_run3.log`
- Test cases: `edgeai-repo/ai-models/fine-tuning/test-cases/logistics_test_cases.json`

---

**Generated**: deploy_to_android
**Date**: 2024-11-17
