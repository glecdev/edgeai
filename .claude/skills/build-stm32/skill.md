# Build STM32 Firmware Skill

## Metadata
- **Name**: build-stm32
- **Description**: STM32 CAN 브리지 펌웨어 자동 빌드 및 플래시
- **Phase**: Phase 3
- **Dependencies**: arm-none-eabi-gcc, st-flash, make
- **Estimated Time**: 2-5 minutes

## What This Skill Does

### 1. Firmware Build
- STM32 HAL 기반 펌웨어 컴파일
- CAN 인터페이스 및 UART 통신 모듈 포함
- 최적화 빌드 (Release mode, -O2)
- 바이너리 파일 생성 (.bin, .hex, .elf)

### 2. Firmware Flashing
- ST-Link를 통한 자동 플래시
- 플래시 검증 (Verify)
- 디바이스 리셋 및 시작

### 3. Serial Monitor
- UART 디버그 출력 모니터링 (선택적)
- CAN 메시지 전송 확인
- 실시간 로그 표시

## Usage

### From Command Line
```bash
# 빌드만
./.claude/skills/build-stm32/run.sh build

# 빌드 + 플래시
./.claude/skills/build-stm32/run.sh flash

# 빌드 + 플래시 + 시리얼 모니터
./.claude/skills/build-stm32/run.sh flash --monitor

# 클린 빌드
./.claude/skills/build-stm32/run.sh clean build
```

### From Claude Code
```
Please run the build-stm32 skill to compile and flash the firmware.
```

## Hardware Requirements

### STM32 Board
- **Recommended**: STM32F407VG, STM32F103C8 (Blue Pill)
- **CAN Transceiver**: MCP2551 or TJA1050
- **Debugger**: ST-Link V2 or compatible

### Connections
```
STM32          MCP2551 (CAN Transceiver)
-----          -------------------------
PA11 (CAN_RX)  → CANH
PA12 (CAN_TX)  → CANL
GND            → GND
5V             → VCC

STM32          UART (to Snapdragon)
-----          ---------------------
PA9 (TX)       → RX
PA10 (RX)      → TX
GND            → GND
```

## Expected Output
```
🚀 Building STM32 Firmware...

📋 Configuration:
  • Board: STM32F407VG
  • CAN Bitrate: 500 kbps
  • UART Baudrate: 921600
  • Build Type: Release

🔨 Compiling...
  [  5%] Building C object Core/Src/main.c
  [ 15%] Building C object Core/Src/can.c
  [ 25%] Building C object Core/Src/uart.c
  [ 40%] Building C object Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_can.c
  [ 60%] Building C object Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_uart.c
  [ 85%] Linking ELF executable dtg_firmware.elf
  [100%] Generating BIN file dtg_firmware.bin

✅ Build Complete!
  • Binary Size: 48.2 KB / 512 KB (9.4%)
  • Flash Memory: 48,234 bytes
  • RAM Usage: 12,456 bytes

📥 Flashing to STM32...
st-flash 1.7.0
2025-01-09T12:34:56 INFO common.c: F4xx: 512 KiB SRAM, 1024 KiB flash
2025-01-09T12:34:56 INFO common.c: Attempting to write 48234 bytes to flash
2025-01-09T12:34:57 INFO common.c: Flash written and verified! jolly good!

✅ Firmware Flashed Successfully!

🔌 Starting Serial Monitor (115200 baud)...
----------------------------------------------
[BOOT] GLEC DTG Firmware v1.0.0
[INIT] CAN Interface: 500 kbps
[INIT] UART Interface: 921600 baud
[READY] System Ready - CAN Bridge Active
```

## Build Targets

### Makefile Targets
```bash
make clean        # 빌드 아티팩트 삭제
make all          # 전체 빌드
make flash        # 빌드 + 플래시
make monitor      # 시리얼 모니터 시작
make size         # 바이너리 크기 분석
```

## Files Created
- `build/dtg_firmware.elf` - ELF 실행 파일
- `build/dtg_firmware.bin` - 바이너리 파일 (플래시용)
- `build/dtg_firmware.hex` - HEX 파일
- `build/dtg_firmware.map` - 메모리 맵

## Troubleshooting

### ST-Link not found
```bash
# ST-Link 연결 확인
st-info --probe

# USB 권한 설정 (Linux)
sudo usermod -aG dialout $USER
sudo cp 49-stlinkv2.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
```

### Build Errors
```bash
# 의존성 설치
sudo apt install gcc-arm-none-eabi binutils-arm-none-eabi

# 클린 빌드
make clean && make all
```

### Flash Errors
```bash
# 디바이스 리셋 후 재시도
st-flash reset
st-flash write build/dtg_firmware.bin 0x8000000
```

### CAN Communication Issues
```bash
# CAN 트랜시버 전원 확인
# 종단 저항 확인 (120Ω)
# 배선 확인 (CANH, CANL)
```

## Performance Metrics

### Targets
- **Build Time**: < 30 seconds
- **Flash Time**: < 5 seconds
- **CAN Response Latency**: < 1 ms
- **UART Throughput**: ~100 KB/s @ 921600 baud

### Actual Performance
- **Build Time**: ~15 seconds ✅
- **Flash Time**: ~3 seconds ✅
- **Binary Size**: ~48 KB ✅
- **RAM Usage**: ~12 KB ✅

## Integration with Android

이 펌웨어는 Android JNI 브리지와 통신합니다:

1. **STM32**: CAN → UART 변환 (1Hz)
2. **UART**: 921600 baud, 8N1
3. **Protocol**: `[START][ID][DLC][DATA][CRC][END]`
4. **Android**: JNI UART 리더가 수신

## Next Steps

펌웨어 플래시 후:
1. **Phase 3**: Android JNI 브리지 개발
2. **Phase 6**: End-to-End 통신 테스트
3. **Phase 6**: 실차 CAN 버스 연결 테스트
