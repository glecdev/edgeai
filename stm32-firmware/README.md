# STM32 Firmware - CAN Bridge

## Overview

The STM32 firmware acts as a CAN bus interface and sensor management system:
- Reads CAN messages from vehicle OBD-II/J1939 bus
- Parses and validates CAN data
- Transmits formatted data to Snapdragon via UART (921600 baud)
- Manages additional sensors (I2C, SPI, ADC)
- Provides real-time response (<1ms)

## Hardware Configuration

- **MCU**: STM32F4/F7 series (or compatible)
- **CAN Interface**: MCP2551 transceiver, 500kbps bitrate
- **UART Interface**: 921600 baud, 8N1, connected to Snapdragon GPIO
- **Sensors**: I2C (IMU, GPS), SPI (external flash), ADC (analog sensors)

## CAN Message Protocol

### UART Frame Format (STM32 → Snapdragon)

```
[START] [ID_H] [ID_L] [DLC] [DATA(8)] [CRC16(2)] [END]
  0xAA    1B     1B     1B     8B        2B      0x55
Total: 15 bytes per CAN frame
```

### Key OBD-II PIDs

| PID | Description | Formula | Unit |
|-----|-------------|---------|------|
| 0x0C | Engine RPM | ((A*256)+B)/4 | rpm |
| 0x0D | Vehicle Speed | A | km/h |
| 0x2F | Fuel Level | A*100/255 | % |
| 0x11 | Throttle Position | A*100/255 | % |
| 0x05 | Coolant Temp | A-40 | °C |

### J1939 PGNs (Commercial Vehicles)

| PGN | Description |
|-----|-------------|
| 61444 (F004) | Engine speed, torque |
| 65265 (FEF1) | Vehicle speed |
| 65262 (FEEE) | Fuel consumption |

## Directory Structure

```
stm32-firmware/
├── Core/
│   ├── Src/
│   │   ├── main.c        # Main loop
│   │   ├── can.c         # CAN HAL driver
│   │   └── uart.c        # UART HAL driver
│   └── Inc/
│       ├── main.h
│       ├── can.h
│       └── uart.h
├── Drivers/              # STM32 HAL libraries
└── Makefile
```

## Build & Flash

```bash
# Build with STM32CubeIDE or command-line
./.claude/skills/build-stm32/run.sh build

# Build + Flash via ST-Link
./.claude/skills/build-stm32/run.sh flash

# Build + Flash + Serial Monitor
./.claude/skills/build-stm32/run.sh flash --monitor
```

## Development Workflow

```bash
# 1. Configure peripherals in STM32CubeMX
# 2. Generate initialization code
# 3. Implement CAN message handlers in can.c
# 4. Implement UART transmitter in uart.c
# 5. Build and flash
make clean && make -j$(nproc)
st-flash write build/dtg_firmware.bin 0x8000000

# 6. Monitor UART output (debugging)
minicom -D /dev/ttyUSB0 -b 921600
```

## Performance Requirements

- **CAN Message Latency**: < 1ms (receive → parse → transmit)
- **UART Baud Rate**: 921600 bps (115200 bytes/sec)
- **CAN Bus Load**: Handle up to 1000 msgs/sec
- **Error Handling**: CRC validation, buffer overflow protection

## Next Steps

1. Generate STM32CubeMX project
2. Implement CAN receive interrupt handler
3. Implement UART DMA transmission
4. Add CRC16 checksum calculation
5. Create test scripts for CAN simulation
