# RuvLLM ESP32 - Ready to Flash

Complete, cross-platform flashable implementation of RuvLLM for ESP32.

## Features

- **Tiny LLM Inference**: INT8 quantized transformer
- **RAG System**: Knowledge storage with HNSW vector search
- **Multi-Chip Clusters**: Pipeline parallelism across multiple ESP32s
- **UART Interface**: Interactive serial console (115200 baud)
- **Cross-Platform**: Windows, macOS, Linux support

## Quick Start

### Option 1: One-Line Install

**Linux/macOS:**
```bash
git clone https://github.com/ruvnet/ruvector
cd ruvector/examples/ruvLLM/esp32-flash
./install.sh              # Install deps + build
./install.sh flash        # Flash to auto-detected port
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/ruvnet/ruvector
cd ruvector\examples\ruvLLM\esp32-flash
.\install.ps1             # Install deps (restart PowerShell after)
.\install.ps1 build       # Build
.\install.ps1 flash COM6  # Flash
```

### Option 2: Makefile

```bash
make install              # Install deps + build
make flash PORT=/dev/ttyUSB0
make monitor              # Serial monitor
```

### Option 3: Docker (No Local Toolchain Needed)

```bash
# Build
docker build -t ruvllm-esp32 .
docker run -v $(pwd):/app ruvllm-esp32 build

# Flash (Linux - needs device access)
docker run -v $(pwd):/app -v /dev:/dev --privileged ruvllm-esp32 flash /dev/ttyUSB0
```

## Platform-Specific Notes

### Windows

```powershell
# Install prerequisites
winget install Rustlang.Rust.MSVC

# Install ESP32 toolchain
cargo install espup
espup install
cargo install espflash ldproxy

# RESTART PowerShell

# Build and flash
cd ruvector\examples\ruvLLM\esp32-flash
cargo build --release
espflash flash --port COM6 --monitor target\xtensa-esp32-espidf\release\ruvllm-esp32-flash
```

### macOS

```bash
# Install prerequisites
brew install rustup
rustup-init -y
source ~/.cargo/env

# Install ESP32 toolchain
cargo install espup
espup install
source ~/export-esp.sh
cargo install espflash ldproxy

# Build and flash
cd ruvector/examples/ruvLLM/esp32-flash
cargo build --release
espflash flash --port /dev/cu.usbserial-0001 --monitor target/xtensa-esp32-espidf/release/ruvllm-esp32-flash
```

### Linux

```bash
# Install prerequisites (Debian/Ubuntu)
sudo apt install build-essential pkg-config libudev-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install ESP32 toolchain
cargo install espup
espup install
source ~/export-esp.sh
cargo install espflash ldproxy

# Add user to dialout group (for serial access)
sudo usermod -a -G dialout $USER
# Log out and back in

# Build and flash
cd ruvector/examples/ruvLLM/esp32-flash
cargo build --release
espflash flash --port /dev/ttyUSB0 --monitor target/xtensa-esp32-espidf/release/ruvllm-esp32-flash
```

## Cluster Setup (Multi-Chip)

For running larger models across multiple ESP32s:

### 1. Generate Cluster Config

```bash
# Create config for 5-chip cluster
./install.sh cluster 5
# or
make cluster CHIPS=5
```

### 2. Edit `cluster.toml`

```toml
[cluster]
name = "my-cluster"
chips = 5
topology = "pipeline"

[[chips.nodes]]
id = 1
role = "master"
port = "/dev/ttyUSB0"  # Edit for your system
layers = [0, 1]

[[chips.nodes]]
id = 2
role = "worker"
port = "/dev/ttyUSB1"
layers = [2, 3]
# ... more chips
```

### 3. Flash All Chips

```bash
./cluster-flash.sh
# or
make cluster-flash
```

### 4. Monitor Cluster

```bash
./cluster-monitor.sh   # Opens tmux with all serial monitors
```

## Usage

Once flashed, connect via serial (115200 baud):

```
========================================
RuvLLM ESP32 Ready!
========================================
Commands:
- gen <text>  : Generate tokens from prompt
- add <text>  : Add knowledge to RAG
- ask <query> : Query the knowledge base
- stats       : Show statistics
- help        : Show this help
========================================
>
```

### Example Session

```
> add The meeting is at 3pm in room 401
Added knowledge #4

> add John's phone number is 555-1234
Added knowledge #5

> ask when is the meeting
Found: The meeting is at 3pm in room 401

> ask john phone
Found: John's phone number is 555-1234

> gen hello
Generated: h, e, l, l, o, ?, w, o, r, l

> stats
Tokens: 10, Knowledge: 6
```

## Troubleshooting

### "Permission denied" on serial port

**Linux:**
```bash
sudo usermod -a -G dialout $USER
# Log out and back in
```

**macOS:**
No special permissions needed.

**Windows:**
Run PowerShell as Administrator.

### "Failed to connect to ESP32"

1. Hold **BOOT** button while clicking flash
2. Check correct COM port in Device Manager
3. Try different USB cable (data cable, not charge-only)
4. Close other serial monitors (Arduino IDE, PuTTY)

### Build errors

```bash
# Re-run toolchain setup
espup install
source ~/export-esp.sh  # Linux/macOS

# Restart terminal on Windows
```

### ESP32 variant selection

Edit `.cargo/config.toml`:

```toml
# For ESP32-S3:
target = "xtensa-esp32s3-espidf"

# For ESP32-C3 (RISC-V):
target = "riscv32imc-esp-espidf"
```

## Memory & Performance

| Component | RAM | Flash |
|-----------|-----|-------|
| LLM Model | ~20 KB | ~16 KB |
| RAG Index (16 entries) | ~2 KB | — |
| UART Buffer | 1 KB | — |
| Stack | 8 KB | — |
| **Total** | **~31 KB** | **~16 KB** |

| Operation | Time (ESP32 @ 240MHz) |
|-----------|----------------------|
| Token generation | ~2-5 ms/token |
| Vector search | ~1 ms |
| Embedding | <1 ms |

## Files

```
esp32-flash/
├── Cargo.toml              # Rust project config
├── build.rs                # ESP-IDF build script
├── sdkconfig.defaults      # ESP32 SDK config
├── Makefile                # Cross-platform make targets
├── Dockerfile              # Docker build environment
├── .cargo/config.toml      # Cargo target config
├── src/main.rs             # Complete implementation
├── install.sh              # Linux/macOS installer
├── install.ps1             # Windows installer
├── cluster.example.toml    # Cluster config example
├── cluster-flash.sh        # Multi-chip flash (Linux/macOS)
├── cluster-flash.ps1       # Multi-chip flash (Windows)
├── cluster-monitor.sh      # Multi-pane serial monitor
└── README.md               # This file
```

## License

MIT
