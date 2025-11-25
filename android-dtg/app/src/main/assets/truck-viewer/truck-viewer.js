/**
 * GLEC DTG 3D Truck Viewer
 *
 * Features:
 * - Real-time 3D truck visualization
 * - Diagnostic code-based fault highlighting (red alerts)
 * - WebView <-> Kotlin JavaScript bridge
 * - Camera controls (orbit, preset views)
 * - HUD overlay with live vehicle data
 *
 * Bridge Functions (called from Kotlin):
 * - updateVehicleData(jsonString): Update real-time vehicle metrics
 * - highlightFault(partName, diagnosticCode): Highlight faulty part in red
 * - clearFaults(): Clear all fault highlights
 * - setCameraView(viewType): Change camera preset
 */

class TruckViewer {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.truckModel = null;
        this.highlightedParts = new Map(); // Map<partName, originalMaterial>

        // Part name mappings for diagnostic codes
        this.diagnosticCodeMapping = {
            'P0118': 'engine',          // Engine Coolant Temperature Circuit High
            'P0217': 'engine',          // Engine Coolant Over Temperature
            'P0420': 'exhaust',         // Catalyst System Efficiency Below Threshold
            'P0128': 'radiator',        // Coolant Thermostat Temperature Below Regulating Temp
            'P0300': 'engine',          // Random/Multiple Cylinder Misfire Detected
            'P0171': 'fuel_system',     // System Too Lean (Bank 1)
            'P0442': 'fuel_tank',       // EVAP System Leak Detected (small leak)
            'P0174': 'fuel_system',     // System Too Lean (Bank 2)
            'P0401': 'egr_valve',       // Exhaust Gas Recirculation Flow Insufficient
            'P0505': 'throttle',        // Idle Control System Malfunction
            'C0035': 'front_wheel_left',  // Left Front Wheel Speed Sensor Circuit
            'C0040': 'front_wheel_right', // Right Front Wheel Speed Sensor Circuit
            'C0045': 'rear_wheel_left',   // Left Rear Wheel Speed Sensor Circuit
            'C0050': 'rear_wheel_right',  // Right Rear Wheel Speed Sensor Circuit
            'B0001': 'airbag',          // Driver Airbag Circuit
            'U0100': 'ecu',             // Lost Communication with ECU/PCM
        };

        this.init();
    }

    init() {
        console.log('[TruckViewer] Initializing...');

        // Setup scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e);
        this.scene.fog = new THREE.Fog(0x1a1a2e, 20, 100);

        // Setup camera
        const container = document.getElementById('canvas-container');
        const aspect = container.clientWidth / container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
        this.camera.position.set(8, 4, 12);

        // Setup renderer
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: false,
            powerPreference: 'high-performance' // Optimize for mobile
        });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Limit pixel ratio for performance
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;
        container.appendChild(this.renderer.domElement);

        // Setup controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.minDistance = 5;
        this.controls.maxDistance = 30;
        this.controls.maxPolarAngle = Math.PI / 2;

        // Setup lights
        this.setupLights();

        // Setup ground
        this.setupGround();

        // Load truck model (placeholder for now, will load actual GLB/GLTF model)
        this.loadTruckModel();

        // Setup camera control buttons
        this.setupCameraControls();

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());

        // Start animation loop
        this.animate();

        console.log('[TruckViewer] Initialization complete');
    }

    setupLights() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        // Directional light (sun)
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(10, 20, 10);
        dirLight.castShadow = true;
        dirLight.shadow.camera.top = 10;
        dirLight.shadow.camera.bottom = -10;
        dirLight.shadow.camera.left = -10;
        dirLight.shadow.camera.right = 10;
        dirLight.shadow.mapSize.width = 2048;
        dirLight.shadow.mapSize.height = 2048;
        this.scene.add(dirLight);

        // Hemisphere light
        const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.4);
        hemiLight.position.set(0, 20, 0);
        this.scene.add(hemiLight);

        // Spotlight for dramatic effect
        const spotLight = new THREE.SpotLight(0xffffff, 0.5);
        spotLight.position.set(0, 10, 0);
        spotLight.angle = Math.PI / 6;
        spotLight.penumbra = 0.2;
        spotLight.castShadow = true;
        this.scene.add(spotLight);
    }

    setupGround() {
        const groundGeometry = new THREE.CircleGeometry(50, 64);
        const groundMaterial = new THREE.MeshStandardMaterial({
            color: 0x2a2a3e,
            roughness: 0.8,
            metalness: 0.2
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);

        // Grid helper
        const gridHelper = new THREE.GridHelper(50, 50, 0x00d9ff, 0x2a2a3e);
        gridHelper.material.opacity = 0.2;
        gridHelper.material.transparent = true;
        this.scene.add(gridHelper);
    }

    loadTruckModel() {
        // Create a placeholder truck model (simple box truck geometry)
        // In production, this will load an actual GLB/GLTF truck model
        const truckGroup = new THREE.Group();
        truckGroup.name = 'truck';

        // Truck body (cargo box)
        const bodyGeometry = new THREE.BoxGeometry(3, 2, 8);
        const bodyMaterial = new THREE.MeshStandardMaterial({
            color: 0x3498db,
            roughness: 0.5,
            metalness: 0.3
        });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.name = 'cargo_box';
        body.position.set(0, 2, 0);
        body.castShadow = true;
        body.receiveShadow = true;
        truckGroup.add(body);

        // Truck cab
        const cabGeometry = new THREE.BoxGeometry(3, 2, 3);
        const cabMaterial = new THREE.MeshStandardMaterial({
            color: 0xe74c3c,
            roughness: 0.4,
            metalness: 0.4
        });
        const cab = new THREE.Mesh(cabGeometry, cabMaterial);
        cab.name = 'cab';
        cab.position.set(0, 2, 5.5);
        cab.castShadow = true;
        cab.receiveShadow = true;
        truckGroup.add(cab);

        // Engine (visible part under hood)
        const engineGeometry = new THREE.BoxGeometry(2, 1, 1.5);
        const engineMaterial = new THREE.MeshStandardMaterial({
            color: 0x555555,
            roughness: 0.3,
            metalness: 0.8
        });
        const engine = new THREE.Mesh(engineGeometry, engineMaterial);
        engine.name = 'engine';
        engine.position.set(0, 1, 7);
        engine.castShadow = true;
        truckGroup.add(engine);

        // Wheels
        const wheelGeometry = new THREE.CylinderGeometry(0.5, 0.5, 0.4, 32);
        const wheelMaterial = new THREE.MeshStandardMaterial({
            color: 0x222222,
            roughness: 0.8
        });

        const wheelPositions = [
            { name: 'front_wheel_left', x: -1.5, y: 0.5, z: 5.5 },
            { name: 'front_wheel_right', x: 1.5, y: 0.5, z: 5.5 },
            { name: 'rear_wheel_left', x: -1.5, y: 0.5, z: -2 },
            { name: 'rear_wheel_right', x: 1.5, y: 0.5, z: -2 },
        ];

        wheelPositions.forEach(pos => {
            const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial.clone());
            wheel.name = pos.name;
            wheel.rotation.z = Math.PI / 2;
            wheel.position.set(pos.x, pos.y, pos.z);
            wheel.castShadow = true;
            truckGroup.add(wheel);
        });

        // Exhaust pipe
        const exhaustGeometry = new THREE.CylinderGeometry(0.1, 0.1, 2, 16);
        const exhaustMaterial = new THREE.MeshStandardMaterial({
            color: 0x888888,
            roughness: 0.6,
            metalness: 0.7
        });
        const exhaust = new THREE.Mesh(exhaustGeometry, exhaustMaterial);
        exhaust.name = 'exhaust';
        exhaust.position.set(-1.2, 2, 6);
        exhaust.castShadow = true;
        truckGroup.add(exhaust);

        // Fuel tank
        const fuelTankGeometry = new THREE.CylinderGeometry(0.5, 0.5, 2, 32);
        const fuelTankMaterial = new THREE.MeshStandardMaterial({
            color: 0x444444,
            roughness: 0.5,
            metalness: 0.6
        });
        const fuelTank = new THREE.Mesh(fuelTankGeometry, fuelTankMaterial);
        fuelTank.name = 'fuel_tank';
        fuelTank.rotation.z = Math.PI / 2;
        fuelTank.position.set(1, 0.8, -1);
        fuelTank.castShadow = true;
        truckGroup.add(fuelTank);

        this.truckModel = truckGroup;
        this.scene.add(truckGroup);

        // Hide loading screen
        setTimeout(() => {
            const loading = document.getElementById('loading');
            loading.classList.add('hide');
        }, 500);

        console.log('[TruckViewer] Truck model loaded');
    }

    setupCameraControls() {
        const buttons = document.querySelectorAll('.control-btn');
        buttons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const view = e.target.getAttribute('data-view');
                this.setCameraView(view);

                // Update active button
                buttons.forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
            });
        });
    }

    setCameraView(viewType) {
        const duration = 1000; // Animation duration in ms
        const startPos = this.camera.position.clone();
        const startTime = Date.now();

        let targetPos;
        switch (viewType) {
            case 'front':
                targetPos = new THREE.Vector3(0, 3, 15);
                break;
            case 'side':
                targetPos = new THREE.Vector3(15, 3, 0);
                break;
            case 'top':
                targetPos = new THREE.Vector3(0, 20, 0.1);
                break;
            case 'perspective':
            default:
                targetPos = new THREE.Vector3(8, 4, 12);
                break;
        }

        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Ease-in-out cubic
            const eased = progress < 0.5
                ? 4 * progress * progress * progress
                : 1 - Math.pow(-2 * progress + 2, 3) / 2;

            this.camera.position.lerpVectors(startPos, targetPos, eased);
            this.controls.target.set(0, 2, 0);
            this.controls.update();

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        animate();
        console.log(`[TruckViewer] Camera view changed to: ${viewType}`);
    }

    /**
     * Update vehicle data from Kotlin
     * @param {string} jsonString - JSON string with vehicle data
     * Example: {"speed": 85, "rpm": 2500, "fuel": 75, "temp": 92, "load": 1500}
     */
    updateVehicleData(jsonString) {
        try {
            const data = JSON.parse(jsonString);

            // Update HUD values
            if (data.speed !== undefined) {
                const speedEl = document.getElementById('speed-value');
                speedEl.textContent = `${data.speed} km/h`;
                speedEl.className = 'hud-value' + (data.speed > 100 ? ' warning' : '');
            }

            if (data.rpm !== undefined) {
                const rpmEl = document.getElementById('rpm-value');
                rpmEl.textContent = `${data.rpm} RPM`;
                rpmEl.className = 'hud-value' + (data.rpm > 3000 ? ' warning' : '');
            }

            if (data.fuel !== undefined) {
                const fuelEl = document.getElementById('fuel-value');
                fuelEl.textContent = `${data.fuel}%`;
                fuelEl.className = 'hud-value' + (data.fuel < 20 ? ' danger' : data.fuel < 40 ? ' warning' : '');
            }

            if (data.temp !== undefined) {
                const tempEl = document.getElementById('temp-value');
                tempEl.textContent = `${data.temp}°C`;
                tempEl.className = 'hud-value' + (data.temp > 105 ? ' danger' : data.temp > 95 ? ' warning' : '');
            }

            if (data.load !== undefined) {
                const loadEl = document.getElementById('load-value');
                loadEl.textContent = `${data.load} kg`;
            }

            console.log('[TruckViewer] Vehicle data updated:', data);
        } catch (error) {
            console.error('[TruckViewer] Failed to parse vehicle data:', error);
        }
    }

    /**
     * Highlight a faulty part in red based on diagnostic code
     * @param {string} diagnosticCode - OBD-II/J1939 diagnostic code (e.g., "P0118")
     * @param {string} faultDescription - Human-readable fault description
     */
    highlightFault(diagnosticCode, faultDescription) {
        const partName = this.diagnosticCodeMapping[diagnosticCode];

        if (!partName) {
            console.warn(`[TruckViewer] Unknown diagnostic code: ${diagnosticCode}`);
            return;
        }

        if (!this.truckModel) {
            console.error('[TruckViewer] Truck model not loaded yet');
            return;
        }

        // Find the part in the truck model
        const part = this.truckModel.getObjectByName(partName);

        if (!part) {
            console.warn(`[TruckViewer] Part not found: ${partName}`);
            return;
        }

        // Store original material if not already highlighted
        if (!this.highlightedParts.has(partName)) {
            this.highlightedParts.set(partName, part.material);
        }

        // Apply red fault material
        const faultMaterial = new THREE.MeshStandardMaterial({
            color: 0xff3b30,
            emissive: 0xff3b30,
            emissiveIntensity: 0.5,
            roughness: 0.3,
            metalness: 0.5
        });
        part.material = faultMaterial;

        // Show fault indicator
        const faultIndicator = document.getElementById('fault-indicator');
        const faultText = document.getElementById('fault-text');
        faultText.textContent = `${diagnosticCode}: ${faultDescription}`;
        faultIndicator.classList.add('show');

        console.log(`[TruckViewer] Fault highlighted: ${diagnosticCode} -> ${partName}`);
    }

    /**
     * Clear all fault highlights
     */
    clearFaults() {
        this.highlightedParts.forEach((originalMaterial, partName) => {
            const part = this.truckModel.getObjectByName(partName);
            if (part) {
                part.material = originalMaterial;
            }
        });

        this.highlightedParts.clear();

        // Hide fault indicator
        const faultIndicator = document.getElementById('fault-indicator');
        faultIndicator.classList.remove('show');

        console.log('[TruckViewer] All faults cleared');
    }

    onWindowResize() {
        const container = document.getElementById('canvas-container');
        const aspect = container.clientWidth / container.clientHeight;

        this.camera.aspect = aspect;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(container.clientWidth, container.clientHeight);
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}

// Initialize viewer when DOM is ready
let viewer;
document.addEventListener('DOMContentLoaded', () => {
    viewer = new TruckViewer();

    // Expose global functions for Kotlin bridge
    window.updateVehicleData = (jsonString) => viewer.updateVehicleData(jsonString);
    window.highlightFault = (code, desc) => viewer.highlightFault(code, desc);
    window.clearFaults = () => viewer.clearFaults();
    window.setCameraView = (view) => viewer.setCameraView(view);

    console.log('[TruckViewer] Bridge functions exposed to Kotlin');
});

// Test data update (remove in production)
setTimeout(() => {
    if (viewer) {
        viewer.updateVehicleData('{"speed": 85, "rpm": 2500, "fuel": 75, "temp": 92, "load": 1500}');

        // Test fault highlighting after 3 seconds
        setTimeout(() => {
            viewer.highlightFault('P0118', '엔진 냉각수 온도 센서 이상');
        }, 3000);
    }
}, 2000);
