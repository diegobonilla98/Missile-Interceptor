import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

const WORLD_SCALE = 0.6;
const GROUND_SIZE = 4200;
const SKY_RADIUS = 9000;

const state = {
  simulation: null,
  frames: [],
  time: 0,
  duration: 0,
  playing: false,
  playbackSpeed: 1,
  cameraMode: "orbit",
  showWind: true,
  showTrails: true,
  intercepted: false,
  outcome: "idle",
};

const container = document.getElementById("canvas-container");
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x070b18, 0.00018);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 1, 16000);
camera.position.set(900, 700, 1300);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(700, 200, 200);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.495;
controls.minDistance = 60;
controls.maxDistance = 6000;

const composer = new EffectComposer(renderer);
const renderPass = new RenderPass(scene, camera);
composer.addPass(renderPass);
const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.85, 0.55, 0.16);
composer.addPass(bloomPass);
const outputPass = new OutputPass();
composer.addPass(outputPass);

function buildSky() {
  const geometry = new THREE.SphereGeometry(SKY_RADIUS, 48, 32);
  const material = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    uniforms: {
      topColor: { value: new THREE.Color(0x05081a) },
      midColor: { value: new THREE.Color(0x10243f) },
      horizonColor: { value: new THREE.Color(0xff8b66) },
      groundColor: { value: new THREE.Color(0x0a0d18) },
    },
    vertexShader: `
      varying vec3 vWorldPosition;
      void main() {
        vec4 worldPosition = modelMatrix * vec4(position, 1.0);
        vWorldPosition = worldPosition.xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      varying vec3 vWorldPosition;
      uniform vec3 topColor;
      uniform vec3 midColor;
      uniform vec3 horizonColor;
      uniform vec3 groundColor;
      void main() {
        float h = normalize(vWorldPosition).y;
        float horizon = pow(clamp(1.0 - abs(h) * 1.6, 0.0, 1.0), 3.0);
        vec3 col = mix(groundColor, midColor, smoothstep(-0.05, 0.35, h));
        col = mix(col, topColor, smoothstep(0.25, 0.95, h));
        col += horizonColor * horizon * 0.32;
        gl_FragColor = vec4(col, 1.0);
      }
    `,
  });
  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);
}

function buildStars() {
  const count = 1400;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const r = SKY_RADIUS * 0.94;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(Math.random() * 1.7 - 0.45);
    positions[i * 3] = Math.cos(theta) * Math.sin(phi) * r;
    positions[i * 3 + 1] = Math.cos(phi) * r;
    positions[i * 3 + 2] = Math.sin(theta) * Math.sin(phi) * r;
    const tone = 0.55 + Math.random() * 0.45;
    colors[i * 3] = tone;
    colors[i * 3 + 1] = tone * (0.85 + Math.random() * 0.15);
    colors[i * 3 + 2] = tone * (0.95 + Math.random() * 0.05);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const material = new THREE.PointsMaterial({
    size: 6,
    vertexColors: true,
    transparent: true,
    opacity: 0.95,
    sizeAttenuation: false,
    depthWrite: false,
  });
  const points = new THREE.Points(geometry, material);
  scene.add(points);
  return points;
}

function buildTerrain() {
  const segments = 220;
  const geometry = new THREE.PlaneGeometry(GROUND_SIZE * 2, GROUND_SIZE * 2, segments, segments);
  geometry.rotateX(-Math.PI / 2);
  const positions = geometry.attributes.position;
  for (let i = 0; i < positions.count; i++) {
    const x = positions.getX(i);
    const z = positions.getZ(i);
    const dist = Math.sqrt(x * x + z * z);
    const ridge = Math.sin(x * 0.0021 + Math.cos(z * 0.0017) * 1.5) * Math.cos(z * 0.0023);
    const detail = Math.sin(x * 0.0067 + 1.7) * Math.cos(z * 0.0073 + 0.4) * 0.5;
    const noise = Math.sin(x * 0.012 + z * 0.011) * 0.3 + Math.cos(x * 0.025 - z * 0.018) * 0.18;
    let height = (ridge * 220 + detail * 90 + noise * 36);
    height *= Math.max(0, (dist - 700) / 2300);
    height = Math.max(height, -3);
    positions.setY(i, height);
  }
  positions.needsUpdate = true;
  geometry.computeVertexNormals();

  const material = new THREE.MeshStandardMaterial({
    color: 0x0c1a2c,
    roughness: 0.92,
    metalness: 0.05,
    flatShading: true,
    emissive: 0x040a16,
    emissiveIntensity: 0.4,
  });
  const terrain = new THREE.Mesh(geometry, material);
  terrain.receiveShadow = true;
  scene.add(terrain);
}

function buildGridFloor() {
  const grid = new THREE.GridHelper(GROUND_SIZE * 1.4, 70, 0x2c4a78, 0x14233c);
  grid.position.y = 0.5;
  grid.material.transparent = true;
  grid.material.opacity = 0.55;
  grid.material.depthWrite = false;
  scene.add(grid);

  const ringGeometry = new THREE.RingGeometry(45, 50, 64);
  ringGeometry.rotateX(-Math.PI / 2);
  const ringMaterial = new THREE.MeshBasicMaterial({ color: 0xff8a3d, transparent: true, opacity: 0.85, side: THREE.DoubleSide });
  return { grid, ringGeometry, ringMaterial };
}

function buildLights() {
  const hemi = new THREE.HemisphereLight(0xa3c1ff, 0x081020, 0.45);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xffe2c0, 1.05);
  sun.position.set(-1500, 1800, 700);
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0x7da9ff, 0.35);
  fill.position.set(1200, 800, -800);
  scene.add(fill);
}

function vecFromSim(arr) {
  return new THREE.Vector3(arr[0] * WORLD_SCALE, arr[2] * WORLD_SCALE, -arr[1] * WORLD_SCALE);
}

function vecToSimXY(vec) {
  return [vec.x / WORLD_SCALE, -vec.z / WORLD_SCALE];
}

class TrailRibbon {
  constructor(color, maxPoints = 1024, glowColor = null) {
    this.maxPoints = maxPoints;
    this.points = [];
    this.geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(maxPoints * 3);
    const opacities = new Float32Array(maxPoints);
    this.geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    this.geometry.setAttribute("opacity", new THREE.BufferAttribute(opacities, 1));
    this.geometry.setDrawRange(0, 0);
    const material = new THREE.ShaderMaterial({
      uniforms: {
        baseColor: { value: new THREE.Color(color) },
        glowColor: { value: glowColor ? new THREE.Color(glowColor) : new THREE.Color(color) },
      },
      vertexShader: `
        attribute float opacity;
        varying float vOpacity;
        void main() {
          vOpacity = opacity;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 baseColor;
        uniform vec3 glowColor;
        varying float vOpacity;
        void main() {
          vec3 col = mix(baseColor, glowColor, vOpacity);
          gl_FragColor = vec4(col, vOpacity);
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.line = new THREE.Line(this.geometry, material);
    this.line.frustumCulled = false;
    this.material = material;
  }

  reset() {
    this.points.length = 0;
    this.geometry.setDrawRange(0, 0);
  }

  push(point) {
    this.points.push(point.clone());
    if (this.points.length > this.maxPoints) {
      this.points.shift();
    }
    const positions = this.geometry.attributes.position.array;
    const opacities = this.geometry.attributes.opacity.array;
    const n = this.points.length;
    for (let i = 0; i < n; i++) {
      const p = this.points[i];
      positions[i * 3] = p.x;
      positions[i * 3 + 1] = p.y;
      positions[i * 3 + 2] = p.z;
      opacities[i] = Math.pow(i / Math.max(n - 1, 1), 1.6) * 0.95;
    }
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.opacity.needsUpdate = true;
    this.geometry.setDrawRange(0, n);
  }

  setVisible(visible) {
    this.line.visible = visible;
  }
}

class ThrustParticles {
  constructor(color, glowColor, count = 220) {
    this.count = count;
    this.alive = new Array(count).fill(false);
    this.positions = new Float32Array(count * 3);
    this.velocities = new Float32Array(count * 3);
    this.lifeTimes = new Float32Array(count);
    this.maxLife = new Float32Array(count);
    this.sizes = new Float32Array(count);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    geometry.setAttribute("size", new THREE.BufferAttribute(this.sizes, 1));
    geometry.setAttribute("life", new THREE.BufferAttribute(this.lifeTimes, 1));

    const texture = ThrustParticles.makeTexture();
    const material = new THREE.ShaderMaterial({
      uniforms: {
        baseColor: { value: new THREE.Color(color) },
        glowColor: { value: new THREE.Color(glowColor) },
        spriteTexture: { value: texture },
      },
      vertexShader: `
        attribute float size;
        attribute float life;
        varying float vLife;
        void main() {
          vLife = life;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * (320.0 / -mvPosition.z);
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        uniform vec3 baseColor;
        uniform vec3 glowColor;
        uniform sampler2D spriteTexture;
        varying float vLife;
        void main() {
          float t = clamp(vLife, 0.0, 1.0);
          vec4 sprite = texture2D(spriteTexture, gl_PointCoord);
          vec3 col = mix(glowColor, baseColor, 1.0 - t);
          float alpha = sprite.a * t;
          gl_FragColor = vec4(col * (0.6 + t * 0.6), alpha);
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.points = new THREE.Points(geometry, material);
    this.points.frustumCulled = false;
    this.geometry = geometry;
    this.cursor = 0;
  }

  static makeTexture() {
    if (ThrustParticles._cachedTexture) return ThrustParticles._cachedTexture;
    const size = 64;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    gradient.addColorStop(0, "rgba(255, 255, 255, 1.0)");
    gradient.addColorStop(0.35, "rgba(255, 240, 220, 0.85)");
    gradient.addColorStop(0.65, "rgba(255, 160, 100, 0.35)");
    gradient.addColorStop(1, "rgba(255, 100, 50, 0.0)");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    ThrustParticles._cachedTexture = texture;
    return texture;
  }

  spawn(origin, direction, count = 4, baseSpeed = 30) {
    for (let i = 0; i < count; i++) {
      const idx = this.cursor;
      this.cursor = (this.cursor + 1) % this.count;
      this.positions[idx * 3] = origin.x;
      this.positions[idx * 3 + 1] = origin.y;
      this.positions[idx * 3 + 2] = origin.z;
      const spread = 0.35;
      const dx = direction.x + (Math.random() - 0.5) * spread;
      const dy = direction.y + (Math.random() - 0.5) * spread;
      const dz = direction.z + (Math.random() - 0.5) * spread;
      const speed = baseSpeed * (0.6 + Math.random() * 0.8);
      this.velocities[idx * 3] = dx * speed;
      this.velocities[idx * 3 + 1] = dy * speed;
      this.velocities[idx * 3 + 2] = dz * speed;
      this.maxLife[idx] = 0.4 + Math.random() * 0.4;
      this.lifeTimes[idx] = 1.0;
      this.sizes[idx] = 36 + Math.random() * 26;
      this.alive[idx] = true;
    }
  }

  update(dt) {
    let any = false;
    for (let i = 0; i < this.count; i++) {
      if (!this.alive[i]) continue;
      this.lifeTimes[i] -= dt / Math.max(this.maxLife[i], 0.001);
      if (this.lifeTimes[i] <= 0.0) {
        this.alive[i] = false;
        this.sizes[i] = 0;
        continue;
      }
      this.positions[i * 3] += this.velocities[i * 3] * dt;
      this.positions[i * 3 + 1] += this.velocities[i * 3 + 1] * dt;
      this.positions[i * 3 + 2] += this.velocities[i * 3 + 2] * dt;
      this.velocities[i * 3] *= 0.93;
      this.velocities[i * 3 + 1] *= 0.93;
      this.velocities[i * 3 + 2] *= 0.93;
      this.sizes[i] *= 0.98;
      any = true;
    }
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.size.needsUpdate = true;
    this.geometry.attributes.life.needsUpdate = true;
  }

  reset() {
    this.alive.fill(false);
    for (let i = 0; i < this.count; i++) {
      this.lifeTimes[i] = 0;
      this.sizes[i] = 0;
    }
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.size.needsUpdate = true;
    this.geometry.attributes.life.needsUpdate = true;
  }
}

class Missile {
  constructor(bodyColor = 0xc8c8d2, accentColor = 0xff5b5b, finColor = 0x0e131e) {
    this.group = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.CylinderGeometry(2.0, 2.6, 16, 18),
      new THREE.MeshStandardMaterial({ color: bodyColor, metalness: 0.55, roughness: 0.32 })
    );
    body.position.y = 0;
    this.group.add(body);

    const nose = new THREE.Mesh(
      new THREE.ConeGeometry(2.0, 6, 18),
      new THREE.MeshStandardMaterial({ color: accentColor, metalness: 0.2, roughness: 0.4, emissive: accentColor, emissiveIntensity: 0.25 })
    );
    nose.position.y = 11;
    this.group.add(nose);

    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(2.5, 0.35, 8, 24),
      new THREE.MeshStandardMaterial({ color: 0x14202f, metalness: 0.6, roughness: 0.4 })
    );
    ring.rotation.x = Math.PI / 2;
    ring.position.y = -3;
    this.group.add(ring);

    const finMaterial = new THREE.MeshStandardMaterial({ color: finColor, metalness: 0.5, roughness: 0.45 });
    for (let i = 0; i < 4; i++) {
      const fin = new THREE.Mesh(new THREE.BoxGeometry(0.4, 5, 4), finMaterial);
      const angle = (i / 4) * Math.PI * 2;
      fin.position.set(Math.cos(angle) * 2.4, -6, Math.sin(angle) * 2.4);
      fin.rotation.y = -angle;
      this.group.add(fin);
    }

    const flame = new THREE.Mesh(
      new THREE.ConeGeometry(1.6, 4.5, 16),
      new THREE.MeshBasicMaterial({ color: 0xffae5b, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false })
    );
    flame.position.y = -10.5;
    flame.rotation.x = Math.PI;
    this.flame = flame;
    this.group.add(flame);

    const halo = new THREE.PointLight(accentColor, 0.0, 220, 2.0);
    halo.position.y = 2;
    this.group.add(halo);
    this.halo = halo;

    this.bodyLength = 22;
  }

  setVisible(visible) {
    this.group.visible = visible;
  }

  orientToVelocity(velocityVector) {
    if (velocityVector.lengthSq() < 1e-3) return;
    const up = new THREE.Vector3(0, 1, 0);
    const dir = velocityVector.clone().normalize();
    const quaternion = new THREE.Quaternion().setFromUnitVectors(up, dir);
    this.group.quaternion.slerp(quaternion, 0.45);
  }

  setThrustIntensity(intensity) {
    intensity = Math.max(0, Math.min(intensity, 1.5));
    this.flame.material.opacity = 0.6 * intensity + 0.05;
    this.flame.scale.set(0.8 + intensity * 0.6, 0.7 + intensity * 1.4, 0.8 + intensity * 0.6);
    this.halo.intensity = 0.65 * intensity;
  }
}

class Target {
  constructor() {
    this.group = new THREE.Group();
    const ringGeometry = new THREE.RingGeometry(36, 42, 64);
    ringGeometry.rotateX(-Math.PI / 2);
    this.outerRing = new THREE.Mesh(
      ringGeometry,
      new THREE.MeshBasicMaterial({ color: 0xffd58a, transparent: true, opacity: 0.9, side: THREE.DoubleSide })
    );
    this.outerRing.position.y = 0.5;
    this.group.add(this.outerRing);

    const beamGeo = new THREE.CylinderGeometry(2, 6, 360, 16, 1, true);
    beamGeo.translate(0, 180, 0);
    const beamMat = new THREE.ShaderMaterial({
      uniforms: { time: { value: 0 } },
      vertexShader: `
        varying float vY;
        void main() {
          vY = position.y;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        varying float vY;
        void main() {
          float t = clamp(vY / 360.0, 0.0, 1.0);
          float pulse = 0.55 + 0.45 * sin(time * 2.4 - vY * 0.04);
          float alpha = (1.0 - t) * 0.55 * pulse;
          gl_FragColor = vec4(1.0, 0.8, 0.45, alpha);
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });
    this.beam = new THREE.Mesh(beamGeo, beamMat);
    this.group.add(this.beam);

    const corePulse = new THREE.Mesh(
      new THREE.SphereGeometry(8, 18, 14),
      new THREE.MeshBasicMaterial({ color: 0xff9447 })
    );
    corePulse.position.y = 6;
    this.group.add(corePulse);
    this.core = corePulse;

    const halo = new THREE.PointLight(0xffaa55, 1.4, 600, 1.5);
    halo.position.y = 60;
    this.group.add(halo);
    this.halo = halo;
  }

  setPosition(point) {
    this.group.position.set(point.x, 0, point.z);
  }

  update(time) {
    this.beam.material.uniforms.time.value = time;
    this.outerRing.material.opacity = 0.7 + 0.25 * Math.sin(time * 3.5);
    const scale = 1.0 + 0.08 * Math.sin(time * 4.0);
    this.core.scale.setScalar(scale);
    this.halo.intensity = 1.1 + 0.6 * Math.sin(time * 4.0);
  }
}

class WindField {
  constructor(count = 320) {
    this.count = count;
    this.positions = new Float32Array(count * 3);
    this.basePositions = new Float32Array(count * 3);
    this.opacity = new Float32Array(count);
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    this.geometry.setAttribute("alpha", new THREE.BufferAttribute(this.opacity, 1));
    const span = 2400;
    for (let i = 0; i < count; i++) {
      const x = (Math.random() - 0.5) * span;
      const z = (Math.random() - 0.5) * span;
      const y = 30 + Math.random() * 600;
      this.basePositions[i * 3] = x;
      this.basePositions[i * 3 + 1] = y;
      this.basePositions[i * 3 + 2] = z;
      this.positions[i * 3] = x;
      this.positions[i * 3 + 1] = y;
      this.positions[i * 3 + 2] = z;
      this.opacity[i] = 0.2 + Math.random() * 0.6;
    }
    const material = new THREE.ShaderMaterial({
      uniforms: { baseColor: { value: new THREE.Color(0x5fffc0) } },
      vertexShader: `
        attribute float alpha;
        varying float vAlpha;
        void main() {
          vAlpha = alpha;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = 4.0 * (320.0 / -mv.z);
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: `
        uniform vec3 baseColor;
        varying float vAlpha;
        void main() {
          vec2 d = gl_PointCoord - vec2(0.5);
          float r = length(d);
          if (r > 0.5) discard;
          float a = vAlpha * (1.0 - r * 2.0);
          gl_FragColor = vec4(baseColor, a * 0.55);
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.points = new THREE.Points(this.geometry, material);
    this.points.frustumCulled = false;
  }

  update(time, windVector) {
    for (let i = 0; i < this.count; i++) {
      const phase = i * 0.13 + time * 0.4;
      const drift = Math.sin(phase) * 22;
      this.positions[i * 3] = this.basePositions[i * 3] + drift + windVector.x * 4.5;
      this.positions[i * 3 + 1] = this.basePositions[i * 3 + 1] + Math.sin(phase * 0.6) * 12 + windVector.y * 1.2;
      this.positions[i * 3 + 2] = this.basePositions[i * 3 + 2] + Math.cos(phase * 0.8) * 22 + windVector.z * 4.5;
    }
    this.geometry.attributes.position.needsUpdate = true;
  }

  setVisible(visible) {
    this.points.visible = visible;
  }
}

class Explosion {
  constructor() {
    this.group = new THREE.Group();
    this.group.visible = false;

    this.shockwave = new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 24),
      new THREE.MeshBasicMaterial({ color: 0xfff1b8, transparent: true, opacity: 0.6, depthWrite: false, blending: THREE.AdditiveBlending })
    );
    this.group.add(this.shockwave);

    this.flash = new THREE.PointLight(0xffe7b8, 0, 1200, 1.7);
    this.group.add(this.flash);

    const count = 220;
    const positions = new Float32Array(count * 3);
    const sizes = new Float32Array(count);
    const velocities = new Float32Array(count * 3);
    const lifeTimes = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const dir = new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).normalize();
      const speed = 80 + Math.random() * 220;
      velocities[i * 3] = dir.x * speed;
      velocities[i * 3 + 1] = dir.y * speed;
      velocities[i * 3 + 2] = dir.z * speed;
      sizes[i] = 18 + Math.random() * 32;
      lifeTimes[i] = 0.6 + Math.random() * 0.7;
    }
    this.particleVelocities = velocities;
    this.particleLifeTimes = lifeTimes;
    this.particleMaxLife = Float32Array.from(lifeTimes);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("size", new THREE.BufferAttribute(sizes, 1));
    geometry.setAttribute("life", new THREE.BufferAttribute(lifeTimes, 1));
    this.particleGeometry = geometry;

    const texture = ThrustParticles.makeTexture();
    const material = new THREE.ShaderMaterial({
      uniforms: { spriteTexture: { value: texture } },
      vertexShader: `
        attribute float size;
        attribute float life;
        varying float vLife;
        void main() {
          vLife = life;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * (320.0 / -mv.z);
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: `
        uniform sampler2D spriteTexture;
        varying float vLife;
        void main() {
          float t = clamp(vLife, 0.0, 1.0);
          vec3 col = mix(vec3(1.0, 0.55, 0.18), vec3(1.0, 0.95, 0.7), t);
          vec4 sprite = texture2D(spriteTexture, gl_PointCoord);
          gl_FragColor = vec4(col, sprite.a * t);
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.particles = new THREE.Points(geometry, material);
    this.group.add(this.particles);

    this.elapsed = 0;
    this.duration = 1.6;
    this.active = false;
  }

  trigger(position) {
    this.group.position.copy(position);
    const positions = this.particleGeometry.attributes.position.array;
    for (let i = 0; i < this.particleLifeTimes.length; i++) {
      positions[i * 3] = 0;
      positions[i * 3 + 1] = 0;
      positions[i * 3 + 2] = 0;
      this.particleLifeTimes[i] = this.particleMaxLife[i];
    }
    this.particleGeometry.attributes.position.needsUpdate = true;
    this.particleGeometry.attributes.life.needsUpdate = true;
    this.shockwave.scale.setScalar(1);
    this.shockwave.material.opacity = 0.9;
    this.flash.intensity = 8;
    this.elapsed = 0;
    this.active = true;
    this.group.visible = true;
  }

  update(dt) {
    if (!this.active) return;
    this.elapsed += dt;
    const positions = this.particleGeometry.attributes.position.array;
    const lifeAttr = this.particleGeometry.attributes.life.array;
    for (let i = 0; i < this.particleLifeTimes.length; i++) {
      this.particleLifeTimes[i] -= dt;
      const ratio = Math.max(0, this.particleLifeTimes[i] / this.particleMaxLife[i]);
      positions[i * 3] += this.particleVelocities[i * 3] * dt;
      positions[i * 3 + 1] += this.particleVelocities[i * 3 + 1] * dt - 110 * dt * (1 - ratio);
      positions[i * 3 + 2] += this.particleVelocities[i * 3 + 2] * dt;
      this.particleVelocities[i * 3] *= 0.94;
      this.particleVelocities[i * 3 + 1] *= 0.94;
      this.particleVelocities[i * 3 + 2] *= 0.94;
      lifeAttr[i] = ratio;
    }
    this.particleGeometry.attributes.position.needsUpdate = true;
    this.particleGeometry.attributes.life.needsUpdate = true;

    const t = this.elapsed / this.duration;
    this.shockwave.scale.setScalar(1 + t * 240);
    this.shockwave.material.opacity = Math.max(0, 0.9 * (1 - t));
    this.flash.intensity = Math.max(0, 8 * (1 - t * 1.5));

    if (this.elapsed >= this.duration) {
      this.active = false;
      this.group.visible = false;
    }
  }
}

buildSky();
buildStars();
buildLights();
buildTerrain();
const { ringGeometry: launchpadRingGeometry, ringMaterial: launchpadRingMaterial } = buildGridFloor();

const launchpadGroup = new THREE.Group();
const launchpadBase = new THREE.Mesh(
  new THREE.CylinderGeometry(34, 38, 6, 24),
  new THREE.MeshStandardMaterial({ color: 0x16202e, metalness: 0.6, roughness: 0.5 })
);
launchpadBase.position.y = 3;
launchpadGroup.add(launchpadBase);
const launchpadRing = new THREE.Mesh(launchpadRingGeometry, launchpadRingMaterial.clone());
launchpadRing.position.y = 0.6;
launchpadGroup.add(launchpadRing);
scene.add(launchpadGroup);

const interceptorPad = new THREE.Group();
const interceptorPadBase = new THREE.Mesh(
  new THREE.CylinderGeometry(28, 32, 5, 24),
  new THREE.MeshStandardMaterial({ color: 0x0e1a2c, metalness: 0.5, roughness: 0.6 })
);
interceptorPadBase.position.y = 2.5;
interceptorPad.add(interceptorPadBase);
const interceptorPadRing = new THREE.Mesh(launchpadRingGeometry, new THREE.MeshBasicMaterial({ color: 0x5cc5ff, transparent: true, opacity: 0.85, side: THREE.DoubleSide }));
interceptorPadRing.position.y = 0.6;
interceptorPad.add(interceptorPadRing);
scene.add(interceptorPad);

const target = new Target();
scene.add(target.group);

const missileTrail = new TrailRibbon(0xff5b5b, 1024, 0xffe7c2);
const interceptorTrail = new TrailRibbon(0x5cc5ff, 1024, 0xffffff);
scene.add(missileTrail.line);
scene.add(interceptorTrail.line);

const missile = new Missile(0xb9c2cf, 0xff5b5b, 0x0a0e15);
scene.add(missile.group);
const interceptor = new Missile(0xc4d3e6, 0x5cc5ff, 0x0a1426);
scene.add(interceptor.group);

const missileThrust = new ThrustParticles(0xff7d3f, 0xfff1b8, 260);
scene.add(missileThrust.points);
const interceptorThrust = new ThrustParticles(0x6acfff, 0xeaf6ff, 260);
scene.add(interceptorThrust.points);

const wind = new WindField(360);
scene.add(wind.points);

const explosion = new Explosion();
scene.add(explosion.group);

const previewLine = new THREE.Line(
  new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 0)]),
  new THREE.LineDashedMaterial({ color: 0xffd58a, dashSize: 18, gapSize: 12, transparent: true, opacity: 0.6 })
);
scene.add(previewLine);

const plannedPathLine = new THREE.Line(
  new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0)]),
  new THREE.LineDashedMaterial({ color: 0xff9b66, dashSize: 14, gapSize: 10, transparent: true, opacity: 0.45 })
);
plannedPathLine.frustumCulled = false;
plannedPathLine.visible = false;
scene.add(plannedPathLine);

const batteryRing = new THREE.Group();
const batteryRingOuter = new THREE.Mesh(
  new THREE.RingGeometry(0.99, 1.0, 96).rotateX(-Math.PI / 2),
  new THREE.MeshBasicMaterial({ color: 0x5cc5ff, transparent: true, opacity: 0.85, side: THREE.DoubleSide, depthWrite: false })
);
const batteryRingInner = new THREE.Mesh(
  new THREE.RingGeometry(0.0, 1.0, 96).rotateX(-Math.PI / 2),
  new THREE.MeshBasicMaterial({ color: 0x5cc5ff, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending })
);
batteryRingOuter.position.y = 0.4;
batteryRingInner.position.y = 0.3;
batteryRing.add(batteryRingOuter);
batteryRing.add(batteryRingInner);
scene.add(batteryRing);

const targetGhost = new THREE.Mesh(
  new THREE.SphereGeometry(8, 24, 16),
  new THREE.MeshBasicMaterial({ color: 0xffd58a, transparent: true, opacity: 0.0, depthWrite: false, blending: THREE.AdditiveBlending })
);
scene.add(targetGhost);
let targetGhostElapsed = 0;
let targetGhostActive = false;

function updatePreview(showAfterLaunch = false) {
  const settings = readSettings();
  const launch = vecFromSim(settings.launch_position);
  const targetPos = vecFromSim(settings.target_position);
  target.setPosition(targetPos);
  launchpadGroup.position.copy(launch);
  launchpadGroup.position.y = 0;

  const radius = settings.interceptor_battery_radius * WORLD_SCALE;
  batteryRing.position.set(targetPos.x, 0, targetPos.z);
  batteryRing.scale.set(radius, 1.0, radius);
  batteryRing.visible = state.frames.length === 0;

  const azimuth = THREE.MathUtils.degToRad(settings.launch_azimuth_deg);
  const elevation = THREE.MathUtils.degToRad(settings.launch_elevation_deg);
  const direction = new THREE.Vector3(
    Math.cos(elevation) * Math.cos(azimuth),
    Math.sin(elevation),
    -Math.cos(elevation) * Math.sin(azimuth)
  );
  const tip = launch.clone().add(direction.clone().multiplyScalar(420 * WORLD_SCALE));
  const points = [launch.clone().setY(launch.y + 6), tip];
  previewLine.geometry.dispose();
  previewLine.geometry = new THREE.BufferGeometry().setFromPoints(points);
  previewLine.computeLineDistances();
  previewLine.visible = state.frames.length === 0 || showAfterLaunch;
}

const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const raycaster = new THREE.Raycaster();
const pointerNDC = new THREE.Vector2();
const pointerDown = { x: 0, y: 0, t: 0, valid: false };
const CLICK_DRAG_THRESHOLD_PX = 6;
const CLICK_DURATION_MS_MAX = 700;

function pickGroundFromEvent(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointerNDC.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointerNDC.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointerNDC, camera);
  const hit = new THREE.Vector3();
  if (!raycaster.ray.intersectPlane(groundPlane, hit)) return null;
  return hit;
}

function flashTargetGhost(point) {
  targetGhost.position.copy(point);
  targetGhost.position.y = 6;
  targetGhostElapsed = 0;
  targetGhostActive = true;
}

function updateTargetGhost(dt) {
  if (!targetGhostActive) {
    targetGhost.material.opacity = 0;
    return;
  }
  targetGhostElapsed += dt;
  const t = targetGhostElapsed / 0.7;
  if (t >= 1) {
    targetGhostActive = false;
    targetGhost.material.opacity = 0;
    return;
  }
  targetGhost.material.opacity = (1 - t) * 0.8;
  targetGhost.scale.setScalar(1 + t * 5);
}

function applyGroundClick(event) {
  const hit = pickGroundFromEvent(event);
  if (!hit) return;
  const [simX, simY] = vecToSimXY(hit);
  const clampedX = Math.max(-2400, Math.min(2400, Math.round(simX)));
  const clampedY = Math.max(-2400, Math.min(2400, Math.round(simY)));
  document.getElementById("target-x").value = clampedX;
  document.getElementById("target-y").value = clampedY;
  flashTargetGhost(hit);
  updatePreview();
}

window.addEventListener(
  "pointerdown",
  (event) => {
    if (event.button !== 0) return;
    if (event.target !== renderer.domElement) {
      pointerDown.valid = false;
      return;
    }
    pointerDown.x = event.clientX;
    pointerDown.y = event.clientY;
    pointerDown.t = performance.now();
    pointerDown.valid = true;
  },
  true,
);

window.addEventListener(
  "pointerup",
  (event) => {
    if (event.button !== 0) return;
    if (!pointerDown.valid) return;
    pointerDown.valid = false;
    if (event.target !== renderer.domElement) return;
    const dx = event.clientX - pointerDown.x;
    const dy = event.clientY - pointerDown.y;
    const dist2 = dx * dx + dy * dy;
    const elapsedMs = performance.now() - pointerDown.t;
    if (dist2 > CLICK_DRAG_THRESHOLD_PX * CLICK_DRAG_THRESHOLD_PX) return;
    if (elapsedMs > CLICK_DURATION_MS_MAX) return;
    applyGroundClick(event);
  },
  true,
);

renderer.domElement.addEventListener("pointermove", () => {
  renderer.domElement.style.cursor = state.frames.length > 0 ? "grab" : "crosshair";
});

function readSettings() {
  return {
    launch_position: [0, 0, 0],
    launch_azimuth_deg: parseFloat(document.getElementById("azimuth").value),
    launch_elevation_deg: parseFloat(document.getElementById("elevation").value),
    thrust_speed: parseFloat(document.getElementById("thrust").value),
    target_position: [
      parseFloat(document.getElementById("target-x").value),
      parseFloat(document.getElementById("target-y").value),
      0,
    ],
    wind_strength_multiplier: parseFloat(document.getElementById("wind").value),
    jitter_strength_multiplier: parseFloat(document.getElementById("jitter").value),
    noise_seed: parseInt(document.getElementById("seed").value, 10),
    launch_delay: parseFloat(document.getElementById("delay").value),
    interceptor_launch_distance: parseFloat(document.getElementById("standoff").value),
    interceptor_battery_radius: parseFloat(document.getElementById("standoff").value),
    interceptor_battery_altitude_min: 25.0,
    interceptor_battery_altitude_max: 80.0,
    interceptor_initial_speed: parseFloat(document.getElementById("int-speed").value),
    interceptor_spawn_mode: "near_target",
    policy_mode: document.getElementById("policy-mode").value,
    max_simulation_time: 22.0,
    timestep: 0.04,
  };
}

function bindRange(id, valueId, formatter = (v) => v) {
  const input = document.getElementById(id);
  const valueDisplay = document.getElementById(valueId);
  function refresh() {
    valueDisplay.textContent = formatter(input.value);
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    const value = parseFloat(input.value);
    const fill = ((value - min) / (max - min)) * 100;
    input.style.setProperty("--fill", `${fill}%`);
    updatePreview();
  }
  input.addEventListener("input", refresh);
  refresh();
}

bindRange("azimuth", "azimuth-val", (v) => `${parseFloat(v).toFixed(0)}`);
bindRange("elevation", "elevation-val", (v) => `${parseFloat(v).toFixed(0)}`);
bindRange("thrust", "thrust-val", (v) => `${parseFloat(v).toFixed(0)}`);
bindRange("wind", "wind-val", (v) => `${parseFloat(v).toFixed(1)}`);
bindRange("jitter", "jitter-val", (v) => `${parseFloat(v).toFixed(1)}`);
bindRange("delay", "delay-val", (v) => `${parseFloat(v).toFixed(1)}`);
bindRange("standoff", "standoff-val", (v) => `${parseFloat(v).toFixed(0)}`);
bindRange("int-speed", "int-speed-val", (v) => `${parseFloat(v).toFixed(0)}`);

document.getElementById("target-x").addEventListener("input", () => updatePreview());
document.getElementById("target-y").addEventListener("input", () => updatePreview());
document.getElementById("seed").addEventListener("input", () => updatePreview());

document.getElementById("random-seed").addEventListener("click", () => {
  document.getElementById("seed").value = Math.floor(Math.random() * 99999);
  updatePreview();
});

document.getElementById("random-target").addEventListener("click", () => {
  const angle = Math.random() * Math.PI * 2;
  const radius = 600 + Math.random() * 1500;
  const x = Math.cos(angle) * radius;
  const y = Math.sin(angle) * radius;
  document.getElementById("target-x").value = Math.round(x);
  document.getElementById("target-y").value = Math.round(y);
  document.getElementById("azimuth").value = Math.round(THREE.MathUtils.radToDeg(angle));
  bindRange.refresh && bindRange.refresh();
  document.getElementById("azimuth").dispatchEvent(new Event("input"));
});

document.getElementById("camera-mode").addEventListener("change", (event) => {
  state.cameraMode = event.target.value;
});

document.getElementById("play-speed").addEventListener("change", (event) => {
  state.playbackSpeed = parseFloat(event.target.value);
});

document.getElementById("show-wind").addEventListener("change", (event) => {
  state.showWind = event.target.checked;
  wind.setVisible(event.target.checked);
});

document.getElementById("show-trails").addEventListener("change", (event) => {
  state.showTrails = event.target.checked;
  missileTrail.setVisible(event.target.checked);
  interceptorTrail.setVisible(event.target.checked);
});

document.getElementById("play-btn").addEventListener("click", () => {
  if (state.frames.length === 0) return;
  state.playing = !state.playing;
  document.getElementById("play-btn").innerHTML = state.playing ? "&#10074;&#10074;" : "&#9658;";
});

document.getElementById("timeline").addEventListener("input", (event) => {
  if (state.frames.length === 0) return;
  state.time = (parseFloat(event.target.value) / 100) * state.duration;
  state.playing = false;
  document.getElementById("play-btn").innerHTML = "&#9658;";
  applyFrame(state.time);
});

document.getElementById("launch-btn").addEventListener("click", launch);
document.getElementById("reset-btn").addEventListener("click", () => {
  state.simulation = null;
  state.frames = [];
  state.time = 0;
  state.duration = 0;
  state.playing = false;
  state.intercepted = false;
  state.outcome = "idle";
  lastFrameAdvanceIdx = -1;
  lastInterceptorTrailIdx = -1;
  missileTrail.reset();
  interceptorTrail.reset();
  missileThrust.reset();
  interceptorThrust.reset();
  missile.setVisible(false);
  interceptor.setVisible(false);
  explosion.active = false;
  explosion.group.visible = false;
  plannedPathLine.visible = false;
  batteryRing.visible = true;
  document.getElementById("play-btn").innerHTML = "&#9658;";
  document.getElementById("timeline").value = 0;
  document.getElementById("timeline-label").textContent = "0.00 / 0.00 s";
  document.getElementById("status-outcome").textContent = "IDLE";
  document.getElementById("status-outcome").className = "status-pill";
  document.getElementById("status-time").textContent = "t = 0.00s";
  document.getElementById("status-distance").textContent = "d = -";
  document.getElementById("status-min").textContent = "min = -";
  document.getElementById("result-title").textContent = "Awaiting launch";
  document.getElementById("result-badge").textContent = "--";
  document.getElementById("result-badge").className = "badge";
  document.getElementById("result-min").textContent = "-";
  document.getElementById("result-time").textContent = "-";
  document.getElementById("result-frames").textContent = "-";
  document.getElementById("result-outcome").textContent = "-";
  resetCameraView();
  updatePreview();
});

function resetCameraView() {
  document.getElementById("camera-mode").value = "orbit";
  state.cameraMode = "orbit";
  controls.enabled = true;
  const settings = readSettings();
  const launchVec = vecFromSim(settings.launch_position);
  const targetVec = vecFromSim(settings.target_position);
  const midPoint = launchVec.clone().lerp(targetVec, 0.5);
  midPoint.y = 240;
  const offsetDir = new THREE.Vector3().subVectors(targetVec, launchVec).normalize();
  const sideways = new THREE.Vector3(-offsetDir.z, 0, offsetDir.x).normalize();
  const distance = launchVec.distanceTo(targetVec) * 0.85 + 600;
  camera.position.copy(midPoint).add(sideways.multiplyScalar(distance)).add(new THREE.Vector3(0, distance * 0.45, 0));
  controls.target.copy(midPoint);
  controls.update();
}

async function launch() {
  const button = document.getElementById("launch-btn");
  button.disabled = true;
  document.getElementById("loader").classList.add("active");
  state.playing = false;
  document.getElementById("play-btn").innerHTML = "&#9658;";

  const payload = readSettings();

  try {
    const response = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }
    const data = await response.json();
    consumeSimulation(data);
  } catch (error) {
    console.error(error);
    document.getElementById("status-outcome").textContent = "ERROR";
    document.getElementById("status-outcome").className = "status-pill miss";
  } finally {
    button.disabled = false;
    document.getElementById("loader").classList.remove("active");
  }
}

function consumeSimulation(data) {
  state.simulation = data;
  state.frames = data.frames.map((frame) => ({
    t: frame.t,
    mp: vecFromSim(frame.mp),
    mv: vecFromSim(frame.mv),
    ma: frame.ma,
    mt: frame.mt,
    ip: vecFromSim(frame.ip),
    iv: vecFromSim(frame.iv),
    ia: frame.ia,
    ic: frame.ic,
    wm: vecFromSim(frame.wm),
    wi: vecFromSim(frame.wi),
    d: frame.d,
  }));
  state.duration = state.frames.length > 0 ? state.frames[state.frames.length - 1].t : 0;
  state.time = 0;
  state.playing = true;
  state.intercepted = data.outcome === "intercept";
  state.outcome = data.outcome;
  missileTrail.reset();
  interceptorTrail.reset();
  missileThrust.reset();
  interceptorThrust.reset();
  explosion.active = false;
  explosion.group.visible = false;
  missile.setVisible(true);
  interceptor.setVisible(false);

  const interceptorStart = Array.isArray(data.interceptor_launch_position)
    ? vecFromSim(data.interceptor_launch_position)
    : state.frames[0].ip.clone();
  interceptorPad.position.copy(interceptorStart);
  interceptorPad.position.y = 0;
  previewLine.visible = false;
  batteryRing.visible = false;

  if (Array.isArray(data.missile_path_points) && data.missile_path_points.length >= 2) {
    const points = data.missile_path_points.map((p) => vecFromSim(p));
    plannedPathLine.geometry.dispose();
    plannedPathLine.geometry = new THREE.BufferGeometry().setFromPoints(points);
    plannedPathLine.computeLineDistances();
    plannedPathLine.visible = true;
  } else {
    plannedPathLine.visible = false;
  }

  document.getElementById("play-btn").innerHTML = "&#10074;&#10074;";
  const badge = document.getElementById("result-badge");
  const status = document.getElementById("status-outcome");
  if (data.outcome === "intercept") {
    badge.textContent = "INTERCEPT";
    badge.className = "badge intercept";
    status.className = "status-pill intercept";
    status.textContent = "INTERCEPT";
  } else if (data.outcome === "missile_impact") {
    badge.textContent = "TARGET HIT";
    badge.className = "badge miss";
    status.className = "status-pill miss";
    status.textContent = "TARGET HIT";
  } else if (data.outcome === "interceptor_crash") {
    badge.textContent = "INT. CRASH";
    badge.className = "badge miss";
    status.className = "status-pill miss";
    status.textContent = "INT. CRASH";
  } else {
    badge.textContent = "MISS";
    badge.className = "badge miss";
    status.className = "status-pill miss";
    status.textContent = "MISS";
  }
  document.getElementById("result-title").textContent = "Mission complete";
  document.getElementById("result-min").textContent = `${data.min_distance.toFixed(1)} m`;
  document.getElementById("result-time").textContent = data.intercept_time >= 0 ? `${data.intercept_time.toFixed(2)} s` : "—";
  document.getElementById("result-frames").textContent = `${data.frames.length}`;
  document.getElementById("result-outcome").textContent = data.outcome;
}

function pickFrameIndex(time) {
  if (state.frames.length === 0) return 0;
  let lo = 0;
  let hi = state.frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (state.frames[mid].t < time) lo = mid + 1;
    else hi = mid;
  }
  return Math.max(0, lo - 1);
}

function applyFrame(time) {
  if (state.frames.length === 0) return;
  const idx = pickFrameIndex(time);
  const next = Math.min(idx + 1, state.frames.length - 1);
  const a = state.frames[idx];
  const b = state.frames[next];
  const span = Math.max(b.t - a.t, 1e-4);
  const k = THREE.MathUtils.clamp((time - a.t) / span, 0, 1);

  const missilePos = a.mp.clone().lerp(b.mp, k);
  const missileVel = a.mv.clone().lerp(b.mv, k);
  const interceptorPos = a.ip.clone().lerp(b.ip, k);
  const interceptorVel = a.iv.clone().lerp(b.iv, k);

  missile.group.position.copy(missilePos);
  if (a.ma) missile.orientToVelocity(missileVel);
  missile.setVisible(a.ma);
  missile.setThrustIntensity(a.mt && missileVel.length() > 5 ? Math.min(1.2, missileVel.length() / 200) : 0.0);

  interceptor.group.position.copy(interceptorPos);
  interceptor.setVisible(a.ia);
  if (a.ia) {
    interceptor.orientToVelocity(interceptorVel);
    const actionMag = Math.sqrt(a.ic[0] ** 2 + a.ic[1] ** 2 + a.ic[2] ** 2);
    interceptor.setThrustIntensity(0.45 + actionMag * 1.2);
  }

  document.getElementById("status-time").textContent = `t = ${time.toFixed(2)}s`;
  document.getElementById("status-distance").textContent = `d = ${a.d.toFixed(1)}m`;
  document.getElementById("timeline-label").textContent = `${time.toFixed(2)} / ${state.duration.toFixed(2)} s`;

  return { idx, missilePos, missileVel, interceptorPos, interceptorVel };
}

let lastFrameAdvanceIdx = -1;
let lastInterceptorTrailIdx = -1;

function pushTrails(idx) {
  if (idx === lastFrameAdvanceIdx) return;
  for (let i = Math.max(0, lastFrameAdvanceIdx + 1); i <= idx; i++) {
    const frame = state.frames[i];
    if (frame.ma) {
      missileTrail.push(frame.mp);
    }
    if (frame.ia && lastInterceptorTrailIdx < i) {
      interceptorTrail.push(frame.ip);
      lastInterceptorTrailIdx = i;
    }
  }
  lastFrameAdvanceIdx = idx;
}

function emitParticles(frame, missilePos, interceptorPos, dt) {
  if (frame.mt) {
    const dir = frame.mv.clone().multiplyScalar(-1).normalize();
    missileThrust.spawn(missilePos.clone().add(dir.clone().multiplyScalar(8)), dir, 3, 38);
  }
  if (frame.ia) {
    const ic = frame.ic;
    const actionMag = Math.sqrt(ic[0] ** 2 + ic[1] ** 2 + ic[2] ** 2);
    if (actionMag > 0.05) {
      const dir = frame.iv.clone().multiplyScalar(-1).normalize();
      const intensity = Math.max(2, Math.round(actionMag * 6));
      interceptorThrust.spawn(interceptorPos.clone().add(dir.clone().multiplyScalar(8)), dir, intensity, 50);
    }
  }
}

function updateCamera(missilePos, interceptorPos, dt) {
  if (state.cameraMode === "orbit") {
    controls.enabled = true;
    return;
  }
  controls.enabled = false;
  const target = new THREE.Vector3();
  let desiredCameraPosition = new THREE.Vector3();
  if (state.cameraMode === "missile") {
    target.copy(missilePos);
    desiredCameraPosition.copy(missilePos).add(new THREE.Vector3(-80, 70, 110));
  } else if (state.cameraMode === "interceptor") {
    target.copy(interceptorPos);
    desiredCameraPosition.copy(interceptorPos).add(new THREE.Vector3(-110, 80, 140));
  } else {
    target.copy(missilePos.clone().lerp(interceptorPos, 0.5));
    const offset = missilePos.clone().sub(interceptorPos);
    const side = new THREE.Vector3(-offset.z, 0, offset.x).normalize();
    desiredCameraPosition.copy(target).add(side.multiplyScalar(360)).add(new THREE.Vector3(0, 200, 0));
  }
  const lerp = 1 - Math.exp(-dt * 3.5);
  camera.position.lerp(desiredCameraPosition, lerp);
  controls.target.lerp(target, lerp);
  camera.lookAt(controls.target);
}

const clock = new THREE.Clock();

function animate() {
  const dt = Math.min(clock.getDelta(), 0.05);
  const elapsed = clock.elapsedTime;

  if (state.frames.length > 0) {
    if (state.playing) {
      state.time += dt * state.playbackSpeed;
      if (state.time >= state.duration) {
        state.time = state.duration;
        state.playing = false;
        document.getElementById("play-btn").innerHTML = "&#9658;";
      }
    }
    document.getElementById("timeline").value = (state.time / Math.max(state.duration, 1e-6)) * 100;
    const frameInfo = applyFrame(state.time);
    if (frameInfo) {
      pushTrails(frameInfo.idx);
      const frame = state.frames[frameInfo.idx];
      emitParticles(frame, frameInfo.missilePos, frameInfo.interceptorPos, dt * state.playbackSpeed);
      updateCamera(frameInfo.missilePos, frameInfo.interceptorPos, dt);
      const minDistance = state.simulation ? state.simulation.min_distance : 0;
      document.getElementById("status-min").textContent = `min = ${minDistance.toFixed(1)}m`;
      if (state.intercepted && !explosion.active && state.simulation && state.time >= state.simulation.intercept_time && state.simulation.intercept_time >= 0) {
        explosion.trigger(frameInfo.interceptorPos.clone());
      }
    }
  } else {
    controls.enabled = true;
  }

  missileThrust.update(dt);
  interceptorThrust.update(dt);
  explosion.update(dt);
  updateTargetGhost(dt);

  const ambientWind = new THREE.Vector3(Math.sin(elapsed * 0.3) * 6, 0, Math.cos(elapsed * 0.4) * 6);
  if (state.frames.length > 0 && state.showWind) {
    const idx = pickFrameIndex(state.time);
    ambientWind.add(state.frames[idx].wm.clone().multiplyScalar(0.6));
  }
  if (state.showWind) wind.update(elapsed, ambientWind);
  target.update(elapsed);

  controls.update();
  composer.render();
  requestAnimationFrame(animate);
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
});

missile.setVisible(false);
interceptor.setVisible(false);
resetCameraView();
updatePreview();
animate();
