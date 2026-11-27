// The Pro button's emblem: a small faceted gem, scale-green with a sun-amber
// core light, turning slowly. Hover or focus speeds it up and warms the glow.
// Loaded lazily after the page is interactive; respects reduced motion and
// simply leaves the static fallback in place if WebGL is unavailable.

import * as THREE from "./vendor/three.module.min.js";

export function mountProGem(canvas, button) {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "low-power" });
  } catch {
    return null;
  }
  const size = () => {
    const { width, height } = canvas.getBoundingClientRect();
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(height, 1);
    camera.updateProjectionMatrix();
  };

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 20);
  camera.position.set(0, 0, 4.2);

  const gem = new THREE.Group();
  const geometry = new THREE.IcosahedronGeometry(1, 0);
  const body = new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({ color: 0x1f9e6e, roughness: 0.28, metalness: 0.35, flatShading: true }),
  );
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(geometry),
    new THREE.LineBasicMaterial({ color: 0xbff0d4, transparent: true, opacity: 0.55 }),
  );
  gem.add(body, edges);
  gem.rotation.set(0.5, 0.3, 0);
  scene.add(gem);

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(2.5, 3, 4);
  scene.add(key);
  const sun = new THREE.PointLight(0xf2a93b, 0, 6);
  sun.position.set(-1.6, -1.2, 2.2);
  scene.add(sun);

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let hot = false;
  let speed = 0.35;
  let glow = 0.6;
  let last = performance.now();
  let frame = 0;
  let visible = true;

  const render = (now) => {
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    const targetSpeed = hot ? 1.8 : 0.35;
    const targetGlow = hot ? 6 : 0.6;
    speed += (targetSpeed - speed) * Math.min(dt * 4, 1);
    glow += (targetGlow - glow) * Math.min(dt * 5, 1);
    gem.rotation.y += speed * dt;
    gem.rotation.x += speed * 0.35 * dt;
    gem.position.y = Math.sin(now / 900) * 0.06;
    sun.intensity = glow;
    renderer.render(scene, camera);
    if (!reduceMotion.matches && visible) frame = requestAnimationFrame(render);
  };

  const start = () => {
    cancelAnimationFrame(frame);
    last = performance.now();
    frame = requestAnimationFrame(render);
  };
  const setHot = (value) => {
    hot = value;
    if (reduceMotion.matches) {
      sun.intensity = value ? 6 : 0.6;
      renderer.render(scene, camera);
    }
  };

  button.addEventListener("pointerenter", () => setHot(true));
  button.addEventListener("pointerleave", () => setHot(false));
  button.addEventListener("focus", () => setHot(true));
  button.addEventListener("blur", () => setHot(false));
  // Pause when the card scrolls out of view or the tab is hidden.
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    if (visible) start();
  }).observe(canvas);
  document.addEventListener("visibilitychange", () => {
    visible = !document.hidden;
    if (visible) start();
  });
  window.addEventListener("resize", size);
  size();
  sun.intensity = glow;
  renderer.render(scene, camera);
  if (!reduceMotion.matches) start();
  canvas.dataset.ready = "true";
  return { dispose: () => { cancelAnimationFrame(frame); renderer.dispose(); } };
}
