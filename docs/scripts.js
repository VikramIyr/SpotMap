// Enhanced animations and interactions for SpotMAP website

document.addEventListener('DOMContentLoaded', function() {
    initScrollAnimations();
    initParallaxEffect();
    initSmoothScrolling();
    initImageHoverEffects();
    initButtonAnimations();
    initTypewriterEffect();
    
    setTimeout(() => {
        init3DViewer();
    }, 1000);
});

// Scroll-triggered animations
function initScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
                
                if (entry.target.classList.contains('team-grid') || 
                    entry.target.classList.contains('demos-grid') ||
                    entry.target.classList.contains('tools-grid')) {
                    const children = entry.target.children;
                    Array.from(children).forEach((child, index) => {
                        setTimeout(() => {
                            child.style.opacity = '1';
                            child.style.transform = 'translateY(0)';
                        }, index * 100);
                    });
                }
            }
        });
    }, observerOptions);

    const elementsToAnimate = document.querySelectorAll('.section, .card, .team-member, .demo-card');
    elementsToAnimate.forEach(el => {
        observer.observe(el);
    });
}

// Parallax effect for header background
function initParallaxEffect() {
    const header = document.querySelector('.header');
    const headerBg = document.querySelector('.header-bg');
    
    if (header && headerBg) {
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            const rate = scrolled * -0.5;
            headerBg.style.transform = `translateY(${rate}px)`;
        });
    }
}

// Smooth scrolling for internal links
function initSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Enhanced image hover effects
function initImageHoverEffects() {
    const images = document.querySelectorAll('.pipeline-img, .tool-logo, .member-image img');
    
    images.forEach(img => {
        img.addEventListener('mouseenter', () => {
            img.style.transform = 'scale(1.05)';
            img.style.filter = 'brightness(1.1)';
        });
        
        img.addEventListener('mouseleave', () => {
            img.style.transform = 'scale(1)';
            img.style.filter = 'brightness(1)';
        });
    });
}

// Button hover animations
function initButtonAnimations() {
    const buttons = document.querySelectorAll('.btn');
    
    buttons.forEach(button => {
        button.addEventListener('mouseenter', () => {
            button.style.transform = 'translateY(-2px) scale(1.02)';
            button.style.boxShadow = '0 10px 20px rgba(0, 0, 0, 0.2)';
        });
        
        button.addEventListener('mouseleave', () => {
            button.style.transform = 'translateY(0) scale(1)';
            button.style.boxShadow = 'none';
        });
        
        button.addEventListener('click', () => {
            button.style.transform = 'translateY(0) scale(0.98)';
            setTimeout(() => {
                button.style.transform = 'translateY(-2px) scale(1.02)';
            }, 150);
        });
    });
}

// Typewriter effect for main title
function initTypewriterEffect() {
    const title = document.querySelector('.main-title');
    if (!title) return;
    
    const text = title.textContent;
    const isContactPage = document.title.includes('Contact');
    
    if (!isContactPage && text) {
        title.textContent = '';
        title.style.opacity = '1';
        
        let index = 0;
        const typeSpeed = 30;
        
        function typeWriter() {
            if (index < text.length) {
                title.innerHTML = text.substring(0, index + 1);
                index++;
                setTimeout(typeWriter, typeSpeed);
            } else {
                title.innerHTML += '<span class="cursor">|</span>';
                setTimeout(() => {
                    const cursor = title.querySelector('.cursor');
                    if (cursor) cursor.remove();
                }, 1000);
            }
        }
        
        setTimeout(typeWriter, 500);
    }
}

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
    .cursor {
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    .animate-in {
        opacity: 1 !important;
        transform: translateY(0) !important;
    }
    
    .team-member, .demo-card {
        opacity: 0;
        transform: translateY(20px);
        transition: all 0.6s ease;
    }
    
    .contact-link {
        transition: all 0.3s ease;
    }
    
    .contact-link:hover {
        transform: translateX(5px);
    }
    
    img {
        transition: opacity 0.3s ease;
    }
    
    img:not([src]) {
        opacity: 0;
    }
    
    .card, .team-member, .demo-card, .image-card {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .tool-logo {
        animation: float 6s ease-in-out infinite;
    }
    
    .tool-logo:nth-child(2) { animation-delay: -1s; }
    .tool-logo:nth-child(3) { animation-delay: -2s; }
    .tool-logo:nth-child(4) { animation-delay: -3s; }
    .tool-logo:nth-child(5) { animation-delay: -4s; }
    .tool-logo:nth-child(6) { animation-delay: -5s; }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
`;

document.head.appendChild(style);

window.addEventListener('load', () => {
    document.body.classList.add('loaded');
    
    const fadeInElements = document.querySelectorAll('.fade-in');
    fadeInElements.forEach((el, index) => {
        setTimeout(() => {
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
        }, index * 100);
    });
});

document.querySelectorAll('img').forEach(img => {
    img.addEventListener('load', () => {
        img.style.opacity = '1';
    });
    
    img.addEventListener('error', () => {
        img.style.opacity = '0.5';
    });
});

function addScrollProgress() {
    const progressBar = document.createElement('div');
    progressBar.className = 'scroll-progress';
    progressBar.innerHTML = '<div class="scroll-progress-bar"></div>';
    document.body.appendChild(progressBar);
    
    window.addEventListener('scroll', () => {
        const scrolled = (window.scrollY / (document.documentElement.scrollHeight - window.innerHeight)) * 100;
        document.querySelector('.scroll-progress-bar').style.width = scrolled + '%';
    });
}

const progressStyle = document.createElement('style');
progressStyle.textContent = `
    .scroll-progress {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: rgba(255, 255, 255, 0.1);
        z-index: 9999;
    }
    
    .scroll-progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #6366f1);
        width: 0%;
        transition: width 0.3s ease;
    }
`;

document.head.appendChild(progressStyle);
addScrollProgress();

// Enhanced 3D Viewer with File Loading Support
let scene, camera, renderer, controls;
let currentScene = 'pointcloud';
let wireframeMode = false;
let sceneObjects = [];
let plyLoader, objLoader;

function init3DViewer() {
    const container = document.getElementById('three-viewer');
    if (!container) return;

    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);

    // Camera setup
    camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(5, 5, 5);

    // Renderer setup
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    // Initialize loaders
    plyLoader = new THREE.PLYLoader();
    objLoader = new THREE.OBJLoader();

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    const pointLight = new THREE.PointLight(0x00ffff, 0.6, 50);
    pointLight.position.set(-10, 10, -5);
    scene.add(pointLight);

    // Controls setup
    setupControls();

    // Initial scene
    loadScene(currentScene);

    // Event listeners
    setupViewerEventListeners();

    // Animation loop
    animate3D();
}

function setupControls() {
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };
    let rotationSpeed = 0.005;
    let zoomSpeed = 0.001;

    renderer.domElement.addEventListener('mousedown', (e) => {
        isDragging = true;
        previousMousePosition = { x: e.clientX, y: e.clientY };
    });

    renderer.domElement.addEventListener('mousemove', (e) => {
        if (!isDragging) return;

        const deltaMove = {
            x: e.clientX - previousMousePosition.x,
            y: e.clientY - previousMousePosition.y
        };

        if (e.button === 0) {
            const spherical = new THREE.Spherical();
            spherical.setFromVector3(camera.position);
            spherical.theta -= deltaMove.x * rotationSpeed;
            spherical.phi += deltaMove.y * rotationSpeed;
            spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
            camera.position.setFromSpherical(spherical);
            camera.lookAt(0, 0, 0);
        }

        previousMousePosition = { x: e.clientX, y: e.clientY };
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
    });

    renderer.domElement.addEventListener('wheel', (e) => {
        e.preventDefault();
        const distance = camera.position.length();
        const newDistance = distance + e.deltaY * zoomSpeed * distance;
        camera.position.normalize().multiplyScalar(Math.max(2, Math.min(50, newDistance)));
    });
}

function loadScene(sceneType) {
    showLoadingIndicator(true);
    
    // Clear existing objects
    sceneObjects.forEach(obj => scene.remove(obj));
    sceneObjects = [];

    switch (sceneType) {
        case 'pointcloud':
            createPointCloud();
            break;
        case 'reconstruction':
            create3DReconstruction();
            break;
        case 'robot':
            createRobotModel();
            break;
        case 'custom':
            // Custom file loading handled by file input
            break;
    }
    
    showLoadingIndicator(false);
}

function loadCustomFile(file) {
    if (!file) return;
    
    showLoadingIndicator(true);
    
    // Clear existing objects
    sceneObjects.forEach(obj => scene.remove(obj));
    sceneObjects = [];
    
    const fileExtension = file.name.split('.').pop().toLowerCase();
    const reader = new FileReader();
    
    reader.onload = function(e) {
        const contents = e.target.result;
        
        try {
            if (fileExtension === 'ply') {
                loadPLYFromBuffer(contents);
            } else if (fileExtension === 'obj') {
                loadOBJFromText(contents);
            }
        } catch (error) {
            console.error('Error loading file:', error);
            alert('Error loading file. Please check the file format and try again.');
        }
        
        showLoadingIndicator(false);
    };
    
    if (fileExtension === 'ply') {
        reader.readAsArrayBuffer(file);
    } else {
        reader.readAsText(file);
    }
}

function loadPLYFromBuffer(buffer) {
    try {
        const geometry = plyLoader.parse(buffer);
        
        // Create material for point cloud
        const material = new THREE.PointsMaterial({
            size: 0.02,
            vertexColors: geometry.hasColors,
            color: geometry.hasColors ? 0xffffff : 0x00ff88
        });
        
        const pointCloud = new THREE.Points(geometry, material);
        
        // Center the model
        const box = new THREE.Box3().setFromObject(pointCloud);
        const center = box.getCenter(new THREE.Vector3());
        pointCloud.position.sub(center);
        
        scene.add(pointCloud);
        sceneObjects.push(pointCloud);
        
        // Adjust camera to fit the model
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        camera.position.set(maxDim, maxDim, maxDim);
        camera.lookAt(0, 0, 0);
        
        console.log('PLY file loaded successfully');
    } catch (error) {
        console.error('Error parsing PLY file:', error);
        alert('Error parsing PLY file. Please check the file format.');
    }
}

function loadOBJFromText(text) {
    try {
        const object = objLoader.parse(text);
        
        // Apply material to all meshes
        object.traverse(function(child) {
            if (child.isMesh) {
                child.material = new THREE.MeshPhongMaterial({
                    color: 0x00ff88,
                    wireframe: wireframeMode
                });
                child.castShadow = true;
                child.receiveShadow = true;
            }
        });
        
        // Center the model
        const box = new THREE.Box3().setFromObject(object);
        const center = box.getCenter(new THREE.Vector3());
        object.position.sub(center);
        
        scene.add(object);
        sceneObjects.push(object);
        
        // Adjust camera to fit the model
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        camera.position.set(maxDim, maxDim, maxDim);
        camera.lookAt(0, 0, 0);
        
        console.log('OBJ file loaded successfully');
    } catch (error) {
        console.error('Error parsing OBJ file:', error);
        alert('Error parsing OBJ file. Please check the file format.');
    }
}

function showLoadingIndicator(show) {
    const indicator = document.getElementById('loading-indicator');
    if (indicator) {
        indicator.style.display = show ? 'block' : 'none';
    }
}

function createPointCloud() {
    const particleCount = 5000;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount; i++) {
        const cluster = Math.floor(Math.random() * 3);
        let x, y, z;

        switch (cluster) {
            case 0:
                x = (Math.random() - 0.5) * 10;
                y = Math.random() * 0.5 - 2;
                z = (Math.random() - 0.5) * 10;
                break;
            case 1:
                x = Math.random() > 0.5 ? 5 : -5;
                y = Math.random() * 4 - 1;
                z = (Math.random() - 0.5) * 8;
                break;
            case 2:
                const angle = Math.random() * Math.PI * 2;
                const radius = Math.random() * 2 + 1;
                x = Math.cos(angle) * radius;
                y = Math.random() * 2;
                z = Math.sin(angle) * radius;
                break;
        }

        positions[i * 3] = x;
        positions[i * 3 + 1] = y;
        positions[i * 3 + 2] = z;

        const intensity = (y + 3) / 6;
        colors[i * 3] = intensity;
        colors[i * 3 + 1] = 0.5 + intensity * 0.5;
        colors[i * 3 + 2] = 1 - intensity;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
        size: 0.05,
        vertexColors: true,
        transparent: true,
        opacity: 0.8
    });

    const pointCloud = new THREE.Points(geometry, material);
    scene.add(pointCloud);
    sceneObjects.push(pointCloud);
}

function create3DReconstruction() {
    const roomGroup = new THREE.Group();

    const floorGeometry = new THREE.PlaneGeometry(10, 10);
    const floorMaterial = new THREE.MeshLambertMaterial({ color: 0x808080 });
    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -2;
    roomGroup.add(floor);

    const wallMaterial = new THREE.MeshLambertMaterial({ color: 0xcccccc });
    
    const backWall = new THREE.Mesh(new THREE.PlaneGeometry(10, 6), wallMaterial);
    backWall.position.set(0, 1, -5);
    roomGroup.add(backWall);

    const leftWall = new THREE.Mesh(new THREE.PlaneGeometry(10, 6), wallMaterial);
    leftWall.rotation.y = Math.PI / 2;
    leftWall.position.set(-5, 1, 0);
    roomGroup.add(leftWall);

    const objects = [
        { geometry: new THREE.BoxGeometry(1, 1, 1), position: [2, -1.5, 2], color: 0xff4444 },
        { geometry: new THREE.CylinderGeometry(0.5, 0.5, 1.5), position: [-2, -1.25, 1], color: 0x44ff44 },
        { geometry: new THREE.SphereGeometry(0.7), position: [0, -1.3, -2], color: 0x4444ff }
    ];

    objects.forEach(obj => {
        const material = new THREE.MeshPhongMaterial({ color: obj.color });
        const mesh = new THREE.Mesh(obj.geometry, material);
        mesh.position.set(...obj.position);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        roomGroup.add(mesh);
    });

    scene.add(roomGroup);
    sceneObjects.push(roomGroup);
}

function createRobotModel() {
    const robotGroup = new THREE.Group();

    const bodyGeometry = new THREE.BoxGeometry(2, 0.8, 1);
    const bodyMaterial = new THREE.MeshPhongMaterial({ color: 0xffdd00 });
    const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
    body.position.y = 0.5;
    robotGroup.add(body);

    const legGeometry = new THREE.CylinderGeometry(0.1, 0.1, 1);
    const legMaterial = new THREE.MeshPhongMaterial({ color: 0x333333 });

    const legPositions = [
        [0.8, -0.5, 0.4], [0.8, -0.5, -0.4],
        [-0.8, -0.5, 0.4], [-0.8, -0.5, -0.4]
    ];

    legPositions.forEach(pos => {
        const leg = new THREE.Mesh(legGeometry, legMaterial);
        leg.position.set(...pos);
        robotGroup.add(leg);
    });

    const headGeometry = new THREE.SphereGeometry(0.3);
    const headMaterial = new THREE.MeshPhongMaterial({ color: 0x0088ff });
    const head = new THREE.Mesh(headGeometry, headMaterial);
    head.position.set(1.2, 0.8, 0);
    robotGroup.add(head);

    robotGroup.userData = { rotationSpeed: 0.01 };

    scene.add(robotGroup);
    sceneObjects.push(robotGroup);
}

function setupViewerEventListeners() {
    const sceneSelector = document.getElementById('scene-selector');
    const wireframeBtn = document.getElementById('toggle-wireframe');
    const resetBtn = document.getElementById('reset-camera');
    const fileInput = document.getElementById('file-input');
    const fileUploadGroup = document.getElementById('file-upload-group');

    if (sceneSelector) {
        sceneSelector.addEventListener('change', (e) => {
            currentScene = e.target.value;
            
            if (currentScene === 'custom') {
                fileUploadGroup.style.display = 'flex';
            } else {
                fileUploadGroup.style.display = 'none';
                loadScene(currentScene);
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                loadCustomFile(file);
            }
        });
    }

    if (wireframeBtn) {
        wireframeBtn.addEventListener('click', () => {
            wireframeMode = !wireframeMode;
            sceneObjects.forEach(obj => {
                obj.traverse(child => {
                    if (child.material && child.material.wireframe !== undefined) {
                        child.material.wireframe = wireframeMode;
                    }
                });
            });
        });
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            camera.position.set(5, 5, 5);
            camera.lookAt(0, 0, 0);
        });
    }

    window.addEventListener('resize', () => {
        const container = document.getElementById('three-viewer');
        if (container && camera && renderer) {
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }
    });
}

function animate3D() {
    requestAnimationFrame(animate3D);

    sceneObjects.forEach(obj => {
        if (obj.userData && obj.userData.rotationSpeed) {
            obj.rotation.y += obj.userData.rotationSpeed;
        }
    });

    if (renderer && scene && camera) {
        renderer.render(scene, camera);
    }
}
