/* ═══════════════════════════════════════════════════════════
   BEV-NET // OCCUPANCY INTELLIGENCE — MASTER JS
   API calls unchanged. UI interactions upgraded.
   ═══════════════════════════════════════════════════════════ */

// --- GLOBAL STATE ---
let allSamples = [];
let featuredSamples = [];
let currentMode = 'dataset';
let selectedIdx = null;
let currentProbGrid = null;
let currentGtGrid = null;

const uploadFiles = {
    up_front: null, up_front_left: null, up_front_right: null,
    up_back: null, up_back_left: null, up_back_right: null
};

// --- 1. INIT & DATA LOADING ---
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const res = await fetch('/api/samples');
        allSamples = await res.json();
        featuredSamples = allSamples.filter(s => s.featured);
        populateDropdown();
        setupUploadGrid();
    } catch (e) {
        console.error("Failed to load dataset samples:", e);
    }
});

function populateDropdown() {
    const select = document.getElementById('sampleSelect');
    select.innerHTML = '<option value="">— SELECT A SCENE —</option>';

    const groupFeatured = document.createElement('optgroup');
    groupFeatured.label = '── FEATURED SCENES ──';

    const groupAll = document.createElement('optgroup');
    groupAll.label = `── nuScenes VALIDATION (${allSamples.length} SAMPLES) ──`;

    const featuredLabels = {
        80:  "Dense Traffic Scene    ← hand-picked best",
        103: "Open Road Scene        ← far-field showcase",
        82:  "Pedestrian Scene       ← near-ego accuracy",
        104: "Urban Intersection     ← complex geometry",
        116: "Highway Merge          ← high-speed tracking"
    };

    allSamples.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s.index;
        if (s.featured) {
            opt.textContent = `★  ${featuredLabels[s.index] || "Featured Scene"}`;
            groupFeatured.appendChild(opt);
        } else {
            const sceneNum = s.scene_name.replace('scene-', '');
            const desc = s.description ? s.description.substring(0, 45) + (s.description.length > 45 ? '…' : '') : 'Standard Drive';
            opt.textContent = `SCN-${sceneNum} — ${desc}`;
            groupAll.appendChild(opt);
        }
    });

    select.appendChild(groupFeatured);
    select.appendChild(groupAll);
}

// --- 2. MODE SWITCHING ---
function setMode(mode) {
    currentMode = mode;

    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById(`tab-${mode}`).classList.add('active');

    document.getElementById('ctrl-dataset').style.display = mode === 'dataset' ? 'block' : 'none';
    document.getElementById('ctrl-random').style.display  = mode === 'random'  ? 'block' : 'none';
    document.getElementById('ctrl-upload').style.display  = mode === 'upload'  ? 'block' : 'none';
    document.getElementById('preview-section').style.display = mode === 'upload' ? 'none' : 'block';
    document.getElementById('output-section').classList.add('hidden');

    const modeBadge = document.getElementById('mode-badge');
    const labels = { dataset: 'MODE: DATASET BROWSER', random: 'MODE: FEATURED SCENES', upload: 'MODE: CUSTOM UPLOAD' };
    if (modeBadge) modeBadge.textContent = labels[mode] || mode.toUpperCase();

    if (mode === 'upload') {
        checkAllFilesUploaded();
    } else if (mode === 'dataset') {
        document.getElementById('sampleSelect').value = "";
        clearGrid();
    } else if (mode === 'random') {
        pickRandomFeatured();
    }
}

// --- 3. INPUT HANDLING ---
document.getElementById('sampleSelect').addEventListener('change', (e) => {
    selectedIdx = e.target.value;
    if (selectedIdx) loadCameraImages(selectedIdx);
    else clearGrid();
});

function pickRandomFeatured() {
    if (!featuredSamples.length) return;
    const s = featuredSamples[Math.floor(Math.random() * featuredSamples.length)];
    selectedIdx = s.index;
    document.getElementById('spec-idx').textContent = s.index;
    document.getElementById('spec-loc').textContent = s.scene_name;
    document.getElementById('spec-desc').textContent = s.description || "N/A";
    loadCameraImages(selectedIdx);
}

async function loadCameraImages(idx) {
    const res = await fetch(`/api/sample-preview/${idx}`);
    const data = await res.json();
    const grid = document.getElementById('camGrid');
    grid.innerHTML = data.cam_images.map(b64 => `<img src="data:image/jpeg;base64,${b64}">`).join('');
    const btn = document.getElementById('runBtn');
    btn.disabled = false;
    btn.querySelector('.btn-run-text').textContent = 'EXECUTE INFERENCE PIPELINE';
}

function clearGrid() {
    const labels = ['CAM_FRONT', 'CAM_F_LEFT', 'CAM_F_RIGHT', 'CAM_BACK', 'CAM_B_LEFT', 'CAM_B_RIGHT'];
    document.getElementById('camGrid').innerHTML = labels.map(l =>
        `<div class="upload-slot"><span class="cam-label">${l}</span></div>`
    ).join('');
    document.getElementById('runBtn').disabled = true;
}

// --- 4. UPLOAD DRAG & DROP ---
function setupUploadGrid() {
    Object.keys(uploadFiles).forEach(id => {
        const inputEl = document.getElementById(id);
        const dropZone = document.getElementById(`dz_${id}`);
        if (!inputEl || !dropZone) return;

        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                inputEl.files = e.dataTransfer.files;
                handleFileUpload(id, e.dataTransfer.files[0], dropZone);
            }
        });
        inputEl.addEventListener('change', (e) => {
            if (e.target.files.length) handleFileUpload(id, e.target.files[0], dropZone);
        });
    });
}

function handleFileUpload(id, file, dropZone) {
    uploadFiles[id] = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        let oldImg = dropZone.querySelector('img');
        if (oldImg) oldImg.remove();
        const img = document.createElement('img');
        img.src = e.target.result;
        dropZone.insertBefore(img, dropZone.firstChild);
        const inner = dropZone.querySelector('.dz-inner');
        if (inner) inner.querySelector('.dz-text').style.color = '#e2e8f0';
    };
    reader.readAsDataURL(file);
    checkAllFilesUploaded();
}

function checkAllFilesUploaded() {
    const allReady = Object.values(uploadFiles).every(f => f !== null);
    const btn = document.getElementById('runBtn');
    btn.disabled = !allReady;
    const textEl = btn.querySelector('.btn-run-text');
    if (textEl) textEl.textContent = allReady ? 'RUN FIXED CALIBRATION INFERENCE' : 'UPLOAD ALL 6 CAMERAS TO CONTINUE';
}

// --- 5. INFERENCE ---
async function runModel() {
    const btn = document.getElementById('runBtn');
    btn.disabled = true;
    const textEl = btn.querySelector('.btn-run-text');
    if (textEl) textEl.textContent = 'PROCESSING...';

    // Show loading
    if (window._showLoading) window._showLoading();

    try {
        let data;
        if (currentMode === 'dataset' || currentMode === 'random') {
            const res = await fetch(`/api/predict/sample/${selectedIdx}`, { method: 'POST' });
            data = await res.json();
        } else if (currentMode === 'upload') {
            const fd = new FormData();
            Object.keys(uploadFiles).forEach(k => fd.append(k, uploadFiles[k] || new Blob()));
            const res = await fetch(`/api/predict/upload`, { method: 'POST', body: fd });
            data = await res.json();
        }

        currentProbGrid = data.pred_grid;
        currentGtGrid = data.gt_grid || null;
        const thresh = parseFloat(document.getElementById('threshSlider').value);

        if (currentMode === 'upload') {
            document.querySelectorAll('.mc-val').forEach(el => el.textContent = 'N/A');
            document.querySelectorAll('.mc-bar-fill').forEach(el => el.style.width = '0%');
            document.querySelectorAll('.metric-card').forEach(el => el.style.opacity = '0.5');
            document.getElementById('errorLegend').innerHTML = `
                <div class="legend-item"><span class="dot pred"></span><span>OCCUPIED (PRED)</span></div>
                <div class="legend-item"><span class="dot" style="background:#1e2a3a;border:1px solid #334155;"></span><span>FREE</span></div>
            `;
        } else {
            document.querySelectorAll('.metric-card').forEach(el => el.style.opacity = '1');
            document.getElementById('errorLegend').innerHTML = `
                <div class="legend-item"><span class="dot tp"></span><span>TRUE POSITIVE</span></div>
                <div class="legend-item"><span class="dot fp"></span><span>FALSE ALARM</span></div>
                <div class="legend-item"><span class="dot fn"></span><span>MISSED</span></div>
            `;

            // DWE from server
            document.getElementById('val_dwe').textContent = data.metrics.dwe.toFixed(4);
            document.getElementById('bar_dwe').style.width = `${Math.max(100 - (data.metrics.dwe * 100), 0)}%`;

            recalculateLiveMetrics(currentProbGrid, thresh, currentGtGrid);
        }

        renderHeatmap(currentProbGrid);
        renderBinaryMap(currentProbGrid, thresh, currentGtGrid);

        if (window._hideLoading) window._hideLoading();

        const outSection = document.getElementById('output-section');
        outSection.classList.remove('hidden');
        setTimeout(() => { outSection.scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 200);

    } catch (e) {
        if (window._hideLoading) window._hideLoading();
        console.error(e);
        alert("INFERENCE FAILED — Check server connection.");
    } finally {
        btn.disabled = false;
        if (textEl) textEl.textContent = currentMode === 'upload' ? 'RUN FIXED CALIBRATION INFERENCE' : 'EXECUTE INFERENCE PIPELINE';
    }
}

// --- 6. LIVE SLIDER ---
document.getElementById('threshSlider').addEventListener('input', (e) => {
    const thresh = parseFloat(e.target.value);
    document.getElementById('threshVal').textContent = thresh.toFixed(2);

    if (currentProbGrid) {
        renderBinaryMap(currentProbGrid, thresh, currentGtGrid);
        if (currentGtGrid && currentMode !== 'upload') {
            recalculateLiveMetrics(currentProbGrid, thresh, currentGtGrid);
        }
    }
});

document.getElementById('toggleRings').addEventListener('change', () => {
    if (currentProbGrid) {
        renderHeatmap(currentProbGrid);
        renderBinaryMap(currentProbGrid, parseFloat(document.getElementById('threshSlider').value), currentGtGrid);
    }
});

function recalculateLiveMetrics(probGrid, thresh, gtGrid) {
    let tp = 0, fp = 0, fn = 0;
    const size = probGrid.length;
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const pred = probGrid[y][x] > thresh ? 1 : 0;
            const gt   = gtGrid[y][x];
            if (pred === 1 && gt === 1) tp++;
            else if (pred === 1 && gt === 0) fp++;
            else if (pred === 0 && gt === 1) fn++;
        }
    }
    const iou  = tp / (tp + fp + fn + 1e-6);
    const prec = tp / (tp + fp + 1e-6);
    const rec  = tp / (tp + fn + 1e-6);

    updateMetricCard('iou',  iou);
    updateMetricCard('prec', prec);
    updateMetricCard('rec',  rec);
}

function updateMetricCard(id, val) {
    const valEl = document.getElementById(`val_${id}`);
    const barEl = document.getElementById(`bar_${id}`);
    if (!valEl || !barEl) return;

    valEl.textContent = val.toFixed(4);
    barEl.style.width = `${Math.min(val * 100, 100)}%`;

    if (id === 'iou') {
        if (val > 0.35) barEl.style.background = 'linear-gradient(90deg, #10b981, #059669)';
        else if (val > 0.25) barEl.style.background = 'linear-gradient(90deg, #f59e0b, #d97706)';
        else barEl.style.background = 'linear-gradient(90deg, #ef4444, #b91c1c)';
    }
}

// --- 7. CANVAS RENDERING ---
function renderHeatmap(grid) {
    const canvas = document.getElementById('canvasHeatmap');
    const ctx    = canvas.getContext('2d');
    const size   = grid.length;
    const scale  = canvas.width / size;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const p = grid[y][x];
            if (p < 0.05) {
                ctx.fillStyle = '#040a0f';
            } else if (p < 0.3) {
                const t = p / 0.3;
                const r = Math.round(0 + t * 0);
                const g = Math.round(14 + t * (80 - 14));
                const b = Math.round(26 + t * (130 - 26));
                ctx.fillStyle = `rgb(${r},${g},${b})`;
            } else if (p < 0.6) {
                const t = (p - 0.3) / 0.3;
                const r = Math.round(0 + t * 0);
                const g = Math.round(80 + t * (180 - 80));
                const b = Math.round(130 + t * (220 - 130));
                ctx.fillStyle = `rgb(${r},${g},${b})`;
            } else if (p < 0.85) {
                const t = (p - 0.6) / 0.25;
                const r = Math.round(0 + t * 255);
                const g = Math.round(180 + t * (184 - 180));
                const b = Math.round(220 + t * (0 - 220));
                ctx.fillStyle = `rgb(${r},${g},${b})`;
            } else {
                const t = (p - 0.85) / 0.15;
                const r = Math.round(255);
                const g = Math.round(184 + t * (255 - 184));
                const b = Math.round(0 + t * 255);
                ctx.fillStyle = `rgb(${r},${g},${b})`;
            }
            ctx.fillRect(x * scale, (size - 1 - y) * scale, scale, scale);
        }
    }

    if (document.getElementById('toggleRings').checked) drawRangeRings(ctx, canvas.width, canvas.height, scale);
    drawEgo(ctx, canvas.width, canvas.height);
}

function renderBinaryMap(probGrid, threshold, gtGrid) {
    const canvas = document.getElementById('canvasBinary');
    const ctx    = canvas.getContext('2d');
    const size   = probGrid.length;
    const scale  = canvas.width / size;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#040a0f';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const pred = probGrid[y][x] > threshold ? 1 : 0;
            if (gtGrid) {
                const gt = gtGrid[y][x];
                if      (pred === 1 && gt === 1) ctx.fillStyle = '#10b981';
                else if (pred === 1 && gt === 0) ctx.fillStyle = '#ef4444';
                else if (pred === 0 && gt === 1) ctx.fillStyle = '#f59e0b';
                else continue;
            } else {
                if (pred === 1) ctx.fillStyle = '#00d4ff';
                else continue;
            }
            ctx.fillRect(x * scale, (size - 1 - y) * scale, scale, scale);
        }
    }

    if (document.getElementById('toggleRings').checked) drawRangeRings(ctx, canvas.width, canvas.height, scale);
    drawEgo(ctx, canvas.width, canvas.height);
}

function drawRangeRings(ctx, w, h, scale) {
    const cx = w / 2, cy = h / 2;
    const rings = [25, 50, 75];
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.2)';
    ctx.lineWidth = 1;

    rings.forEach((r, idx) => {
        const radius = r * scale;
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
        ctx.stroke();

        ctx.fillStyle = 'rgba(0, 212, 255, 0.5)';
        ctx.font = `10px 'Share Tech Mono', monospace`;
        ctx.fillText(`${(idx + 1) * 10}m`, cx + radius + 4, cy - 4);
    });
}

function drawEgo(ctx, w, h) {
    const cx = w / 2, cy = h / 2;

    // Glow
    ctx.shadowColor = 'rgba(0,212,255,0.8)';
    ctx.shadowBlur = 10;
    ctx.fillStyle = 'rgba(0,212,255,0.15)';
    ctx.beginPath();
    ctx.arc(cx, cy, 12, 0, Math.PI * 2);
    ctx.fill();

    // Body
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#e2e8f0';
    ctx.fillRect(cx - 5, cy - 9, 10, 18);

    // Nose indicator (front)
    ctx.fillStyle = '#00d4ff';
    ctx.fillRect(cx - 5, cy - 11, 10, 3);

    // Tail
    ctx.fillStyle = '#ef4444';
    ctx.fillRect(cx - 4, cy + 7, 8, 2);

    ctx.shadowBlur = 0;
}

// --- 8. HOVER TOOLTIPS ---
const canvases = [document.getElementById('canvasHeatmap'), document.getElementById('canvasBinary')];
const tooltip  = document.getElementById('bevTooltip');

canvases.forEach(canvas => {
    canvas.addEventListener('mousemove', (e) => {
        if (!currentProbGrid) return;

        const rect   = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const mouseX = (e.clientX - rect.left) * scaleX;
        const mouseY = (e.clientY - rect.top)  * scaleY;

        const cellSize = canvas.width / currentProbGrid.length;
        const gridX = Math.floor(mouseX / cellSize);
        const gridY = currentProbGrid.length - 1 - Math.floor(mouseY / cellSize);

        if (gridX >= 0 && gridX < 200 && gridY >= 0 && gridY < 200) {
            const dx   = (gridX - 100) * 0.4;
            const dy   = (gridY - 100) * 0.4;
            const dist = Math.sqrt(dx * dx + dy * dy).toFixed(1);
            const prob = currentProbGrid[gridY][gridX].toFixed(3);
            const thresh  = parseFloat(document.getElementById('threshSlider').value);
            const isOcc   = parseFloat(prob) > thresh;

            let gtText = '<span style="color:#4a6b8a">N/A — UPLOAD MODE</span>';
            if (currentGtGrid) {
                const gt = currentGtGrid[gridY][gridX];
                if (isOcc && gt)   gtText = `<span style="color:#10b981">OCCUPIED // TRUE POSITIVE ✓</span>`;
                else if (isOcc && !gt) gtText = `<span style="color:#ef4444">FREE // FALSE POSITIVE ✗</span>`;
                else if (!isOcc && gt) gtText = `<span style="color:#f59e0b">OCCUPIED // MISSED (FN) ✗</span>`;
                else                   gtText = `<span style="color:#4a6b8a">FREE // TRUE NEGATIVE ✓</span>`;
            }

            tooltip.innerHTML = `
                <p>POS  X:<span class="hl">${dx >= 0 ? '+' : ''}${dx.toFixed(1)}m</span>  Y:<span class="hl">${dy >= 0 ? '+' : ''}${dy.toFixed(1)}m</span>  Δ:<span class="hl">${dist}m</span></p>
                <hr>
                <p>CONFIDENCE  <span class="hl">${prob}</span></p>
                <p>PREDICTION  ${isOcc ? '<span style="color:#00d4ff;font-weight:bold">■ OCCUPIED</span>' : '<span style="color:#4a6b8a">□ FREE</span>'}</p>
                <p>GROUND TRUTH  ${gtText}</p>
            `;

            tooltip.style.left = (e.clientX + 18) + 'px';
            tooltip.style.top  = (e.clientY + 18) + 'px';
            tooltip.classList.remove('hidden');
        }
    });

    canvas.addEventListener('mouseleave', () => tooltip.classList.add('hidden'));
});