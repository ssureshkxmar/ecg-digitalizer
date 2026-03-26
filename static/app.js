document.addEventListener('DOMContentLoaded', () => {
    /* ── Element References ──────────────────────────────── */
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');

    // Layout sections
    const uploadSec = document.getElementById('upload-section');
    const loadSec = document.getElementById('loading-section');
    const mainImageDisplay = document.getElementById('main-image-display');
    const galleryPanel = document.getElementById('gallery-panel-section');
    const thumbnailsBar = document.getElementById('viewport-thumbnails');

    // Buttons & inputs
    const resetBtn = document.getElementById('reset-btn');
    const csvBtn = document.getElementById('csv-btn');
    const optGrid = document.getElementById('opt-grid');
    const optWavelet = document.getElementById('opt-wavelet');
    const optNotch = document.getElementById('opt-notch');

    // Telemetry
    const telemStatus = document.getElementById('telem-status');
    const telemQuality = document.getElementById('telem-quality');
    const telemGrade = document.getElementById('telem-grade');
    const telemScore = document.getElementById('telem-score');
    const telemGrid = document.getElementById('telem-grid');
    const telemFilter = document.getElementById('telem-filter');

    // Clinical
    const clinHR = document.getElementById('clin-hr');
    const clinBeats = document.getElementById('clin-beats');
    const clinPR = document.getElementById('clin-pr');
    const clinQRS = document.getElementById('clin-qrs');
    const clinQT = document.getElementById('clin-qt');
    const clinQTc = document.getElementById('clin-qtc');

    // Status bar
    const statusFile = document.getElementById('status-file');
    const statusMessage = document.getElementById('status-message');
    const statusStage = document.getElementById('status-stage');
    const statusTimeDisplay = document.getElementById('status-time-display');

    let currentCoords = null;
    let currentImages = null;
    let pipeTimer = null;

    /* ── Clock ───────────────────────────────────────────── */
    function updateClock() {
        const now = new Date();
        const timeStr = now.toLocaleTimeString('en-US', { hour12: false });
        const dateStr = now.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        document.getElementById('topbar-time').textContent = `${dateStr} ${timeStr}`;
        statusTimeDisplay.textContent = timeStr;
    }
    updateClock();
    setInterval(updateClock, 1000);

    /* ── Settings listeners ──────────────────────────────── */
    optGrid.addEventListener('change', () => {
        telemGrid.textContent = optGrid.checked ? 'ENABLED' : 'DISABLED';
    });
    optNotch.addEventListener('change', () => {
        telemFilter.textContent = optNotch.value === 'None' ? 'OFF' : optNotch.value + 'Hz';
    });

    /* ── Drag & Drop ─────────────────────────────────────── */
    // dropzone is now a label linked to file-input, no manual click listener needed

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(e =>
        dropzone.addEventListener(e, ev => { ev.preventDefault(); ev.stopPropagation(); }, false));
    ['dragenter', 'dragover'].forEach(e =>
        dropzone.addEventListener(e, () => dropzone.classList.add('dragover'), false));
    ['dragleave', 'drop'].forEach(e =>
        dropzone.addEventListener(e, () => dropzone.classList.remove('dragover'), false));
    dropzone.addEventListener('drop', e => handle(e.dataTransfer.files), false);
    fileInput.addEventListener('change', function () { handle(this.files); });

    function handle(files) {
        if (files.length && files[0].type.startsWith('image/')) upload(files[0]);
        else alert('Upload a valid image (PNG, JPG, etc.).');
    }

    /* ── Pipeline animation ──────────────────────────────── */
    function animPipe() {
        const s = document.querySelectorAll('.pipeline-steps .step');
        let i = 0;
        s.forEach(el => el.classList.remove('active', 'done'));
        pipeTimer = setInterval(() => {
            s.forEach(el => el.classList.remove('active'));
            if (i > 0) s[i - 1].classList.add('done');
            if (i < s.length) {
                s[i].classList.add('active');
                statusStage.textContent = `STAGE: ${i + 1}/9`;
                i++;
            } else {
                clearInterval(pipeTimer);
            }
        }, 400);
    }

    function stopPipe() {
        clearInterval(pipeTimer);
        document.querySelectorAll('.pipeline-steps .step').forEach(s => {
            s.classList.remove('active');
            s.classList.add('done');
        });
        statusStage.textContent = 'STAGE: COMPLETE';
    }

    /* ── Upload ──────────────────────────────────────────── */
    async function upload(file) {
        uploadSec.classList.add('hidden');
        mainImageDisplay.classList.add('hidden');
        loadSec.classList.remove('hidden');
        telemStatus.textContent = 'PROCESSING';
        telemStatus.style.color = 'var(--orange)';
        statusMessage.textContent = 'Processing ' + file.name + '...';
        statusFile.textContent = file.name.toUpperCase();
        animPipe();

        const fd = new FormData();
        fd.append('file', file);
        fd.append('grid_suppression', optGrid.checked);
        fd.append('wavelet_denoise', optWavelet.checked);
        fd.append('notch_filter', optNotch.value);

        try {
            const res = await fetch('/api/extract', { method: 'POST', body: fd });
            if (!res.ok) throw new Error((await res.json()).error || 'Failed to process image');
            const data = await res.json();
            stopPipe();
            setTimeout(() => render(data), 300);
        } catch (e) {
            stopPipe();
            console.error(e);
            alert('Error: ' + e.message);
            loadSec.classList.add('hidden');
            uploadSec.classList.remove('hidden');
            telemStatus.textContent = 'ERROR';
            telemStatus.style.color = 'var(--red)';
            statusMessage.textContent = 'Error: ' + e.message;
            statusStage.textContent = 'STAGE: FAILED';
        }
    }

    /* ── Render ───────────────────────────────────────────── */
    function render(data) {
        loadSec.classList.add('hidden');
        mainImageDisplay.classList.remove('hidden');
        galleryPanel.classList.remove('hidden');
        currentCoords = data.coordinates;
        currentImages = data.images;

        telemStatus.textContent = 'COMPLETE';
        telemStatus.style.color = 'var(--green)';
        statusMessage.textContent = 'Analysis complete — All stages finished';

        renderTelemetry(data.quality);
        renderClinical(data.wave_detection, data.clinical);
        renderQualityBars(data.quality);
        renderFlags(data.clinical);
        renderExtraction(data.extraction_info, data.wave_detection);
        renderGallery(data.images, optGrid.checked);
        renderThumbnails(data.images, data.signal_plot);

        // Set initial main view to signal plot
        document.getElementById('gallery-image').src = data.signal_plot || '';
        document.getElementById('signal-plot').src = data.signal_plot || '';
        document.getElementById('annotated-image').src = data.images.annotated || '';
        document.getElementById('confidence-image').src = data.images.confidence_map || '';
        if (data.images.overlay) {
            document.getElementById('overlay-image').src = data.images.overlay || '';
        }

        // Set main view label
        document.getElementById('main-view-label').textContent = 'SIGNAL PLOT';
        document.getElementById('main-view-sublabel').textContent = 'Reconstructed Digital ECG';

        // Activate first thumbnail
        document.querySelectorAll('.thumb-card').forEach(t => t.classList.remove('active'));
        document.getElementById('thumb-signal').classList.add('active');
    }

    /* ── Telemetry ───────────────────────────────────────── */
    function renderTelemetry(q) {
        const grade = (q.grade || 'UNKNOWN');
        const score = q.overall_score || 0;
        telemQuality.textContent = grade;
        telemQuality.style.color = score >= 70 ? 'var(--green)' : score >= 40 ? 'var(--orange)' : 'var(--red)';
        telemGrade.textContent = grade;
        telemGrade.style.color = telemQuality.style.color;
        telemScore.textContent = score + '/100';
        telemScore.style.color = telemQuality.style.color;
    }

    /* ── Clinical Metrics ────────────────────────────────── */
    function renderClinical(wd, clin) {
        const met = (wd && wd.metrics) || {};
        clinHR.textContent = met.heart_rate_bpm != null ? met.heart_rate_bpm + ' BPM' : '—';
        clinBeats.textContent = met.num_beats != null ? met.num_beats : '—';
        clinPR.textContent = met.avg_PR_ms != null ? met.avg_PR_ms + ' ms' : '—';
        clinQRS.textContent = met.avg_QRS_ms != null ? met.avg_QRS_ms + ' ms' : '—';
        clinQT.textContent = met.avg_QT_ms != null ? met.avg_QT_ms + ' ms' : '—';
        clinQTc.textContent = met.QTc_ms != null ? met.QTc_ms + ' ms' : '—';

        // Color the heart rate
        if (met.heart_rate_bpm != null) {
            const hr = parseFloat(met.heart_rate_bpm);
            clinHR.style.color = (hr >= 60 && hr <= 100) ? 'var(--green)' : 'var(--orange)';
        }
    }

    /* ── Quality Bars ────────────────────────────────────── */
    function renderQualityBars(q) {
        const container = document.getElementById('quality-bars');
        container.innerHTML = '';
        const metrics = q.metrics || {};
        const labels = {
            blur: 'Sharpness', motion: 'Stability', resolution: 'Resolution',
            skew: 'Alignment', shadow: 'Lighting', contrast: 'Contrast',
            visibility: 'Visibility'
        };
        for (const [k, label] of Object.entries(labels)) {
            const v = metrics[k];
            if (!v) continue;
            const score = v.score || 0;
            const col = score >= 70 ? 'green' : score >= 40 ? 'yellow' : 'red';
            const item = document.createElement('div');
            item.className = 'quality-bar-item';
            item.innerHTML = `
                <div class="qb-top">
                    <span class="qb-label">${label}</span>
                    <span class="qb-val">${score}</span>
                </div>
                <div class="qb-track">
                    <div class="qb-fill ${col}" style="width:${score}%"></div>
                </div>`;
            container.appendChild(item);
        }
        if (container.children.length === 0) {
            container.innerHTML = '<div style="font-size:10px;color:var(--text-dim);text-align:center;padding:8px;">No quality data</div>';
        }
    }

    /* ── Clinical Flags ──────────────────────────────────── */
    function renderFlags(clin) {
        const fl = document.getElementById('flags-list');
        fl.innerHTML = '';
        const flags = (clin && clin.flags) || [];
        if (flags.length === 0) {
            fl.innerHTML = '<div class="warning-item ok"><i class="fa-solid fa-circle-check"></i> All clinical parameters within normal range.</div>';
            return;
        }
        flags.forEach(f => {
            const d = document.createElement('div');
            d.className = 'warning-item warn';
            d.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> <div><strong>${f.metric}:</strong> ${f.issue}</div>`;
            fl.appendChild(d);
        });
    }

    /* ── Extraction Info ─────────────────────────────────── */
    function renderExtraction(ei, wd) {
        const c = document.getElementById('extraction-info');
        c.innerHTML = '';
        const mr = (ei && ei.method_results) || {};
        const items = [
            [ei ? ei.total_points : 0, 'Total Points'],
            [ei ? (ei.method_agreement * 100).toFixed(0) + '%' : '—', 'Method Agreement'],
        ];
        for (const [name, info] of Object.entries(mr)) {
            items.push([info.num_points, name.replace('_', ' ')]);
        }
        const di = (wd && wd.detection_info) || {};
        if (di.avg_r_confidence) items.push([(di.avg_r_confidence * 100).toFixed(0) + '%', 'R-Peak Confidence']);

        items.forEach(([val, label]) => {
            const d = document.createElement('div');
            d.className = 'ext-row';
            d.innerHTML = `<span class="ext-label">${label}</span><span class="ext-value">${val}</span>`;
            c.appendChild(d);
        });
    }

    /* ── Gallery (sidebar) ───────────────────────────────── */
    function renderGallery(images, gridEnabled) {
        const names = {
            original: 'Original',
            enhanced: 'Enhanced',
            grid_colour: 'Grid (Colour)',
            grid_fft: 'Grid (FFT)',
            grid_morphological: 'Grid (Morph)',
            grid_removed_fused: 'Grid Fused',
            binary: 'Binary',
            cleaned: 'Cleaned',
            skeleton: 'Skeleton',
            overlay: 'Overlay',
            confidence_map: 'Confidence',
            annotated: 'Annotated',
        };
        const tabs = document.getElementById('gallery-tabs');
        tabs.innerHTML = '';
        let first = true;

        for (const [k, label] of Object.entries(names)) {
            if (!images[k]) continue;
            if (!gridEnabled && (k === 'grid_colour' || k === 'grid_fft' || k === 'grid_morphological' || k === 'grid_removed_fused')) {
                if (k !== 'grid_removed_fused') continue;
            }

            const t = document.createElement('div');
            t.className = 'gallery-tab' + (first ? ' active' : '');

            let displayLabel = label;
            if (!gridEnabled && k === 'grid_removed_fused') displayLabel = 'Base (No Grid)';

            t.innerHTML = `<i class="fa-solid fa-image"></i> ${displayLabel}`;
            t.addEventListener('click', () => {
                document.querySelectorAll('.gallery-tab').forEach(el => el.classList.remove('active'));
                t.classList.add('active');
                const img = document.getElementById('gallery-image');
                img.style.opacity = 0;
                setTimeout(() => {
                    img.src = images[k];
                    img.style.opacity = 1;
                }, 150);
                document.getElementById('main-view-label').textContent = displayLabel.toUpperCase();
                document.getElementById('main-view-sublabel').textContent = 'Pipeline Stage';
                // Deactivate thumbnails
                document.querySelectorAll('.thumb-card').forEach(tc => tc.classList.remove('active'));
            });
            tabs.appendChild(t);
            if (first) first = false;
        }
    }

    /* ── Thumbnails ──────────────────────────────────────── */
    function renderThumbnails(images, signalPlot) {
        // Set thumbnail images
        if (signalPlot) document.getElementById('signal-plot').src = signalPlot;
        if (images.annotated) document.getElementById('annotated-image').src = images.annotated;
        if (images.confidence_map) document.getElementById('confidence-image').src = images.confidence_map;
        if (images.overlay) document.getElementById('overlay-image').src = images.overlay;
    }

    // Thumbnail click handlers
    document.querySelectorAll('.thumb-card').forEach(card => {
        card.addEventListener('click', () => {
            const view = card.dataset.view;
            const img = document.getElementById('gallery-image');
            let src = '';
            let label = '';
            let sublabel = '';

            switch (view) {
                case 'signal':
                    src = document.getElementById('signal-plot').src;
                    label = 'SIGNAL PLOT';
                    sublabel = 'Reconstructed Digital ECG';
                    break;
                case 'annotated':
                    src = document.getElementById('annotated-image').src;
                    label = 'ANNOTATED';
                    sublabel = 'PQRST Wave Detection';
                    break;
                case 'confidence':
                    src = document.getElementById('confidence-image').src;
                    label = 'CONFIDENCE';
                    sublabel = 'Extraction Confidence Map';
                    break;
                case 'overlay':
                    src = document.getElementById('overlay-image').src;
                    label = 'OVERLAY';
                    sublabel = 'Signal Overlay View';
                    break;
            }

            if (src) {
                img.style.opacity = 0;
                setTimeout(() => { img.src = src; img.style.opacity = 1; }, 150);
                document.getElementById('main-view-label').textContent = label;
                document.getElementById('main-view-sublabel').textContent = sublabel;
            }

            document.querySelectorAll('.thumb-card').forEach(t => t.classList.remove('active'));
            card.classList.add('active');

            // Deactivate gallery tabs
            document.querySelectorAll('.gallery-tab').forEach(g => g.classList.remove('active'));
        });
    });

    /* ── Icon sidebar nav highlights ─────────────────────── */
    document.querySelectorAll('.icon-btn[id^="nav-"]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.icon-sidebar-top .icon-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    /* ── Actions ──────────────────────────────────────────── */
    resetBtn.addEventListener('click', () => {
        mainImageDisplay.classList.add('hidden');
        galleryPanel.classList.add('hidden');
        uploadSec.classList.remove('hidden');
        loadSec.classList.add('hidden');
        fileInput.value = '';
        currentCoords = null;
        currentImages = null;
        telemStatus.textContent = 'IDLE';
        telemStatus.style.color = 'var(--green)';
        telemQuality.textContent = '—';
        telemGrade.textContent = '—';
        telemScore.textContent = '—';
        clinHR.textContent = '—';
        clinBeats.textContent = '—';
        clinPR.textContent = '—';
        clinQRS.textContent = '—';
        clinQT.textContent = '—';
        clinQTc.textContent = '—';
        document.getElementById('quality-bars').innerHTML = '';
        document.getElementById('flags-list').innerHTML = '<div class="warning-item ok"><i class="fa-solid fa-circle-check"></i> No data loaded</div>';
        document.getElementById('extraction-info').innerHTML = '';
        document.getElementById('gallery-tabs').innerHTML = '';
        statusFile.textContent = 'NO FILE';
        statusMessage.textContent = 'Ready';
        statusStage.textContent = 'STAGE: IDLE';
        document.getElementById('main-view-label').textContent = 'DIGITALIZER';
        document.getElementById('main-view-sublabel').textContent = '10-Stage Architecture';
        document.querySelectorAll('.thumb-card').forEach(t => t.classList.remove('active'));
    });

    csvBtn.addEventListener('click', () => {
        if (!currentCoords || !currentCoords.length) {
            statusMessage.textContent = 'No data to export';
            return;
        }
        let csv = 'X_Pixel,Y_Pixel,Time_s,Voltage_mV\n';
        currentCoords.forEach(p => csv += `${p.x},${p.y},${p.time_s},${p.voltage_mv}\n`);
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = 'ecg_signal_calibrated.csv';
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        URL.revokeObjectURL(url);
        statusMessage.textContent = 'CSV exported successfully';
    });

    /* ── Crosshair interaction ───────────────────────────── */
    const canvasWrapper = document.querySelector('.viewport-canvas-wrapper');
    canvasWrapper.addEventListener('mousemove', (e) => {
        if (mainImageDisplay.classList.contains('hidden')) return;
        const rect = canvasWrapper.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        document.querySelector('.crosshair-h').style.top = y + 'px';
        document.querySelector('.crosshair-v').style.left = x + 'px';
    });
});
