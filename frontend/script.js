const fileInput = document.getElementById('fileInput');
const dropzone = document.getElementById('dropzone');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const runAnalysisBtn = document.getElementById('runAnalysisBtn');
const classResult = document.getElementById('classResult');

// Result Images
const resOriginal = document.getElementById('resOriginal');
const resUnet = document.getElementById('resUnet');
const resSAUnet = document.getElementById('resSAUnet');

// Metrics
const bmUnetDice = document.getElementById('bmUnetDice');
const bmSAUnetDice = document.getElementById('bmSAUnetDice');
const modelStatus = document.getElementById('modelStatus');

let selectedFile = null;

// Backend API URL (FastAPI default)
const API_URL = 'http://127.0.0.1:8081';

// Handle file selection
fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

// Drag and drop
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'var(--primary)';
});

dropzone.addEventListener('dragleave', () => {
    dropzone.style.borderColor = 'var(--primary-light)';
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'var(--primary-light)';
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        resOriginal.src = e.target.result;
        previewContainer.style.display = 'block';
        dropzone.style.display = 'none';
        runAnalysisBtn.disabled = false;
        
        // Reset results
        resUnet.src = 'placeholder.png';
        resSAUnet.src = 'placeholder.png';
        classResult.innerText = 'Waiting for analysis...';
    };
    reader.readAsDataURL(file);
}

runAnalysisBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    runAnalysisBtn.disabled = true;
    runAnalysisBtn.innerText = 'Analyzing...';
    classResult.innerText = 'Analyzing...';

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch(`${API_URL}/analyze`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        
        // Update images
        resUnet.src = data.unet_mask;
        resSAUnet.src = data.sa_unet_mask;
        
        // Update classification
        if (!data.is_skin_image) {
            // Not a dermoscopic image – show rejection banner
            resUnet.src    = data.unet_mask;
            resSAUnet.src  = data.sa_unet_mask;
            classResult.innerHTML = `
                <div style="background:rgba(239, 68, 68, 0.1);border:1px solid rgba(239, 68, 68, 0.3);border-radius:12px;padding:1.5rem;text-align:center;">
                    <div style="font-size:2.5rem;margin-bottom:0.5rem;">🚫</div>
                    <div style="font-weight:700;color:#fca5a5;font-size:1.2rem;margin:0.5rem 0;">Not a Skin Lesion Image</div>
                    <div style="font-size:0.95rem;color:var(--text-muted);line-height:1.4;">Please upload a dermoscopic skin lesion image.<br>Non-skin images are rejected.</div>
                    <div style="margin-top:1rem;padding:0.75rem;background:rgba(0,0,0,0.3);border-radius:8px;border:1px solid rgba(255,255,255,0.05);">
                        <strong style="color:var(--text-muted);">All disease scores: 0%</strong>
                    </div>
                </div>`;
        } else {
            // Valid skin image – show probability bars
            const allProbs = Object.entries(data.predicted_class_probs);
            const maxProb  = Math.max(...allProbs.map(([,p]) => p));
            const barColors = {
                'Common Nevus':  '#06b6d4',
                'Atypical Nevus':'#fbbf24',
                'Melanoma':      '#f87171'
            };
            let probsHtml = '';
            for (const [cls, prob] of allProbs) {
                const color    = barColors[cls] || '#06b6d4';
                const isTop    = prob === maxProb;
                const barWidth = prob.toFixed(1);
                probsHtml += `
                    <div style="margin-bottom:1.25rem;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                            <span style="font-weight:${isTop?'600':'400'};color:${isTop?color:'var(--text-muted)'};font-size:1.05rem;">${cls}${isTop?' ← Predicted':''}</span>
                            <strong style="color:${color};text-shadow:0 0 10px rgba(255,255,255,0.1);">${prob}%</strong>
                        </div>
                        <div style="background:rgba(255,255,255,0.1);border-radius:6px;height:12px;overflow:hidden;box-shadow:inset 0 2px 4px rgba(0,0,0,0.3);">
                            <div style="width:${barWidth}%;background:${color};height:100%;border-radius:6px;transition:width 0.8s cubic-bezier(0.4, 0, 0.2, 1);box-shadow:0 0 10px ${color};"></div>
                        </div>
                    </div>`;
            }
            classResult.innerHTML = probsHtml;
        }

        // Update status
        const status = data.models_loaded;
        function badge(loaded) {
            return loaded
                ? '<span style="color:#22d3ee;font-weight:600;text-shadow:0 0 8px rgba(34, 211, 238, 0.4);">✅ Loaded</span>'
                : '<span style="color:#f87171;font-weight:600;text-shadow:0 0 8px rgba(248, 113, 113, 0.4);">❌ Untrained</span>';
        }
        modelStatus.innerHTML = `
            <div>U-Net: ${badge(status.unet)}</div>
            <div>SA U-Net: ${badge(status.sa_unet)}</div>
            <div>Classifier: ${badge(status.classifier)}</div>
        `;

        // Dummy benchmark stats for demo (In a real app, this comes from test evaluation JSON)
        bmUnetDice.innerText = '0.842';
        bmSAUnetDice.innerText = '0.905';

    } catch (error) {
        console.error('Error during analysis:', error);
        classResult.innerText = 'Error connecting to backend.';
        alert('Could not connect to the backend API. Make sure FastAPI is running on port 8081.');
    } finally {
        runAnalysisBtn.disabled = false;
        runAnalysisBtn.innerText = 'Run Analysis';
    }
});
