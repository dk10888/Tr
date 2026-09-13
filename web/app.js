document.addEventListener('DOMContentLoaded', () => {
    // State Variables
    let selectedFile = null;
    let processedData = null;

    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const uploadInner = document.getElementById('uploadInner');
    const previewInner = document.getElementById('previewInner');
    const imagePreview = document.getElementById('imagePreview');
    const changeImgBtn = document.getElementById('changeImgBtn');

    const confSlider = document.getElementById('confSlider');
    const sliderValue = document.getElementById('sliderValue');

    const spinnerCard = document.getElementById('spinnerCard');
    const spinnerText = document.getElementById('spinnerText');
    const resultsSection = document.getElementById('resultsSection');

    const metricTotal = document.getElementById('metricTotal');
    const metricFood = document.getElementById('metricFood');
    const metricDiscard = document.getElementById('metricDiscard');
    const metricTime = document.getElementById('metricTime');

    const tableFood = document.getElementById('tableFood').querySelector('tbody');
    const tableDiscard = document.getElementById('tableDiscard').querySelector('tbody');
    const tableCompare = document.getElementById('tableCompare').querySelector('tbody');
    const annotatedImage = document.getElementById('annotatedImage');

    const countFoodTab = document.getElementById('countFoodTab');
    const countDiscardTab = document.getElementById('countDiscardTab');

    const btnDownloadCsv = document.getElementById('btnDownloadCsv');
    const btnDownloadTxt = document.getElementById('btnDownloadTxt');
    const btnDownloadJson = document.getElementById('btnDownloadJson');

    // Slider listener
    confSlider.addEventListener('input', (e) => {
        sliderValue.textContent = `${e.target.value}%`;
        if (selectedFile) {
            processReceipt();
        }
    });

    // Model choice radio listener
    document.querySelectorAll('input[name="modelChoice"]').forEach(radio => {
        radio.addEventListener('change', () => {
            if (selectedFile) {
                processReceipt();
            }
        });
    });

    // Drag & Drop
    dropzone.addEventListener('click', (e) => {
        if (e.target !== changeImgBtn && !changeImgBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-over');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('drag-over');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    changeImgBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid receipt image file (PNG/JPG).');
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadInner.classList.add('hidden');
            previewInner.classList.remove('hidden');
            processReceipt();
        };
        reader.readAsDataURL(file);
    }

    // Process Receipt API Call
    async function processReceipt() {
        if (!selectedFile) return;

        // Show spinner
        spinnerCard.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        const modelChoice = document.querySelector('input[name="modelChoice"]:checked').value;
        const confThreshold = parseFloat(confSlider.value) / 100.0;

        spinnerText.textContent = `Running RapidOCR & ${modelChoice === 'compare' ? 'Side-by-Side Model Inference' : modelChoice.toUpperCase()}...`;

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('model_choice', modelChoice);
        formData.append('conf_threshold', confThreshold.toString());

        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`Server returned ${response.status}: ${errText}`);
            }

            processedData = await response.json();
            renderResults(processedData);
        } catch (err) {
            alert(`Error processing receipt: ${err.message}`);
        } finally {
            spinnerCard.classList.add('hidden');
        }
    }

    // Render Results
    function renderResults(data) {
        resultsSection.classList.remove('hidden');

        // Metrics
        metricTotal.textContent = data.total_lines || 0;
        metricFood.textContent = data.food_items.length;
        metricDiscard.textContent = data.discarded_items.length;
        metricTime.textContent = `${data.elapsed_ms} ms`;

        countFoodTab.textContent = data.food_items.length;
        countDiscardTab.textContent = data.discarded_items.length;

        // Table 1: Food Items
        tableFood.innerHTML = '';
        if (data.food_items.length === 0) {
            tableFood.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-secondary);">No food items detected above confidence threshold.</td></tr>`;
        } else {
            data.food_items.forEach((item, idx) => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${idx + 1}</td>
                    <td><strong>${escapeHtml(item.text)}</strong></td>
                    <td><span class="conf-badge conf-badge-green">${(item.confidence * 100).toFixed(1)}%</span></td>
                    <td><span class="badge badge-success">🍏 Food</span></td>
                `;
                tableFood.appendChild(tr);
            });
        }

        // Table 2: Discarded Items
        tableDiscard.innerHTML = '';
        if (data.discarded_items.length === 0) {
            tableDiscard.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-secondary);">No discarded lines.</td></tr>`;
        } else {
            data.discarded_items.forEach((item, idx) => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${idx + 1}</td>
                    <td>${escapeHtml(item.text)}</td>
                    <td><span class="conf-badge conf-badge-gray">${(item.confidence * 100).toFixed(1)}%</span></td>
                    <td><span class="badge badge-secondary">Non-Food</span></td>
                `;
                tableDiscard.appendChild(tr);
            });
        }

        // Table 3: Comparison
        tableCompare.innerHTML = '';
        if (data.comparison && data.comparison.length > 0) {
            data.comparison.forEach((comp, idx) => {
                const tr = document.createElement('tr');
                const agree = comp.agree;
                tr.innerHTML = `
                    <td>${idx + 1}</td>
                    <td>${escapeHtml(comp.text)}</td>
                    <td>${comp.tiny_pred.toUpperCase()} (${(comp.tiny_conf * 100).toFixed(1)}%)</td>
                    <td>${comp.mini_pred.toUpperCase()} (${(comp.mini_conf * 100).toFixed(1)}%)</td>
                    <td><span class="agree-badge ${agree ? 'agree-yes' : 'agree-no'}">${agree ? '✅ Agree' : '⚠️ Differ'}</span></td>
                `;
                tableCompare.appendChild(tr);
            });
        } else {
            tableCompare.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-secondary);">Select "Side-by-Side Comparison" mode to view benchmark matrix.</td></tr>`;
        }

        // Annotated Image
        if (data.annotated_image_base64) {
            annotatedImage.src = `data:image/jpeg;base64,${data.annotated_image_base64}`;
        }
    }

    // Tabs Switcher
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const targetId = btn.getAttribute('data-tab');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // Exports
    btnDownloadCsv.addEventListener('click', () => {
        if (!processedData || !processedData.food_items) return;
        let csv = '#,Food Item Name,Confidence\n';
        processedData.food_items.forEach((item, idx) => {
            csv += `"${idx + 1}","${item.text.replace(/"/g, '""')}","${(item.confidence * 100).toFixed(1)}%"\n`;
        });
        downloadFile(csv, 'food_items.csv', 'text/csv');
    });

    btnDownloadTxt.addEventListener('click', () => {
        if (!processedData || !processedData.food_items) return;
        const txt = processedData.food_items.map(i => i.text).join('\n');
        downloadFile(txt, 'food_items.txt', 'text/plain');
    });

    btnDownloadJson.addEventListener('click', () => {
        if (!processedData || !processedData.food_items) return;
        const jsonStr = JSON.stringify(processedData.food_items, null, 2);
        downloadFile(jsonStr, 'food_items.json', 'application/json');
    });

    function downloadFile(content, fileName, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});
