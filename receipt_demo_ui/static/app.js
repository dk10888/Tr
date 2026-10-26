document.addEventListener("DOMContentLoaded", () => {
  // Mode Elements
  const modeTabs = document.querySelectorAll(".mode-tab");
  const imageModeView = document.getElementById("imageModeView");
  const textModeView = document.getElementById("textModeView");
  const presetSection = document.getElementById("presetSection");

  // Image Upload Elements
  const uploadDropzone = document.getElementById("uploadDropzone");
  const imageFileInput = document.getElementById("imageFileInput");
  const dropzoneEmpty = document.getElementById("dropzoneEmpty");
  const dropzonePreview = document.getElementById("dropzonePreview");
  const imagePreviewImg = document.getElementById("imagePreviewImg");
  const previewFilename = document.getElementById("previewFilename");
  const previewFilesize = document.getElementById("previewFilesize");
  const btnProcessImage = document.getElementById("btnProcessImage");
  const btnClearImage = document.getElementById("btnClearImage");
  const sampleThumbBtns = document.querySelectorAll(".sample-thumb-btn");

  // Text Elements
  const receiptInput = document.getElementById("receiptInput");
  const charLineCount = document.getElementById("charLineCount");
  const btnProcess = document.getElementById("btnProcess");
  const btnClear = document.getElementById("btnClear");
  const btnPaste = document.getElementById("btnPaste");
  const presetButtons = document.getElementById("presetButtons");

  // Metrics
  const metricItems = document.getElementById("metricItems");
  const metricFiltered = document.getElementById("metricFiltered");
  const metricSpend = document.getElementById("metricSpend");
  const metricConf = document.getElementById("metricConf");
  const metricLatency = document.getElementById("metricLatency");
  const metricPipelineTime = document.getElementById("metricPipelineTime");

  // Breakdown
  const spendBarContainer = document.getElementById("spendBarContainer");
  const spendTagsGrid = document.getElementById("spendTagsGrid");
  const breakdownCount = document.getElementById("breakdownCount");

  // Tables
  const itemsTbody = document.getElementById("itemsTbody");
  const noiseTbody = document.getElementById("noiseTbody");
  const tabItemCount = document.getElementById("tabItemCount");
  const tabNoiseCount = document.getElementById("tabNoiseCount");
  const jsonOutput = document.getElementById("jsonOutput");
  const btnCopyJson = document.getElementById("btnCopyJson");

  // Live Sandbox
  const sandboxInput = document.getElementById("sandboxInput");
  const sandboxResult = document.getElementById("sandboxResult");

  let presetsCache = {};
  let currentJsonPayload = {};
  let currentImageBase64 = null;

  // Category Color Map
  const categoryColors = {
    household: "#06b6d4",
    beverages: "#3b82f6",
    meat: "#ef4444",
    pantry: "#f97316",
    snacks: "#eab308",
    produce: "#10b981",
    dairy: "#8b5cf6",
    bakery: "#f59e0b",
    alcohol: "#ec4899",
    frozen: "#14b8a6",
    non_item: "#64748b",
  };

  // ── 1. Mode Switching ───────────────────────────────────────────────────────
  modeTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      modeTabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const mode = tab.dataset.mode;

      if (mode === "image") {
        imageModeView.style.display = "block";
        textModeView.style.display = "none";
        presetSection.style.display = "none";
      } else {
        imageModeView.style.display = "none";
        textModeView.style.display = "block";
        presetSection.style.display = "flex";
      }
    });
  });

  // ── 2. Image Upload & Dropzone ──────────────────────────────────────────────
  uploadDropzone.addEventListener("click", () => imageFileInput.click());

  uploadDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadDropzone.classList.add("dragover");
  });

  uploadDropzone.addEventListener("dragleave", () => {
    uploadDropzone.classList.remove("dragover");
  });

  uploadDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadDropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  imageFileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload an image file (PNG, JPG, WEBP).");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      currentImageBase64 = e.target.result;
      displayImagePreview(currentImageBase64, file.name, (file.size / 1024).toFixed(1) + " KB");
      btnProcessImage.disabled = false;
      // Auto-trigger scan
      runImageProcessing();
    };
    reader.readAsDataURL(file);
  }

  function displayImagePreview(b64Url, name, sizeStr) {
    imagePreviewImg.src = b64Url;
    previewFilename.textContent = name;
    previewFilesize.textContent = sizeStr;
    dropzoneEmpty.style.display = "none";
    dropzonePreview.style.display = "block";
  }

  btnClearImage.addEventListener("click", (e) => {
    e.stopPropagation();
    currentImageBase64 = null;
    imageFileInput.value = "";
    dropzoneEmpty.style.display = "flex";
    dropzonePreview.style.display = "none";
    btnProcessImage.disabled = true;
  });

  // Sample Images click handler
  sampleThumbBtns.forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const imgUrl = btn.dataset.img;
      const title = btn.getAttribute("title") || "sample_receipt.jpg";

      btnProcessImage.disabled = true;
      btnProcessImage.querySelector("span").textContent = "Loading sample...";

      try {
        const resp = await fetch(imgUrl);
        const blob = await resp.blob();
        const reader = new FileReader();
        reader.onload = () => {
          currentImageBase64 = reader.result;
          displayImagePreview(currentImageBase64, title, (blob.size / 1024).toFixed(1) + " KB");
          btnProcessImage.disabled = false;
          runImageProcessing();
        };
        reader.readAsDataURL(blob);
      } catch (err) {
        console.error("Failed to load sample image:", err);
      }
    });
  });

  btnProcessImage.addEventListener("click", runImageProcessing);

  async function runImageProcessing() {
    if (!currentImageBase64) return;

    btnProcessImage.disabled = true;
    const btnSpan = btnProcessImage.querySelector("span");
    btnSpan.textContent = "Scanning with RapidOCR...";

    try {
      const res = await fetch("/api/process_image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: currentImageBase64 }),
      });

      btnSpan.textContent = "Classifying with MobileBERT...";
      const data = await res.json();
      currentJsonPayload = data;

      // Copy OCR raw text to text tab input for inspection/editing
      if (data.raw_text) {
        receiptInput.value = data.raw_text;
        updateCharCount();
      }

      renderResults(data);
    } catch (err) {
      console.error("Image processing error:", err);
      alert("Error scanning image: " + err.message);
    } finally {
      btnProcessImage.disabled = false;
      btnSpan.textContent = "Scan Image & Classify";
    }
  }

  // ── 3. Fetch Presets (Text Mode) ───────────────────────────────────────────
  async function loadPresets() {
    try {
      const res = await fetch("/api/presets");
      presetsCache = await res.json();
      if (presetsCache.walmart) {
        receiptInput.value = presetsCache.walmart.text;
        updateCharCount();
      }
    } catch (e) {
      console.warn("Could not fetch presets:", e);
    }
  }

  presetButtons.addEventListener("click", (e) => {
    const btn = e.target.closest(".preset-btn");
    if (!btn) return;

    document.querySelectorAll(".preset-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");

    const key = btn.dataset.preset;
    if (presetsCache[key]) {
      receiptInput.value = presetsCache[key].text;
      updateCharCount();
      runProcessing();
    }
  });

  // ── 4. Editor Helpers ───────────────────────────────────────────────────────
  function updateCharCount() {
    const val = receiptInput.value;
    const lines = val ? val.split("\n").length : 0;
    const chars = val.length;
    charLineCount.textContent = `${lines} lines • ${chars.toLocaleString()} chars`;
  }

  receiptInput.addEventListener("input", updateCharCount);

  btnClear.addEventListener("click", () => {
    receiptInput.value = "";
    updateCharCount();
    receiptInput.focus();
  });

  btnPaste.addEventListener("click", async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (text) {
        receiptInput.value = text;
        updateCharCount();
        runProcessing();
      }
    } catch (err) {
      alert("Clipboard access denied. Please paste manually into the text area.");
    }
  });

  // Keyboard shortcut: Cmd+Enter or Ctrl+Enter
  window.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      runProcessing();
    }
  });

  btnProcess.addEventListener("click", runProcessing);

  // ── 5. Tabs Handling ────────────────────────────────────────────────────────
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.dataset.tab;
      const targetContent = document.getElementById(targetId);
      if (targetContent) targetContent.classList.add("active");
    });
  });

  btnCopyJson.addEventListener("click", () => {
    navigator.clipboard.writeText(JSON.stringify(currentJsonPayload, null, 2));
    btnCopyJson.textContent = "Copied!";
    setTimeout(() => {
      btnCopyJson.textContent = "Copy JSON";
    }, 1800);
  });

  // ── 6. Text Processing Call ─────────────────────────────────────────────────
  async function runProcessing() {
    const text = receiptInput.value.trim();
    if (!text) {
      alert("Please enter or paste receipt text first.");
      return;
    }

    btnProcess.disabled = true;
    btnProcess.querySelector("span").textContent = "Processing...";

    try {
      const res = await fetch("/api/process_receipt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const data = await res.json();
      currentJsonPayload = data;

      renderResults(data);
    } catch (err) {
      console.error("Processing failed:", err);
      alert("Error processing receipt: " + err.message);
    } finally {
      btnProcess.disabled = false;
      btnProcess.querySelector("span").textContent = "Run Structurer & MobileBERT";
    }
  }

  // ── 7. Render Results ───────────────────────────────────────────────────────
  function renderResults(data) {
    const { items, filtered_lines, summary, category_breakdown } = data;

    // Metrics
    metricItems.textContent = summary.total_items;
    metricFiltered.textContent = `${summary.filtered_lines_count} filtered noise`;
    metricSpend.textContent = `$${summary.total_spend.toFixed(2)}`;
    metricConf.textContent = `${summary.average_confidence}%`;
    metricLatency.textContent = `${summary.average_latency_ms} ms`;
    
    // Total pipeline string (shows OCR time if present)
    if (summary.ocr_latency_ms) {
      metricPipelineTime.textContent = `OCR: ${summary.ocr_latency_ms}ms • Total: ${summary.total_pipeline_ms}ms`;
    } else {
      metricPipelineTime.textContent = `Total pipeline: ${summary.total_pipeline_ms} ms`;
    }

    tabItemCount.textContent = items.length;
    tabNoiseCount.textContent = filtered_lines.length;
    breakdownCount.textContent = `${items.length} items parsed`;

    // Render Category Breakdown Bar
    renderSpendBreakdown(category_breakdown, summary.total_spend);

    // Render Items Table
    renderItemsTable(items);

    // Render Noise Table
    renderNoiseTable(filtered_lines);

    // Render JSON
    jsonOutput.querySelector("code").textContent = JSON.stringify(data, null, 2);
  }

  function renderSpendBreakdown(breakdown, totalSpend) {
    spendBarContainer.innerHTML = "";
    spendTagsGrid.innerHTML = "";

    if (!breakdown || breakdown.length === 0) {
      spendBarContainer.innerHTML = `<div style="padding:4px; font-size:0.7rem; color:var(--text-dim);">No items found</div>`;
      return;
    }

    breakdown.forEach((item) => {
      const color = categoryColors[item.category] || "#94a3b8";

      // Segment in bar
      const seg = document.createElement("div");
      seg.className = "spend-bar-segment";
      seg.style.width = `${item.percentage}%`;
      seg.style.backgroundColor = color;
      seg.title = `${item.category}: $${item.total_spend.toFixed(2)} (${item.percentage}%)`;
      spendBarContainer.appendChild(seg);

      // Tag Pill below
      const tag = document.createElement("div");
      tag.className = "spend-tag";
      tag.innerHTML = `
        <span class="spend-tag-dot" style="background:${color}"></span>
        <span style="font-weight:700; text-transform:capitalize;">${item.category}</span>
        <span class="spend-tag-pct">${item.percentage}%</span>
        <span class="spend-tag-amt">$${item.total_spend.toFixed(2)}</span>
      `;
      spendTagsGrid.appendChild(tag);
    });
  }

  function renderItemsTable(items) {
    itemsTbody.innerHTML = "";

    if (items.length === 0) {
      itemsTbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding:30px; color:var(--text-dim);">No structured items recognized. Check noise tab.</td></tr>`;
      return;
    }

    items.forEach((it, idx) => {
      const tr = document.createElement("tr");

      const catBadge = `<span class="cat-badge cat-${it.category}">${it.category}</span>`;
      const fillWidth = Math.min(Math.max(it.confidence, 5), 100);
      const confBar = `
        <div class="conf-bar-wrapper">
          <div class="conf-bar"><div class="conf-fill" style="width: ${fillWidth}%"></div></div>
          <span class="conf-text">${it.confidence}%</span>
        </div>
      `;

      const unitPriceStr = it.unit_price !== null ? `$${it.unit_price.toFixed(2)}` : "-";
      const totalPriceStr = it.total_price !== null ? `$${it.total_price.toFixed(2)}` : "-";

      tr.innerHTML = `
        <td class="cell-num">${idx + 1}</td>
        <td class="cell-qty">${it.quantity} <span style="font-size:0.68rem; color:var(--text-dim); font-weight:normal;">${it.unit}</span></td>
        <td class="cell-name">${escapeHtml(it.item_name)}</td>
        <td><span class="cell-bert">${escapeHtml(it.bert_query || it.item_name)}</span></td>
        <td>${catBadge}</td>
        <td>${confBar}</td>
        <td class="cell-price" style="text-align:right;">${unitPriceStr}</td>
        <td class="cell-price" style="text-align:right; color:var(--accent-green);">${totalPriceStr}</td>
      `;

      itemsTbody.appendChild(tr);
    });
  }

  function renderNoiseTable(filtered) {
    noiseTbody.innerHTML = "";

    if (filtered.length === 0) {
      noiseTbody.innerHTML = `<tr><td colspan="3" style="text-align:center; padding:20px; color:var(--text-dim);">No noise lines filtered.</td></tr>`;
      return;
    }

    filtered.forEach((f) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="cell-num">${f.line_no}</td>
        <td style="font-family:var(--font-mono); color:#cbd5e1;">${escapeHtml(f.text)}</td>
        <td><span class="noise-reason">${escapeHtml(f.reason)}</span></td>
      `;
      noiseTbody.appendChild(tr);
    });
  }

  // ── 8. Live Single-Item Sandbox ─────────────────────────────────────────────
  let sandboxTimer = null;
  sandboxInput.addEventListener("input", () => {
    clearTimeout(sandboxTimer);
    sandboxTimer = setTimeout(runSandboxClassification, 220);
  });

  async function runSandboxClassification() {
    const line = sandboxInput.value.trim();
    if (!line) {
      sandboxResult.innerHTML = `<span style="color:var(--text-dim);">Type a receipt item above to test live...</span>`;
      return;
    }

    try {
      const res = await fetch("/api/classify_single", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ line }),
      });
      const data = await res.json();

      if (data.is_item) {
        const s = data.structured;
        const p = data.prediction;
        sandboxResult.innerHTML = `
          <div class="sandbox-result-row">
            <span><strong>Item Name:</strong> ${escapeHtml(s.item_name)}</span>
            <span class="cat-badge cat-${p.category}">${p.category} (${p.confidence}%)</span>
          </div>
          <div class="sandbox-result-row" style="color:var(--text-muted); font-size:0.72rem;">
            <span>Qty: <strong>${s.quantity} ${s.unit}</strong> • Unit: <strong>$${s.unit_price}</strong> • Total: <strong style="color:var(--accent-green);">$${s.total_price}</strong></span>
            <span>MobileBERT: <strong>${p.latency_ms} ms</strong></span>
          </div>
        `;
      } else {
        sandboxResult.innerHTML = `
          <div class="sandbox-result-row">
            <span style="color:#fb7185;">Filtered as Noise:</span>
            <span class="noise-reason">${escapeHtml(data.reason)}</span>
          </div>
        `;
      }
    } catch (e) {
      sandboxResult.innerHTML = `<span style="color:#ef4444;">Inference error</span>`;
    }
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  async function loadModelInfo() {
    try {
      const res = await fetch("/api/info");
      const info = await res.json();
      if (info && info.num_classes) {
        const chipClasses = document.getElementById("chipClasses");
        if (chipClasses) chipClasses.innerHTML = `<span>${info.num_classes} Categories</span>`;

        const catCardTitle = document.getElementById("catCardTitle");
        if (catCardTitle) catCardTitle.textContent = `Model Class Taxonomy (${info.num_classes} Classes)`;

        const catPills = document.getElementById("catPills");
        if (catPills && info.classes) {
          catPills.innerHTML = info.classes
            .map((c) => `<span class="cat-badge cat-${c}">${c}</span>`)
            .join("");
        }
      }
    } catch (e) {
      console.warn("Could not load model info:", e);
    }
  }

  // Initialize
  loadModelInfo();
  loadPresets();
  runSandboxClassification();

  // Load first sample image as default
  if (sampleThumbBtns.length > 0) {
    sampleThumbBtns[0].click();
  }
});
