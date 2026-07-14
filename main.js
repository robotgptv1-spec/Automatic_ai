(() => {
  "use strict";

  const state = {
    sessionId: null,
    columns: [],
    targetColumn: null,
    featureColumns: [],
    taskType: "auto",
    problemMode: null,
    trainMeta: null, // {feature_columns, categorical_features, categories, class_names}
  };

  // ---------- helpers ----------
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  function setStepActive(n) {
    $$(".step").forEach((el) => {
      const step = Number(el.dataset.step);
      el.classList.toggle("is-active", step === n);
      el.classList.toggle("is-done", step < n);
    });
    [1, 2, 3, 4].forEach((n2) => {
      const panel = $(`#panel-${n2}`);
      if (panel) panel.hidden = n2 !== n;
    });
  }

  function showError(el, message) {
    el.hidden = false;
    el.textContent = message;
  }
  function hideError(el) {
    el.hidden = true;
    el.textContent = "";
  }

  async function api(path, opts) {
    const res = await fetch(path, opts);
    let data;
    try {
      data = await res.json();
    } catch (e) {
      throw new Error("Server returned an unexpected response.");
    }
    if (!res.ok) throw new Error(data.error || "Request failed.");
    return data;
  }

  // ---------- STEP 1: upload ----------
  const dropzone = $("#dropzone");
  const fileInput = $("#fileInput");
  const uploadError = $("#uploadError");

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("is-dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("is-dragover");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  });
  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) handleUpload(file);
  });

  async function handleUpload(file) {
    hideError(uploadError);
    const fd = new FormData();
    fd.append("file", file);
    try {
      const data = await api("/api/upload", { method: "POST", body: fd });
      state.sessionId = data.session_id;
      state.columns = data.columns;
      renderDatasetSummary(data);
      buildConfigureStep(data.columns);
    } catch (e) {
      showError(uploadError, e.message);
    }
  }

  function renderDatasetSummary(data) {
    $("#sumRows").textContent = data.n_rows.toLocaleString();
    $("#sumCols").textContent = data.n_cols;
    $("#sumFile").textContent = data.filename;

    const table = $("#previewTable");
    const cols = data.columns.map((c) => c.name);
    const thead = `<thead><tr>${cols.map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr></thead>`;
    const tbody = `<tbody>${data.preview
      .map(
        (row) =>
          `<tr>${cols.map((c) => `<td>${escapeHtml(String(row[c] ?? ""))}</td>`).join("")}</tr>`
      )
      .join("")}</tbody>`;
    table.innerHTML = thead + tbody;

    $("#datasetSummary").hidden = false;
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));
  }

  $("#toConfigureBtn").addEventListener("click", () => setStepActive(2));

  // ---------- STEP 2: configure ----------
  const targetSelect = $("#targetSelect");
  const featureChips = $("#featureChips");
  const configError = $("#configError");

  function buildConfigureStep(columns) {
    targetSelect.innerHTML = columns
      .map((c) => `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)} (${c.dtype})</option>`)
      .join("");
    // default target = last column, common convention
    targetSelect.value = columns[columns.length - 1].name;
    renderFeatureChips();
  }

  function renderFeatureChips() {
    const target = targetSelect.value;
    featureChips.innerHTML = state.columns
      .map((c) => {
        const disabled = c.name === target;
        const selected = !disabled;
        return `<button type="button" class="feature-chip ${selected ? "is-selected" : ""} ${
          disabled ? "is-disabled" : ""
        }" data-col="${escapeHtml(c.name)}" ${disabled ? "disabled" : ""}>${escapeHtml(c.name)}</button>`;
      })
      .join("");
  }

  targetSelect.addEventListener("change", renderFeatureChips);

  featureChips.addEventListener("click", (e) => {
    const btn = e.target.closest(".feature-chip");
    if (!btn || btn.disabled) return;
    btn.classList.toggle("is-selected");
  });

  $("#taskTypeSeg").addEventListener("click", (e) => {
    const btn = e.target.closest(".seg-btn");
    if (!btn) return;
    $$("#taskTypeSeg .seg-btn").forEach((b) => b.classList.remove("is-active"));
    btn.classList.add("is-active");
    state.taskType = btn.dataset.value;
  });

  // hyperparam sliders
  const epochsRange = $("#epochsRange"), epochsVal = $("#epochsVal");
  const lrRange = $("#lrRange"), lrVal = $("#lrVal");
  const batchRange = $("#batchRange"), batchVal = $("#batchVal");
  const testRange = $("#testRange"), testVal = $("#testVal");

  epochsRange.addEventListener("input", () => (epochsVal.textContent = epochsRange.value));
  batchRange.addEventListener("input", () => (batchVal.textContent = batchRange.value));
  testRange.addEventListener("input", () => (testVal.textContent = `${testRange.value}%`));
  lrRange.addEventListener("input", () => (lrVal.textContent = (lrRange.value / 10000).toFixed(4)));
  lrVal.textContent = (lrRange.value / 10000).toFixed(4);

  $("#toTrainBtn").addEventListener("click", async () => {
    hideError(configError);
    const target = targetSelect.value;
    const features = $$("#featureChips .feature-chip.is-selected").map((b) => b.dataset.col);
    if (features.length === 0) {
      showError(configError, "Select at least one feature column.");
      return;
    }
    state.targetColumn = target;
    state.featureColumns = features;

    try {
      await api("/api/configure", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: state.sessionId,
          target_column: target,
          feature_columns: features,
          task_type: state.taskType,
        }),
      });
      setStepActive(3);
      runTraining();
    } catch (e) {
      showError(configError, e.message);
    }
  });

  // ---------- STEP 3: train ----------
  const consoleBody = $("#consoleBody");
  let chartCtx = null;

  function consoleLine(html) {
    const line = document.createElement("div");
    line.className = "console-line";
    line.innerHTML = html;
    consoleBody.appendChild(line);
    consoleBody.scrollTop = consoleBody.scrollHeight;
  }

  async function runTraining() {
    consoleBody.innerHTML = "";
    $("#toPredictBtn").hidden = true;
    $("#finalTrainLoss").textContent = "\u2014";
    $("#finalTestLoss").textContent = "\u2014";
    $("#finalMetric").textContent = "\u2014";

    consoleLine(`<span class="console-line-meta">$</span> autoai train --target ${escapeHtml(state.targetColumn)} --features ${state.featureColumns.length}`);
    consoleLine(`<span class="k">task_type</span> <span class="v">${escapeHtml(state.taskType)}</span> <span class="k">epochs</span> <span class="v">${epochsRange.value}</span> <span class="k">lr</span> <span class="v">${(lrRange.value/10000).toFixed(4)}</span>`);
    consoleLine(`<span class="console-line-meta">building feature matrix, fitting scaler...</span>`);

    let data;
    try {
      data = await api("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: state.sessionId,
          epochs: Number(epochsRange.value),
          lr: Number(lrRange.value) / 10000,
          batch_size: Number(batchRange.value),
          test_size: Number(testRange.value) / 100,
        }),
      });
    } catch (e) {
      consoleLine(`<span style="color:var(--red)">error: ${escapeHtml(e.message)}</span>`);
      return;
    }

    state.problemMode = data.problem_mode;
    state.trainMeta = data;

    const metricLabel = data.problem_mode === "regression" ? "R\u00B2" : "accuracy";
    $("#metricLabel").textContent = metricLabel === "R\u00B2" ? "R\u00B2 score" : "Accuracy";

    initChart(data.log);

    // Replay the log with a light streaming animation for the "live console" feel
    for (let i = 0; i < data.log.length; i++) {
      const row = data.log[i];
      await sleep(data.log.length > 60 ? 8 : Math.max(6, 260 - i * 2));
      consoleLine(
        `<span class="k">epoch</span> <span class="v">${String(row.epoch).padStart(3, " ")}</span>` +
        `  <span class="k">train_loss</span> <span class="v">${row.train_loss.toFixed(4)}</span>` +
        `  <span class="k">test_loss</span> <span class="v">${row.test_loss.toFixed(4)}</span>` +
        `  <span class="k">${escapeHtml(row.metric_name)}</span> <span class="v">${row.metric_value.toFixed(4)}</span>`
      );
      updateChart(i);
    }
    consoleLine(`<span class="console-line-meta">done.</span><span class="console-cursor"></span>`);

    const final = data.final;
    $("#finalTrainLoss").textContent = final.train_loss.toFixed(4);
    $("#finalTestLoss").textContent = final.test_loss.toFixed(4);
    $("#finalMetric").textContent = final.metric_value.toFixed(4);

    buildPredictForm(data);
    $("#toPredictBtn").hidden = false;
    $("#downloadModelBtn").href = `/api/download_model/${state.sessionId}`;
  }

  function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  function initChart(log) {
    const canvas = $("#lossChart");
    chartCtx = { ctx: canvas.getContext("2d"), canvas, log };
    drawChart(0);
  }
  function updateChart(uptoIndex) {
    drawChart(uptoIndex);
  }
  function drawChart(uptoIndex) {
    if (!chartCtx) return;
    const { ctx, canvas, log } = chartCtx;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const rows = log.slice(0, uptoIndex + 1);
    if (rows.length < 2) return;

    const pad = 24;
    const allVals = rows.flatMap((r) => [r.train_loss, r.test_loss]);
    const maxV = Math.max(...allVals, 0.0001);
    const minV = Math.min(...allVals, 0);

    const xFor = (i) => pad + (i / (log.length - 1)) * (W - pad * 2);
    const yFor = (v) => H - pad - ((v - minV) / (maxV - minV || 1)) * (H - pad * 2);

    // gridlines
    ctx.strokeStyle = "#1A222D";
    ctx.lineWidth = 1;
    for (let g = 0; g <= 3; g++) {
      const y = pad + (g / 3) * (H - pad * 2);
      ctx.beginPath();
      ctx.moveTo(pad, y);
      ctx.lineTo(W - pad, y);
      ctx.stroke();
    }

    const drawLine = (key, color) => {
      ctx.beginPath();
      rows.forEach((r, i) => {
        const x = xFor(i), y = yFor(r[key]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
    };
    drawLine("train_loss", "#FF6B35");
    drawLine("test_loss", "#22D3EE");

    ctx.font = "10px JetBrains Mono";
    ctx.fillStyle = "#FF6B35";
    ctx.fillText("train", pad, 14);
    ctx.fillStyle = "#22D3EE";
    ctx.fillText("test", pad + 40, 14);
  }

  $("#toPredictBtn").addEventListener("click", () => setStepActive(4));

  // ---------- STEP 4: predict ----------
  const predictForm = $("#predictForm");
  const predictResult = $("#predictResult");

  function buildPredictForm(data) {
    const categories = data.categories || {};
    predictForm.innerHTML = data.feature_columns
      .map((col) => {
        if (data.categorical_features.includes(col)) {
          const opts = (categories[col] || [])
            .map((v) => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`)
            .join("");
          return `<div class="pf-field"><label>${escapeHtml(col)}</label><select name="${escapeHtml(col)}">${opts}</select></div>`;
        }
        return `<div class="pf-field"><label>${escapeHtml(col)}</label><input type="number" step="any" name="${escapeHtml(col)}" placeholder="0" required></div>`;
      })
      .join("") + `<button class="primary-btn pf-submit" type="submit">Run prediction</button>`;
  }

  predictForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(predictForm);
    const features = {};
    fd.forEach((v, k) => (features[k] = v));

    try {
      const result = await api("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.sessionId, features }),
      });
      renderPredictResult(result);
    } catch (e2) {
      predictResult.innerHTML = `<div class="error-box">${escapeHtml(e2.message)}</div>`;
    }
  });

  function renderPredictResult(result) {
    if (result.probabilities) {
      const rows = Object.entries(result.probabilities)
        .sort((a, b) => b[1] - a[1])
        .map(
          ([label, p]) => `
          <div class="prob-row"><span>${escapeHtml(label)}</span><span>${(p * 100).toFixed(1)}%</span></div>
          <div class="prob-bar-track"><div class="prob-bar-fill" style="width:${p * 100}%"></div></div>
        `
        )
        .join("");
      predictResult.innerHTML = `
        <div class="result-label">Predicted class</div>
        <div class="result-value">${escapeHtml(result.prediction)}</div>
        ${rows}
      `;
    } else {
      predictResult.innerHTML = `
        <div class="result-label">Predicted value</div>
        <div class="result-value">${result.prediction}</div>
      `;
    }
  }

  // ---------- reset ----------
  $("#resetBtn").addEventListener("click", async () => {
    if (state.sessionId) {
      try {
        await api("/api/reset", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: state.sessionId }),
        });
      } catch (e) {
        /* ignore */
      }
    }
    window.location.reload();
  });
})();
