// InferOps Web Dashboard JavaScript Application

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  loadFleetData();
  setupVramCalculator();
  setupChatPlayground();
  connectWebSocket();

  document.getElementById("btnRefresh")?.addEventListener("click", () => {
    loadFleetData();
  });
});

function setupTabs() {
  const navBtns = document.querySelectorAll(".nav-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  navBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      navBtns.forEach((b) => b.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPane = document.getElementById(`tab-${targetTab}`);
      if (targetPane) targetPane.classList.add("active");

      // Update titles
      const titles = {
        fleet: ["Inference Fleet", "Bare-metal multi-engine control plane for vLLM & SGLang"],
        vram: ["VRAM Sizer", "Analytical pre-flight memory calculator"],
        playground: ["Chat Playground", "Interactive test workbench with live token telemetry"],
      };
      if (titles[targetTab]) {
        document.getElementById("pageTitle").textContent = titles[targetTab][0];
        document.getElementById("pageSubtitle").textContent = titles[targetTab][1];
      }
    });
  });
}

async function loadFleetData() {
  try {
    const [gpusRes, modelsRes] = await Promise.all([
      fetch("/api/gpus"),
      fetch("/api/models"),
    ]);

    if (gpusRes.ok) {
      const gpus = await gpusRes.json();
      renderGpuCards(gpus);
    }
    if (modelsRes.ok) {
      const models = await modelsRes.json();
      renderModelsTable(models);
      updatePlaygroundModelsDropdown(models);
    }
  } catch (err) {
    console.error("Failed to load fleet data:", err);
  }
}

function renderGpuCards(gpus) {
  const container = document.getElementById("gpuCardsContainer");
  if (!container) return;

  if (!gpus || gpus.length === 0) {
    container.innerHTML = `
      <div class="gpu-card">
        <div class="gpu-header">
          <span class="gpu-name">No Discrete NVIDIA GPU Detected</span>
          <span class="gpu-badge">CPU / Fallback</span>
        </div>
        <p style="font-size: 13px; color: var(--text-muted);">
          InferOps is running in host or emulation mode. Ensure NVIDIA drivers and nvidia-smi are installed for hardware acceleration.
        </p>
      </div>`;
    return;
  }

  container.innerHTML = gpus
    .map((g) => {
      const pct = g.total_memory_gb > 0 ? Math.round((g.used_memory_gb / g.total_memory_gb) * 100) : 0;
      return `
      <div class="gpu-card">
        <div class="gpu-header">
          <span class="gpu-name">GPU ${g.index}: ${g.name}</span>
          <span class="gpu-badge">${g.temperature_c !== null ? g.temperature_c + "°C" : "N/A"}</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill" style="width: ${pct}%"></div>
        </div>
        <div class="gpu-stats-row">
          <span>VRAM: ${g.used_memory_gb} / ${g.total_memory_gb} GB (${pct}%)</span>
          <span>Load: ${g.utilization_gpu_pct}%</span>
        </div>
      </div>`;
    })
    .join("");
}

function renderModelsTable(models) {
  const container = document.getElementById("modelsTableContainer");
  if (!container) return;

  if (!models || models.length === 0) {
    container.innerHTML = `<div class="loading-state">No model definitions found in configs/models/. Run <code>inferops model create</code> to register one.</div>`;
    return;
  }

  container.innerHTML = models
    .map((m) => {
      const isHealthy = m.status === "HEALTHY";
      return `
      <div class="model-row-card">
        <div class="model-info-block">
          <h4>${m.name}</h4>
          <div class="model-tags">
            <span class="tag-badge">Engine: ${m.engine.toUpperCase()}</span>
            <span class="tag-badge">Port: ${m.port}</span>
            <span class="tag-badge">GPUs: ${m.gpus || "0"}</span>
            <span class="tag-badge">${m.model}</span>
          </div>
        </div>
        <div style="display: flex; align-items: center; gap: 16px;">
          <span class="status-badge status-${m.status}">${m.status}</span>
          <div class="model-actions">
            ${
              isHealthy
                ? `<button class="btn btn-secondary" onclick="stopModel('${m.name}')">Stop</button>`
                : `<button class="btn btn-primary" onclick="startModel('${m.name}')">Start</button>`
            }
          </div>
        </div>
      </div>`;
    })
    .join("");
}

async function startModel(name) {
  try {
    const res = await fetch(`/api/models/${name}/start`, { method: "POST" });
    if (res.ok) {
      loadFleetData();
    } else {
      const err = await res.json();
      alert("Failed to start model: " + (err.detail || res.statusText));
    }
  } catch (e) {
    alert("Error: " + e.message);
  }
}

async function stopModel(name) {
  try {
    const res = await fetch(`/api/models/${name}/stop`, { method: "POST" });
    if (res.ok) {
      loadFleetData();
    }
  } catch (e) {
    alert("Error: " + e.message);
  }
}

function setupVramCalculator() {
  const btn = document.getElementById("btnCalculateVram");
  btn?.addEventListener("click", async () => {
    const model = document.getElementById("sizerModelInput").value.trim();
    const context = parseInt(document.getElementById("sizerContextSelect").value, 10);
    const dtype = document.getElementById("sizerDtypeSelect").value;
    const tp = parseInt(document.getElementById("sizerTpInput").value, 10);

    const quant = dtype === "awq" || dtype === "gptq" ? dtype : null;

    try {
      const res = await fetch("/api/vram/estimate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: jsonBody({
          model: model,
          context_length: context,
          dtype: dtype,
          quantization: quant,
          tensor_parallel_size: tp,
        }),
      });

      if (!res.ok) throw new Error("Calculation failed");
      const data = await res.json();

      document.getElementById("resWeights").textContent = `${data.weights_gb} GB`;
      document.getElementById("resKv").textContent = `${data.kv_cache_gb} GB`;
      document.getElementById("resOverhead").textContent = `${data.cuda_overhead_gb} GB`;
      document.getElementById("resPerGpu").textContent = `${data.per_gpu_gb} GB`;

      const banner = document.getElementById("vramVerdictBanner");
      banner.className = `verdict-banner ${data.fits ? "fits" : "overflow"}`;
      banner.textContent = data.suggestion || (data.fits ? "Model will fit in target GPU memory." : "Insufficient GPU memory.");

      document.getElementById("vramResultBox").classList.remove("hidden");
    } catch (e) {
      alert("Failed to estimate VRAM: " + e.message);
    }
  });
}

function updatePlaygroundModelsDropdown(models) {
  const sel = document.getElementById("playgroundModelSelect");
  if (!sel) return;
  const currentVal = sel.value;
  sel.innerHTML = '<option value="">Select a running model</option>';

  const running = models.filter((m) => m.status === "HEALTHY");
  running.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.name;
    opt.textContent = `${m.name} (${m.engine.toUpperCase()} - Port ${m.port})`;
    sel.appendChild(opt);
  });

  if (currentVal && running.some((m) => m.name === currentVal)) {
    sel.value = currentVal;
  }
}

function setupChatPlayground() {
  const sendBtn = document.getElementById("btnSendMessage");
  const chatInput = document.getElementById("chatInput");

  const send = async () => {
    const prompt = chatInput.value.trim();
    const model = document.getElementById("playgroundModelSelect").value;
    if (!prompt) return;
    if (!model) {
      alert("Please select a running model from the dropdown first.");
      return;
    }

    appendChatMessage("user", prompt);
    chatInput.value = "";

    const assistantMsgNode = appendChatMessage("assistant", "");
    const bubble = assistantMsgNode.querySelector(".msg-bubble");
    bubble.textContent = "...";

    const startTime = performance.now();
    let firstTokenTime = null;
    let tokenCount = 0;

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: jsonBody({
          model: model,
          messages: [{ role: "user", content: prompt }],
          stream: true,
        }),
      });

      if (!resp.ok) {
        bubble.textContent = "Error: " + resp.statusText;
        return;
      }

      bubble.textContent = "";
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.slice(6).trim();
            if (dataStr === "[DONE]") continue;
            try {
              const parsed = JSON.parse(dataStr);
              const delta = parsed.choices?.[0]?.delta?.content || "";
              if (delta) {
                if (firstTokenTime === null) {
                  firstTokenTime = performance.now() - startTime;
                }
                tokenCount++;
                bubble.textContent += delta;
              }
            } catch (err) {}
          }
        }

        // Update live stats
        const elapsedSec = (performance.now() - startTime) / 1000;
        const tokPerSec = elapsedSec > 0 ? (tokenCount / elapsedSec).toFixed(1) : "--";
        const ttftStr = firstTokenTime !== null ? `${Math.round(firstTokenTime)}ms` : "--";
        document.getElementById("chatStats").textContent = `Tokens: ${tokenCount} | TTFT: ${ttftStr} | Tok/s: ${tokPerSec}`;
      }
    } catch (e) {
      bubble.textContent = "Connection error: " + e.message;
    }
  };

  sendBtn?.addEventListener("click", send);
  chatInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
}

function appendChatMessage(role, text) {
  const container = document.getElementById("chatMessages");
  const msg = document.createElement("div");
  msg.className = `message ${role}`;
  msg.innerHTML = `<div class="msg-bubble">${escapeHtml(text)}</div>`;
  container.appendChild(msg);
  container.scrollTop = container.scrollHeight;
  return msg;
}

function escapeHtml(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function jsonBody(obj) {
  return JSON.stringify(obj);
}

function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

  const ws = new WebSocket(wsUrl);
  ws.onopen = () => {
    document.getElementById("daemonStatusText").textContent = "Connected (Live)";
  };
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.gpus) renderGpuCards(data.gpus);
    } catch (e) {}
  };
  ws.onclose = () => {
    document.getElementById("daemonStatusText").textContent = "Disconnected (Retrying)";
    setTimeout(connectWebSocket, 3000);
  };
}
