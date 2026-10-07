// InferOps Web Dashboard JavaScript Application

let logWebSocket = null;

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  loadFleetData();
  setupVramCalculator();
  setupChatPlayground();
  setupLiveConsole();
  setupAnalytics();
  connectWebSocket();

  document.getElementById("btnRefresh")?.addEventListener("click", () => {
    loadFleetData();
    loadAnalyticsData();
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
        logs: ["Live Console", "Real-time streaming process log monitor"],
        analytics: ["Cost & Analytics", "Token consumption accounting & commercial cost savings"],
      };
      if (titles[targetTab]) {
        document.getElementById("pageTitle").textContent = titles[targetTab][0];
        document.getElementById("pageSubtitle").textContent = titles[targetTab][1];
      }

      if (targetTab === "analytics") {
        loadAnalyticsData();
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
      updateLogModelsDropdown(models);
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
    container.innerHTML = `
      <div style="padding: 24px; text-align: center; color: var(--text-muted);">
        No model configs found in configs/models/*.yaml. Run <code>inferops init</code> or <code>inferops model create</code>.
      </div>`;
    return;
  }

  const rows = models
    .map((m) => {
      const isHealthy = m.status === "healthy";
      const isStarting = m.status === "starting";
      const badgeClass = isHealthy ? "badge-healthy" : isStarting ? "badge-starting" : "badge-stopped";

      return `
      <tr>
        <td><strong>${m.name}</strong><br><small style="color:var(--text-muted);">${m.model}</small></td>
        <td><span class="engine-tag">${m.engine.toUpperCase()}</span></td>
        <td>:${m.port}</td>
        <td>${m.gpus || "0"}</td>
        <td><span class="status-badge ${badgeClass}">${m.status.toUpperCase()}</span></td>
        <td>
          ${
            isHealthy || isStarting
              ? `<button class="btn btn-secondary btn-sm" onclick="stopModel('${m.name}')">Stop</button>`
              : `<button class="btn btn-primary btn-sm" onclick="startModel('${m.name}')">Start</button>`
          }
        </td>
      </tr>`;
    })
    .join("");

  container.innerHTML = `
    <table class="data-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Engine</th>
          <th>Port</th>
          <th>GPUs</th>
          <th>Status</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        ${rows}
      </tbody>
    </table>`;
}

function updatePlaygroundModelsDropdown(models) {
  const select = document.getElementById("playgroundModelSelect");
  if (!select) return;

  const currentVal = select.value;
  select.innerHTML = '<option value="">Select a model</option>';

  models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.name;
    opt.textContent = `${m.name} (${m.engine} - ${m.status})`;
    select.appendChild(opt);
  });

  if (currentVal) select.value = currentVal;
}

function updateLogModelsDropdown(models) {
  const select = document.getElementById("logModelSelect");
  if (!select) return;

  const currentVal = select.value;
  select.innerHTML = '<option value="">Select model to tail</option>';

  models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.name;
    opt.textContent = `${m.name} (${m.status})`;
    select.appendChild(opt);
  });

  if (currentVal) select.value = currentVal;
}

window.startModel = async function (name) {
  try {
    const res = await fetch(`/api/models/${name}/start`, { method: "POST" });
    if (res.ok) {
      setTimeout(loadFleetData, 1000);
    } else {
      const err = await res.json();
      alert("Failed to start model: " + (err.detail || "Unknown error"));
    }
  } catch (e) {
    alert("Network error: " + e.message);
  }
};

window.stopModel = async function (name) {
  try {
    const res = await fetch(`/api/models/${name}/stop`, { method: "POST" });
    if (res.ok) {
      setTimeout(loadFleetData, 1000);
    }
  } catch (e) {
    alert("Network error: " + e.message);
  }
};

function setupVramCalculator() {
  const btn = document.getElementById("btnCalculateVram");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    const model = document.getElementById("sizerModelInput").value.trim();
    const ctx = parseInt(document.getElementById("sizerContextSelect").value, 10);
    const dtype = document.getElementById("sizerDtypeSelect").value;
    const tp = parseInt(document.getElementById("sizerTpInput").value, 10);

    const quant = ["awq", "gptq", "fp8"].includes(dtype) ? dtype : null;
    const actualDtype = quant ? "auto" : dtype;

    btn.disabled = true;
    btn.textContent = "Calculating...";

    try {
      const res = await fetch("/api/vram/estimate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: model,
          context_length: ctx,
          dtype: actualDtype,
          quantization: quant,
          tensor_parallel_size: tp,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        document.getElementById("resWeights").textContent = `${data.weights_gb} GB`;
        document.getElementById("resKv").textContent = `${data.kv_cache_gb} GB`;
        document.getElementById("resOverhead").textContent = `${data.cuda_overhead_gb} GB`;
        document.getElementById("resPerGpu").textContent = `${data.per_gpu_gb} GB`;

        const banner = document.getElementById("vramVerdictBanner");
        banner.className = data.fits ? "verdict-banner pass" : "verdict-banner fail";
        banner.innerHTML = `<strong>${data.fits ? "[PASS] Model Fits Memory" : "[WARNING] Out of Memory Risk"}</strong><br>${data.suggestion}`;

        document.getElementById("vramResultBox").classList.remove("hidden");
      }
    } catch (e) {
      alert("Error: " + e.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Calculate Memory Footprint";
    }
  });
}

function setupChatPlayground() {
  const sendBtn = document.getElementById("btnSendMessage");
  const chatInput = document.getElementById("chatInput");
  const modelSelect = document.getElementById("playgroundModelSelect");
  const messagesContainer = document.getElementById("chatMessages");

  const send = async () => {
    const text = chatInput.value.trim();
    const model = modelSelect.value;
    if (!text) return;
    if (!model) {
      alert("Please select a running model first.");
      return;
    }

    appendChatMessage("user", text);
    chatInput.value = "";

    const assistantMsg = appendChatMessage("assistant", "...");
    const bubble = assistantMsg.querySelector(".msg-bubble");
    bubble.textContent = "";

    const startTime = performance.now();
    let firstTokenTime = null;
    let tokenCount = 0;

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: model,
          messages: [{ role: "user", content: text }],
          stream: true,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        bubble.textContent = "Error: " + (err.detail || res.statusText);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

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
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
              }
            } catch (e) {}
          }
        }

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

function setupLiveConsole() {
  const modelSelect = document.getElementById("logModelSelect");
  const terminal = document.getElementById("terminalOutput");
  const statusChip = document.getElementById("logConnectionStatus");
  const autoScrollChk = document.getElementById("chkAutoScroll");
  const clearBtn = document.getElementById("btnClearLogs");

  clearBtn?.addEventListener("click", () => {
    terminal.textContent = "";
  });

  modelSelect?.addEventListener("change", () => {
    const model = modelSelect.value;
    if (!model) {
      if (logWebSocket) {
        logWebSocket.close();
        logWebSocket = null;
      }
      statusChip.textContent = "Idle";
      statusChip.className = "status-chip";
      return;
    }

    if (logWebSocket) {
      logWebSocket.close();
      logWebSocket = null;
    }

    terminal.textContent = `[inferops] Connecting to live log stream for '${model}'...\n`;
    statusChip.textContent = "Connecting";
    statusChip.className = "status-chip";

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/logs/${model}`;

    logWebSocket = new WebSocket(wsUrl);

    logWebSocket.onopen = () => {
      statusChip.textContent = "Streaming";
      statusChip.className = "status-chip active";
    };

    logWebSocket.onmessage = (event) => {
      terminal.textContent += event.data;
      if (autoScrollChk.checked) {
        terminal.scrollTop = terminal.scrollHeight;
      }
    };

    logWebSocket.onclose = () => {
      statusChip.textContent = "Closed";
      statusChip.className = "status-chip";
    };

    logWebSocket.onerror = () => {
      statusChip.textContent = "Error";
      statusChip.className = "status-chip";
    };
  });
}

function setupAnalytics() {
  loadAnalyticsData();
}

async function loadAnalyticsData() {
  try {
    const res = await fetch("/api/analytics/usage");
    if (!res.ok) return;

    const data = await res.json();
    document.getElementById("statCostSaved").textContent = `$${data.total_savings_usd.toFixed(4)}`;
    document.getElementById("statTotalTokens").textContent = data.grand_total_tokens.toLocaleString();
    document.getElementById("statTotalRequests").textContent = data.total_requests.toLocaleString();
    document.getElementById("statAvgLatency").textContent = `${data.avg_latency_ms.toFixed(1)} ms`;

    const container = document.getElementById("analyticsTableContainer");
    if (!container) return;

    if (!data.by_model || data.by_model.length === 0) {
      container.innerHTML = `
        <div style="padding: 24px; text-align: center; color: var(--text-muted);">
          No inference requests recorded yet. Query models through the OpenAI gateway to accumulate token savings data.
        </div>`;
      return;
    }

    const rows = data.by_model
      .map(
        (m) => `
        <tr>
          <td><strong>${m.model}</strong></td>
          <td>${m.requests.toLocaleString()}</td>
          <td>${m.tokens.toLocaleString()}</td>
          <td class="text-success">$${m.savings.toFixed(4)}</td>
        </tr>`
      )
      .join("");

    container.innerHTML = `
      <table class="data-table">
        <thead>
          <tr>
            <th>Model Identifier</th>
            <th>Requests</th>
            <th>Total Tokens</th>
            <th>Commercial Cost Saved</th>
          </tr>
        </thead>
        <tbody>
          ${rows}
        </tbody>
      </table>`;
  } catch (e) {
    console.error("Failed to load analytics:", e);
  }
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
