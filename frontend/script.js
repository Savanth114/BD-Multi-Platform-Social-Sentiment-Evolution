const API_BASE = "http://127.0.0.1:5000";

let platformSentData = [];
let platformChartInstance = null;

async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error("HTTP error " + response.status);
  }
  return await response.json();
}

// ===== METRICS =====
async function loadMetrics() {
  try {
    const data = await fetchJSON(`${API_BASE}/metrics`);

    const vals = Object.values(data);
    const acc = vals[0];
    const prec = vals[1];
    const rec = vals[2];
    const f1 = vals[3];

    document.getElementById("accValue").textContent = acc
      ? acc.toFixed(3)
      : "--";
    document.getElementById("precValue").textContent = prec
      ? prec.toFixed(3)
      : "--";
    document.getElementById("recValue").textContent = rec
      ? rec.toFixed(3)
      : "--";
    document.getElementById("f1Value").textContent = f1
      ? f1.toFixed(3)
      : "--";

    const labels = Object.keys(data);
    const values = vals;

    const ctx = document.getElementById("metricsChart").getContext("2d");
    new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Score",
            data: values
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          tooltip: { enabled: true },
          legend: { display: false }
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 1.0
          }
        }
      }
    });
  } catch (err) {
    console.error("Error loading metrics:", err);
  }
}

// ===== GLOBAL LABEL DISTRIBUTION =====
async function loadLabelDistribution() {
  try {
    const data = await fetchJSON(`${API_BASE}/label-distribution`);

    const labels = data.map((d) => d.sentiment_category);
    const counts = data.map((d) => d.count);

    const ctx = document.getElementById("labelChart").getContext("2d");
    new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Number of posts",
            data: counts
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          tooltip: { enabled: true },
          legend: { display: false }
        },
        scales: {
          y: { beginAtZero: true }
        }
      }
    });
  } catch (err) {
    console.error("Error loading label distribution:", err);
  }
}

// ===== PLATFORM SENTIMENT (INTERACTIVE) =====
async function loadPlatformSentiment() {
  try {
    const data = await fetchJSON(`${API_BASE}/platform-sentiment`);
    platformSentData = data;

    const platforms = [...new Set(data.map((row) => row.platform))].sort();

    const select = document.getElementById("platformSelect");
    select.innerHTML = "";

    platforms.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p;
      opt.textContent = p;
      select.appendChild(opt);
    });

    select.addEventListener("change", () => {
      updatePlatformView(select.value);
    });

    if (platforms.length > 0) {
      select.value = platforms[0];
      updatePlatformView(platforms[0]);
    }
  } catch (err) {
    console.error("Error loading platform sentiment:", err);
  }
}

function updatePlatformView(selectedPlatform) {
  const rows = platformSentData.filter(
    (row) => row.platform === selectedPlatform
  );

  const tbody = document.getElementById("platformBody");
  tbody.innerHTML = "";

  rows.forEach((row) => {
    const tr = document.createElement("tr");

    const tdPlatform = document.createElement("td");
    tdPlatform.textContent = row.platform;

    const tdSent = document.createElement("td");
    tdSent.textContent = row.sentiment_category;

    const tdCount = document.createElement("td");
    tdCount.textContent = row.count;

    tr.appendChild(tdPlatform);
    tr.appendChild(tdSent);
    tr.appendChild(tdCount);

    tbody.appendChild(tr);
  });

  const sentimentMap = {};
  rows.forEach((row) => {
    const s = row.sentiment_category;
    const c = Number(row.count) || 0;
    sentimentMap[s] = (sentimentMap[s] || 0) + c;
  });

  const labels = Object.keys(sentimentMap);
  const counts = Object.values(sentimentMap);

  const ctx = document.getElementById("platformChart").getContext("2d");
  const title = document.getElementById("platformChartTitle");
  const summary = document.getElementById("platformSummary");

  title.textContent = `Sentiment counts for ${selectedPlatform}`;
  summary.textContent = `Total posts: ${counts.reduce(
    (a, b) => a + b,
    0
  )} · Negative / Neutral / Positive distribution for ${selectedPlatform}.`;

  if (platformChartInstance) {
    platformChartInstance.data.labels = labels;
    platformChartInstance.data.datasets[0].data = counts;
    platformChartInstance.data.datasets[0].label =
      `Sentiment counts for ${selectedPlatform}`;
    platformChartInstance.update();
  } else {
    platformChartInstance = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: `Sentiment counts for ${selectedPlatform}`,
            data: counts
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          tooltip: { enabled: true },
          legend: { display: false }
        },
        scales: {
          y: { beginAtZero: true }
        }
      }
    });
  }
}

// ===== INITIAL LOAD =====
loadMetrics();
loadLabelDistribution();
loadPlatformSentiment();
