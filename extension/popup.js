const API_URL = "https://comment-sentiment-analysis-project.onrender.com";
let allComments = [];
let activeFilter = "ALL";
let visibleCount = 25;

document.addEventListener("DOMContentLoaded", () => {
  const output = document.getElementById("output");

  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0]?.url || "";
    const match = url.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/)([\w-]{11})/);

    if (!match) {
      output.innerHTML = '<div class="box error">Open a YouTube video page (youtube.com/watch?v=...) and click the extension icon again.</div>';
      return;
    }

    const videoId = match[1];
    output.innerHTML = `<div class="box status">Video: <b>${videoId}</b><br>Fetching up to 500 comments and analyzing... this can take ~15s.</div>`;

    try {
      const resp = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ videoId }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || `Backend error (${resp.status})`);
      render(data);
    } catch (error) {
      output.innerHTML = `<div class="box error">${error.message}<br><br>Is the backend running? (python fastapi-backend/main.py)</div>`;
    }
  });

  function render(data) {
    allComments = data.comments;
    const pct = data.percentages;
    const m = data.metrics;

    output.innerHTML = `
      <div class="box">
        <div class="section-title">SENTIMENT ANALYSIS</div>
        <div class="total">Total Comments: ${data.totalComments}</div>
        ${bar("positive", "POSITIVE", pct.POSITIVE)}
        ${bar("neutral", "NEUTRAL", pct.NEUTRAL)}
        ${bar("negative", "NEGATIVE", pct.NEGATIVE)}
        <div class="metrics-grid">
          <div class="metric"><span>Avg Comment Length</span><b>${m.avgCommentLength} words</b></div>
          <div class="metric"><span>Unique Authors</span><b>${m.uniqueAuthors}</b></div>
          <div class="metric"><span>Avg Sentiment Score</span><b>${m.avgSentimentScore}/10</b></div>
          <div class="metric"><span>Spam Flagged</span><b>${m.spamCount}</b></div>
        </div>
      </div>

      <div class="box">
        <div class="section-title">SUMMARY OF COMMENTS${data.summarySource === "stats" ? "" : " · " + escapeHtml(data.summarySource)}</div>
        <p class="summary-text">${escapeHtml(data.summary)}</p>
      </div>

      <div class="box">
        <div class="section-title">KEY THEMES (WORD CLOUD)</div>
        <div class="wordcloud">${wordCloud(data.themes)}</div>
      </div>

      <div class="box">
        <div class="section-title">TREND OVER TIME</div>
        <div id="trend-plot"></div>
        <div id="trend-fallback"></div>
      </div>

      <div class="box">
        <div class="section-title">DETAILED INSIGHTS</div>
        <div class="filters">
          ${["ALL", "POSITIVE", "NEUTRAL", "NEGATIVE", "SPAM"].map(
            (f) => `<button class="filter-btn ${f === activeFilter ? "active" : ""}" data-filter="${f}">${f}</button>`
          ).join("")}
        </div>
        <div id="comment-list"></div>
        <button id="load-more" class="load-more"></button>
      </div>

      <button id="export-csv" class="export-btn">EXPORT CSV</button>
    `;

    renderComments();
    drawTrend(data.trend);

    output.querySelectorAll(".filter-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        activeFilter = btn.dataset.filter;
        visibleCount = 25;
        output.querySelectorAll(".filter-btn").forEach((b) => b.classList.toggle("active", b === btn));
        renderComments();
      });
    });

    document.getElementById("load-more").addEventListener("click", () => {
      visibleCount += 25;
      renderComments();
    });

    document.getElementById("export-csv").addEventListener("click", () => exportCsv(data));
  }

  function filteredComments() {
    if (activeFilter === "ALL") return allComments;
    if (activeFilter === "SPAM") return allComments.filter((c) => c.spam === 1);
    return allComments.filter((c) => c.label === activeFilter);
  }

  function renderComments() {
    const list = document.getElementById("comment-list");
    const filtered = filteredComments();
    const shown = filtered.slice(0, visibleCount);
    list.innerHTML = shown
      .map(
        (item, i) => `
      <div class="comment-item">
        <div class="text"><b>${i + 1}.</b> ${escapeHtml(item.comment)}</div>
        <span class="tag ${item.label}">Sentiment: ${item.sentiment} (${item.label})</span>
        ${item.spam ? '<span class="tag SPAM">SPAM?</span>' : ""}
      </div>`
      )
      .join("");
    const more = document.getElementById("load-more");
    more.textContent = filtered.length > visibleCount ? `SHOW MORE (${filtered.length - visibleCount} left)` : "";
  }

  function bar(cls, label, value) {
    return `
      <div class="bar-row ${cls}">
        <div class="bar-label"><span>${label}</span><span>${value}%</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:${value}%"></div></div>
      </div>`;
  }

  function wordCloud(themes) {
    if (!themes.length) return "<i>No themes found.</i>";
    const max = themes[0].count;
    return themes
      .map((t) => {
        const size = 11 + Math.round((t.count / max) * 15);
        return `<span class="cloud-word" style="font-size:${size}px" title="${t.count} mentions">${escapeHtml(t.word)}</span>`;
      })
      .join(" ");
  }

  function drawTrend(trend) {
    const el = document.getElementById("trend-plot");
    if (trend.length < 2) {
      el.remove();
      document.getElementById("trend-fallback").innerHTML =
        "<i>Not enough dated comments to build a trend.</i>";
      return;
    }
    const months = trend.map((t) => t.month);
    const trace = (name, color) => ({
      x: months,
      y: trend.map((t) => t[name]),
      name,
      mode: "lines+markers",
      line: { color, width: 3 },
      marker: { color, size: 7 },
      hovertemplate: "%{y}%<br>%{x}<br>" + name + "<extra></extra>",
    });
    Plotly.newPlot(
      el,
      [trace("NEGATIVE", "#FF4D4D"), trace("NEUTRAL", "#8A8A8A"), trace("POSITIVE", "#00D97C")],
      {
        title: { text: "Monthly Sentiment Percentage Over Time", font: { size: 13 } },
        margin: { l: 42, r: 8, t: 34, b: 70 },
        paper_bgcolor: "#fff",
        plot_bgcolor: "#fff",
        font: { family: "Arial, sans-serif", size: 10 },
        xaxis: { tickangle: -45, gridcolor: "#E8E8E8" },
        yaxis: {
          range: [0, 105],
          ticksuffix: "%",
          title: { text: "Percentage of Comments", font: { size: 10 } },
          gridcolor: "#E8E8E8",
        },
        legend: { orientation: "h", y: -0.45, x: 0.05, font: { size: 10 } },
      },
      { displayModeBar: false, responsive: true }
    );
  }

  function exportCsv(data) {
    const header = "comment,author,timestamp,sentiment,label,spam";
    const rows = data.comments.map((c) =>
      [
        `"${(c.comment || "").replace(/"/g, '""')}"`,
        `"${(c.author || "").replace(/"/g, '""')}"`,
        c.timestamp,
        c.sentiment,
        c.label,
        c.spam,
      ].join(",")
    );
    const csv = [header, ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `sentiment_${data.videoId}.csv`;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }
});
