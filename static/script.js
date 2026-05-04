const newsText = document.getElementById("newsText");
const predictBtn = document.getElementById("predictBtn");
const refreshHistoryBtn = document.getElementById("refreshHistoryBtn");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const downloadReportBtn = document.getElementById("downloadReportBtn");
const resultBox = document.getElementById("resultBox");
const predictionText = document.getElementById("predictionText");
const finalMessage = document.getElementById("finalMessage");
const confidenceText = document.getElementById("confidenceText");
const confidenceFill = document.getElementById("confidenceFill");
const validationQuery = document.getElementById("validationQuery");
const sourceList = document.getElementById("sourceList");
const historyList = document.getElementById("historyList");
const statusMessage = document.getElementById("statusMessage");
const totalCount = document.getElementById("totalCount");
const realCount = document.getElementById("realCount");
const fakeCount = document.getElementById("fakeCount");
let latestReportPayload = null;

function redirectToLogin() {
    window.location.href = "/login";
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.innerText = text;
    return div.innerHTML;
}

function renderHistory(historyItems) {
    if (!historyItems || historyItems.length === 0) {
        historyList.innerHTML = '<p class="empty-state">No predictions yet.</p>';
        return;
    }

    historyList.innerHTML = historyItems
        .map((item) => {
            const badgeClass = item.prediction === "REAL" ? "real" : "fake";
            const shortText =
                item.news_text.length > 180
                    ? `${item.news_text.slice(0, 180)}...`
                    : item.news_text;

            return `
                <div class="history-item">
                    <div class="history-meta">
                        <span class="badge ${badgeClass}">${item.prediction}</span>
                        <span>Confidence: ${item.confidence}%</span>
                        <span>${item.created_at}</span>
                    </div>
                    <p class="history-text">${escapeHtml(shortText)}</p>
                </div>
            `;
        })
        .join("");
}

function renderStats(stats) {
    totalCount.textContent = stats?.total_predictions ?? 0;
    realCount.textContent = stats?.real_count ?? 0;
    fakeCount.textContent = stats?.fake_count ?? 0;
}

function renderSources(sources) {
    if (!sources || sources.length === 0) {
        sourceList.innerHTML = '<p class="empty-state">No supporting articles were found.</p>';
        return;
    }

    sourceList.innerHTML = sources
        .map((source) => {
            return `
                <a class="source-item" href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer">
                    <div class="source-title">${escapeHtml(source.title)}</div>
                    <div class="source-meta">
                        <span>${escapeHtml(source.source_name)}</span>
                        <span>Support: ${source.support_score}</span>
                    </div>
                </a>
            `;
        })
        .join("");
}

function setStatus(message, type = "") {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${type}`.trim();
}

async function loadHistory() {
    try {
        setStatus("Loading history...");
        const response = await fetch("/history?limit=20");
        if (response.status === 401) {
            redirectToLogin();
            return;
        }
        const payload = await response.json();
        renderHistory(payload.items || []);
        renderStats(payload.stats || {});
        setStatus("History updated.", "success");
    } catch (error) {
        historyList.innerHTML =
            '<p class="empty-state">Unable to load history right now.</p>';
        setStatus("Could not load history.", "error");
    }
}

async function predictNews() {
    const text = newsText.value.trim();

    if (!text) {
        alert("Please enter some news text.");
        return;
    }

    predictBtn.disabled = true;
    predictBtn.textContent = "Verifying...";
    setStatus("Running AI analysis and checking the web...");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ news_text: text }),
        });
        if (response.status === 401) {
            redirectToLogin();
            return;
        }

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Verification failed.");
        }

        const finalVerdict = result.verification?.final_verdict || "UNCERTAIN";
        const finalConfidence = result.verification?.final_confidence ?? 0;

        predictionText.textContent = finalVerdict;
        finalMessage.textContent =
            result.verification?.final_message || "Verification completed.";
        confidenceText.textContent = `${finalConfidence}%`;
        confidenceFill.style.width = `${Math.max(0, Math.min(100, finalConfidence))}%`;
        validationQuery.textContent = result.verification?.query || "Not available";
        resultBox.classList.remove("hidden", "real", "fake");

        const verdictClass = finalVerdict.includes("REAL")
            ? "real"
            : finalVerdict.includes("FAKE")
              ? "fake"
              : "uncertain";
        resultBox.classList.add(verdictClass);
        renderSources(result.verification?.sources || []);
        latestReportPayload = {
            news_text: text,
            verification: result.verification,
        };
        downloadReportBtn.disabled = false;

        renderHistory(result.history);
        renderStats(result.stats || {});
        setStatus("News verification completed.", "success");
    } catch (error) {
        setStatus(error.message, "error");
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = "Verify News";
    }
}

async function downloadReport() {
    if (!latestReportPayload) {
        setStatus("Please verify news first before downloading a report.", "error");
        return;
    }

    downloadReportBtn.disabled = true;
    downloadReportBtn.textContent = "Preparing PDF...";
    setStatus("Generating PDF report...");

    try {
        const response = await fetch("/report", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(latestReportPayload),
        });
        if (response.status === 401) {
            redirectToLogin();
            return;
        }

        if (!response.ok) {
            const errorPayload = await response.json();
            throw new Error(errorPayload.error || "Could not generate PDF report.");
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = "news_verification_report.pdf";
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        window.URL.revokeObjectURL(url);
        setStatus("PDF report downloaded.", "success");
    } catch (error) {
        setStatus(error.message, "error");
    } finally {
        downloadReportBtn.disabled = false;
        downloadReportBtn.textContent = "Download Report";
    }
}

async function clearHistory() {
    const confirmed = window.confirm("Clear all saved prediction history?");

    if (!confirmed) {
        return;
    }

    clearHistoryBtn.disabled = true;
    clearHistoryBtn.textContent = "Clearing...";
    setStatus("Clearing saved history...");

    try {
        const response = await fetch("/history/clear", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
        });
        if (response.status === 401) {
            redirectToLogin();
            return;
        }

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Could not clear history.");
        }

        renderHistory(result.items || []);
        renderStats(result.stats || {});
        setStatus(result.message || "History cleared.", "success");
    } catch (error) {
        setStatus(error.message, "error");
    } finally {
        clearHistoryBtn.disabled = false;
        clearHistoryBtn.textContent = "Clear History";
    }
}

predictBtn.addEventListener("click", predictNews);
refreshHistoryBtn.addEventListener("click", loadHistory);
clearHistoryBtn.addEventListener("click", clearHistory);
downloadReportBtn.addEventListener("click", downloadReport);
newsText.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        predictNews();
    }
});
window.addEventListener("DOMContentLoaded", loadHistory);
