
const API_URL = "http://127.0.0.1:8000";

// Category counts
const stats = {
  "Bug Report": 0,
  "Feature Request": 0,
  "Privacy-Irrelevant": 0
};

let totalReviews = 0;

// Data will remain in memory only until page refresh.
let savedReviews = [];
let csvSavedReviews = [];

// -----------------------------
// RESET CATEGORY COUNTS
// -----------------------------

function resetStats() {
  totalReviews = 0;

  Object.keys(stats).forEach(category => {
    stats[category] = 0;
  });
}

// -----------------------------
// INITIALIZE DASHBOARD
// -----------------------------

function initializeDashboard() {
  resetStats();

  const tbody = document.querySelector(".recent-reviews tbody");

  // Remove hardcoded sample reviews.
  if (tbody) {
    tbody.innerHTML = "";
  }

  // Clear search field.
  if (searchInput) {
    searchInput.value = "";
  }

  // Remove old upload status.
  if (csvUploadStatus) {
    csvUploadStatus.textContent = "";
  }

  updateDashboard();
}

// -----------------------------
// SEARCH REVIEWS
// -----------------------------

const searchInput = document.querySelector(".search-box input");

if (searchInput) {
  searchInput.addEventListener("input", function () {
    const searchText = searchInput.value.toLowerCase();

    document.querySelectorAll(".recent-reviews tbody tr")
      .forEach(row => {
        row.style.display = row.innerText.toLowerCase()
          .includes(searchText) ? "" : "none";
      });
  });
}

// -----------------------------
// ADD SINGLE REVIEW
// -----------------------------

const singleReviewCard = document.querySelector(".single-box");

if (singleReviewCard) {
  singleReviewCard.style.cursor = "pointer";
  singleReviewCard.addEventListener("click", openReviewModal);
}

function openReviewModal() {
  if (document.querySelector(".review-modal-overlay")) return;

  const overlay = document.createElement("div");
  overlay.className = "review-modal-overlay";

  overlay.innerHTML = `
    <div class="review-modal">
      <h2>Add Single Review</h2>
      <p>Enter a customer review to classify it using FORCORA AI.</p>

      <textarea
        id="reviewInput"
        placeholder="Enter customer review..."
        maxlength="5000"
      ></textarea>

      <p id="reviewError" style="color:red;"></p>

      <div class="modal-actions">
        <button id="cancelReview">Cancel</button>
        <button id="submitReview">Analyze Review</button>
      </div>
    </div>
  `;

  document.body.appendChild(overlay);

  const style = document.createElement("style");
  style.id = "review-modal-style";

  style.textContent = `
    .review-modal-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.5);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 9999;
      padding: 20px;
    }

    .review-modal {
      background: white;
      width: 100%;
      max-width: 480px;
      padding: 28px;
      border-radius: 14px;
      box-shadow: 0 10px 35px rgba(0,0,0,0.2);
    }

    .review-modal h2 {
      margin-top: 0;
    }

    .review-modal textarea {
      width: 100%;
      min-height: 130px;
      padding: 12px;
      border: 1px solid #ddd;
      border-radius: 8px;
      resize: vertical;
      box-sizing: border-box;
      font: inherit;
    }

    .modal-actions {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      margin-top: 15px;
    }

    .modal-actions button {
      padding: 10px 18px;
      border: none;
      border-radius: 7px;
      cursor: pointer;
    }

    #submitReview {
      background: #6254ed;
      color: white;
    }
  `;

  document.head.appendChild(style);

  document.getElementById("cancelReview").onclick = closeReviewModal;
  document.getElementById("submitReview").onclick = submitReview;
  document.getElementById("reviewInput").focus();
}

function closeReviewModal() {
  document.querySelector(".review-modal-overlay")?.remove();
  document.getElementById("review-modal-style")?.remove();
}

// -----------------------------
// SINGLE REVIEW API
// -----------------------------

async function submitReview() {
  const input = document.getElementById("reviewInput");
  const error = document.getElementById("reviewError");
  const button = document.getElementById("submitReview");

  const review = input.value.trim();

  if (!review) {
    error.textContent = "Please enter a review.";
    return;
  }

  button.disabled = true;
  button.textContent = "Analyzing...";
  error.textContent = "";

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ review })
    });

    if (!response.ok) {
      throw new Error("Prediction failed. Please try again.");
    }

    const result = await response.json();
    const category = result.predicted_category;

    if (stats[category] === undefined) {
      throw new Error("Unknown category returned by the model.");
    }

    const date = new Date().toLocaleDateString("en-GB", {
      day: "2-digit",
      month: "short",
      year: "numeric"
    });

    const newReview = {
      review,
      result,
      date
    };

    // Keep this review in memory for the current page session.
    // Backend saves the permanent record in PostgreSQL.
    savedReviews.push(newReview);

    stats[category]++;
    totalReviews++;

    addReviewToTable(review, result, date);
    updateDashboard();

    closeReviewModal();

    alert(
      `Category: ${category}\n` +
      `Confidence: ${(result.confidence * 100).toFixed(1)}%`
    );

  } catch (err) {
    error.textContent = err.message;
    button.disabled = false;
    button.textContent = "Analyze Review";
  }
}

// -----------------------------
// ADD REVIEW TO TABLE
// -----------------------------

function addReviewToTable(review, result, date, prepend = true) {
  const tbody = document.querySelector(".recent-reviews tbody");

  if (!tbody) return;

  const categoryClasses = {
    "Bug Report": "bug-badge",
    "Feature Request": "feature-badge",
    "Privacy-Irrelevant": "privacy-badge"
  };

  const category = result.predicted_category;
  const badgeClass = categoryClasses[category] || "";

  const row = document.createElement("tr");

  const cells = [
    "",
    review,
    null,
    `${(result.confidence * 100).toFixed(1)}%`,
    date
  ];

  cells.forEach((value, index) => {
    const cell = document.createElement("td");

    if (index === 2) {
      const badge = document.createElement("span");
      badge.className = `badge ${badgeClass}`;
      badge.textContent = category;
      cell.appendChild(badge);
    } else {
      cell.textContent = value;
    }

    row.appendChild(cell);
  });

  if (prepend) {
    tbody.prepend(row);
  } else {
    tbody.appendChild(row);
  }

  updateRowNumbers();

  if (searchInput) {
    const searchText = searchInput.value.toLowerCase();

    row.style.display = row.innerText.toLowerCase()
      .includes(searchText) ? "" : "none";
  }
}

function updateRowNumbers() {
  document.querySelectorAll(".recent-reviews tbody tr")
    .forEach((row, index) => {
      row.cells[0].textContent = index + 1;
    });
}

// -----------------------------
// UPDATE DASHBOARD CARDS
// -----------------------------

function updateDashboard() {
  const cards = document.querySelectorAll(".cards .card");

  if (cards[0]) {
    cards[0].querySelector("h2").textContent =
      totalReviews.toLocaleString("en-IN");
  }

  if (cards[1]) {
    cards[1].querySelector("h2").textContent =
      stats["Privacy-Irrelevant"].toLocaleString("en-IN");
  }

  if (cards[2]) {
    cards[2].querySelector("h2").textContent =
      stats["Bug Report"].toLocaleString("en-IN");
  }

  if (cards[3]) {
    cards[3].querySelector("h2").textContent =
      stats["Feature Request"].toLocaleString("en-IN");
  }

  updateCharts();
}

// -----------------------------
// UPDATE DONUT AND BAR CHARTS
// -----------------------------

function updateCharts() {
  const bug = stats["Bug Report"];
  const feature = stats["Feature Request"];
  const privacy = stats["Privacy-Irrelevant"];

  const total = bug + feature + privacy;

  const donut = document.querySelector(".category-donut");

  const donutTotal = document.querySelector(
    ".category-donut .donut-center h2"
  );

  const legendValues = document.querySelectorAll(
    ".category-content .legend-item strong"
  );

  const barItems = document.querySelectorAll(
    ".category-bar-chart .bar-item"
  );

  // Reset all charts when dashboard is empty.
  if (!total) {
    if (donut) {
      donut.style.background = "conic-gradient(#e5e7eb 0% 100%)";
    }

    if (donutTotal) {
      donutTotal.textContent = "0";
    }

    legendValues.forEach(element => {
      element.textContent = "0.0%";
    });

    barItems.forEach(item => {
      const valueElement = item.querySelector(".bar-value");
      const bar = item.querySelector(".bar");

      if (valueElement) {
        valueElement.textContent = "0";
      }

      if (bar) {
        bar.style.height = "0px";
      }
    });

    return;
  }

  const bugPercent = (bug / total) * 100;
  const featurePercent = (feature / total) * 100;
  const privacyPercent = (privacy / total) * 100;

  if (donut) {
    const bugEnd = bugPercent;
    const featureEnd = bugPercent + featurePercent;

    donut.style.background = `
      conic-gradient(
        #f04452 0% ${bugEnd}%,
        #7c5cf1 ${bugEnd}% ${featureEnd}%,
        #1fb66d ${featureEnd}% 100%
      )
    `;
  }

  if (donutTotal) {
    donutTotal.textContent = total.toLocaleString("en-IN");
  }

  const percentages = [
    bugPercent,
    featurePercent,
    privacyPercent
  ];

  legendValues.forEach((element, index) => {
    element.textContent = `${percentages[index].toFixed(1)}%`;
  });

  const values = [bug, feature, privacy];
  const maxValue = Math.max(...values, 1);

  barItems.forEach((item, index) => {
    const valueElement = item.querySelector(".bar-value");
    const bar = item.querySelector(".bar");

    if (valueElement) {
      valueElement.textContent = values[index].toLocaleString("en-IN");
    }

    if (bar) {
      const height = values[index] === 0
        ? 0
        : Math.max(8, (values[index] / maxValue) * 180);

      bar.style.height = `${height}px`;
    }
  });
}

// -----------------------------
// CSV FILE UPLOAD
// -----------------------------

const uploadCard = document.querySelector(".upload-box");
const csvFileInput = document.getElementById("csvFileInput");
const csvUploadStatus = document.getElementById("csvUploadStatus");

if (uploadCard && csvFileInput) {
  uploadCard.addEventListener("click", (event) => {
    if (event.target !== csvFileInput) {
      csvFileInput.click();
    }
  });

  csvFileInput.addEventListener("change", uploadCSV);
}

// -----------------------------
// CSV PARSER
// -----------------------------

function parseCSV(text) {
  const rows = [];
  let row = [];
  let value = "";
  let insideQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    const nextChar = text[i + 1];

    if (char === '"' && insideQuotes && nextChar === '"') {
      value += '"';
      i++;
    } else if (char === '"') {
      insideQuotes = !insideQuotes;
    } else if (char === "," && !insideQuotes) {
      row.push(value);
      value = "";
    } else if ((char === "\n" || char === "\r") && !insideQuotes) {
      if (char === "\r" && nextChar === "\n") i++;

      row.push(value);

      if (row.some(cell => cell.length > 0)) {
        rows.push(row);
      }

      row = [];
      value = "";
    } else {
      value += char;
    }
  }

  if (value.length > 0 || row.length > 0) {
    row.push(value);

    if (row.some(cell => cell.length > 0)) {
      rows.push(row);
    }
  }

  return rows;
}

// -----------------------------
// CSV FILE UPLOAD AND CLASSIFICATION
// -----------------------------

async function uploadCSV() {
  const file = csvFileInput.files[0];

  if (!file) return;

  if (!file.name.toLowerCase().endsWith(".csv")) {
    if (csvUploadStatus) {
      csvUploadStatus.textContent = "Please select a CSV file.";
    }

    csvFileInput.value = "";
    return;
  }

  if (csvUploadStatus) {
    csvUploadStatus.textContent = "Uploading and analyzing reviews...";
  }

  const formData = new FormData();

  formData.append("file", file);
  formData.append("review_column", "Review");

  try {
    const response = await fetch(`${API_URL}/predict/bulk`, {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      let message = "CSV processing failed.";

      try {
        const errorData = await response.json();
        message = errorData.detail || message;
      } catch {
        // Keep default error message.
      }

      throw new Error(message);
    }

    // Backend returns a classified CSV file.
    const csvBlob = await response.blob();
    const csvText = await csvBlob.text();

    const parsedRows = parseCSV(
      csvText.replace(/^\uFEFF/, "")
    );

    if (parsedRows.length < 2) {
      throw new Error("The CSV response contains no review results.");
    }

    const headers = parsedRows[0].map(header => header.trim());

    const reviewIndex = headers.indexOf("Review");
    const categoryIndex = headers.indexOf("predicted_category");
    const confidenceIndex = headers.indexOf("confidence");

    if (
      reviewIndex === -1 ||
      categoryIndex === -1 ||
      confidenceIndex === -1
    ) {
      throw new Error("Required prediction columns were not found.");
    }

    const date = new Date().toLocaleDateString("en-GB", {
      day: "2-digit",
      month: "short",
      year: "numeric"
    });

    // Prepare CSV reviews for the current page session.
    const newCSVReviews = [];

    parsedRows.slice(1).forEach(row => {
      const review = (row[reviewIndex] || "").trim();
      const category = (row[categoryIndex] || "").trim();

      const confidence = Number(
        (row[confidenceIndex] || "").trim()
      );

      if (!review || stats[category] === undefined) return;

      const result = {
        predicted_category: category,
        confidence: Number.isFinite(confidence) ? confidence : 0
      };

      newCSVReviews.push({
        review,
        result,
        date
      });
    });

    if (newCSVReviews.length === 0) {
      throw new Error("No valid classified reviews were found.");
    }

    // Keep CSV results in memory only.
    // PostgreSQL stores the permanent records.
    csvSavedReviews = newCSVReviews;

    // Rebuild dashboard using current-session reviews.
    resetStats();

    const tbody = document.querySelector(".recent-reviews tbody");

    if (tbody) {
      tbody.innerHTML = "";
    }

    // Add CSV reviews.
    csvSavedReviews.forEach(item => {
      const category = item.result.predicted_category;

      stats[category]++;
      totalReviews++;

      addReviewToTable(
        item.review,
        item.result,
        item.date,
        false
      );
    });

    // Add Single Reviews from the current page session.
    savedReviews.forEach(item => {
      const category = item.result?.predicted_category;

      if (stats[category] !== undefined) {
        stats[category]++;
        totalReviews++;

        addReviewToTable(
          item.review,
          item.result,
          item.date,
          false
        );
      }
    });

    updateDashboard();

    const totalRows = response.headers.get("X-Total-Rows");
    const classifiedRows = response.headers.get("X-Classified-Rows");
    const skippedRows = response.headers.get("X-Skipped-Rows");

    if (csvUploadStatus) {
      csvUploadStatus.textContent = classifiedRows !== null
        ? `Successfully classified ${classifiedRows} of ${totalRows} rows. Skipped: ${skippedRows || 0}.`
        : `Successfully classified ${newCSVReviews.length} reviews from ${file.name}.`;
    }

    // Download classified CSV.
    const downloadUrl = URL.createObjectURL(csvBlob);
    const downloadLink = document.createElement("a");

    downloadLink.href = downloadUrl;
    downloadLink.download = "forcora_classified_reviews.csv";

    document.body.appendChild(downloadLink);
    downloadLink.click();
    downloadLink.remove();

    setTimeout(() => URL.revokeObjectURL(downloadUrl), 1000);

  } catch (error) {
    console.error("CSV upload error:", error);

    if (csvUploadStatus) {
      csvUploadStatus.textContent = error.message;
    }
  } finally {
    csvFileInput.value = "";
  }
}

// -----------------------------
// MOBILE HAMBURGER SIDEBAR
// -----------------------------

const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebarToggle");
const sidebarClose = document.getElementById("sidebarClose");
const sidebarOverlay = document.getElementById("sidebarOverlay");

function openSidebar() {
  if (!sidebar || !sidebarOverlay) return;

  sidebar.classList.add("open");
  sidebarOverlay.classList.add("show");
  document.body.classList.add("sidebar-open");

  if (sidebarToggle) {
    sidebarToggle.setAttribute("aria-expanded", "true");
  }
}

function closeSidebar() {
  if (!sidebar || !sidebarOverlay) return;

  sidebar.classList.remove("open");
  sidebarOverlay.classList.remove("show");
  document.body.classList.remove("sidebar-open");

  if (sidebarToggle) {
    sidebarToggle.setAttribute("aria-expanded", "false");
  }
}

if (sidebarToggle) {
  sidebarToggle.addEventListener("click", openSidebar);
}

if (sidebarClose) {
  sidebarClose.addEventListener("click", closeSidebar);
}

if (sidebarOverlay) {
  sidebarOverlay.addEventListener("click", closeSidebar);
}

// Close sidebar after clicking a menu item on mobile.
if (sidebar) {
  sidebar.querySelectorAll(".sidebar-menu li, .bottom-item")
    .forEach(item => {
      item.addEventListener("click", () => {
        if (window.innerWidth <= 700) {
          closeSidebar();
        }
      });
    });
}

// Close sidebar with Escape key.
document.addEventListener("keydown", function(event) {
  if (event.key === "Escape") {
    closeSidebar();
  }
});

// Close sidebar when switching back to desktop.
window.addEventListener("resize", function() {
  if (window.innerWidth > 700) {
    closeSidebar();
  }
});

// -----------------------------
// START DASHBOARD
// -----------------------------

initializeDashboard();
