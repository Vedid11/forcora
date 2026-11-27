
// ==========================================
// FORCORA - JAVASCRIPT
// ==========================================

const API_URL = "https://forcora-api.onrender.com";


const STORAGE_KEY = "forcora_added_reviews";
const CSV_STORAGE_KEY = "forcora_csv_reviews";

const stats = {
  "Bug Report": 0,
  "Feature Request": 0,
  "Privacy-Irrelevant": 0
};

let totalReviews = 0;
let savedReviews = [];
let csvSavedReviews = [];

// ==========================================
// STORAGE
// ==========================================

function loadSavedData() {
  try {
    savedReviews = JSON.parse(
      localStorage.getItem(STORAGE_KEY) || "[]"
    );

    csvSavedReviews = JSON.parse(
      localStorage.getItem(CSV_STORAGE_KEY) || "[]"
    );

    if (!Array.isArray(savedReviews)) savedReviews = [];
    if (!Array.isArray(csvSavedReviews)) csvSavedReviews = [];

  } catch (error) {
    console.error("Error loading saved reviews:", error);
    savedReviews = [];
    csvSavedReviews = [];
  }
}

function saveSingleReviews() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify(savedReviews)
  );
}

function saveCSVReviews() {
  try {
    localStorage.setItem(
      CSV_STORAGE_KEY,
      JSON.stringify(csvSavedReviews)
    );
  } catch (error) {
    console.error("Could not save CSV reviews:", error);

    throw new Error(
      "Reviews were classified, but browser storage is full."
    );
  }
}

function resetStats() {
  totalReviews = 0;

  Object.keys(stats).forEach(category => {
    stats[category] = 0;
  });
}

// ==========================================
// SIDEBAR NAVIGATION
// ==========================================

const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebarToggle");
const sidebarClose = document.getElementById("sidebarClose");
const sidebarOverlay = document.getElementById("sidebarOverlay");

function openSidebar() {
  sidebar?.classList.add("open");
  sidebarOverlay?.classList.add("show");
  document.body.classList.add("sidebar-open");

  sidebarToggle?.setAttribute("aria-expanded", "true");
}

function closeSidebar() {
  sidebar?.classList.remove("open");
  sidebarOverlay?.classList.remove("show");
  document.body.classList.remove("sidebar-open");

  sidebarToggle?.setAttribute("aria-expanded", "false");
}

sidebarToggle?.addEventListener("click", openSidebar);
sidebarClose?.addEventListener("click", closeSidebar);
sidebarOverlay?.addEventListener("click", closeSidebar);

document.addEventListener("keydown", event => {
  if (event.key === "Escape") {
    closeSidebar();
    closeReviewModal();
  }
});

// ==========================================
// PAGE NAVIGATION
// ==========================================

function showPage(pageId) {
  const target = document.getElementById(pageId);

  if (!target) return;

  document.querySelectorAll(".page-section").forEach(section => {
    section.classList.toggle(
      "active",
      section.id === pageId
    );
  });

  document.querySelectorAll("[data-page]").forEach(item => {
    const active = item.dataset.page === pageId;

    item.classList.toggle("active", active);

    if (item.matches("li")) {
      if (active) {
        item.setAttribute("aria-current", "page");
      } else {
        item.removeAttribute("aria-current");
      }
    }
  });

  closeSidebar();

  window.scrollTo({
    top: 0,
    behavior: "smooth"
  });
}

document.querySelectorAll("[data-page]").forEach(item => {
  item.addEventListener("click", () => {
    showPage(item.dataset.page);
  });
});

document.getElementById("goToUpload")?.addEventListener("click", () => {
  showPage("upload");
});

// ==========================================
// SEARCH AND FILTER ELEMENTS
// ==========================================

const searchInput = document.getElementById("globalSearch");
const reviewSearch = document.getElementById("reviewSearch");
const categoryFilter = document.getElementById("categoryFilter");

const addSingleReviewButton =
  document.getElementById("addSingleReview");

// ==========================================
// REVIEW FILTERS
// ==========================================

function applyReviewFilters() {
  const query = (
    reviewSearch?.value ||
    searchInput?.value ||
    ""
  ).trim().toLowerCase();

  const category = categoryFilter?.value || "all";

  const rows = document.querySelectorAll(
    "#explorerTableBody tr"
  );

  let visible = 0;

  rows.forEach(row => {
    const matchesText = row.innerText
      .toLowerCase()
      .includes(query);

    const matchesCategory =
      category === "all" ||
      row.cells[2]?.innerText.trim() === category;

    const show = matchesText && matchesCategory;

    row.style.display = show ? "" : "none";

    if (show) visible++;
  });

  const count = document.getElementById("explorerReviewCount");
  const empty = document.getElementById("explorerEmpty");

  if (count) {
    count.textContent = `${visible} Reviews`;
  }

  if (empty) {
    empty.style.display = visible ? "none" : "block";
  }
}

searchInput?.addEventListener("input", () => {
  if (reviewSearch) {
    reviewSearch.value = searchInput.value;
  }

  const query = searchInput.value.toLowerCase();

  document.querySelectorAll(
    "#overview .recent-reviews tbody tr"
  ).forEach(row => {
    row.style.display = row.innerText
      .toLowerCase()
      .includes(query) ? "" : "none";
  });

  applyReviewFilters();
});

reviewSearch?.addEventListener(
  "input",
  applyReviewFilters
);

categoryFilter?.addEventListener(
  "change",
  applyReviewFilters
);

// ==========================================
// SINGLE REVIEW MODAL
// ==========================================

if (addSingleReviewButton) {
  addSingleReviewButton.addEventListener(
    "click",
    openReviewModal
  );
}

function openReviewModal() {
  if (document.querySelector(".review-modal-overlay")) {
    return;
  }

  const overlay = document.createElement("div");
  overlay.className = "review-modal-overlay";

  overlay.innerHTML = `
    <div class="review-modal">
      <h2>Add Single Review</h2>

      <p>
        Enter a customer review to classify it using FORCORA.
      </p>

      <textarea
        id="reviewInput"
        placeholder="Enter customer review..."
        maxlength="5000"
      ></textarea>

      <p id="reviewError" class="review-error"></p>

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
      margin-bottom: 10px;
    }

    .review-modal > p {
      color: #667085;
      font-size: 14px;
      margin-bottom: 18px;
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

    .review-modal textarea:focus {
      outline: none;
      border-color: #6254ed;
    }

    .review-error {
      color: red !important;
      margin-top: 8px;
      font-size: 13px !important;
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

    #cancelReview {
      background: #eef0f4;
      color: #344054;
    }

    #submitReview {
      background: #6254ed;
      color: white;
    }

    #submitReview:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
  `;

  document.head.appendChild(style);

  document.getElementById("cancelReview").onclick =
    closeReviewModal;

  document.getElementById("submitReview").onclick =
    submitReview;

  document.getElementById("reviewInput").focus();
}

function closeReviewModal() {
  document
    .querySelector(".review-modal-overlay")
    ?.remove();

  document
    .getElementById("review-modal-style")
    ?.remove();
}

// ==========================================
// SINGLE REVIEW API
// ==========================================

async function submitReview() {
  const input = document.getElementById("reviewInput");
  const error = document.getElementById("reviewError");
  const button = document.getElementById("submitReview");

  if (!input || !error || !button) return;

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
      throw new Error(
        "Prediction failed. Please try again."
      );
    }

    const result = await response.json();
    const category = result.predicted_category;

    if (stats[category] === undefined) {
      throw new Error(
        "Unknown category returned by the model."
      );
    }

    const date = new Date().toLocaleDateString(
      "en-GB",
      {
        day: "2-digit",
        month: "short",
        year: "numeric"
      }
    );

    const newReview = {
      review,
      result,
      date
    };

    savedReviews.push(newReview);
    saveSingleReviews();

    stats[category]++;
    totalReviews++;

    addReviewToTable(
      review,
      result,
      date
    );

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

// ==========================================
// ADD REVIEW TO TABLE
// ==========================================

function addReviewToTable(
  review,
  result,
  date,
  prepend = true
) {
  const categoryClasses = {
    "Bug Report": "bug-badge",
    "Feature Request": "feature-badge",
    "Privacy-Irrelevant": "privacy-badge"
  };

  const category = result.predicted_category;

  const badgeClass =
    categoryClasses[category] || "";

  const targets = [
    document.querySelector(
      "#overview .recent-reviews tbody"
    ),
    document.getElementById("explorerTableBody")
  ].filter(Boolean);

  targets.forEach(tbody => {
    const row = document.createElement("tr");

    const values = [
      "",
      review,
      category,
      `${(result.confidence * 100).toFixed(1)}%`,
      date
    ];

    values.forEach((value, index) => {
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
  });

  updateRowNumbers();
  applyReviewFilters();
}

function updateRowNumbers() {
  document.querySelectorAll(
    ".recent-reviews tbody"
  ).forEach(tbody => {
    Array.from(tbody.rows).forEach((row, index) => {
      row.cells[0].textContent = index + 1;
    });
  });
}

// ==========================================
// UPDATE DASHBOARD
// ==========================================

function updateDashboard() {
  const cards = document.querySelectorAll(
    "#overview .cards .card"
  );

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

  const analyticsValues = {
    analyticsTotal: totalReviews,
    analyticsBugs: stats["Bug Report"],
    analyticsFeatures: stats["Feature Request"],
    analyticsPrivacy: stats["Privacy-Irrelevant"]
  };

  Object.entries(analyticsValues).forEach(
    ([id, value]) => {
      const element = document.getElementById(id);

      if (element) {
        element.textContent =
          value.toLocaleString("en-IN");
      }
    }
  );

  updateCharts();
}

// ==========================================
// UPDATE DONUT AND BAR CHARTS
// ==========================================

function updateCharts() {
  const bug = stats["Bug Report"] || 0;
  const feature = stats["Feature Request"] || 0;
  const privacy = stats["Privacy-Irrelevant"] || 0;

  const total = bug + feature + privacy;

  const values = [bug, feature, privacy];

  const percentages = total
    ? values.map(value => (value / total) * 100)
    : [0, 0, 0];

  // Donut chart background
  const donutBackground = total
    ? `conic-gradient(
        #ef4c55 0% ${percentages[0]}%,
        #7c5cf0 ${percentages[0]}% ${percentages[0] + percentages[1]}%,
        #20b76a ${percentages[0] + percentages[1]}% 100%
      )`
    : "conic-gradient(#e8edf4 0% 100%)";

  document.querySelectorAll(".category-donut").forEach(donut => {
    donut.style.background = donutBackground;

    const center = donut.querySelector(".donut-center h2");

    if (center) {
      center.textContent = total.toLocaleString("en-IN");
    }
  });

  // Category legend percentages
  document.querySelectorAll(
    ".category-content .legend-item strong"
  ).forEach((element, index) => {

    // Prevent undefined percentage errors
    if (index >= percentages.length) {
      return;
    }

    const percentage = Number(percentages[index] ?? 0);

    element.textContent = `${percentage.toFixed(1)}%`;
  });

  // Category bar chart
  document.querySelectorAll(".category-bar-chart").forEach(chart => {
    const items = chart.querySelectorAll(".bar-item");

    const maxValue = Math.max(...values, 1);

    items.forEach((item, index) => {
      const value = values[index] ?? 0;

      const valueElement = item.querySelector(".bar-value");
      const bar = item.querySelector(".bar");

      if (valueElement) {
        valueElement.textContent =
          value.toLocaleString("en-IN");
      }

      if (bar) {
        bar.style.height = value === 0
          ? "0px"
          : `${Math.max(
              8,
              (value / maxValue) * 180
            )}px`;
      }
    });
  });
}

// ==========================================
// CSV FILE UPLOAD
// ==========================================

const csvFileInput =
  document.getElementById("csvFileInput");

const csvUploadStatus =
  document.getElementById("csvUploadStatus");

csvFileInput?.addEventListener(
  "change",
  uploadCSV
);

// ==========================================
// CSV PARSER
// ==========================================

function parseCSV(text) {
  const rows = [];

  let row = [];
  let value = "";
  let insideQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    const nextChar = text[i + 1];

    if (
      char === '"' &&
      insideQuotes &&
      nextChar === '"'
    ) {
      value += '"';
      i++;

    } else if (char === '"') {
      insideQuotes = !insideQuotes;

    } else if (char === "," && !insideQuotes) {
      row.push(value);
      value = "";

    } else if (
      (char === "\n" || char === "\r") &&
      !insideQuotes
    ) {
      if (char === "\r" && nextChar === "\n") {
        i++;
      }

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

// ==========================================
// CSV UPLOAD AND BULK CLASSIFICATION
// ==========================================

async function uploadCSV() {
  const file = csvFileInput?.files[0];

  if (!file) return;

  if (!file.name.toLowerCase().endsWith(".csv")) {
    if (csvUploadStatus) {
      csvUploadStatus.textContent =
        "Please select a CSV file.";
    }

    csvFileInput.value = "";
    return;
  }

  if (csvUploadStatus) {
    csvUploadStatus.textContent =
      "Uploading and analyzing reviews...";
  }

  const formData = new FormData();

  formData.append("file", file);
  formData.append("review_column", "Review");

  try {
    const response = await fetch(
      `${API_URL}/predict/bulk`,
      {
        method: "POST",
        body: formData
      }
    );

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

    // Backend returns classified CSV.
    const csvBlob = await response.blob();
    const csvText = await csvBlob.text();

    const parsedRows = parseCSV(
      csvText.replace(/^\uFEFF/, "")
    );

    if (parsedRows.length < 2) {
      throw new Error(
        "The CSV response contains no review results."
      );
    }

    const headers = parsedRows[0].map(
      header => header.trim()
    );

    const reviewIndex = headers.indexOf("Review");

    const categoryIndex =
      headers.indexOf("predicted_category");

    const confidenceIndex =
      headers.indexOf("confidence");

    if (
      reviewIndex === -1 ||
      categoryIndex === -1 ||
      confidenceIndex === -1
    ) {
      throw new Error(
        "Required prediction columns were not found."
      );
    }

    const date = new Date().toLocaleDateString(
      "en-GB",
      {
        day: "2-digit",
        month: "short",
        year: "numeric"
      }
    );

    const newCSVReviews = [];

    parsedRows.slice(1).forEach(row => {
      const review = (
        row[reviewIndex] || ""
      ).trim();

      const category = (
        row[categoryIndex] || ""
      ).trim();

      const confidence = Number(
        (row[confidenceIndex] || "").trim()
      );

      if (
        !review ||
        stats[category] === undefined
      ) {
        return;
      }

      const result = {
        predicted_category: category,
        confidence: Number.isFinite(confidence)
          ? confidence
          : 0
      };

      newCSVReviews.push({
        review,
        result,
        date
      });
    });

    if (newCSVReviews.length === 0) {
      throw new Error(
        "No valid classified reviews were found."
      );
    }

    // Save uploaded CSV reviews.
    csvSavedReviews = newCSVReviews;
    saveCSVReviews();

    // Rebuild dashboard.
    resetStats();

    const overviewBody = document.querySelector(
      "#overview .recent-reviews tbody"
    );

    if (overviewBody) {
      overviewBody.innerHTML = "";
    }

    const explorerBody = document.getElementById(
      "explorerTableBody"
    );

    if (explorerBody) {
      explorerBody.innerHTML = "";
    }

    // Restore CSV reviews.
    csvSavedReviews.forEach(item => {
      const category =
        item.result.predicted_category;

      stats[category]++;
      totalReviews++;

      addReviewToTable(
        item.review,
        item.result,
        item.date,
        false
      );
    });

    // Restore previously saved single reviews.
    savedReviews.forEach(item => {
      const category =
        item.result?.predicted_category;

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

    const totalRows =
      response.headers.get("X-Total-Rows");

    const classifiedRows =
      response.headers.get("X-Classified-Rows");

    const skippedRows =
      response.headers.get("X-Skipped-Rows");

    if (csvUploadStatus) {
      csvUploadStatus.textContent =
        classifiedRows !== null
          ? `Successfully classified ${classifiedRows} of ${totalRows} rows. Skipped: ${skippedRows || 0}.`
          : `Successfully classified ${newCSVReviews.length} reviews from ${file.name}.`;
    }

    // Download classified CSV.
    const downloadUrl =
      URL.createObjectURL(csvBlob);

    const downloadLink =
      document.createElement("a");

    downloadLink.href = downloadUrl;

    downloadLink.download =
      "forcora_classified_reviews.csv";

    document.body.appendChild(downloadLink);

    downloadLink.click();
    downloadLink.remove();

    setTimeout(() => {
      URL.revokeObjectURL(downloadUrl);
    }, 1000);

  } catch (error) {
    console.error("CSV upload error:", error);

    if (csvUploadStatus) {
      csvUploadStatus.textContent = error.message;
    }

  } finally {
    csvFileInput.value = "";
  }
}

// ==========================================
// INITIALIZE DASHBOARD
// ==========================================


function initializeDashboard() {

  // Clear previously saved browser data on every page load
  localStorage.removeItem(STORAGE_KEY);
  localStorage.removeItem(CSV_STORAGE_KEY);

  // Reset in-memory data
  savedReviews = [];
  csvSavedReviews = [];

  // Reset all dashboard statistics
  resetStats();

  // Clear Overview recent reviews
  const overviewBody = document.querySelector(
    "#overview .recent-reviews tbody"
  );

  if (overviewBody) {
    overviewBody.innerHTML = "";
  }

  // Clear Review Explorer
  const explorerBody = document.getElementById(
    "explorerTableBody"
  );

  if (explorerBody) {
    explorerBody.innerHTML = "";
  }

  // Update dashboard with zero values
  updateDashboard();
}

// ==========================================
// START APPLICATION
// ==========================================

initializeDashboard();
showPage("overview");
applyReviewFilters();
