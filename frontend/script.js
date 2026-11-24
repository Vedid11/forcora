const searchInput = document.querySelector(".search-box input");
const rows = document.querySelectorAll(".recent-reviews tbody tr");

searchInput.addEventListener("keyup", function () {
  const searchText = searchInput.value.toLowerCase();

  rows.forEach(function (row) {
    const rowText = row.innerText.toLowerCase();
    row.style.display = rowText.includes(searchText) ? "" : "none";
  });
});
