// Mobile menu toggle
const mobileMenuButton = document.getElementById("mobile-menu-button");
const mobileMenu = document.getElementById("mobile-menu");

mobileMenuButton.addEventListener("click", function () {
  mobileMenu.classList.toggle("hidden");
});

// Close mobile menu when clicking on a menu item
const mobileMenuItems = document.querySelectorAll("#mobile-menu a");
mobileMenuItems.forEach((item) => {
  item.addEventListener("click", function () {
    mobileMenu.classList.add("hidden");
  });
});
