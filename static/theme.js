let isDarkMode = false;

document.getElementById("mode_button").addEventListener("click", function() {
    isDarkMode =!isDarkMode;
    document.body.classList.toggle("dark-mode", isDarkMode);
});