document.getElementById("faq_button").addEventListener("click", function() {
    document.getElementById("faq_popup").style.display = "block";
});

document.getElementsByClassName("close")[0].addEventListener("click", function() {
    document.getElementById("faq_popup").style.display = "none";
});
