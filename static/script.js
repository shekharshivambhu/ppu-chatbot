const chatForm = document.getElementById("chat-form");
const chatBox = document.getElementById("chat-box");

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("user-input");
  const userText = input.value;
  addMessage("You", userText, "user");
  input.value = "";

  const formData = new FormData();
  formData.append("query", userText);

  const response = await fetch("/chat", {
    method: "POST",
    body: formData
  });

  const data = await response.json();
  addMessage("Bot", data.reply, "bot");
});

function addMessage(sender, text, className) {
  const div = document.createElement("div");
  div.className = `message ${className}`;
  div.innerHTML = `<span class="${className}">${sender}:</span> ${text}`;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}
