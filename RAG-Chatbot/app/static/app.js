const messages = document.getElementById('messages');
const form = document.getElementById('composer');
const queryInput = document.getElementById('query');
const providerSelect = document.getElementById('provider');

function appendMessage(text, from='bot', snippet=''){
  const el = document.createElement('div');
  el.className = 'message ' + (from==='user' ? 'user' : 'bot');
  const bubble = document.createElement('div');
  bubble.className = 'bubble ' + (from==='user' ? 'user' : '');
  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = from === 'user' ? 'You' : 'Assistant';
  bubble.appendChild(meta);
  const p = document.createElement('div');
  p.innerHTML = text.replace(/\n/g, '<br>');
  bubble.appendChild(p);
  if(snippet){
    const s = document.createElement('small');
    s.className = 'snip';
    s.textContent = snippet;
    bubble.appendChild(s);
  }
  el.appendChild(bubble);
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
}

form.addEventListener('submit', async (ev)=>{
  ev.preventDefault();
  const q = queryInput.value.trim();
  if(!q) return;
  appendMessage(q, 'user');
  queryInput.value = '';

  const payload = { query: q };
  try{
    appendMessage('Thinking...', 'bot');
    const res = await fetch('/api/chat', { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    const data = await res.json();
    // remove the 'Thinking...' placeholder
    messages.removeChild(messages.lastChild);
    if(data.answer){
      appendMessage(data.answer, 'bot');
    } else {
      appendMessage('No answer (check server logs).', 'bot');
    }
    // show citations if present
    if(Array.isArray(data.citations) && data.citations.length){
      data.citations.forEach((c, idx)=>{
        appendMessage(`Document ${idx+1}: ${c.content}`, 'bot', `id=${c.id}`)
      })
    }
  }catch(err){
    messages.removeChild(messages.lastChild);
    appendMessage('Error: '+err.message, 'bot');
  }
});

// sample welcome
appendMessage('Welcome — ask a question and I will retrieve supporting documents.', 'bot');
