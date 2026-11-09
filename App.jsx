import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

const API = "http://localhost:8000";

const BotAvatar = () => <div className="avatar bot-avatar">A</div>;
const UserAvatar = () => <div className="avatar user-avatar">U</div>;

const CATEGORY_ICONS = { goal: "🎯", preference: "⚙️", project: "🛠️" };

function MemoryPanel({ onClose }) {
  const [facts, setFacts] = useState([]);
  const [memories, setMemories] = useState([]);
  const [editingId, setEditingId] = useState(null);
  const [editText, setEditText] = useState("");

  useEffect(() => {
    fetch(`${API}/facts`).then(r => r.json()).then(setFacts);
    fetch(`${API}/memories`).then(r => r.json()).then(setMemories);
  }, []);

  async function saveFact(id) {
    await fetch(`${API}/facts/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fact: editText }),
    });
    setFacts(prev => prev.map(f => f.id === id ? { ...f, fact: editText } : f));
    setEditingId(null);
  }

  async function deleteFact(id) {
    await fetch(`${API}/facts/${id}`, { method: "DELETE" });
    setFacts(prev => prev.filter(f => f.id !== id));
  }

  async function deleteMemory(id) {
    await fetch(`${API}/memories/${id}`, { method: "DELETE" });
    setMemories(prev => prev.filter(m => m.id !== id));
  }

  const grouped = facts.reduce((acc, f) => {
    (acc[f.category] = acc[f.category] || []).push(f);
    return acc;
  }, {});

  return (
    <div className="memory-panel">
      <div className="memory-panel-header">
        <span>✦ Memory</span>
        <button onClick={onClose}>✕</button>
      </div>
      <div className="memory-panel-body">
        <div className="memory-section-title">Facts</div>
        {Object.keys(grouped).length === 0 && <p className="memory-empty">No facts stored yet.</p>}
        {Object.entries(grouped).map(([cat, items]) => (
          <div key={cat} className="memory-category">
            <div className="memory-category-label">{CATEGORY_ICONS[cat]} {cat}</div>
            {items.map(f => (
              <div key={f.id} className="memory-fact-row">
                {editingId === f.id ? (
                  <>
                    <input className="memory-edit-input" value={editText} onChange={e => setEditText(e.target.value)} autoFocus />
                    <button className="memory-action-btn save" onClick={() => saveFact(f.id)}>✓</button>
                    <button className="memory-action-btn cancel" onClick={() => setEditingId(null)}>✕</button>
                  </>
                ) : (
                  <>
                    <span className="memory-fact-text">{f.fact}</span>
                    <button className="memory-action-btn edit" onClick={() => { setEditingId(f.id); setEditText(f.fact); }}>✎</button>
                    <button className="memory-action-btn delete" onClick={() => deleteFact(f.id)}>✕</button>
                  </>
                )}
              </div>
            ))}
          </div>
        ))}
        <div className="memory-section-title" style={{ marginTop: "20px" }}>Summaries</div>
        {memories.length === 0 && <p className="memory-empty">No summaries stored yet.</p>}
        {memories.map(m => (
          <div key={m.id} className="memory-summary-row">
            <span className="memory-fact-text">{m.summary}</span>
            <button className="memory-action-btn delete" onClick={() => deleteMemory(m.id)}>✕</button>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loadingId, setLoadingId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [memoryOpen, setMemoryOpen] = useState(false);
  const [docs, setDocs] = useState([]);
  const fileRef = useRef(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/conversations`)
      .then(r => r.json())
      .then(data => {
        setConversations(data);
        if (data.length > 0) loadChat(data[0].id);
      });
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loadingId]);

  useEffect(() => {
    if (activeId) {
      fetch(`${API}/conversations/${activeId}/documents`)
        .then(r => r.json()).then(setDocs);
    } else {
      setDocs([]);
    }
  }, [activeId]);

  async function loadChat(id) {
    setActiveId(id);
    setInput("");
    const res = await fetch(`${API}/conversations/${id}/messages`);
    const data = await res.json();
    setMessages(data);
  }

  async function startNewChat() {
    if (activeId) {
      await fetch(`${API}/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversationId: activeId }),
      });
    }
    const id = crypto.randomUUID();
    await fetch(`${API}/conversations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversationId: id, title: "New Chat" }),
    });
    setConversations(prev => [{ id, title: "New Chat" }, ...prev]);
    setActiveId(id);
    setInput("");
    setMessages([]);
  }

  const sendMessage = async () => {
    if (!input.trim()) return;
    setMessages(prev => [...prev, { role: "user", text: input }]);
    setInput("");
    const sendingId = activeId;
    setLoadingId(sendingId);
    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversationId: sendingId, message: input }),
      });
      const data = await res.json();
      setMessages(prev => [...prev, { role: "bot", text: data.reply }]);
      setConversations(prev =>
        prev.map(c => c.id === sendingId && c.title === "New Chat" ? { ...c, title: input.slice(0, 30) } : c)
      );
    } catch {
      setMessages(prev => [...prev, { role: "bot", text: "Error: could not reach the server." }]);
    } finally {
      setLoadingId(null);
    }
  };

  async function deleteChat(e, id) {
    e.stopPropagation();
    await fetch(`${API}/conversations/${id}`, { method: "DELETE" });
    setConversations(prev => prev.filter(c => c.id !== id));
    if (activeId === id) {
      setActiveId(null);
      setMessages([]);
    }
  }

  async function uploadPDF(e) {
    const file = e.target.files[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    const r = await fetch(`${API}/conversations/${activeId}/upload`, { method: "POST", body: form });
    const data = await r.json();
    if (data.ok) setDocs(prev => [...prev, { filename: data.filename }]);
    e.target.value = "";
  }

  async function deleteDoc(filename) {
    await fetch(`${API}/conversations/${activeId}/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
    setDocs(prev => prev.filter(d => d.filename !== filename));
  }

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="app">
      <aside className={`sidebar ${sidebarOpen ? "" : "collapsed"}`}>
        <div className="sidebar-header">
          <span className="logo">{sidebarOpen ? "✦ Aldric" : "✦"}</span>
          <button className="toggle-btn" onClick={() => setSidebarOpen(o => !o)}>
            {sidebarOpen ? "←" : "→"}
          </button>
        </div>
        <div className="sidebar-body">
          {sidebarOpen && <button className="new-chat-btn" onClick={startNewChat}>+ New Chat</button>}
          {sidebarOpen && <button className="memory-btn" onClick={() => setMemoryOpen(true)}>✦ Memory</button>}
          <div className="conversation-list">
            {conversations.map(c => (
              <div key={c.id} className="conversation-item-wrapper">
                <button
                  className={`conversation-item ${c.id === activeId ? "active" : ""}`}
                  onClick={() => loadChat(c.id)}
                >
                  {c.title}
                </button>
                <button className="delete-chat-btn" onClick={e => deleteChat(e, c.id)} aria-label="Delete">✕</button>
              </div>
            ))}
          </div>
        </div>
      </aside>

      {memoryOpen && <MemoryPanel onClose={() => setMemoryOpen(false)} />}

      <main className="chat-main">
        <header className="chat-header">
          <button className="mobile-menu-btn" onClick={() => setSidebarOpen(o => !o)}>☰</button>
          <span>Aldric the Ancient</span>
          {activeId && (
            <div className="doc-area">
              {docs.map(d => (
                <span key={d.filename} className="doc-tag">
                  📎 {d.filename}
                  <button onClick={() => deleteDoc(d.filename)}>✕</button>
                </span>
              ))}
              <button className="upload-btn" onClick={() => fileRef.current.click()}>📎 Upload PDF</button>
              <input ref={fileRef} type="file" accept=".pdf" style={{ display: "none" }} onChange={uploadPDF} />
            </div>
          )}
        </header>

        <div className="messages">
          {!activeId && (
            <div className="empty-state">
              <div className="empty-icon">✦</div>
              <h2>Welcome to Aldric</h2>
              <p>Select a chat or start a new one to begin.</p>
            </div>
          )}
          {messages.length === 0 && activeId && loadingId !== activeId && (
            <div className="empty-state">
              <div className="empty-icon">✦</div>
              <h2>How can I assist thee?</h2>
              <p>Ask me anything, and I shall impart my ancient wisdom.</p>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`message-row ${msg.role} animate-in`}>
              {msg.role === "bot" && <BotAvatar />}
              <div className="bubble">
                {msg.role === "bot" ? <ReactMarkdown>{msg.text}</ReactMarkdown> : msg.text}
              </div>
              {msg.role === "user" && <UserAvatar />}
            </div>
          ))}

          {activeId && loadingId === activeId && (
            <div className="message-row bot">
              <BotAvatar />
              <div className="bubble typing">
                <span /><span /><span />
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div className="input-area">
          <div className="input-box">
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Message Aldric..."
              rows={1}
            />
            <button onClick={sendMessage} disabled={loadingId === activeId} aria-label="Send">↑</button>
          </div>
          <p className="disclaimer">Aldric may produce inaccurate information.</p>
        </div>
      </main>
    </div>
  );
}
