import { useMemo, useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(
  /\/$/,
  "",
);

export default function App() {
  const [query, setQuery] = useState("yellow soup");
  const [mode, setMode] = useState("text");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const onSubmit = async (evt) => {
    evt.preventDefault();
    setError("");
    if (!query.trim()) {
      setError("Enter a query to search.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, mode, limit: 12 }),
      });

      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail?.detail || "Search failed");
      }

      const data = await response.json();
      setResults(Array.isArray(data.results) ? data.results : []);
    } catch (err) {
      setError(err.message);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header className="hero">
        <div>
          <div className="brand">
            <img className="logo" src="/cocoindex-favicon.png" alt="CocoIndex logo" />
            <h1 className="title">CocoIndex + LanceDB Search</h1>
          </div>
          <p className="subtitle">
            Search for recipes using natural language via text or image embeddings.
          </p>
        </div>
      </header>

      <section className="panel">
        <form className="form" onSubmit={onSubmit}>
          <div className="input-row">
            <input
              className="query-input"
              type="text"
              placeholder="Try: yellow soup, spicy curry, latte art"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <button className="button" type="submit" disabled={loading}>
              {loading ? "Searching..." : "Search"}
            </button>
          </div>

          <div className="controls">
            <div className="segmented">
              <button
                type="button"
                className={`segment ${mode === "text" ? "active" : ""}`}
                onClick={() => setMode("text")}
              >
                Text search
              </button>
              <button
                type="button"
                className={`segment ${mode === "image" ? "active" : ""}`}
                onClick={() => setMode("image")}
              >
                Image search
              </button>
            </div>
            <span className="status">Backend: {API_BASE_URL}</span>
          </div>

          {error && <span className="status" style={{ color: "#dc2626" }}>{error}</span>}
        </form>
      </section>

      <section className="panel">
        {results.length === 0 && !loading ? (
          <div className="empty">No results yet — run a search to see matches.</div>
        ) : (
          <div className="grid">
            {results.map((item) => {
              const imageUrl = item.image_url ? `${API_BASE_URL}${item.image_url}` : null;
              return (
                <article key={item.id} className="card">
                  {imageUrl ? (
                    <img src={imageUrl} alt={item.title || "Recipe image"} className="thumb" />
                  ) : (
                    <div className="thumb" />
                  )}
                  <div className="card-body">
                    <h3 className="card-title">{item.title || "Untitled recipe"}</h3>
                    <div className="meta">
                      {item.category && <span className="pill">{item.category}</span>}
                      {item.is_vegetarian && <span className="pill">Vegetarian</span>}
                      {item.has_nuts && <span className="pill">Contains nuts</span>}
                    </div>
                    {Array.isArray(item.ingredients) && item.ingredients.length > 0 && (
                      <p className="ingredients">
                        {item.ingredients.slice(0, 4).join(" • ")}
                        {item.ingredients.length > 4 ? " ..." : ""}
                      </p>
                    )}
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
